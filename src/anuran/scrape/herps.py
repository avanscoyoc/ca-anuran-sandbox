"""Scrape call recordings and descriptions from californiaherps.com.

Research use only: the site is (c) californiaherps.com, all rights reserved.
Scraped audio lives under data/ (gitignored) and must not be redistributed.
robots.txt asks for Crawl-delay: 10, which CRAWL_DELAY honours.

Outputs:
  data/raw/herps/html/<page>.html          cached pages
  data/raw/herps/audio/<code>/<file>.mp3   recordings
  data/interim/herps_recordings.parquet    one row per recording
  data/interim/herps_species_text.jsonl    description text per species
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import yaml
from bs4 import BeautifulSoup

from anuran.config import INTERIM, ROOT, load_species
from anuran.config import RAW as RAW_ROOT
from anuran.scrape.fetch import Fetcher

BASE = "https://www.californiaherps.com/"
PAGES = urljoin(BASE, "frogs/pages/")
CRAWL_DELAY = 10.0
RAW = RAW_ROOT / "herps"

CALL_SECTION = re.compile(r"^\s*[A-Z][\w'() -]{0,40}\s(?:Calls?|Sounds|Vocalizations)\s*$")
CALL_TYPES = ("advertisement", "release", "chorus", "distress", "territorial", "encounter", "aggressive", "underwater")


def load_overrides(path: Path = ROOT / "configs" / "label_overrides.yaml") -> dict:
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def fetch_page(fetch: Fetcher, url: str) -> str | None:
    """Pages are cached under data/raw/herps/html (a 404 is cached as "404")."""
    cache = RAW / "html" / Path(urlparse(url).path).name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = fetch.get(url)
        cache.write_bytes(r.content if r is not None else b"404")
    text = cache.read_text(encoding="latin-1")
    return None if text == "404" else text


@dataclass
class Recording:
    url: str
    caption: str
    section: str | None = None
    context: str | None = None
    duration_s: float | None = None
    recordist: str | None = None
    call_types: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _norm_url(href: str, page_url: str) -> str:
    # the site mixes http://www..., /sounds/... and relative links
    url = urljoin(page_url, href.strip())
    p = urlparse(url)
    return f"https://www.californiaherps.com{p.path}"


def parse_caption(caption: str, section: str | None) -> dict:
    low = caption.lower()
    m = re.search(r"(\d+(?:\.\d+)?)[\s-]*(second|minute)", low)
    duration = float(m.group(1)) * (60 if m and m.group(2) == "minute" else 1) if m else None
    m = re.search(r"courtesy of ([^.]+)", caption, re.I)
    recordist = _clean(m.group(1)) if m else None
    types = [t for t in CALL_TYPES if t in low]
    if section:
        sec = section.lower()
        types += [t for t in CALL_TYPES if t in sec and t not in types]
    return {"duration_s": duration, "recordist": recordist, "call_types": types or ["unspecified"]}


def parse_sounds_page(html: str, page_url: str) -> tuple[list[Recording], dict]:
    """Return recordings and species-level text from one sounds page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    recs: list[Recording] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        if not a["href"].strip().lower().endswith(".mp3"):
            continue
        url = _norm_url(a["href"], page_url)
        if url in seen:
            continue
        seen.add(url)
        # caption sits in the next table cell of the same row
        td = a.find_parent("td")
        cap_td = td.find_next_sibling("td") if td else None
        caption = _clean(cap_td.get_text(" ")) if cap_td else _clean(a.get_text(" "))
        heading = a.find_previous(string=CALL_SECTION)
        ctx = a.find_previous(string=re.compile(r"^\s*The following", re.I))
        rec = Recording(
            url=url,
            caption=caption,
            section=_clean(heading) if heading else None,
            context=_clean(ctx.find_parent().get_text(" ")) if ctx else None,
        )
        for k, v in parse_caption(caption, rec.section).items():
            setattr(rec, k, v)
        recs.append(rec)

    text = _clean(soup.get_text(" "))
    desc = [
        _clean(td.get_text(" "))
        for td in soup.find_all("td")
        if re.search(r"vocalizations? of|can be described|call (?:is|sounds|consists)", td.get_text(" "), re.I)
        and not td.find("td")  # innermost cell only
    ]
    sonograms = sorted({_norm_url(i["src"], page_url) for i in soup.find_all("img", src=True) if "sonogram" in i["src"].lower()})
    return recs, {"description": list(dict.fromkeys(desc)), "page_text": text, "sonograms": sonograms}


def name_matcher(species: list[dict]):
    """Return f(text) -> species codes named in text (longest name wins, so
    'Baja California Treefrog' is not also read as 'California Treefrog')."""
    names = sorted(
        ((n, sp["code"]) for sp in species for n in [sp["common"], sp["scientific"], *sp.get("aliases", [])]),
        key=lambda x: -len(x[0]),
    )
    pats = [(re.compile(r"\b" + re.escape(n).replace(r"\ ", r"[\s-]+") + r"s?\b", re.I), c) for n, c in names]

    def match(text: str) -> list[str]:
        found = []
        for pat, code in pats:
            text, n = pat.subn(" ", text)
            if n and code not in found:
                found.append(code)
        return found

    return match


def assign_labels(df: pd.DataFrame, species: list[dict], overrides: dict | None = None) -> pd.DataFrame:
    """Pages sometimes host other species' clips (e.g. Sierran Treefrog on the
    P. regilla page). Captions open with "This is a recording of ... <species>",
    so only the first sentence can change the label; species named later are
    secondary (background) labels. Relabels and ambiguities are flagged for
    review, and `overrides` (mp3 filename -> {label, secondary}) wins last."""
    match = name_matcher(species)
    comparison = re.compile(r"\b(compare|similar|same as|identical|unlike|resembl)", re.I)

    def parse(code: str, caption: str) -> tuple[str, str, bool, list[str]]:
        sents = [s for s in re.split(r"(?<=[.!?)])\s+", caption) if not comparison.search(s)]
        first = match(sents[0]) if sents else []
        rest = [c for s in sents[1:] for c in match(s)]
        if not first or code in first:
            label, source, review = code, "page", False
        elif len(first) == 1:
            label, source, review = first[0], "caption", True
        else:
            label, source, review = code, "page", True
        secondary = [c for c in dict.fromkeys(first + rest) if c != label]
        return label, source, review, secondary

    parsed = pd.DataFrame(
        [parse(c, cap) for c, cap in zip(df["code"], df["caption"])],
        columns=["label", "label_source", "needs_review", "secondary_species"],
        index=df.index,
    )
    df = df.drop(columns=parsed.columns, errors="ignore").join(parsed)
    for i, fname in df["url"].map(lambda u: Path(urlparse(u).path).name).items():
        if fname in (overrides or {}):
            o = overrides[fname]
            df.at[i, "label"] = o["label"]
            df.at[i, "secondary_species"] = o.get("secondary", [])
            df.at[i, "label_source"], df.at[i, "needs_review"] = "override", False
    conflict = df.groupby("url")["label"].transform("nunique") > 1
    df["needs_review"] = df["needs_review"] | conflict
    return df


def strip_boilerplate(texts: list[dict], min_pages: int = 4) -> None:
    """Drop sentences repeated on many pages (generic call-type definitions)."""
    split = lambda s: [x.strip() for x in re.split(r"(?<=[.!?\"])\s+", s) if x.strip()]
    counts: dict[str, int] = {}
    for t in texts:
        for s in {s for d in t["description"] for s in split(d)}:
            counts[s] = counts.get(s, 0) + 1
    for t in texts:
        kept = [" ".join(s for s in split(d) if counts[s] < min_pages) for d in t["description"]]
        t["description"] = [d for d in kept if d]


def scrape(delay: float = CRAWL_DELAY, download: bool = True) -> pd.DataFrame:
    fetch = Fetcher(delay)
    species = load_species()
    rows, texts = [], []
    for sp in species:
        if sp.get("vocal", True) is False:
            print(f"[skip] {sp['common']}: non-vocal")
            continue
        page_url = urljoin(PAGES, f"{sp['slug']}.sounds.html")
        html = fetch_page(fetch, page_url)
        recs, info = parse_sounds_page(html, page_url) if html else ([], {"description": [], "page_text": "", "sonograms": []})
        for href in sp.get("index_audio", []):
            url = _norm_url(href, BASE)
            if url not in {r.url for r in recs}:
                recs.append(Recording(url=url, caption="clip from frogscalls.html index", section="index"))
        print(f"[{sp['code']}] {sp['common']}: page={'yes' if html else 'no'} recordings={len(recs)}")
        texts.append({"code": sp["code"], "scientific": sp["scientific"], "page_url": page_url if html else None, **info})
        for i, rec in enumerate(recs):
            dest = RAW / "audio" / sp["code"] / Path(urlparse(rec.url).path).name
            ok = fetch.file(rec.url, dest) if download else dest.exists()
            rows.append({
                "recording_id": f"herps_{sp['code']}_{i:02d}",
                "code": sp["code"],
                "scientific": sp["scientific"],
                "source": "californiaherps",
                "page_url": page_url if html else None,
                "path": str(dest.relative_to(ROOT)) if ok else None,
                **asdict(rec),
            })
    df = assign_labels(pd.DataFrame(rows), species, load_overrides())
    # the same mp3 can sit on several pages; keep one row so it can't straddle splits
    df = df.drop_duplicates("url", keep="first").reset_index(drop=True)
    strip_boilerplate(texts)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM / "herps_recordings.parquet", index=False)
    with open(INTERIM / "herps_species_text.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=CRAWL_DELAY, help="seconds between requests (robots.txt: 10)")
    ap.add_argument("--no-download", action="store_true", help="parse pages only, skip mp3 downloads")
    args = ap.parse_args()
    df = scrape(args.delay, download=not args.no_download)
    print(f"\n{len(df)} recordings, {df['path'].notna().sum()} downloaded -> {INTERIM / 'herps_recordings.parquet'}")


if __name__ == "__main__":
    main()
