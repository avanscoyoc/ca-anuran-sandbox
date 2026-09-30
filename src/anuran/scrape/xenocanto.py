"""Download xeno-canto frog recordings for the configured taxa (API v3).

Requires a free API key (https://xeno-canto.org/account) in $XC_API_KEY.
Keeps quality A/B by default. Names are matched with genus synonyms
(Lithobates/Rana, Anaxyrus/Incilius/Bufo); P. regilla s.l. and A. boreas without
subspecies map to their acoustic group, as in the iNat scraper.

Outputs:
  data/raw/xc/audio/<label>/XC<id>.<ext>
  data/interim/xc_recordings.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from anuran.config import INTERIM, RAW, ROOT, vocal_species
from anuran.scrape.fetch import Fetcher

API = "https://xeno-canto.org/api/3/recordings"
SYNONYMS = {"Lithobates": ["Rana"], "Rana": ["Lithobates"], "Anaxyrus": ["Bufo"], "Incilius": ["Bufo"]}
# binomials whose label depends on subspecies, or that XC may use sensu lato
GROUP_BINOMIALS = {"pseudacris regilla": "PACH", "anaxyrus boreas": "WETO", "bufo boreas": "WETO"}


def _binomials() -> dict[str, tuple[str, str]]:
    """lowercase 'genus species[ ssp]' -> (label, rank), including genus synonyms."""
    out = {}
    for s in vocal_species():
        parts = s["scientific"].split()
        for gen in [parts[0], *SYNONYMS.get(parts[0], [])]:
            out[" ".join([gen, *parts[1:]]).lower()] = (s["code"], "species")
            if len(parts) == 3 and parts[1] == parts[2]:  # nominate ssp, e.g. A. woodhousii woodhousii
                out.setdefault(f"{gen} {parts[1]}".lower(), (s["code"], "species"))
    for name, group in GROUP_BINOMIALS.items():
        out[name] = (group, "group")
    return out


NAMES = _binomials()


def classify(gen: str, sp: str, ssp: str = "") -> tuple[str, str] | None:
    key = f"{gen} {sp}".lower()
    if ssp and f"{key} {ssp}".lower() in NAMES:
        return NAMES[f"{key} {ssp}".lower()]
    return NAMES.get(key)


def recording_row(rec: dict) -> dict | None:
    hit = classify(rec.get("gen", ""), rec.get("sp", ""), rec.get("ssp", ""))
    if hit is None:
        return None
    also = [classify(*a.split()[:2]) for a in rec.get("also") or [] if len(a.split()) >= 2]
    lat, lon = rec.get("lat"), rec.get("lon", rec.get("lng"))
    return {
        "recording_id": f"xc_{rec['id']}",
        "source": "xc",
        "source_id": str(rec["id"]),
        "label": hit[0],
        "label_rank": hit[1],
        "taxon_name": " ".join(x for x in (rec.get("gen"), rec.get("sp"), rec.get("ssp")) if x),
        "secondary_species": sorted({a[0] for a in also if a and a[0] != hit[0]}),
        "recordist": rec.get("rec"),
        "license": rec.get("lic"),
        "quality": rec.get("q"),
        "call_types": [t.strip() for t in (rec.get("type") or "").split(",") if t.strip()],
        "lat": float(lat) if lat not in (None, "") else None,
        "lon": float(lon) if lon not in (None, "") else None,
        "date": rec.get("date"),
        "place": ", ".join(x for x in (rec.get("loc"), rec.get("cnt")) if x),
        "caption": rec.get("rmk") or "",
        "page_url": f"https://xeno-canto.org/{rec['id']}",
        "url": rec.get("file"),
        "ext": Path(rec.get("file-name") or "x.mp3").suffix.lower() or ".mp3",
    }


def search(fetch: Fetcher, key: str, gen: str, sp: str) -> list[dict]:
    recs, page = [], 1
    while True:
        d = fetch.json(API, {"query": f'grp:frogs gen:{gen} sp:{sp}', "key": key, "page": page, "per_page": 500})
        recs += d.get("recordings", [])
        if page >= int(d.get("numPages", 0) or 0):
            return recs
        page += 1


def scrape(key: str, qualities: set[str], delay: float, download: bool = True) -> pd.DataFrame:
    fetch = Fetcher(delay)
    queries = sorted({tuple(n.split()[:2]) for n in NAMES})
    rows, seen = [], set()
    for gen, sp in queries:
        recs = [r for r in search(fetch, key, gen.capitalize(), sp) if r["id"] not in seen]
        seen.update(r["id"] for r in recs)
        new = [row for r in recs if r.get("q") in qualities and (row := recording_row(r))]
        print(f"[xc] {gen} {sp}: {len(recs)} recordings, {len(new)} kept (q in {sorted(qualities)})", flush=True)
        rows += new
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    paths = []
    for r in df.itertuples():
        dest = RAW / "xc" / "audio" / r.label / f"XC{r.source_id}{r.ext}"
        ok = bool(r.url) and (fetch.file(r.url, dest) if download else dest.exists())
        paths.append(str(dest.relative_to(ROOT)) if ok else None)
    df["path"] = paths
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.drop(columns="ext").to_parquet(INTERIM / "xc_recordings.parquet", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality", default="AB", help="xeno-canto quality ratings to keep (default AB)")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    key = os.environ.get("XC_API_KEY")
    if not key:
        raise SystemExit("Set XC_API_KEY (free key: https://xeno-canto.org/account)")
    df = scrape(key, set(args.quality.upper()), args.delay, download=not args.no_download)
    print(f"\n{len(df)} recordings -> {INTERIM / 'xc_recordings.parquet'}")


if __name__ == "__main__":
    main()
