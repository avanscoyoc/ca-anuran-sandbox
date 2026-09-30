"""Download research-grade iNaturalist sound observations for the configured taxa.

Filters: quality_grade=research, captive=false, CC-licensed sounds only.
Two pulls per taxon: the oldest --limit observations, plus up to --recent observed after
Perch 2.0's training-data download (leakage-safe evaluation).
Species/subspecies taxa are queried before group taxa (e.g. P. regilla s.l.), so an
observation identified finer than the group keeps its finer label.

Outputs:
  data/raw/inat/audio/<label>/<sound_id>.<ext>
  data/interim/inat_recordings.parquet
"""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

import pandas as pd

from anuran.config import INTERIM, RAW, ROOT, load_groups, vocal_species
from anuran.scrape.fetch import Fetcher

API = "https://api.inaturalist.org/v1/observations"
# Perch 2.0's XC/iNat training data was downloaded March 2025 (Perch 2.0 paper, arXiv:2508.04665)
PERCH_CUTOFF = "2025-04-01"
EXT = {"audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav"}


def targets() -> list[tuple[str, str, int]]:
    """(label, label_rank, taxon_id), finest ranks first."""
    sp = [(s["code"], "species", t) for s in vocal_species() for t in s.get("inat", [])]
    gr = [(g["code"], "group", t) for g in load_groups() for t in g.get("inat", [])]
    return sp + gr


def observations(fetch: Fetcher, taxon_id: int, limit: int, observed_after: str | None = None) -> list[dict]:
    out, id_above = [], 0
    extra = {"d1": observed_after} if observed_after else {}
    while len(out) < limit:
        page = fetch.json(API, {
            "taxon_id": taxon_id, "sounds": "true", "quality_grade": "research", "captive": "false",
            "order_by": "id", "order": "asc", "id_above": id_above, "per_page": 200, **extra,
        }).get("results", [])
        if not page:
            break
        out += page
        id_above = page[-1]["id"]
    return out[:limit]


def sound_rows(obs: dict, label: str, label_rank: str) -> list[dict]:
    lat, lon = (float(x) for x in obs["location"].split(",")) if obs.get("location") else (None, None)
    rows = []
    for snd in obs.get("sounds") or []:
        if not snd.get("license_code") or not snd.get("file_url"):
            continue  # all-rights-reserved or embedded (e.g. SoundCloud) sounds
        ctype = snd.get("file_content_type") or ""
        ext = EXT.get(ctype) or mimetypes.guess_extension(ctype) or Path(snd["file_url"].split("?")[0]).suffix
        rows.append({
            "recording_id": f"inat_{snd['id']}",
            "source": "inat",
            "source_id": str(obs["id"]),
            "label": label,
            "label_rank": label_rank,
            "taxon_name": obs["taxon"]["name"],
            "recordist": obs["user"]["login"],
            "license": snd["license_code"],
            "lat": lat, "lon": lon,
            "positional_accuracy_m": obs.get("positional_accuracy"),
            "date": obs.get("observed_on"),
            "place": obs.get("place_guess"),
            "caption": obs.get("description") or "",
            "page_url": obs.get("uri"),
            "url": snd["file_url"],
            "ext": ext,
        })
    return rows


def scrape(limit: int, delay: float, download: bool = True, recent_limit: int = 0) -> pd.DataFrame:
    fetch = Fetcher(delay)
    rows, seen = [], set()
    for label, rank, taxon in targets():
        obs = observations(fetch, taxon, limit)
        if recent_limit:  # observed after Perch 2.0's training download: leakage-safe test data
            obs += observations(fetch, taxon, recent_limit, observed_after=PERCH_CUTOFF)
        obs = [o for o in {o["id"]: o for o in obs}.values() if o["id"] not in seen]
        seen.update(o["id"] for o in obs)
        new = [r for o in obs for r in sound_rows(o, label, rank)]
        print(f"[{label}] taxon {taxon}: {len(obs)} observations, {len(new)} licensed sounds", flush=True)
        rows += new
    df = pd.DataFrame(rows)
    paths = []
    for r in df.itertuples():
        dest = RAW / "inat" / "audio" / r.label / f"{r.recording_id.removeprefix('inat_')}{r.ext}"
        ok = fetch.file(r.url, dest) if download else dest.exists()
        paths.append(str(dest.relative_to(ROOT)) if ok else None)
    df["path"] = paths
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.drop(columns="ext").to_parquet(INTERIM / "inat_recordings.parquet", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=250, help="max observations per taxon")
    ap.add_argument("--recent", type=int, default=100,
                    help=f"extra observations per taxon observed on/after {PERCH_CUTOFF} (0 = off)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (iNat asks <=1 req/s)")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    df = scrape(args.limit, args.delay, download=not args.no_download, recent_limit=args.recent)
    print(f"\n{len(df)} sounds, {df['path'].notna().sum()} downloaded -> {INTERIM / 'inat_recordings.parquet'}")


if __name__ == "__main__":
    main()
