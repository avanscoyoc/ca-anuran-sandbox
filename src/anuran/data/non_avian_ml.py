"""CDFW ARU clips -> data/interim/cdfw_recordings.parquet (a manifest source like herps/inat/xc).

data/raw/cdfw_aru/audio/<CODE>/<name>_<clip_number>.wav, 3 s each, labels from the folder
(confirmed correct by the user 2026-10-01; PSRE maps to the PACH group like everywhere else).

No site/date/recorder metadata came with the clips, and clip numbers are contiguous per
species, so there are no ID gaps to split on. Neighbouring numbers do share recording
conditions (noise-floor spectra of adjacent clips are much closer than random pairs and
decorrelate over ~20-100 clips), so contiguous blocks of BLOCK clips are the proxy for a
site/night group. Results grouped this way are "proxy-grouped"; swap in real site groups
if CDFW metadata turns up.
"""

from __future__ import annotations

import re

import pandas as pd
import soundfile as sf

from anuran.config import INTERIM, RAW, ROOT

CDFW_DIR = RAW / "cdfw_aru" / "audio"
BLOCK = 50  # clips per proxy group
CLIP_NUMBER = re.compile(r"_(\d+)\.wav$")


def proxy_blocks(clip_numbers: pd.Series, block: int = BLOCK) -> pd.Series:
    """Block index by rank of clip number (contiguous runs of `block` clips)."""
    return (clip_numbers.rank(method="first").astype(int) - 1) // block


def build() -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in CDFW_DIR.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.wav")):
            info = sf.info(f)
            rows.append({
                "recording_id": f"cdfw_{d.name}_{CLIP_NUMBER.search(f.name).group(1)}",
                "source": "cdfw",
                "source_id": f.name,
                "label": d.name,
                "clip_number": int(CLIP_NUMBER.search(f.name).group(1)),
                "recordist": "cdfw",
                "license": "CDFW (not redistributable)",
                "path": str(f.relative_to(ROOT)),
                "sample_rate": info.samplerate,
                "secondary_species": [],
                "call_types": [],
            })
    df = pd.DataFrame(rows)
    df["block"] = df.groupby("label")["clip_number"].transform(proxy_blocks)
    df.to_parquet(INTERIM / "cdfw_recordings.parquet", index=False)
    return df


if __name__ == "__main__":
    out = build()
    print(out.groupby("label").agg(clips=("recording_id", "size"), blocks=("block", "nunique"),
                                   sample_rate=("sample_rate", "first")).to_string())
    print(f"-> {INTERIM / 'cdfw_recordings.parquet'}")
