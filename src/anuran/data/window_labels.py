"""Weak window labels from recording-level labels.

A recording label says the class calls *somewhere*. Per window:
  pos        single-window recording; or activity >= MIN_ACTIVITY_DB and, if the encoder
             knows the class, its score is within ZS_MARGIN of the recording's best window
  uncertain  everything else: excluded from training, never used as a negative
Every recording keeps at least one pos window (its best by encoder score, else activity).
Secondary species (mixed choruses) are masked for the whole recording: neither pos nor neg.

The relative encoder criterion (vs. the recording's own best window) avoids keeping
only windows the pretrained model already finds easy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anuran.config import acoustic_group
from anuran.data.manifest import PROCESSED

MIN_ACTIVITY_DB = 10.0
ZS_MARGIN = 4.0  # logits


def label_windows(windows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    groups = acoustic_group()
    m = manifest.set_index("recording_id")
    w = windows[["recording_id", "start_s", "end_s", "activity_db"]].copy()
    w["acoustic_class"] = w["recording_id"].map(m["acoustic_class"])
    w["fold"] = w["recording_id"].map(m["fold"])
    w["source"] = w["recording_id"].map(m["source"])

    zs = pd.Series(np.nan, index=w.index)
    for cls in w["acoustic_class"].unique():
        col = f"zs_{cls}"
        if col in windows:
            rows = w["acoustic_class"] == cls
            zs[rows] = windows.loc[rows, col]
    w["zs_own"] = zs

    g = w.groupby("recording_id")
    n = g["start_s"].transform("size")
    best_zs = g["zs_own"].transform("max")
    active = w["activity_db"] >= MIN_ACTIVITY_DB
    near_best = zs.isna() | (zs >= best_zs - ZS_MARGIN)
    pos = (n == 1) | (active & near_best)

    # guarantee one pos window per recording
    rank_key = w["zs_own"].fillna(w["activity_db"])
    best_idx = rank_key.groupby(w["recording_id"]).idxmax()
    pos[best_idx.values] = True
    w["state"] = np.where(pos, "pos", "uncertain")

    sec = m["secondary_species"].map(lambda s: sorted({groups.get(c, c) for c in s}))
    w["masked_classes"] = w["recording_id"].map(sec)
    w["masked_classes"] = [
        [c for c in mc if c != own] for mc, own in zip(w["masked_classes"], w["acoustic_class"])
    ]
    return w


def run(encoder: str) -> pd.DataFrame:
    d = PROCESSED / "embeddings" / encoder
    w = label_windows(pd.read_parquet(d / "windows.parquet"), pd.read_parquet(PROCESSED / "manifest.parquet"))
    w.to_parquet(d / "window_labels.parquet", index=False)
    share = w.groupby("acoustic_class")["state"].apply(lambda s: (s == "pos").mean()).round(2)
    print(f"{(w['state'] == 'pos').mean():.0%} of {len(w)} windows pos; per class:\n{share.to_string()}")
    return w


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="birdnet_v24")
    run(ap.parse_args().encoder)
