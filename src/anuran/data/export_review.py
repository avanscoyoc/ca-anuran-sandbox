"""Export windows as a clip table for jupyter_bioacoustic (strong-label experiment).

  data/processed/review/<encoder>_windows.parquet

Columns: audio_path (absolute), start_time, end_time, plus display columns
(recording_id, acoustic_class, common_name, weak_label, activity_db, baseline scores,
source, fold). Example:

    from jupyter_bioacoustic import BioacousticAnnotator
    BioacousticAnnotator(
        data="data/processed/review/birdnet_v24_windows.parquet",
        audio="audio_path",  # per-row audio column
        display_columns=["common_name", "weak_label", "score_own", "top_class", "start_time"],
        output="data/processed/review/strong_labels.parquet",
    ).open()

Check the annotator's README for current keys (e.g. audio_column) before use.
"""

from __future__ import annotations

import argparse

import pandas as pd

from anuran.config import ROOT, load_groups, vocal_species
from anuran.data.manifest import PROCESSED
from anuran.train_baseline import RESULTS


def export(encoder: str = "birdnet_v24") -> pd.DataFrame:
    d = PROCESSED / "embeddings" / encoder
    lab = pd.read_parquet(d / "window_labels.parquet")
    man = pd.read_parquet(PROCESSED / "manifest.parquet").set_index("recording_id")
    names = {s["code"]: s["common"] for s in vocal_species()} | {g["code"]: g["common"] for g in load_groups()}
    out = pd.DataFrame({
        "audio_path": lab["recording_id"].map(man["path"]).map(lambda p: str(ROOT / p)),
        "start_time": lab["start_s"],
        "end_time": lab["end_s"],
        "recording_id": lab["recording_id"],
        "acoustic_class": lab["acoustic_class"],
        "common_name": lab["acoustic_class"].map(names),
        "weak_label": lab["state"],
        "activity_db": lab["activity_db"].round(1),
        "source": lab["source"],
        "fold": lab["fold"],
    })
    oof = RESULTS / f"{encoder}_baseline" / "oof_windows.parquet"
    if oof.exists():
        o = pd.read_parquet(oof)
        assert (o["recording_id"].values == out["recording_id"].values).all()
        out[["score_own", "top_class", "top_score"]] = o[["score_own", "top_class", "top_score"]].values
    dest = PROCESSED / "review" / f"{encoder}_windows.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"{len(out)} windows -> {dest}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="birdnet_v24")
    export(ap.parse_args().encoder)
