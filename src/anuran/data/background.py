"""Verified frog-free background clips -> negative windows for the detector head.

Keeps clips in data/raw/background/ that labels/background_verification.csv marks
frog_present == False (last answer wins), drops clips below MIN_SR (BirdNET uses up to
15 kHz, so 32 and 48 kHz clips look alike; lower rates would be a shortcut), and assigns
folds stratified by category (clips have no site info, so this split is looser than
the frog recordings' recordist/site grouping).

Outputs (per encoder):
  data/processed/embeddings/<encoder>/background_windows.parquet
  data/processed/embeddings/<encoder>/background_embeddings.npy
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold

from anuran.config import RAW, ROOT
from anuran.data.manifest import N_FOLDS, PROCESSED

BG_DIR = RAW / "background"
VERIFICATION = ROOT / "labels" / "background_verification.csv"
MIN_SR = 32000


def clips(seed: int = 0) -> pd.DataFrame:
    v = pd.read_csv(VERIFICATION).drop_duplicates("clip_id", keep="last")
    frog_free = set(v.loc[~v["frog_present"].astype(bool), "clip_id"])
    rows = []
    for f in sorted(BG_DIR.glob("*.wav")):
        if f.stem not in frog_free:
            continue  # frog present, or never reviewed
        info = sf.info(f)
        rows.append({"clip_id": f.stem, "path": str(f.relative_to(ROOT)), "category": f.stem.rsplit("_", 1)[0],
                     "sample_rate": info.samplerate, "duration_s": info.duration})
    df = pd.DataFrame(rows)
    df = df[df["sample_rate"] >= MIN_SR].reset_index(drop=True)
    cv = StratifiedKFold(n_splits=min(N_FOLDS, len(df)), shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, test) in enumerate(cv.split(df, df["category"])):
        df.loc[test, "fold"] = k
    return df


def embed(encoder: str) -> None:
    from anuran.features.embed_audio import embed_recording
    from anuran.features.encoders import ENCODERS

    df = clips()
    enc = ENCODERS[encoder]()
    rows, embs = [], []
    for r in df.itertuples():
        out = embed_recording(enc, r.path, r.duration_s, enc.window_s / 2)
        embs.append(out["embedding"])
        for i in range(len(out["start_s"])):
            rows.append({"clip_id": r.clip_id, "category": r.category, "fold": r.fold,
                         "start_s": float(out["start_s"][i]), "end_s": float(out["end_s"][i])})
    d = PROCESSED / "embeddings" / encoder
    pd.DataFrame(rows).to_parquet(d / "background_windows.parquet", index=False)
    np.save(d / "background_embeddings.npy", np.concatenate(embs))
    print(f"{len(df)} frog-free clips (>= {MIN_SR} Hz), {len(rows)} windows -> {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="birdnet_v24")
    embed(ap.parse_args().encoder)
