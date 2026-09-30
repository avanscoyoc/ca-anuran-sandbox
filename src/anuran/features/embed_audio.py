"""Embed every manifest recording in fixed windows with a frozen encoder.

Per-recording cache (resumable; new manifest rows only cost their own audio):
  data/processed/embeddings/<encoder>/cache/<recording_id>.npz
Assembled outputs, row-aligned:
  data/processed/embeddings/<encoder>/windows.parquet   recording_id, start_s, end_s, activity_db, zs_<class>...
  data/processed/embeddings/<encoder>/embeddings.npy    (n_windows, dim) float16
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from anuran.config import ROOT
from anuran.data.manifest import PROCESSED
from anuran.data.windows import activity_db, decode, frame, slice_windows
from anuran.features.encoders import ENCODERS

EMB_DIR = PROCESSED / "embeddings"
BATCH = 16


def embed_recording(enc, path: str, duration_s: float, hop_s: float) -> dict:
    audio = decode(str(ROOT / path), enc.sample_rate)
    spans = frame(max(duration_s, len(audio) / enc.sample_rate), enc.window_s, hop_s)
    x = slice_windows(audio, enc.sample_rate, spans)
    embs, scores = [], {}
    batch = getattr(enc, "batch_size", BATCH)
    for i in range(0, len(x), batch):
        chunk = x[i: i + batch]
        n = len(chunk)
        if n < batch:  # fixed batch shape: avoids re-allocating the model per call
            chunk = np.concatenate([chunk, np.zeros((batch - n, chunk.shape[1]), chunk.dtype)])
        e, s = enc(chunk)
        embs.append(e[:n])
        for c, v in s.items():
            scores.setdefault(c, []).append(v[:n])
    return {
        "start_s": np.array([s for s, _ in spans], dtype=np.float32),
        "end_s": np.array([e for _, e in spans], dtype=np.float32),
        "activity_db": activity_db(audio, enc.sample_rate, spans),
        "embedding": np.concatenate(embs).astype(np.float16),
        **{f"zs_{c}": np.concatenate(v).astype(np.float32) for c, v in scores.items()},
    }


def run(encoder: str, hop_s: float | None = None, limit: int | None = None) -> None:
    manifest = pd.read_parquet(PROCESSED / "manifest.parquet")
    enc = ENCODERS[encoder]()
    hop_s = hop_s or enc.window_s / 2
    cache = EMB_DIR / encoder / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in manifest.itertuples() if not (cache / f"{r.recording_id}.npz").exists()]
    print(f"{encoder}: {len(manifest) - len(todo)} cached, {len(todo)} to embed "
          f"(known classes: {sorted(enc.class_indices)})", flush=True)
    todo = todo[:limit]
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        out = embed_recording(enc, r.path, r.duration_s, hop_s)
        tmp = cache / f"{r.recording_id}.part.npz"
        np.savez(tmp, **out)
        tmp.rename(cache / f"{r.recording_id}.npz")
        if k % 50 == 0 or k == len(todo):
            rate = (time.time() - t0) / k
            print(f"  {k}/{len(todo)}  {rate:.2f} s/recording, ~{rate * (len(todo) - k) / 60:.0f} min left", flush=True)
    assemble(encoder, manifest)


def assemble(encoder: str, manifest: pd.DataFrame) -> None:
    """Concatenate cached recordings (manifest order) into windows.parquet + embeddings.npy."""
    cache = EMB_DIR / encoder / "cache"
    rows, embs = [], []
    for rid in manifest["recording_id"]:
        f = cache / f"{rid}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        embs.append(z["embedding"])
        cols = {k: z[k] for k in z.files if k != "embedding"}
        rows.append(pd.DataFrame({"recording_id": rid, **cols}))
    win = pd.concat(rows, ignore_index=True)
    np.save(EMB_DIR / encoder / "embeddings.npy", np.concatenate(embs))
    win.to_parquet(EMB_DIR / encoder / "windows.parquet", index=False)
    print(f"assembled {len(win)} windows from {len(rows)} recordings -> {EMB_DIR / encoder}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="birdnet_v24", choices=sorted(ENCODERS))
    ap.add_argument("--hop", type=float, default=None, help="seconds (default: half a window)")
    ap.add_argument("--limit", type=int, default=None, help="embed at most N new recordings (smoke test)")
    args = ap.parse_args()
    run(args.encoder, args.hop, args.limit)


if __name__ == "__main__":
    main()
