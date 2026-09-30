"""Fixed-length analysis windows over a recording, plus a per-window activity score.

Windows are (start_s, end_s) rows; audio is never cut to disk. Short recordings get
one window, zero-padded by the encoder. The last window is shifted back to end at the
recording's end so the tail is never dropped.
"""

from __future__ import annotations

import subprocess

import numpy as np


def frame(duration_s: float, window_s: float, hop_s: float) -> list[tuple[float, float]]:
    if duration_s <= window_s:
        return [(0.0, window_s)]
    starts = list(np.arange(0.0, duration_s - window_s + 1e-9, hop_s))
    if duration_s - (starts[-1] + window_s) > 1e-6:
        starts.append(duration_s - window_s)
    return [(round(float(s), 3), round(float(s) + window_s, 3)) for s in starts]


def decode(path: str, sr: int) -> np.ndarray:
    """Mono float32 at `sr` via ffmpeg (handles mp3/m4a/wav alike)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def slice_windows(audio: np.ndarray, sr: int, spans: list[tuple[float, float]]) -> np.ndarray:
    n = int(round((spans[0][1] - spans[0][0]) * sr))
    out = np.zeros((len(spans), n), dtype=np.float32)
    for i, (s, _) in enumerate(spans):
        seg = audio[int(round(s * sr)): int(round(s * sr)) + n]
        out[i, : len(seg)] = seg
    return out


def activity_db(audio: np.ndarray, sr: int, spans: list[tuple[float, float]],
                fmin: float = 100.0, fmax: float = 10000.0, n_fft: int = 1024, hop: int = 512) -> np.ndarray:
    """Per window: 95th-percentile band energy minus the recording's noise floor
    (20th percentile of all frames), in dB. High = something is calling."""
    if len(audio) < n_fft:
        return np.zeros(len(spans), dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = spec[:, (freqs >= fmin) & (freqs <= min(fmax, sr / 2))].sum(axis=1)
    db = 10 * np.log10(band + 1e-10)
    floor = np.percentile(db, 20)
    t = np.arange(len(db)) * hop / sr
    out = []
    for s, e in spans:
        w = db[(t >= s) & (t < e)]
        out.append(np.percentile(w, 95) - floor if len(w) else 0.0)
    return np.asarray(out, dtype=np.float32)
