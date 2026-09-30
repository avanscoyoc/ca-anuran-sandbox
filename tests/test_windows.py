import numpy as np

from anuran.data.windows import activity_db, frame, slice_windows
from anuran.features.encoders import class_label_indices


def test_frame_short_clip_is_one_padded_window():
    assert frame(0.3, 5.0, 2.5) == [(0.0, 5.0)]


def test_frame_covers_recording_end():
    spans = frame(12.0, 5.0, 2.5)
    assert spans[0] == (0.0, 5.0) and spans[-1] == (7.0, 12.0)
    assert all(b[0] - a[0] <= 2.5 for a, b in zip(spans, spans[1:]))


def test_slice_windows_pads():
    x = slice_windows(np.ones(100, np.float32), 100, [(0.0, 5.0)])
    assert x.shape == (1, 500) and x[0, :100].sum() == 100 and x[0, 100:].sum() == 0


def test_activity_detects_call():
    sr = 16000
    rng = np.random.default_rng(0)
    audio = 0.001 * rng.standard_normal(sr * 10).astype(np.float32)
    t = np.arange(sr) / sr
    audio[6 * sr: 7 * sr] += 0.5 * np.sin(2 * np.pi * 1000 * t)  # tone in 5-10 s window
    a = activity_db(audio, sr, [(0.0, 5.0), (5.0, 10.0)])
    assert a[1] > 30 and a[0] < 6


def test_class_label_indices_synonyms_and_groups():
    labels = ["Rana catesbeianus", "Pseudacris regilla", "Anaxyrus boreas", "Turdus migratorius"]
    idx = class_label_indices(labels)
    assert idx == {"LICA": [0], "PACH": [1], "WETO": [2]}


def test_label_windows_rules():
    import pandas as pd

    from anuran.data.window_labels import label_windows

    man = pd.DataFrame({
        "recording_id": ["a", "b", "c"], "acoustic_class": ["LICA", "ANEX", "LICA"], "fold": [0, 1, 2],
        "source": ["inat"] * 3, "secondary_species": [["LIPI"], [], []],
    })
    win = pd.DataFrame({
        "recording_id": ["a", "a", "a", "b", "b", "c"],
        "start_s": [0, 2.5, 5, 0, 2.5, 0], "end_s": [5, 7.5, 10, 5, 7.5, 5],
        "activity_db": [25, 25, 2, 3, 15, 1],
        "zs_LICA": [9.0, 2.0, 9.5, 0, 0, -5.0],  # ANEX unknown to encoder
    })
    out = label_windows(win, man)
    assert out["state"].tolist() == ["pos", "uncertain", "pos", "uncertain", "pos", "pos"]
    # a: window 1 active but far below best score; window 2 quiet but it is the best-scoring window (kept)
    # b: no encoder score -> activity decides; c: single window -> pos
    assert out.loc[0, "masked_classes"] == ["LIPI"] and out.loc[3, "masked_classes"] == []
