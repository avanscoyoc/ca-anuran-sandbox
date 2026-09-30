import numpy as np
import pandas as pd
import torch

from anuran.models.head import fit, masked_bce, predict


def test_masked_entries_contribute_nothing():
    logits = torch.tensor([[5.0, -5.0]])
    y = torch.tensor([[0.0, 0.0]])
    pw = torch.ones(2)
    full = masked_bce(logits, y, torch.tensor([[1.0, 1.0]]), pw)
    only_second = masked_bce(logits, y, torch.tensor([[0.0, 1.0]]), pw)
    assert only_second < 0.01 < full  # the wrong first logit is ignored when masked


def test_head_overfits_separable_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16)).astype(np.float32)
    cls = (X[:, 0] > 0).astype(int)
    Y = np.eye(2, dtype=np.float32)[cls]
    Y[:50] = 0  # background rows: all-zero targets
    X[:50] += 5
    model = fit(X, Y, np.ones_like(Y), seed=0, epochs=60, d_proj=32, dropout=0.0, batch_size=32)
    P = predict(model, X)
    assert (P[50:].argmax(1) == cls[50:]).mean() > 0.95 and P[:50].max(1).mean() < 0.3


def test_background_filter_drops_frog_clips(tmp_path, monkeypatch):
    import soundfile as sf

    import anuran.data.background as bgmod

    ids = [f"wind_wind_{i}" for i in range(7)] + [f"gun_gun_{i}" for i in range(5)]
    for cid in ids:
        sf.write(tmp_path / f"{cid}.wav", np.zeros(48000 * 3, np.float32), 48000)
    sf.write(tmp_path / "gun_gun_4.wav", np.zeros(24000 * 3, np.float32), 24000)  # below MIN_SR
    csv = tmp_path / "v.csv"
    answers = [(c, False) for c in ids] + [("wind_wind_2", True)]  # re-answered: frog present
    pd.DataFrame(answers, columns=["clip_id", "frog_present"]).to_csv(csv, index=False)
    monkeypatch.setattr(bgmod, "BG_DIR", tmp_path)
    monkeypatch.setattr(bgmod, "VERIFICATION", csv)
    monkeypatch.setattr(bgmod, "ROOT", tmp_path)
    got = bgmod.clips()
    assert "wind_wind_2" not in set(got["clip_id"])  # last answer (frog) wins
    assert "gun_gun_4" not in set(got["clip_id"])    # 24 kHz dropped
    assert len(got) == 10 and set(got["fold"]) == set(range(5))
