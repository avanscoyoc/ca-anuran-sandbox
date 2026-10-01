import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from anuran.data.cdfw import proxy_blocks
from anuran.eval import metrics, splits


def test_ap_boot_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.2
    s = np.round(rng.random(300), 1)  # many ties
    w = rng.integers(0, 4, 300).astype(float)
    got = metrics.ap_boot(y, s, np.stack([np.ones(300), w]))
    assert got[0] == pytest.approx(average_precision_score(y, s))
    assert got[1] == pytest.approx(average_precision_score(y, s, sample_weight=w))


def test_ap_boot_nan_without_positives():
    assert np.isnan(metrics.ap(np.zeros(5, bool), np.arange(5.0)))


def test_bootstrap_weights_resample_whole_groups():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    W = metrics.bootstrap_weights(groups, 200, seed=1)
    assert W.shape == (200, 6)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 3] == W[:, 5]).all()
    assert (W[:, [0, 2, 3]].sum(1) == 3).all()  # 3 groups drawn per resample


def test_recall_at_precision_and_review_effort():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 1], bool)
    s = np.arange(10, 0, -1, dtype=float)
    r, t = metrics.recall_at_precision(y, s, 0.75)
    assert r == pytest.approx(0.75) and t == 7.0  # top 4: 3 hits
    assert metrics.review_effort(y, s, 0.75) == pytest.approx(0.4)
    assert metrics.review_effort(y, s, 1.0) == pytest.approx(1.0)


def _toy_units(n_per=20):
    rng = np.random.default_rng(0)
    rows = []
    for domain in ("focal", "aru"):
        for cls in ("X", "Y"):
            for i in range(n_per):
                rows.append({"unit_id": f"{domain}{cls}{i}", "kind": "frog", "acoustic_class": cls,
                             "split_group": f"{domain}{cls}{i // 4}", "domain": domain, "fold": i % 5})
    for i in range(30):
        rows.append({"unit_id": f"bg{i}", "kind": "background", "acoustic_class": None,
                     "split_group": f"bg{i}", "domain": "aru", "fold": i % 5})
    u = pd.DataFrame(rows).set_index("unit_id")
    u = u.assign(locked=False, masked=[[] for _ in range(len(u))], leakage_safe=True, medium="air")
    s = pd.DataFrame({c: (u["acoustic_class"] == c) * 0.5 + rng.random(len(u)) * 0.6 for c in ("X", "Y")}, index=u.index)
    return u, s


def test_paired_delta_of_run_with_itself_is_zero():
    u, s = _toy_units()
    d = metrics.paired_delta(u, s, s.copy(), n_boot=100)
    assert (d["delta"] == 0).all() and not d["significant"].any()
    assert not metrics.decide(d)[0]


def test_paired_delta_detects_a_clear_improvement():
    u, s = _toy_units()
    better = s.copy()
    for c in ("X", "Y"):
        better[c] = (u["acoustic_class"] == c) * 1.0 + s[c] * 0.1
    d = metrics.paired_delta(u, s, better, n_boot=200)
    assert d.loc["B_macro_ap", "delta"] > 0 and d.loc["B_macro_ap", "significant"]
    assert metrics.decide(d)[0]


def test_summarize_reports_both_tiers():
    u, s = _toy_units()
    res = metrics.summarize(u, s, n_boot=50)
    assert {"A_macro_ap", "B_macro_ap", "B_detection_ap"} <= set(res["metrics"])
    lo, hi = res["metrics"]["B_macro_ap"]["ci95"]
    assert lo <= res["metrics"]["B_macro_ap"]["value"] <= hi + 1e-9
    assert set(res["per_class_B"].loc[["X", "Y"], "clips"]) == {20}


def test_proxy_blocks_are_contiguous_runs():
    b = proxy_blocks(pd.Series([105, 101, 103, 102, 104, 100]), block=2)
    assert b.tolist() == [2, 0, 1, 1, 2, 0]


def test_cdfw_folds_contiguous_and_last_blocks_locked():
    man = pd.DataFrame({
        "recording_id": [f"c{i}" for i in range(100)], "source": "cdfw", "acoustic_class": "RABO",
        "split_group": [f"cdfw:RABO:b{i // 10:02d}" for i in range(100)],
    })
    c = splits.cdfw_units(man, n_folds=5, lock_frac=0.2)
    assert c.loc[c["locked"], "split_group"].unique().tolist() == ["cdfw:RABO:b08", "cdfw:RABO:b09"]
    assert (c.loc[c["locked"], "fold"] == -1).all()
    dev = c[~c["locked"]]
    assert dev["fold"].is_monotonic_increasing and set(dev["fold"]) == set(range(5))
    assert (dev.groupby("split_group")["fold"].nunique() == 1).all()


def test_splits_hash_catches_tampering(tmp_path, monkeypatch):
    units = pd.DataFrame({"unit_id": ["a", "b"], "kind": "frog", "source": "inat", "acoustic_class": "X",
                          "split_group": ["g1", "g2"], "domain": "focal", "fold": [0, 1], "locked": False})
    monkeypatch.setattr(splits, "SPLITS", tmp_path / "splits.parquet")
    monkeypatch.setattr(splits, "HASH", tmp_path / "splits.sha256")
    units.to_parquet(splits.SPLITS, index=False)
    splits.HASH.write_text(splits.digest(units))
    assert splits.load()[1] == splits.digest(units)
    units.assign(fold=[1, 0]).to_parquet(splits.SPLITS, index=False)
    with pytest.raises(RuntimeError):
        splits.load()


def test_locked_units_never_train():
    from anuran.experiment import Data, train_mask

    win = pd.DataFrame({"unit_id": ["a", "b", "c", "d"], "kind": ["frog", "frog", "background", "background"],
                        "source": ["inat", "cdfw", "background_aru", "background_aru"],
                        "acoustic_class": ["X", "X", None, None], "state": ["pos", "pos", "neg", "neg"],
                        "masked_classes": [[], [], [], []], "fold": [0, -1, 1, -1],
                        "locked": [False, True, False, True]})
    data = Data(np.zeros((4, 2)), win, pd.DataFrame(), ["X"], "h", "birdnet_v24")
    cfg = {"train": {"sources": ["inat", "cdfw"], "background": True}}
    assert train_mask(data, cfg).tolist() == [True, False, True, False]
