"""Evaluation metrics on unit-level scores, with group-bootstrap confidence intervals.

Units are recordings (focal), clips (CDFW, background). A unit's score for a class is
the max over its windows. Tier A = focal dev units; Tier B = ARU dev units (CDFW clips
+ background clips). Locked (Tier C) units are never passed in during development.

Uncertainty: resample split groups with replacement (recordist/site for focal, proxy
block for CDFW, clip for background); a unit's weight is how often its group was drawn.
AP is computed with those weights for all resamples at once (`ap_boot`), which matches
sklearn's average_precision_score (step-wise, ties grouped) for unit weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RARE_MAX = 30


# ---------------------------------------------------------------- core metrics
def ap_boot(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted average precision for each row of W (n_boot, n). NaN where no positives."""
    W = np.atleast_2d(W).astype(np.float64)
    order = np.argsort(-s, kind="stable")
    ss, yy, Wo = s[order], y[order].astype(np.float64), W[:, order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]  # last index of each tie block
    tp = np.cumsum(Wo * yy, axis=1)[:, ends]
    allp = np.cumsum(Wo, axis=1)[:, ends]
    total = tp[:, -1]
    prec = np.divide(tp, allp, out=np.zeros_like(tp), where=allp > 0)
    d_tp = np.diff(tp, axis=1, prepend=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (d_tp * prec).sum(1) / total, np.nan)


def ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(ap_boot(y, s, np.ones((1, len(y))))[0])


def recall_at_precision(y: np.ndarray, s: np.ndarray, precision: float) -> tuple[float, float]:
    """Highest recall reachable with precision >= `precision`, and the score threshold for it."""
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.cumsum(yy)[ends]
    prec = tp / (ends + 1)
    ok = prec >= precision
    if yy.sum() == 0 or not ok.any():
        return 0.0, float("inf")
    best = np.flatnonzero(ok)[np.argmax(tp[ok])]
    return float(tp[best] / yy.sum()), float(ss[ends[best]])


def review_effort(y: np.ndarray, s: np.ndarray, recall: float = 0.9) -> float:
    """Fraction of units a reviewer must listen to, in score order, to find `recall` of the positives."""
    if y.sum() == 0:
        return float("nan")
    hits = np.cumsum(y[np.argsort(-s, kind="stable")])
    return float((np.searchsorted(hits, np.ceil(recall * y.sum())) + 1) / len(y))


def bootstrap_weights(groups: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n_units) weights from resampling groups with replacement."""
    codes, uniq = pd.factorize(pd.Series(groups))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
    return counts[:, codes].astype(np.float64)


def ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    if not len(v):
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


# ---------------------------------------------------------------- pools
def tier_a(units: pd.DataFrame) -> pd.DataFrame:
    return units[(units["domain"] == "focal") & ~units["locked"]]


def tier_b(units: pd.DataFrame) -> pd.DataFrame:
    return units[(units["domain"] == "aru") & ~units["locked"]]


def evaluable_a(units: pd.DataFrame) -> list[str]:
    """Focal classes whose recordings span >= 2 folds (else no training data when tested)."""
    a = tier_a(units)
    n_folds = a.groupby("acoustic_class")["fold"].nunique()
    return sorted(n_folds.index[n_folds >= 2])


def classes_b(units: pd.DataFrame) -> list[str]:
    b = tier_b(units)
    return sorted(b.loc[b["kind"] == "frog", "acoustic_class"].unique())


def _class_ap(pool: pd.DataFrame, scores: pd.DataFrame, cls: str, W: np.ndarray) -> np.ndarray:
    keep = ~pool["masked"].map(lambda m: cls in m).values
    y = (pool["acoustic_class"] == cls).values[keep]
    return ap_boot(y, scores.loc[pool.index, cls].values[keep], W[:, keep])


# ---------------------------------------------------------------- headline metrics
def headline(units: pd.DataFrame, scores: pd.DataFrame, W_a: np.ndarray | None, W_b: np.ndarray | None) -> dict:
    """Headline metrics as arrays over bootstrap rows (row 0 of each W should be all-ones = point estimate).

    `units` must be indexed by unit_id, include `masked` (list of masked classes),
    `leakage_safe` and `medium`, and be aligned with `scores` (unit_id x class).
    """
    out: dict[str, np.ndarray] = {}
    if W_a is not None:
        a = tier_a(units)
        ev = evaluable_a(units)
        n = a["acoustic_class"].value_counts()
        per = {c: _class_ap(a, scores, c, W_a) for c in ev}
        stack = np.stack([per[c] for c in ev])
        rare = np.array([n[c] < RARE_MAX for c in ev])
        out["A_macro_ap"] = np.nanmean(stack, 0)
        if rare.any():
            out["A_macro_ap_rare"] = np.nanmean(stack[rare], 0)
        if (~rare).any():
            out["A_macro_ap_common"] = np.nanmean(stack[~rare], 0)
        for sub in ("leakage_safe", "air"):
            mask = (a["leakage_safe"] if sub == "leakage_safe" else a["medium"] == "air").values
            sa = a[mask]
            out[f"A_macro_ap_{sub}"] = np.nanmean(np.stack([_class_ap(sa, scores, c, W_a[:, mask]) for c in ev]), 0)
        for c in ev:
            out[f"A_ap:{c}"] = per[c]
    if W_b is not None:
        b = tier_b(units)
        cb = classes_b(units)
        if cb:
            per = {c: _class_ap(b, scores, c, W_b) for c in cb}
            out["B_macro_ap"] = np.nanmean(np.stack([per[c] for c in cb]), 0)
            for c in cb:
                out[f"B_ap:{c}"] = per[c]
        frog = (b["kind"] == "frog").values
        out["B_detection_ap"] = ap_boot(frog, scores.loc[b.index].max(axis=1).values, W_b)
    return out


def summarize(units: pd.DataFrame, scores: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Point estimates + 95% CIs for headline metrics, plus threshold-based tables."""
    a, b = tier_a(units), tier_b(units)
    W_a = np.vstack([np.ones(len(a)), bootstrap_weights(a["split_group"].values, n_boot, seed)]) if len(a) else None
    W_b = np.vstack([np.ones(len(b)), bootstrap_weights(b["split_group"].values, n_boot, seed)]) if len(b) else None
    h = headline(units, scores, W_a, W_b)
    metrics = {k: {"value": float(v[0]), "ci95": ci(v[1:])} for k, v in h.items()}
    return {"metrics": metrics, "per_class_A": per_class_a(units, scores), "per_class_B": per_class_b(units, scores),
            "thresholds": thresholds(units, scores)}


def per_class_a(units: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    a = tier_a(units)
    ev = set(evaluable_a(units))
    rows = {}
    for c in sorted(a["acoustic_class"].unique()):
        keep = ~a["masked"].map(lambda m: c in m)
        aa = a[keep]
        y, s = (aa["acoustic_class"] == c).values, scores.loc[aa.index, c].values
        safe, air = aa["leakage_safe"].values, (aa["medium"] == "air").values
        r09, _ = recall_at_precision(y, s, 0.9)
        rows[c] = {"recordings": int(y.sum()), "evaluable": c in ev,
                   "AP": ap(y, s) if c in ev else np.nan,
                   "AP_leakage_safe": ap(y[safe], s[safe]) if c in ev and y[safe].any() else np.nan,
                   "AP_air": ap(y[air], s[air]) if c in ev and y[air].any() else np.nan,
                   "recall@P0.9": r09 if c in ev else np.nan}
    return pd.DataFrame(rows).T.sort_values("recordings")


def per_class_b(units: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """ARU pool: per-class AP / recall@precision / review effort for CDFW classes; for every
    class, how often it fires (>= 0.5) on background and on other species' ARU clips."""
    b = tier_b(units)
    if not len(b):
        return pd.DataFrame()
    s_all = scores.loc[b.index]
    bg = (b["kind"] == "background").values
    hours = bg.sum() * 3 / 3600  # background clips are single 3 s windows
    groups = b["split_group"].values
    rows = {}
    for c in scores.columns:
        y = (b["acoustic_class"] == c).values
        s = s_all[c].values
        row = {"clips": int(y.sum()),
               "bg_fp_per_hour@0.5": float((s[bg] >= 0.5).sum() / hours) if hours else np.nan,
               "fires_on_other_aru_frogs@0.5": float((s[~bg & ~y] >= 0.5).mean()) if (~bg & ~y).any() else np.nan}
        if y.any():
            r09, _ = recall_at_precision(y, s, 0.9)
            r095, _ = recall_at_precision(y, s, 0.95)
            g = pd.DataFrame({"g": groups, "y": y, "s": s}).groupby("g").agg(y=("y", "max"), s=("s", "max"))
            row |= {"AP": ap(y, s), "recall@P0.9": r09, "recall@P0.95": r095,
                    "review_effort@R0.9": review_effort(y, s, 0.9),
                    "recall@0.5": float((s[y] >= 0.5).mean()),
                    "presence_AP(block)": ap(g["y"].values, g["s"].values)}
        rows[c] = row
    return pd.DataFrame(rows).T.sort_values("clips", ascending=False)


def thresholds(units: pd.DataFrame, scores: pd.DataFrame, precision: float = 0.9) -> dict:
    """Per-class score thresholds reaching `precision` on the dev pools (ARU pool if the
    class has ARU clips, else focal). Frozen with the run; Tier C uses them unchanged."""
    out = {}
    a, b = tier_a(units), tier_b(units)
    cb = set(classes_b(units))
    for c in scores.columns:
        pool = b if c in cb else a
        keep = ~pool["masked"].map(lambda m: c in m).values
        y = (pool.loc[keep, "acoustic_class"] == c).values
        if not y.any():
            continue
        _, t = recall_at_precision(y, scores.loc[pool.index[keep], c].values, precision)
        out[c] = {"threshold": t, "pool": "aru" if c in cb else "focal", "precision_target": precision}
    return out


# ---------------------------------------------------------------- paired comparison
def paired_delta(units: pd.DataFrame, scores_a: pd.DataFrame, scores_b: pd.DataFrame,
                 n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """metric(B) - metric(A) on identical group resamples. Same units, same splits."""
    ua, ub = tier_a(units), tier_b(units)
    W_a = np.vstack([np.ones(len(ua)), bootstrap_weights(ua["split_group"].values, n_boot, seed)]) if len(ua) else None
    W_b = np.vstack([np.ones(len(ub)), bootstrap_weights(ub["split_group"].values, n_boot, seed)]) if len(ub) else None
    ha, hb = headline(units, scores_a, W_a, W_b), headline(units, scores_b, W_a, W_b)
    rows = []
    for k in ha:
        if k not in hb:
            continue
        d = hb[k] - ha[k]
        lo, hi = ci(d[1:])
        rows.append({"metric": k, "A": ha[k][0], "B": hb[k][0], "delta": d[0], "ci_lo": lo, "ci_hi": hi,
                     "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows).set_index("metric")


def decide(delta: pd.DataFrame, max_class_drop: float = 0.05) -> tuple[bool, str]:
    """Adopt B over A if the primary metric's paired 95% CI is above 0 and no class drops
    by more than max_class_drop AP. Primary = ARU macro AP when available, else focal rare."""
    primary = "B_macro_ap" if "B_macro_ap" in delta.index else "A_macro_ap_rare"
    p = delta.loc[primary]
    per_class = delta[delta.index.str.contains(":")]
    worst = per_class["delta"].min() if len(per_class) else 0.0
    ok = p["ci_lo"] > 0 and worst >= -max_class_drop
    why = (f"{primary}: {p['delta']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]; "
           f"worst class {per_class['delta'].idxmin() if len(per_class) else '-'} {worst:+.3f}")
    return bool(ok), why
