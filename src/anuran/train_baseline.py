"""Audio-only baseline on cached embeddings (the reference the text additions must beat).

Model: standardized embeddings -> multinomial logistic regression, trained on weak-pos
windows of the other folds. This is species ID *given a call*; a multi-label detector
needs no-frog negatives (background recordings) and comes with the step-4 head.

Evaluation is per recording (score = max over all its windows, incl. uncertain ones),
out-of-fold, pooled across folds. Secondary species are excluded from that class's AP.
Subsets: all; leakage-safe = californiaherps + iNat/XC observed after the encoder's
training data (TRAINING_CUTOFF). Zero-shot = the encoder's own logits for known classes.

Outputs: data/results/<encoder>_baseline/{per_class.csv, summary.md, oof_windows.parquet}
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from anuran.config import ROOT, load_groups, vocal_species
from anuran.data.manifest import FOCAL, PROCESSED
from anuran.features.encoders import TRAINING_CUTOFF

RESULTS = ROOT / "data" / "results"
RARE_MAX = 30  # classes with fewer recordings count as rare


def load(encoder: str, sources: list[str] | None = FOCAL):
    """Embeddings + window tables, restricted to `sources` (default: focal recordings;
    the ARU clips have their own folds, see eval/splits.py). None = everything."""
    d = PROCESSED / "embeddings" / encoder
    X = np.load(d / "embeddings.npy").astype(np.float32)
    win = pd.read_parquet(d / "windows.parquet")
    lab = pd.read_parquet(d / "window_labels.parquet")
    assert len(X) == len(win) == len(lab) and (win["recording_id"].values == lab["recording_id"].values).all()
    man = pd.read_parquet(PROCESSED / "manifest.parquet").set_index("recording_id")
    if sources is not None:
        keep = lab["recording_id"].map(man["source"]).isin(sources).values
        X, win, lab = X[keep], win[keep].reset_index(drop=True), lab[keep].reset_index(drop=True)
        man = man[man["source"].isin(sources)]
    return X, win, lab, man


def fit_predict(X, lab, classes, C: float) -> np.ndarray:
    """Out-of-fold window probabilities (n_windows, n_classes)."""
    proba = np.zeros((len(X), len(classes)), dtype=np.float32)
    for k in sorted(lab["fold"].unique()):
        train = (lab["fold"] != k) & (lab["state"] == "pos")
        test = lab["fold"] == k
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(scaler.transform(X[train]), lab.loc[train, "acoustic_class"])
        p = clf.predict_proba(scaler.transform(X[test]))
        proba[np.flatnonzero(test)[:, None], [classes.index(c) for c in clf.classes_]] = p
        print(f"  fold {k}: train {int(train.sum())} windows, test {int(test.sum())}", flush=True)
    return proba


def per_class_ap(scores: pd.DataFrame, rec: pd.DataFrame, classes: list[str]) -> pd.Series:
    out = {}
    for c in classes:
        if c not in scores:
            continue
        keep = ~rec["masked"].map(lambda m: c in m)
        y = (rec.loc[keep, "acoustic_class"] == c).values
        if y.sum() == 0 or y.all():
            continue
        out[c] = average_precision_score(y, scores.loc[keep, c].values)
    return pd.Series(out, dtype=float)


def evaluate(win_scores: pd.DataFrame, lab: pd.DataFrame, man: pd.DataFrame, classes: list[str], cutoff: str) -> pd.DataFrame:
    rec_scores = win_scores.groupby(lab["recording_id"].values).max()
    rec = man.loc[rec_scores.index]
    rec = rec.assign(masked=lab.groupby("recording_id")["masked_classes"].first().loc[rec.index])
    date = pd.to_datetime(rec["date"], errors="coerce")
    subsets = {
        "all": np.ones(len(rec), bool),
        "leakage_safe": (rec["source"] == "herps") | (date >= cutoff),
    }
    if "medium" in rec:  # underwater recordings kept in training, reported separately
        subsets["air"] = (rec["medium"] == "air").values
    cols = {}
    for name, mask in subsets.items():
        cols[name] = per_class_ap(rec_scores[mask], rec[mask], classes)
        cols[f"n_{name}"] = rec[mask]["acoustic_class"].value_counts()
    return pd.DataFrame(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="birdnet_v24", choices=sorted(TRAINING_CUTOFF))
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    X, win, lab, man = load(args.encoder)
    classes = sorted(lab["acoustic_class"].unique())
    print(f"{len(X)} windows, {len(classes)} classes, {int((lab['state'] == 'pos').sum())} weak-pos", flush=True)
    proba = fit_predict(X, lab, classes, args.C)
    cutoff = TRAINING_CUTOFF[args.encoder]
    trained = evaluate(pd.DataFrame(proba, columns=classes), lab, man, classes, cutoff)

    zs_cols = {c: win[f"zs_{c}"] for c in classes if f"zs_{c}" in win}
    zero = evaluate(pd.DataFrame(zs_cols), lab, man, classes, cutoff)

    names = {s["code"]: s["common"] for s in vocal_species()} | {g["code"]: g["common"] for g in load_groups()}
    n_rec = man["acoustic_class"].value_counts()
    table = pd.DataFrame({
        "common": pd.Series(names).reindex(classes),
        "recordings": n_rec.reindex(classes).fillna(0).astype(int),
        "rare": n_rec.reindex(classes).fillna(0) < RARE_MAX,
        "AP": trained["all"], "AP_leakage_safe": trained["leakage_safe"], "n_safe": trained["n_leakage_safe"],
        "AP_zero_shot": zero["all"], "AP_zero_shot_safe": zero["leakage_safe"],
    }).sort_values("recordings")

    out = RESULTS / f"{args.encoder}_baseline"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_class.csv")
    oof = lab[["recording_id", "start_s", "end_s", "acoustic_class", "state", "fold"]].assign(
        score_own=proba[np.arange(len(lab)), [classes.index(c) for c in lab["acoustic_class"]]],
        top_class=np.array(classes)[proba.argmax(1)],
        top_score=proba.max(1),
    )
    oof.to_parquet(out / "oof_windows.parquet", index=False)

    rec_top = pd.DataFrame(proba, columns=classes).groupby(lab["recording_id"].values).max().idxmax(axis=1)
    acc = (rec_top == man.loc[rec_top.index, "acoustic_class"]).mean()
    both = table.dropna(subset=["AP_zero_shot"])
    lines = [
        f"# {args.encoder} baseline (C={args.C})",
        f"- recording-level top-1 accuracy: {acc:.3f}",
        f"- macro AP, all classes: {table['AP'].mean():.3f} (rare {table.loc[table.rare, 'AP'].mean():.3f}, "
        f"common {table.loc[~table.rare, 'AP'].mean():.3f})",
        f"- macro AP, leakage-safe subset (observed >= {cutoff} or californiaherps): {table['AP_leakage_safe'].mean():.3f} "
        f"({int(table['AP_leakage_safe'].notna().sum())} classes with safe positives)",
        f"- classes known to the encoder ({len(both)}): trained {both['AP'].mean():.3f} vs zero-shot {both['AP_zero_shot'].mean():.3f}",
        "",
        "```", table.round(3).to_string(), "```",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
