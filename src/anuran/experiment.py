"""Config-driven experiments on frozen embeddings + frozen splits, with a results registry.

  python -m anuran.experiment run configs/experiments/e0_focal_bg.yaml
  python -m anuran.experiment compare <run_a> <run_b>        # paired bootstrap deltas + adopt/no

Each run: 5-fold cross-validation over the frozen splits (eval/splits.py). Fold k trains on
dev units outside fold k (training sources' weak-pos windows + background windows) and
scores every dev window in fold k. Locked (Tier C) units are never touched. Unit score =
max over its windows, averaged over seeds. Metrics: eval/metrics.py.

Outputs: data/results/runs/<name>/{config.yaml, units.parquet, metrics.json, per_class_A.csv,
per_class_B.csv, thresholds.json, summary.md}; one row appended to data/results/runs.csv and
one line to .claude/notes/results_log.md (tracked, survives container rebuilds).

Config keys (see configs/experiments/):
  name, encoder, seeds
  train.sources      manifest sources whose weak-pos windows are trained on (herps, inat, xc, non_avian_ml)
  train.background   include verified ARU background windows as all-negative rows
  model.type         mlp (models/head.py) | logreg (multinomial, + a background class if used)
  model.*            mlp: epochs, d_proj, dropout, lr, weight_decay; logreg: C
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anuran.config import ROOT
from anuran.data.manifest import FOCAL, PROCESSED
from anuran.eval import metrics, splits
from anuran.features.encoders import TRAINING_CUTOFF

RUNS = ROOT / "data" / "results" / "runs"
REGISTRY = ROOT / "data" / "results" / "runs.csv"
NOTES_LOG = ROOT / ".claude" / "notes" / "results_log.md"
HEADLINE = ["A_macro_ap", "A_macro_ap_rare", "A_macro_ap_common", "A_macro_ap_leakage_safe",
            "B_macro_ap", "B_detection_ap"]


@dataclass
class Data:
    X: np.ndarray            # (n_windows, d) frog windows then background windows
    win: pd.DataFrame        # per window: unit_id, kind, source, acoustic_class, state, masked_classes, fold, locked
    units: pd.DataFrame      # per dev unit (index unit_id): eval metadata
    classes: list[str]
    splits_hash: str
    encoder: str


def load_data(encoder: str) -> Data:
    d = PROCESSED / "embeddings" / encoder
    Xf = np.load(d / "embeddings.npy").astype(np.float32)
    lab = pd.read_parquet(d / "window_labels.parquet")
    Xb = np.load(d / "background_embeddings.npy").astype(np.float32)
    bgw = pd.read_parquet(d / "background_windows.parquet")
    man = pd.read_parquet(PROCESSED / "manifest.parquet").set_index("recording_id")
    units, h = splits.load()
    u = units.set_index("unit_id")

    frog = pd.DataFrame({"unit_id": lab["recording_id"], "kind": "frog",
                         "source": lab["recording_id"].map(man["source"]),
                         "acoustic_class": lab["acoustic_class"], "state": lab["state"],
                         "masked_classes": lab["masked_classes"]})
    bg = pd.DataFrame({"unit_id": bgw["clip_id"], "kind": "background", "source": "non_avian_ml",
                       "acoustic_class": None, "state": "neg", "masked_classes": [[] for _ in range(len(bgw))]})
    win = pd.concat([frog, bg], ignore_index=True)
    missing = ~win["unit_id"].isin(u.index)
    if missing.any():
        raise RuntimeError(f"{missing.sum()} windows belong to units not in the frozen splits "
                           f"(e.g. {win.loc[missing, 'unit_id'].iloc[0]}); embeddings and splits are out of sync")
    win["fold"] = win["unit_id"].map(u["fold"]).values
    win["locked"] = win["unit_id"].map(u["locked"]).values
    X = np.concatenate([Xf, Xb])

    dev = u[~u["locked"] & u.index.isin(win["unit_id"])].copy()
    first_mask = win.groupby("unit_id")["masked_classes"].first()
    dev["masked"] = dev.index.map(first_mask)
    date = pd.to_datetime(dev.index.map(man["date"]).to_series(index=dev.index), errors="coerce")
    # herps and non-avian-ml clips are not in the encoders' training data; iNat/XC only after the cutoff
    dev["leakage_safe"] = ~dev["source"].isin(["inat", "xc"]) | (date >= TRAINING_CUTOFF[encoder])
    dev["medium"] = dev.index.map(man["medium"]).fillna("air")
    classes = sorted(man.loc[man["source"].isin(FOCAL), "acoustic_class"].unique())
    return Data(X, win, dev, classes, h, encoder)


# ---------------------------------------------------------------- models
def _fit_predict(cfg: dict, Xtr, Ytr, Mtr, Xte, classes, seed: int) -> np.ndarray:
    m = cfg["model"]
    if m["type"] == "mlp":
        from anuran.models.head import fit, predict

        kw = {k: m[k] for k in ("epochs", "d_proj", "dropout", "lr", "weight_decay") if k in m}
        return predict(fit(Xtr, Ytr, Mtr, seed=seed, **kw), Xte)
    if m["type"] == "logreg":
        from sklearn.linear_model import LogisticRegression

        y = np.where(Ytr.any(1), Ytr.argmax(1), len(classes))  # extra index = background
        clf = LogisticRegression(C=m.get("C", 0.1), max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y)
        out = np.zeros((len(Xte), len(classes)), np.float32)
        p = clf.predict_proba(Xte)
        for j, c in enumerate(clf.classes_):
            if c < len(classes):
                out[:, c] = p[:, j]
        return out
    raise ValueError(f"unknown model.type {m['type']}")


def targets(win: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(win), len(classes)), np.float32)
    M = np.ones_like(Y)
    frog = (win["kind"] == "frog").values
    Y[np.flatnonzero(frog), win.loc[frog, "acoustic_class"].map(idx).values] = 1
    for i, masked in zip(np.flatnonzero(frog), win.loc[frog, "masked_classes"]):
        for c in masked:
            if c in idx:
                M[i, idx[c]] = 0
    return Y, M


def train_mask(data: Data, cfg: dict) -> np.ndarray:
    """Windows eligible for training in any fold (fold exclusion is applied per fold)."""
    w = data.win
    frog = (w["kind"] == "frog") & (w["state"] == "pos") & w["source"].isin(cfg["train"]["sources"])
    bg = (w["kind"] == "background") & bool(cfg["train"].get("background", True))
    return ((frog | bg) & ~w["locked"]).values


def cross_validate(data: Data, cfg: dict, allow: np.ndarray | None = None, folds: list[int] | None = None,
                   log=print) -> pd.DataFrame:
    """Unit-level out-of-fold scores (dev units x classes). `allow` further restricts training
    windows (e.g. a learning-curve subsample); `folds` restricts which folds are run."""
    w = data.win
    Y, M = targets(w, data.classes)
    can_train = train_mask(data, cfg) & (allow if allow is not None else True)
    dev = ~w["locked"].values
    P = np.full((len(w), len(data.classes)), np.nan, np.float32)
    seeds = cfg.get("seeds", [0])
    for k in folds if folds is not None else sorted(w.loc[dev, "fold"].unique()):
        tr = can_train & (w["fold"] != k).values
        te = dev & (w["fold"] == k).values
        mu, sd = data.X[tr].mean(0), data.X[tr].std(0) + 1e-6
        acc = np.zeros((te.sum(), len(data.classes)), np.float32)
        for seed in seeds:
            acc += _fit_predict(cfg, (data.X[tr] - mu) / sd, Y[tr], M[tr], (data.X[te] - mu) / sd,
                                data.classes, seed) / len(seeds)
        P[te] = acc
        log(f"  fold {k}: trained on {tr.sum()} windows, scored {te.sum()}")
    scored = ~np.isnan(P[:, 0])
    return pd.DataFrame(P[scored], columns=data.classes).groupby(w.loc[scored, "unit_id"].values).max()


# ---------------------------------------------------------------- run + registry
def _git() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], capture_output=True,
                               text=True, cwd=ROOT).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def _fmt(m: dict, key: str) -> str:
    if key not in m:
        return "–"
    v, (lo, hi) = m[key]["value"], m[key]["ci95"]
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def run(cfg: dict, n_boot: int = 500) -> Path:
    name = cfg["name"]
    data = load_data(cfg["encoder"])
    print(f"{name}: {len(data.classes)} classes, {len(data.units)} dev units, splits {data.splits_hash}", flush=True)
    scores = cross_validate(data, cfg)
    units = data.units.loc[scores.index]
    res = metrics.summarize(units, scores, n_boot=n_boot)

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg, "git": _git(), "splits_hash": data.splits_hash,
            "finished": dt.datetime.now().isoformat(timespec="seconds"), "n_boot": n_boot}
    (out / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    keep = ["kind", "source", "acoustic_class", "split_group", "domain", "fold", "masked", "leakage_safe", "medium"]
    units[keep].join(scores.add_prefix("p_")).reset_index(names="unit_id").to_parquet(out / "units.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(res["metrics"], indent=1))
    (out / "thresholds.json").write_text(json.dumps(res["thresholds"], indent=1))
    res["per_class_A"].to_csv(out / "per_class_A.csv")
    res["per_class_B"].to_csv(out / "per_class_B.csv")

    m = res["metrics"]
    lines = [f"# {name}", f"- git {meta['git']}, splits {data.splits_hash}, {meta['finished']}",
             f"- config: `{json.dumps({k: cfg[k] for k in ('train', 'model', 'seeds')})}`", ""]
    lines += [f"- {k}: {_fmt(m, k)}" for k in HEADLINE]
    lines += ["", "## Tier A (focal, per recording)", "```", res["per_class_A"].round(3).to_string(), "```",
              "", "## Tier B (ARU pool: non-avian-ml frog clips + background)", "```", res["per_class_B"].round(3).to_string(), "```"]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:4 + len(HEADLINE)]))

    row = {"name": name, "finished": meta["finished"], "git": meta["git"], "splits": data.splits_hash,
           **{k: m[k]["value"] for k in HEADLINE if k in m}}
    reg = pd.concat([pd.read_csv(REGISTRY), pd.DataFrame([row])]) if REGISTRY.exists() else pd.DataFrame([row])
    reg.to_csv(REGISTRY, index=False)
    with NOTES_LOG.open("a") as f:
        f.write(f"| {meta['finished'][:10]} | {name} | {meta['git']} | {data.splits_hash} | "
                + " | ".join(_fmt(m, k) for k in HEADLINE) + " |\n")
    return out


def load_run(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = RUNS / name
    u = pd.read_parquet(d / "units.parquet").set_index("unit_id")
    s = u[[c for c in u if c.startswith("p_")]].rename(columns=lambda c: c[2:])
    u = u.drop(columns=s.columns.map(lambda c: f"p_{c}")).assign(locked=False)
    return u, s, yaml.safe_load((d / "config.yaml").read_text())


def compare(a: str, b: str, n_boot: int = 1000) -> pd.DataFrame:
    ua, sa, ca = load_run(a)
    ub, sb, cb = load_run(b)
    if ca["splits_hash"] != cb["splits_hash"]:
        raise SystemExit(f"runs use different splits ({ca['splits_hash']} vs {cb['splits_hash']}); not comparable")
    common = ua.index.intersection(ub.index)
    if len(common) != len(ua) or len(common) != len(ub):
        raise SystemExit("runs scored different units; not comparable")
    delta = metrics.paired_delta(ua.loc[common], sa.loc[common], sb.loc[common], n_boot=n_boot)
    adopt, why = metrics.decide(delta)
    head = delta.loc[[k for k in HEADLINE if k in delta.index]]
    print(f"{b} vs {a} (paired group bootstrap, n={n_boot})")
    print(head.round(3).to_string())
    per = delta[delta.index.str.contains(":")].sort_values("delta")
    print("\nper class (largest drops first):\n" + per.round(3).head(10).to_string())
    print(f"\ndecision: {'ADOPT' if adopt else 'keep A'} -- {why}")
    return delta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config", nargs="+")
    r.add_argument("--n-boot", type=int, default=500)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.cmd == "run":
        for path in args.config:
            run(yaml.safe_load(Path(path).read_text()), args.n_boot)
    else:
        compare(args.a, args.b, args.n_boot)


if __name__ == "__main__":
    main()
