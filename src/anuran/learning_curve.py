"""Learning curve: how many ARU clips (and from how many proxy sites) does a species need?

For the ARU species (ANWO, LICA, PACH, RABO): for each ARU dev fold k and repeat r,
train the experiment model on all focal + background training data outside fold k, plus
a subsample of the ARU frog clips outside fold k: `clips` clips per species drawn from `groups`
proxy blocks per species (0 clips = focal + background only, the E1 setting). Score the
ARU pool of fold k (ARU frog clips + background) and record per-species AP, recall at
precision 0.9, and any-frog detection AP.

  python -m anuran.learning_curve configs/experiments/e0_focal_bg.yaml --repeats 2

Outputs: data/results/learning_curve/{lc.parquet, summary.md}. Proxy blocks are 50
contiguous clips (data/non_avian_ml.py), so `groups` approximates sites/nights, not exactly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anuran.config import ROOT
from anuran.data.non_avian_ml import BLOCK, SOURCE
from anuran.eval import metrics
from anuran.experiment import NOTES_LOG, cross_validate, load_data

OUT = ROOT / "data" / "results" / "learning_curve"
CLIPS = [5, 10, 25, 50, 100, 200, None]  # None = every clip in the chosen groups
GROUPS = [1, 3, None]                    # None = all available groups


def grid() -> list[tuple[int, int | None, int | None]]:
    out = [(0, 0, 0)]
    for g in GROUPS:
        for c in CLIPS:
            if g is not None and c is not None and c > g * BLOCK:
                continue  # more clips than the groups hold
            if g == 1 and c is None:
                continue  # same as 50 clips from 1 group
            out.append((len(out), c, g))
    return out


def sample(win: pd.DataFrame, pool: np.ndarray, clips: int | None, groups: int | None, rng) -> np.ndarray:
    """Boolean mask over windows: the ARU frog training clips for this point of the curve."""
    keep = np.zeros(len(win), bool)
    sub = win[pool]
    for _, cls_rows in sub.groupby("acoustic_class"):
        g_all = sorted(cls_rows["split_group"].unique())
        chosen = g_all if groups is None else list(rng.choice(g_all, min(groups, len(g_all)), replace=False))
        rows = cls_rows.index[cls_rows["split_group"].isin(chosen)].to_numpy()
        if clips is not None and clips < len(rows):
            rows = rng.choice(rows, clips, replace=False)
        keep[rows] = True
    return keep


def run(cfg: dict, repeats: int, seed: int = 0) -> pd.DataFrame:
    cfg = {**cfg, "seeds": cfg.get("seeds", [0])[:1]}
    cfg["train"] = {**cfg["train"], "sources": sorted(set(cfg["train"]["sources"]) | {SOURCE})}
    data = load_data(cfg["encoder"])
    w = data.win.copy()
    w["split_group"] = w["unit_id"].map(data.units["split_group"])
    aru_frog = ((w["source"] == SOURCE) & (w["kind"] == "frog")).values & ~w["locked"].values
    units = data.units
    aru = metrics.tier_b(units)
    classes = metrics.classes_b(units)
    folds = sorted(aru["fold"].unique())
    points = grid()
    print(f"{len(points)} curve points x {len(folds)} folds x {repeats} repeats", flush=True)
    rows, t0, n = [], time.time(), 0
    for r in range(repeats):
        rng = np.random.default_rng([seed, r])
        for k in folds:
            pool = aru_frog & (w["fold"] != k).values
            for pid, clips, groups in points:
                if r > 0 and clips == 0 and groups == 0:
                    continue  # no sampling involved: identical across repeats
                chosen = sample(w, pool, clips, groups, rng) if clips != 0 else np.zeros(len(w), bool)
                allow = ~aru_frog | chosen
                scores = cross_validate(data, cfg, allow=allow, folds=[k], log=lambda *_: None)
                test = aru[aru["fold"] == k]
                s = scores.loc[test.index]
                bg = (test["kind"] == "background").values
                det = metrics.ap((~bg), s.max(axis=1).values)
                n_train = w.loc[chosen].groupby("acoustic_class").size()
                n_groups = w.loc[chosen].groupby("acoustic_class")["split_group"].nunique()
                for c in classes:
                    y = (test["acoustic_class"] == c).values
                    r09, _ = metrics.recall_at_precision(y, s[c].values, 0.9)
                    rows.append({"repeat": r, "fold": k, "point": pid, "clips": clips if clips is not None else -1,
                                 "groups": groups if groups is not None else -1, "class": c,
                                 "n_train_clips": int(n_train.get(c, 0)), "n_train_groups": int(n_groups.get(c, 0)),
                                 "n_test_clips": int(y.sum()), "AP": metrics.ap(y, s[c].values),
                                 "recall@P0.9": r09, "detection_AP": det})
                n += 1
                el = time.time() - t0
                print(f"  r{r} fold {k} clips={clips} groups={groups}: macro AP "
                      f"{np.nanmean([x['AP'] for x in rows[-len(classes):]]):.3f}  "
                      f"[{n} fits, {el / n:.0f} s/fit]", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "lc.parquet", index=False)
    write_summary(df, cfg)
    return df


def write_summary(df: pd.DataFrame, cfg: dict) -> str:
    label = lambda v, all_: all_ if v == -1 else str(v)  # noqa: E731
    df = df.assign(clips_l=df["clips"].map(lambda v: label(v, "all")), groups_l=df["groups"].map(lambda v: label(v, "all")))
    # mean over folds/repeats per class first, then macro over classes
    per = df.groupby(["point", "clips_l", "groups_l", "class"]).agg(
        AP=("AP", "mean"), r09=("recall@P0.9", "mean"), n=("n_train_clips", "mean"))
    macro = per.groupby(["point", "clips_l", "groups_l"]).agg(macro_AP=("AP", "mean"), macro_recall_P09=("r09", "mean"))
    spread = df.groupby(["point", "repeat", "fold"]).agg(m=("AP", "mean"), det=("detection_AP", "first")) \
        .groupby("point").agg(sd=("m", "std"), detection_AP=("det", "mean"))
    macro = macro.join(spread, on="point").reset_index().sort_values("point")
    wide = per["AP"].unstack("class").reset_index().sort_values("point").drop(columns="point")
    ntr = per["n"].unstack("class").reset_index().sort_values("point").drop(columns="point")
    lines = ["# ARU learning curve (non-avian-ml species, proxy-grouped)",
             f"model: `{yaml.safe_dump(cfg['model'], default_flow_style=True).strip()}`, train sources {cfg['train']['sources']} "
             f"+ background; per-species clips sampled from `groups` proxy blocks (50 contiguous clips) outside the test fold",
             "", "## Macro over ANWO/LICA/PACH/RABO (mean over folds x repeats; sd across fold-repeats)", "```",
             macro.drop(columns="point").round(3).to_string(index=False), "```",
             "", "## Per-class AP", "```", wide.round(3).to_string(index=False), "```",
             "", "## Training clips actually used (mean)", "```", ntr.round(0).to_string(index=False), "```"]
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text)
    print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    df = run(cfg, args.repeats)
    full = df[(df["clips"] == -1) & (df["groups"] == -1)]["AP"].mean()
    zero = df[df["clips"] == 0]["AP"].mean()
    with NOTES_LOG.open("a") as f:
        f.write(f"| learning curve | ARU macro AP with 0 ARU frog clips {zero:.3f} -> all clips {full:.3f}; "
                f"see data/results/learning_curve/summary.md |\n")


if __name__ == "__main__":
    main()
