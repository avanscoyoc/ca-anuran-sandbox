"""Frozen evaluation splits: one row per evaluation unit, written once and hash-checked.

  data/processed/splits.parquet   unit_id, kind, source, acoustic_class, split_group, domain, fold, locked
  data/processed/splits.sha256    hash of the above; every run records it, `compare` refuses mismatches

Units: frog recordings (manifest) and verified background clips.
  domain "focal"  herps/iNat/XC recordings (Tier A). Folds = the manifest's recordist/site folds.
  domain "aru"    CDFW clips + ARU background (Tier B). CDFW folds are contiguous runs of
                  proxy blocks (data/cdfw.py) so neighbouring clips rarely straddle folds;
                  background keeps its category-stratified folds (data/background.py).
locked (Tier C, final test only; never trained on or scored during development):
  - the last LOCK_FRAC of each CDFW species' blocks (contiguous, so one boundary)
  - LOCK_FRAC of background clips, stratified by category
  - focal: for classes with >= RARE_MAX recordings only, single-class split groups whose
    recordings are all observed >= LOCK_AFTER (leakage-safe for BirdNET and Perch), up to
    LOCK_FRAC of the class's recordings and half its eligible groups. Rare classes are too
    small to lock; they have no Tier C number.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from anuran.data.manifest import FOCAL, N_FOLDS, PROCESSED

SPLITS = PROCESSED / "splits.parquet"
HASH = PROCESSED / "splits.sha256"
LOCK_FRAC = 0.2
FOCAL_LOCK_FRAC = 0.15
LOCK_AFTER = "2025-04-01"
RARE_MAX = 30  # same rare/common cut as train_baseline
COLUMNS = ["unit_id", "kind", "source", "acoustic_class", "split_group", "domain", "fold", "locked"]


def cdfw_units(man: pd.DataFrame, n_folds: int = N_FOLDS, lock_frac: float = LOCK_FRAC) -> pd.DataFrame:
    """Contiguous folds over each species' proxy blocks; the last blocks are locked."""
    c = man[man["source"] == "cdfw"].copy()
    c["fold"], c["locked"] = -1, False
    for _, idx in c.groupby("acoustic_class").groups.items():
        blocks = sorted(c.loc[idx, "split_group"].unique())  # zero-padded block numbers sort in order
        n_lock = max(1, int(round(lock_frac * len(blocks))))
        dev = blocks[: len(blocks) - n_lock]
        fold_of = {b: i * n_folds // len(dev) for i, b in enumerate(dev)}
        c.loc[idx, "locked"] = ~c.loc[idx, "split_group"].isin(dev)
        c.loc[idx, "fold"] = c.loc[idx, "split_group"].map(fold_of).fillna(-1).astype(int)
    return c.assign(kind="frog", domain="aru")


def focal_units(man: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    f = man[man["source"].isin(FOCAL)].copy()
    f["locked"] = False
    date = pd.to_datetime(f["date"], errors="coerce")
    g = f.assign(safe=date >= LOCK_AFTER).groupby("split_group").agg(
        n_cls=("acoustic_class", "nunique"), cls=("acoustic_class", "first"),
        n=("recording_id", "size"), all_safe=("safe", "all"))
    eligible = g[(g["n_cls"] == 1) & g["all_safe"]]
    totals = f["acoustic_class"].value_counts()
    rng = np.random.default_rng(seed)
    locked_groups = []
    for cls, grp in eligible.groupby("cls"):
        if totals[cls] < RARE_MAX:
            continue
        budget, used = FOCAL_LOCK_FRAC * totals[cls], 0
        for name in rng.permutation(sorted(grp.index))[: int(np.ceil(len(grp) / 2))]:
            if used + grp.at[name, "n"] > budget:
                continue
            locked_groups.append(name)
            used += grp.at[name, "n"]
    f.loc[f["split_group"].isin(locked_groups), "locked"] = True
    return f.assign(kind="frog", domain="focal")


def background_units(seed: int = 0, lock_frac: float = LOCK_FRAC) -> pd.DataFrame:
    from anuran.data.background import clips

    bg = clips(seed)
    rng = np.random.default_rng(seed)
    bg["locked"] = False
    for _, idx in bg.groupby("category").groups.items():
        n_lock = int(round(lock_frac * len(idx)))
        bg.loc[rng.choice(idx, n_lock, replace=False), "locked"] = True
    return pd.DataFrame({
        "unit_id": bg["clip_id"], "kind": "background", "source": "background_aru", "acoustic_class": None,
        "split_group": "bg:" + bg["clip_id"], "domain": "aru", "fold": bg["fold"], "locked": bg["locked"],
    })


def build(seed: int = 0) -> pd.DataFrame:
    man = pd.read_parquet(PROCESSED / "manifest.parquet")
    frogs = pd.concat([focal_units(man, seed), cdfw_units(man)]).rename(columns={"recording_id": "unit_id"})
    units = pd.concat([frogs[COLUMNS], background_units(seed)], ignore_index=True)
    units["fold"] = units["fold"].astype(int)
    units["locked"] = units["locked"].astype(bool)
    assert units["unit_id"].is_unique
    assert ((units["fold"] >= 0) | units["locked"]).all(), "every dev unit needs a fold"
    return units.sort_values("unit_id", ignore_index=True)


def digest(units: pd.DataFrame) -> str:
    canon = units[COLUMNS].sort_values("unit_id").to_csv(index=False).encode()
    return hashlib.sha256(canon).hexdigest()[:16]


def write(force: bool = False, seed: int = 0) -> pd.DataFrame:
    if SPLITS.exists() and not force:
        raise SystemExit(f"{SPLITS} exists; splits are frozen. Re-run with --force only if you mean to "
                         "invalidate every earlier result (runs on different splits can't be compared).")
    units = build(seed)
    units.to_parquet(SPLITS, index=False)
    HASH.write_text(digest(units) + "\n")
    return units


def load() -> tuple[pd.DataFrame, str]:
    """Frozen splits + their hash; fails loudly if the file changed since it was written."""
    units = pd.read_parquet(SPLITS)
    h = digest(units)
    if HASH.read_text().strip() != h:
        raise RuntimeError(f"{SPLITS} does not match {HASH}: splits were modified after freezing")
    return units, h


def describe(units: pd.DataFrame) -> str:
    t = units.assign(cls=units["acoustic_class"].fillna("background")).pivot_table(
        index="cls", columns=["domain", "locked"], values="unit_id", aggfunc="size", fill_value=0)
    return t.to_string()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite frozen splits (invalidates earlier runs)")
    args = ap.parse_args()
    u = write(args.force)
    print(describe(u))
    print(f"-> {SPLITS} (hash {HASH.read_text().strip()})")
