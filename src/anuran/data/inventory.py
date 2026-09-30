"""Per-class inventory of the manifest: what we have, from whom, and what is testable."""

from __future__ import annotations

import pandas as pd

from anuran.config import load_groups, vocal_species
from anuran.data.manifest import PROCESSED


def inventory(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "manifest.parquet") if df is None else df
    by_src = df.pivot_table(index="acoustic_class", columns="source", values="recording_id", aggfunc="count", fill_value=0)
    g = df.groupby("acoustic_class")
    inv = by_src.assign(
        recordings=g.size(),
        minutes=(g["duration_s"].sum() / 60).round(1),
        split_groups=g["split_group"].nunique(),
        folds=g["fold"].nunique(),
        group_level=g["label_rank"].apply(lambda s: int((s == "group").sum())),
    )
    names = {s["code"]: s["common"] for s in vocal_species() if "acoustic_group" not in s}
    names |= {g["code"]: g["common"] for g in load_groups()}
    out = pd.DataFrame({"common": pd.Series(names)}).join(inv, how="left").fillna(0)
    out["testable"] = out["folds"] >= 2  # needs >=2 independent split groups
    return out.sort_values("recordings")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(inventory().to_string())
