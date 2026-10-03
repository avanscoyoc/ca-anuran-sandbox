"""Merge all sources into one clean manifest with recordist-grouped CV folds.

  data/interim/{herps,inat,xc,non_avian_ml}_recordings.parquet -> data/processed/manifest.parquet

Cleaning:
  - every file is fully decoded with ffmpeg; unreadable or <0.2 s files are dropped
  - exact duplicate files (md5) and cross-posted recordings (same date, place to
    ~100 m and duration) are dropped, keeping the first source in SOURCES order
Folds: StratifiedGroupKFold on acoustic_class. iNat/XC are grouped by recordist.
californiaherps is almost all one recordist, so it is grouped by species + county
(short sonogram clips are excerpts of longer recordings from the same site).
non-avian-ml ARU clips (data/non_avian_ml.py) are grouped by proxy block and get fold -1 here: their
folds and the locked test set are assigned with the other ARU data in eval/splits.py, so
adding them leaves the focal folds unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from anuran.config import INTERIM, ROOT, acoustic_group

SOURCES = ["herps", "xc", "inat", "non_avian_ml"]  # dedupe keeps the earliest source
FOCAL = ["herps", "xc", "inat"]  # handheld/focal recordings; non_avian_ml = ARU clips
PROCESSED = ROOT / "data" / "processed"
COLUMNS = [
    "recording_id", "source", "source_id", "label", "label_rank", "acoustic_class", "secondary_species",
    "call_types", "recordist", "license", "quality", "lat", "lon", "date", "place", "caption",
    "page_url", "url", "path", "duration_s", "md5", "medium", "split_group", "fold",
]
MIN_SECONDS = 0.2  # herps single-call exemplars are 0.26-0.97 s
N_FOLDS = 5


def load_source(name: str) -> pd.DataFrame:
    path = INTERIM / f"{name}_recordings.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if name == "herps":
        df = df.assign(
            source_id=df["url"].str.rsplit("/", n=1).str[-1],
            source="herps",
            label_rank="species",
            recordist=df["recordist"].fillna("californiaherps.com"),
            license="all-rights-reserved (research use)",
        ).drop(columns=["duration_s"])  # caption-parsed; measured below
    return df


def probe(path: str) -> dict:
    """Full decode (catches truncated/corrupt files), duration and md5."""
    f = ROOT / path
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-f", "null", "-"], capture_output=True, text=True)
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(f)],
        capture_output=True, text=True,
    )
    try:
        duration = float(json.loads(info.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        duration = None
    return {
        "path": path,
        "duration_s": duration,
        # damaged frames ("Header missing") are logged but the rest of the file decodes
        "decode_error": None if dec.returncode == 0 else (dec.stderr.strip()[-200:] or f"exit {dec.returncode}"),
        "decode_warning": dec.stderr.strip()[:200] or None,
        "md5": hashlib.md5(f.read_bytes()).hexdigest(),
    }


def audio_meta(paths: list[str]) -> pd.DataFrame:
    """Cached per-file checks (data/interim/audio_meta.parquet)."""
    cache = INTERIM / "audio_meta.parquet"
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["path"])
    todo = sorted(set(paths) - set(old["path"]))
    if todo:
        print(f"probing {len(todo)} files...", flush=True)
        with ThreadPoolExecutor(8) as ex:
            new = pd.DataFrame(list(ex.map(probe, todo)))
        old = pd.concat([old, new], ignore_index=True) if len(old) else new
        old.to_parquet(cache, index=False)
    return old.set_index("path")


def _merge_dups(df: pd.DataFrame, key: pd.Series) -> tuple[pd.DataFrame, int]:
    """Keep the first row of each duplicate set. The same audio posted under different
    species (a mixed chorus) keeps the other labels as secondary_species."""
    dup = key.notna() & key.duplicated(keep=False)
    if not dup.any():
        return df, 0
    df = df.copy()
    for _, idx in df[dup].groupby(key[dup]).groups.items():
        first, rest = idx[0], idx[1:]
        extra = {c for i in idx for c in [df.at[i, "label"], *df.at[i, "secondary_species"]]}
        df.at[first, "secondary_species"] = sorted(extra - {df.at[first, "label"]})
        df.loc[rest, "_drop"] = True
    drop = df.get("_drop", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    return df[~drop].drop(columns="_drop"), int(drop.sum())


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("source", key=lambda s: s.map(SOURCES.index), kind="stable")
    df, n_md5 = _merge_dups(df, df["md5"])
    # same recording re-encoded or cross-posted to iNat and XC under different accounts
    parts = [df["date"].astype("string"), df["lat"].round(3).astype("string"),
             df["lon"].round(3).astype("string"), df["duration_s"].round(0).astype("string")]
    key = parts[0].str.cat(parts[1:], sep="|")  # NA if any part is missing
    df, n_key = _merge_dups(df, key)
    print(f"dedupe: {n_md5} identical files, {n_key} cross-posts merged")
    return df


UNDERWATER = re.compile(r"underwater|under water|hydrophone", re.I)


def medium(df: pd.DataFrame) -> pd.Series:
    """'underwater' if the call type or caption says so (kept, but reported separately)."""
    uw = df["call_types"].map(lambda t: "underwater" in t) | df["caption"].fillna("").str.contains(UNDERWATER)
    return uw.map({True: "underwater", False: "air"})


COUNTY = re.compile(r"((?:[A-Z][a-z]+ )*[A-Z][a-z]+) County")


def split_groups(df: pd.DataFrame) -> pd.Series:
    person = df["source"] + ":" + df["recordist"].fillna("unknown").str.strip().str.lower()
    if "context" not in df:
        return person

    def county(row) -> str:
        for text in (row["caption"], row["context"]):
            m = COUNTY.search(text) if isinstance(text, str) else None
            if m:
                return m.group(1)
        return "unknown"  # whole species page becomes one group

    session = "herps:" + df["label"] + ":" + df.apply(county, axis=1)
    out = session.where(df["source"] == "herps", person)
    if "block" in df:
        aru = df["source"] == "non_avian_ml"
        out[aru] = "non_avian_ml:" + df.loc[aru, "label"] + ":b" + df.loc[aru, "block"].astype(int).astype(str).str.zfill(2)
    return out


def assign_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = 0) -> pd.Series:
    groups = split_groups(df)
    folds = pd.Series(-1, index=df.index)
    cv = StratifiedGroupKFold(n_splits=min(n_folds, groups.nunique()), shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(cv.split(df, df["acoustic_class"], groups)):
        folds.iloc[test] = k
    return folds


def build() -> pd.DataFrame:
    df = pd.concat([load_source(s) for s in SOURCES], ignore_index=True)
    df = df[df["path"].notna()]
    for col in ("secondary_species", "call_types"):
        df[col] = df[col].map(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])
    meta = audio_meta(df["path"].tolist())
    df = df.join(meta, on="path")
    bad = df["duration_s"].isna() | df["decode_error"].notna() | (df["duration_s"] < MIN_SECONDS)
    print(f"dropped {int(bad.sum())} unreadable/short files")
    df = df[~bad]
    df = dedupe(df)
    groups = acoustic_group()
    df["acoustic_class"] = df["label"].map(groups)
    unknown = df["acoustic_class"].isna()
    if unknown.any():
        raise ValueError(f"labels not in config: {sorted(df.loc[unknown, 'label'].unique())}")
    df = df.reset_index(drop=True)
    df["medium"] = medium(df)
    df["split_group"] = split_groups(df)
    focal = df["source"].isin(FOCAL)
    df["fold"] = -1
    df.loc[focal, "fold"] = assign_folds(df[focal])
    df = df.reindex(columns=COLUMNS)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "manifest.parquet", index=False)
    return df


if __name__ == "__main__":
    out = build()
    print(f"{len(out)} recordings, {out['duration_s'].sum() / 3600:.1f} h -> {PROCESSED / 'manifest.parquet'}")
