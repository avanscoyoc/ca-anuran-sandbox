"""non-avian-ml ARU clips -> this dataset's own tables (a manifest source like herps/inat/xc).

  pixi run non-avian-ml
    data/interim/non_avian_ml_clips.csv            every clip file: label, site_id, date, recording_id, offsets
    data/interim/non_avian_ml_recordings.parquet   the frog clips, in manifest format

Raw data (configs/datasets/non_avian_ml.yaml), 3 s clips from the non-avian-ml project
(github.com/avanscoyoc/non-avian-ml):
  data/raw/aru/non_avian_ml/<CODE>/<name>_<ID>.wav              frog clips, label = folder (user-confirmed
                                                                 2026-10-01; PSRE maps to the PACH group)
  data/raw/aru/non_avian_ml/background_no_frog/<cat>_<name>_<ID>.wav   noise clips (verified frog-free by ear)
  data/raw/aru/non_avian_ml/all_3s_clips.csv                    the project's clip sheet, keyed by ID

The sheet's source_audio_path gives each clip's site and source recording (see parse_source):
  site_id       recorder folder (BD-33452B, yellow-legged-frog-1) or name prefix (UOLD4, Windmill_R13_Sound_WCenterA)
  recording_id  the source recording's start, YYYYMMDD_HHMMSS; <folder>_<stem> (nutria_clip_1) if it has none
In this file recording_id is the *source* recording; elsewhere (manifest, catalog) recording_id is the clip
(clip_id here) and the source recording is `source_recording`.

The manifest still groups frog clips into proxy blocks of BLOCK contiguous clip IDs. The real sites
(2-5 per species) replace them at the next splits --force.
"""

from __future__ import annotations

import re

import pandas as pd
import soundfile as sf

from anuran.config import ARU_RAW, INTERIM, LABELS, ROOT

NAM_DIR = ARU_RAW / "non_avian_ml"
SHEET = NAM_DIR / "all_3s_clips.csv"
BACKGROUND = "background_no_frog"
VERIFICATION = LABELS / "non_avian_ml" / "background_verification.csv"
CLIPS = INTERIM / "non_avian_ml_clips.csv"
SOURCE = "non_avian_ml"  # manifest source id
BLOCK = 50  # clips per proxy group
LABELS_TO_CODE = {"woodhouses_toad": "ANWO", "american_bullfrog": "LICA", "pacific_chorus_frog": "PSRE",
                  "yellow_legged_frog": "RABO"}

# source_audio_path stems; for the first layout the parent folder is the site
SOURCE_LAYOUTS = [
    re.compile(r"^(?:\w+__\d+__)?(?P<date>\d{8})_(?P<time>\d{6})$"),                # BD-33452B/20230521_000000.WAV
    re.compile(r"^(?P<site>[A-Za-z]+\d*)_\d{4}_(?P<date>\d{8})_(?P<time>\d{6})"),       # UOLD4_2021_20210527_200000_...
    re.compile(r"^\d{4}_\d{2}_\d{2}_(?P<site>.+)_(?P<date>\d{8})_(?P<time>\d{6})$"),   # 2021_10_26_Windmill_R13_Sound_WCenterA_20211013_065000
]


def parse_source(path: str) -> dict:
    """source_audio_path -> site_id, date, recording_id (None where the name doesn't say)."""
    folder, name = (["", *path.split("/")])[-2:]
    stem = name.rsplit(".", 1)[0]
    for layout in SOURCE_LAYOUTS:
        if m := layout.match(stem):
            g = m.groupdict()
            d = g["date"]
            return {"site_id": g.get("site") or folder, "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                    "recording_id": f"{d}_{g['time']}"}
    return {"site_id": None, "date": None, "recording_id": f"{folder}_{stem}"}


def proxy_blocks(clip_numbers: pd.Series, block: int = BLOCK) -> pd.Series:
    """Block index by rank of clip number (contiguous runs of `block` clips)."""
    return (clip_numbers.rank(method="first").astype(int) - 1) // block


def clips(min_sample_rate: int = 32000) -> pd.DataFrame:
    """One row per clip file. `use` says whether it enters training/eval and, if not, why."""
    sheet = pd.read_csv(SHEET).set_index("ID")
    v = pd.read_csv(VERIFICATION).drop_duplicates("clip_id", keep="last").set_index("clip_id")["frog_present"]
    rows = []
    for f in sorted(NAM_DIR.glob("*/*.wav")):
        clip = int(f.stem.rsplit("_", 1)[1])  # the sheet ID is the file's last number
        s = sheet.loc[clip]
        bg = f.parent.name == BACKGROUND
        if not bg and LABELS_TO_CODE[s["common_name"]] != f.parent.name:
            raise ValueError(f"{f}: folder {f.parent.name} but the sheet says {s['common_name']}")
        info = sf.info(f)
        if info.samplerate < min_sample_rate:
            use = "low_sample_rate"
        elif bg and f.stem not in v.index:
            use = "not_reviewed"
        elif bg and bool(v[f.stem]):
            use = "frog_present"
        else:
            use = "yes"
        rows.append({
            "clip_id": f.stem if bg else f"{SOURCE}_{f.parent.name}_{clip}", "file": str(f.relative_to(ROOT)),
            "kind": "background" if bg else "frog", "label": s["common_name"], "code": None if bg else f.parent.name,
            **parse_source(s["source_audio_path"]), "source_audio_path": s["source_audio_path"],
            "clip_start_s": s["start_time_3s"], "clip_end_s": s["end_time_3s"],
            "call_start_s": s["start_time"], "call_end_s": s["end_time"],  # annotated call, in source time
            "sample_rate": info.samplerate, "duration_s": round(info.duration, 4), "sheet_id": clip, "use": use,
        })
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    c = clips()
    INTERIM.mkdir(parents=True, exist_ok=True)
    c.to_csv(CLIPS, index=False)
    frog = c[c["kind"] == "frog"]
    rec = pd.DataFrame({
        "recording_id": frog["clip_id"], "source": SOURCE, "source_id": frog["file"].str.rsplit("/", n=1).str[-1],
        "label": frog["code"], "clip_number": frog["sheet_id"], "recordist": SOURCE,
        "license": "non-avian-ml project (not redistributable)", "path": frog["file"],
        "sample_rate": frog["sample_rate"], "date": frog["date"], "place": frog["site_id"],
        "secondary_species": [[] for _ in range(len(frog))], "call_types": [[] for _ in range(len(frog))],
    })
    rec["block"] = rec.groupby("label")["clip_number"].transform(proxy_blocks)
    rec.to_parquet(INTERIM / f"{SOURCE}_recordings.parquet", index=False)
    return c


if __name__ == "__main__":
    out = build()
    pd.set_option("display.width", 200)
    print(out.groupby(["kind", "use"]).size().to_string(), "\n")
    print(out[out["use"] == "yes"].groupby(["kind", "code"], dropna=False).agg(
        clips=("clip_id", "size"), sites=("site_id", "nunique"), days=("date", "nunique"),
        recordings=("recording_id", "nunique"), no_site=("site_id", lambda s: int(s.isna().sum()))).to_string())
    print(f"-> {CLIPS}\n-> {INTERIM / f'{SOURCE}_recordings.parquet'}")
