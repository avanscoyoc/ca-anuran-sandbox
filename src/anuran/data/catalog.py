"""Dataset catalog: every data card -> one recordings table + one spans table, plus the inventory figure.

  pixi run catalog
    data/processed/catalog/recordings.parquet   one row per audio file
    data/processed/catalog/spans.parquet        one row per label (whole file, clip or box)
    data/results/figures/inventory.{png,csv}    per class: hours, 3 s windows, sites, days by label kind

Each dataset has a card in configs/datasets/<name>.yaml (see the README there). The card's `reader`
picks a function below that turns the dataset's files into the two tables, so new data in a known
format needs only a card. Nothing here embeds audio or touches the manifest, splits or models.

Label kinds (how much a label says about *where* the call is):
  weak      the species calls somewhere in the recording (scraped focal recordings)
  clip      the species calls in this short clip (non-avian-ml 3 s clips)
  box       a time (and frequency) box around each call; with `exhaustive`, unboxed time is absent
  negative  verified free of frogs (whole file)
Spans of dropped labels (e.g. Raven "X") are kept with dropped=True, so what was dropped stays visible.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from anuran.config import DATASETS, ROOT, acoustic_group, load_groups, vocal_species
from anuran.data.manifest import PROCESSED
from anuran.data.windows import frame

CATALOG = PROCESSED / "catalog"
FIGURES = ROOT / "data" / "results" / "figures"
LABEL_KINDS = ("weak", "clip", "box", "negative")
REQUIRED = ("name", "kind", "raw_dir", "reader", "label_kind", "license")
WINDOW_S, HOP_S = 3.0, 1.5  # BirdNET framing (data/windows.py)
BOX_RULE = {"min_overlap_s": 0.2, "min_box_frac": 0.5}  # rana_sierrae authors' rule for labels_2s.csv

REC_COLS = ["recording_id", "dataset", "kind", "label_kind", "exhaustive", "path", "duration_s", "sample_rate",
            "site_id", "date", "lat", "lon", "source_recording", "source_file", "source_start_s", "recordist", "license"]
SPAN_COLS = ["recording_id", "start_s", "end_s", "low_hz", "high_hz", "acoustic_class", "call_type",
             "source_label", "dropped"]


# ---------------------------------------------------------------- cards
def load_cards(folder: Path = DATASETS) -> list[dict]:
    cards = [yaml.safe_load(p.read_text()) for p in sorted(folder.glob("*.yaml"))]
    for c in cards:
        validate(c)
    return cards


def validate(card: dict) -> None:
    missing = [k for k in REQUIRED if k not in card]
    if missing:
        raise ValueError(f"card {card.get('name', '?')}: missing {missing}")
    if card["kind"] not in ("focal", "aru"):
        raise ValueError(f"card {card['name']}: kind must be focal or aru")
    if card["label_kind"] not in LABEL_KINDS:
        raise ValueError(f"card {card['name']}: label_kind must be one of {LABEL_KINDS}")
    if card["reader"] not in READERS:
        raise ValueError(f"card {card['name']}: unknown reader {card['reader']!r} (have {sorted(READERS)})")


# ---------------------------------------------------------------- readers: card -> (recordings, spans)
def read_manifest(card: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recordings already in the manifest (deduped, probed): one whole-file span each."""
    args = card["reader_args"]
    m = pd.read_parquet(PROCESSED / "manifest.parquet")
    m = m[m["source"] == args["source"]].reset_index(drop=True)
    site = {
        "county": m["split_group"].str.split(":").str[-1].where(lambda s: s != "unknown"),
        "latlon": (m["lat"].round(2).astype(str) + "," + m["lon"].round(2).astype(str)).where(m["lat"].notna()),
    }.get(args.get("site_from"), pd.Series(None, index=m.index, dtype=object))
    rec = m.assign(site_id=site, date=pd.to_datetime(m["date"], errors="coerce").dt.strftime("%Y-%m-%d"))
    spans = pd.DataFrame({"recording_id": m["recording_id"], "start_s": 0.0, "end_s": m["duration_s"],
                          "acoustic_class": m["acoustic_class"], "source_label": m["label"], "dropped": False})
    return rec, spans


def read_non_avian_ml(card: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The dataset's own clip table (data/non_avian_ml.py; `pixi run non-avian-ml` writes it). Frog clips ->
    one clip span each; background clips verified frog-free -> one negative span each; use != yes -> skipped."""
    from anuran.data import non_avian_ml as nam

    if not nam.CLIPS.exists():
        raise FileNotFoundError(f"{nam.CLIPS} missing: run `pixi run non-avian-ml` first")
    c = pd.read_csv(nam.CLIPS)
    c = c[c["use"] == "yes"]
    groups = acoustic_group()
    frog = (c["kind"] == "frog").values
    rec = pd.DataFrame({"recording_id": c["clip_id"], "path": c["file"], "duration_s": c["duration_s"],
                        "sample_rate": c["sample_rate"], "label_kind": np.where(frog, "clip", "negative"),
                        "exhaustive": ~frog, "site_id": c["site_id"], "date": c["date"],
                        "source_recording": c["recording_id"], "source_file": c["source_audio_path"],
                        "source_start_s": c["clip_start_s"]})
    spans = pd.DataFrame({"recording_id": c["clip_id"], "start_s": 0.0, "end_s": c["duration_s"],
                          "acoustic_class": c["code"].map(groups).where(frog, None), "source_label": c["label"],
                          "dropped": False})
    return rec.reset_index(drop=True), spans.reset_index(drop=True)


def read_raven(card: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audio files + one Raven selection table each. Labels map via the card (`drop` = kept, flagged)."""
    args, raw = card["reader_args"], ROOT / card["raw_dir"]
    pattern = re.compile(args["filename"]) if args.get("filename") else None
    sites = {s.get("device_id"): s["site_id"] for s in card.get("sites", [])}
    one_site = card["sites"][0]["site_id"] if len(card.get("sites", [])) == 1 else None
    groups = acoustic_group()
    recs, boxes = [], []
    for f in sorted(raw.glob(args["audio_glob"])):
        info = sf.info(f)
        meta = pattern.match(f.stem).groupdict() if pattern else {}
        rid = f"{card['name']}_{f.stem}"
        recs.append({"recording_id": rid, "path": str(f.relative_to(ROOT)), "duration_s": info.duration,
                     "sample_rate": info.samplerate, "site_id": sites.get(meta.get("device"), one_site),
                     "date": pd.to_datetime(meta["date"]).strftime("%Y-%m-%d") if "date" in meta else None})
        table = raw / args["table"].format(stem=f.stem)
        if not table.exists():
            raise FileNotFoundError(f"no Raven table for {f.name}: {table}")
        t = pd.read_csv(table, sep="\t")
        for _, b in t.iterrows():
            raw_label = str(b[args.get("label_column", "annotation")]).strip()
            target = card["labels"].get(raw_label)
            if target is None:
                raise ValueError(f"{table.name}: label {raw_label!r} not in the card's labels")
            drop = target == "drop"
            boxes.append({"recording_id": rid, "start_s": b["Begin Time (s)"], "end_s": b["End Time (s)"],
                          "low_hz": b.get("Low Freq (Hz)"), "high_hz": b.get("High Freq (Hz)"),
                          "acoustic_class": None if drop else groups[target["species"]],
                          "call_type": None if drop else target.get("call_type"),
                          "source_label": raw_label, "dropped": drop})
    return pd.DataFrame(recs), pd.DataFrame(boxes, columns=SPAN_COLS)


READERS = {"manifest": read_manifest, "non_avian_ml": read_non_avian_ml, "raven": read_raven}


# ---------------------------------------------------------------- build
def build(cards: list[dict] | None = None, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs, spans = [], []
    for card in cards if cards is not None else load_cards():
        validate(card)
        r, s = READERS[card["reader"]](card)
        defaults = {"label_kind": card["label_kind"], "exhaustive": bool(card.get("exhaustive", False)),
                    "license": card["license"]}  # a reader may set these per recording
        recs.append(r.assign(dataset=card["name"], kind=card["kind"],
                             **{k: v for k, v in defaults.items() if k not in r}))
        spans.append(s)
    rec = pd.concat(recs, ignore_index=True).reindex(columns=REC_COLS)
    spn = pd.concat(spans, ignore_index=True).reindex(columns=SPAN_COLS)
    spn["dropped"] = spn["dropped"].fillna(False).astype(bool)
    assert rec["recording_id"].is_unique, "recording_id must be unique across datasets"
    if write:
        CATALOG.mkdir(parents=True, exist_ok=True)
        rec.to_parquet(CATALOG / "recordings.parquet", index=False)
        spn.to_parquet(CATALOG / "spans.parquet", index=False)
    return rec, spn


# ---------------------------------------------------------------- windows
def box_hits(win_start: np.ndarray, win_end: np.ndarray, box_start: float, box_end: float,
             min_overlap_s: float = BOX_RULE["min_overlap_s"], min_box_frac: float = BOX_RULE["min_box_frac"]) -> np.ndarray:
    """A window contains a boxed call if they overlap >= min_overlap_s, or > min_box_frac of the box is inside."""
    overlap = np.clip(np.minimum(win_end, box_end) - np.maximum(win_start, box_start), 0, None)
    return (overlap >= min_overlap_s) | (overlap > min_box_frac * (box_end - box_start))


def windows(rec: pd.DataFrame, spans: pd.DataFrame, window_s: float = WINDOW_S, hop_s: float = HOP_S,
            rule: dict = BOX_RULE) -> pd.DataFrame:
    """One row per (window, class) a window counts for; acoustic_class None = verified-negative window.
    weak/clip: every window of the recording counts for its class (we don't know where the call is).
    box: windows passing `box_hits`; with exhaustive labels, windows hitting no box are negative.
    negative: every window is negative."""
    live = spans[~spans["dropped"]]
    by_rec = dict(tuple(live.groupby("recording_id")))
    out = []
    for r in rec.itertuples():
        spans_r = by_rec.get(r.recording_id, live.iloc[:0])
        w = np.array(frame(r.duration_s, window_s, hop_s))
        ws, we = w[:, 0], w[:, 1]
        hit_any = np.zeros(len(w), bool)
        if r.label_kind == "box":
            for cls, b in spans_r.groupby("acoustic_class"):
                hit = np.zeros(len(w), bool)
                for s, e in zip(b["start_s"], b["end_s"]):
                    hit |= box_hits(ws, we, s, e, **rule)
                hit_any |= hit
                out.append(pd.DataFrame({"recording_id": r.recording_id, "start_s": ws[hit], "acoustic_class": cls}))
            neg = ~hit_any if r.exhaustive else np.zeros(len(w), bool)
        elif r.label_kind == "negative":
            neg = np.ones(len(w), bool)
        else:
            for cls in spans_r["acoustic_class"].dropna().unique():
                out.append(pd.DataFrame({"recording_id": r.recording_id, "start_s": ws, "acoustic_class": cls}))
            neg = np.zeros(len(w), bool)
        if neg.any():
            out.append(pd.DataFrame({"recording_id": r.recording_id, "start_s": ws[neg], "acoustic_class": None}))
    win = pd.concat(out, ignore_index=True)
    return win.merge(rec[["recording_id", "dataset", "label_kind"]], on="recording_id")


# ---------------------------------------------------------------- inventory
NEG = "negative"


def inventory(rec: pd.DataFrame, spans: pd.DataFrame, win: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table: one row per (class, label_kind). Class "negative" = verified-absent audio."""
    win = windows(rec, spans) if win is None else win
    win = win.assign(acoustic_class=win["acoustic_class"].fillna(NEG),
                     label_kind=np.where(win["acoustic_class"].isna(), NEG, win["label_kind"]))
    # which recordings carry each (class, label kind); a box file with no live box is a negative recording
    pairs = win[["recording_id", "acoustic_class", "label_kind"]].drop_duplicates()
    pos_rec = pairs[pairs["acoustic_class"] != NEG]
    neg_rec = rec.loc[~rec["recording_id"].isin(pos_rec["recording_id"]) & (rec["exhaustive"] | (rec["label_kind"] == NEG)),
                      ["recording_id"]].assign(acoustic_class=NEG, label_kind=NEG)
    pairs = pd.concat([pos_rec, neg_rec]).merge(rec, on="recording_id", suffixes=("", "_rec"))
    known = pairs["site_id"].notna() & pairs["date"].notna()
    pairs["site_day"] = (pairs["site_id"].fillna("") + "|" + pairs["date"].fillna("")).where(known)
    g = pairs.groupby(["acoustic_class", "label_kind"])
    inv = g.agg(recordings=("recording_id", "nunique"), hours=("duration_s", lambda s: s.sum() / 3600),
                sources=("dataset", lambda s: ", ".join(sorted(set(s)))), sites=("site_id", "nunique"),
                days=("date", "nunique"), site_days=("site_day", "nunique"),
                recordings_no_site=("site_id", lambda s: int(s.isna().sum())))
    inv["windows_3s"] = win.groupby(["acoustic_class", "label_kind"]).size()
    inv = inv.fillna({"windows_3s": 0}).astype({"windows_3s": int}).reset_index()
    return inv[["acoustic_class", "label_kind", "recordings", "hours", "windows_3s", "sources", "sites", "days",
                "site_days", "recordings_no_site"]]


def summary(inv: pd.DataFrame) -> pd.DataFrame:
    """Wide, one row per class: recordings and windows per label kind; hours, sites and days summed over kinds."""
    w = inv.pivot_table(index="acoustic_class", columns="label_kind", values=["recordings", "windows_3s"],
                        aggfunc="sum", fill_value=0)
    w.columns = [f"{k}_{v}".replace("windows_3s", "win").replace("recordings", "rec") for v, k in w.columns]
    g = inv.groupby("acoustic_class")
    w["hours"] = g["hours"].sum().round(2)
    w["sites"], w["days"], w["site_days"] = g["sites"].sum(), g["days"].sum(), g["site_days"].sum()
    w["sources"] = g["sources"].agg(lambda s: ", ".join(sorted({x for v in s for x in v.split(", ")})))
    order = [f"{k}_{m}" for k in LABEL_KINDS for m in ("rec", "win") if f"{k}_{m}" in w]
    return w[order + ["hours", "sites", "days", "site_days", "sources"]]


# ---------------------------------------------------------------- figure
# Reference palette (dataviz skill, references/palette.md): categorical slots 1-3 validate all-pairs
# in light mode; "negative" is not an identity, so it takes the neutral secondary ink.
COLORS = {"weak": "#2a78d6", "clip": "#eb6834", "box": "#1baf7a", NEG: "#52514e"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PANELS = [("hours", "Hours of audio"), ("windows_3s", "3 s windows"), ("sites", "Sites (known)"),
          ("days", "Days (known)")]


def plot(inv: pd.DataFrame, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {s["code"]: s["common"] for s in vocal_species()} | {g["code"]: g["common"] for g in load_groups()}
    totals = inv[inv["acoustic_class"] != NEG].groupby("acoustic_class")["windows_3s"].sum()
    classes = [NEG, *totals.sort_values().index]  # y=0 is the bottom: negatives there, most data at the top
    y = {c: i for i, c in enumerate(classes)}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 0.32 * len(classes) + 1.6), sharey=True,
                             facecolor=SURFACE)
    for ax, (col, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for kind in LABEL_KINDS:
            d = inv[(inv["label_kind"] == kind) & (inv[col] > 0)]
            ax.scatter(d[col], d["acoustic_class"].map(y), s=42, color=COLORS[kind], edgecolor=SURFACE,
                       linewidth=1.5, zorder=3, label=kind)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle=":", zorder=0)
        ax.axhline(0.5, color=INK_2, linewidth=0.8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels([f"{c}  {names[c]}" if c in names else "verified negatives" for c in classes],
                            color=INK, fontsize=9)
    axes[0].set_ylim(-0.7, len(classes) - 0.3)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS[k]) for k in LABEL_KINDS]
    labels = ["weak (species somewhere in recording)", "clip (species in 3 s clip)", "box (annotated call times)",
              "verified negative"]
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="lower left", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.955),
               labelcolor=INK)
    fig.suptitle("Data per acoustic class, by label kind (log scale; no dot = none)", x=0.01, y=1.0,
                 ha="left", fontsize=13, color=INK)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    rec, spans = build()
    inv = inventory(rec, spans)
    FIGURES.mkdir(parents=True, exist_ok=True)
    inv.round(3).to_csv(FIGURES / "inventory.csv", index=False)
    plot(inv, FIGURES / "inventory.png")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(rec.groupby(["dataset", "label_kind"]).agg(recordings=("recording_id", "size"),
                                                     hours=("duration_s", lambda s: round(s.sum() / 3600, 2))).to_string())
    print()
    print(summary(inv).to_string())
    dropped = spans[spans["dropped"]]
    if len(dropped):
        print(f"\ndropped spans (kept in spans.parquet, flagged): {dropped['source_label'].value_counts().to_dict()}")
    print(f"\n-> {CATALOG}/{{recordings,spans}}.parquet, {FIGURES}/inventory.{{png,csv}}")


if __name__ == "__main__":
    main()
