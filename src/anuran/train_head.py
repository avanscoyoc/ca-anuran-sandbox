"""Multi-label detector head on frozen embeddings, with background negatives.

Training rows: weak-pos frog windows (target 1 for their class, secondary species
masked) + verified background windows (all targets 0). Uncertain windows are never
trained on. Out-of-fold predictions on all windows, averaged over seeds.

Reported (per class, recording-level max over windows): AP on all / leakage-safe /
air-only recordings; classes whose recordings sit in a single fold are "not
evaluable" (no training data when tested). Detection: any-frog score (max over
classes) for frog vs background, at window and recording/clip level.

--setting audio         BCE only (the audio-only row of the step-4 ablation)
Text settings (attributes / alignment) plug in via extra losses once the edited
species text is ready.

Outputs: data/results/<encoder>_head_<setting>/{per_class.csv, summary.md, oof.npz}
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from anuran.config import load_groups, vocal_species
from anuran.data.manifest import PROCESSED
from anuran.features.encoders import TRAINING_CUTOFF
from anuran.models.head import fit, predict
from anuran.train_baseline import RARE_MAX, RESULTS, evaluate, load


def targets(lab: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(lab), len(classes)), np.float32)
    M = np.ones_like(Y)
    Y[np.arange(len(lab)), lab["acoustic_class"].map(idx).values] = 1
    for i, masked in enumerate(lab["masked_classes"]):
        for c in masked:
            if c in idx:
                M[i, idx[c]] = 0
    return Y, M


def run(encoder: str, setting: str, seeds: list[int], epochs: int) -> None:
    X, win, lab, man = load(encoder)
    d = PROCESSED / "embeddings" / encoder
    Xb = np.load(d / "background_embeddings.npy").astype(np.float32)
    bg = pd.read_parquet(d / "background_windows.parquet")
    classes = sorted(lab["acoustic_class"].unique())
    Y, M = targets(lab, classes)
    Yb, Mb = np.zeros((len(bg), len(classes)), np.float32), np.ones((len(bg), len(classes)), np.float32)
    pos = (lab["state"] == "pos").values
    print(f"{len(X)} frog windows ({pos.sum()} weak-pos), {len(Xb)} background windows, {len(classes)} classes")

    P, Pb = np.zeros_like(Y), np.zeros_like(Yb)
    folds = sorted(lab["fold"].unique())
    for seed in seeds:
        for k in folds:
            tr, te = pos & (lab["fold"] != k).values, (lab["fold"] == k).values
            trb, teb = (bg["fold"] != k).values, (bg["fold"] == k).values
            Xtr = np.concatenate([X[tr], Xb[trb]])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            model = fit((Xtr - mu) / sd, np.concatenate([Y[tr], Yb[trb]]), np.concatenate([M[tr], Mb[trb]]),
                        seed=seed, epochs=epochs)
            P[te] += predict(model, (X[te] - mu) / sd) / len(seeds)
            Pb[teb] += predict(model, (Xb[teb] - mu) / sd) / len(seeds)
        print(f"  seed {seed} done", flush=True)

    cutoff = TRAINING_CUTOFF[encoder]
    ev = evaluate(pd.DataFrame(P, columns=classes), lab, man, classes, cutoff)
    n_rec = man["acoustic_class"].value_counts()
    evaluable = man.groupby("acoustic_class")["fold"].nunique() >= 2
    uw_share = man.groupby("acoustic_class")["medium"].apply(lambda s: (s == "underwater").mean())
    names = {s["code"]: s["common"] for s in vocal_species()} | {g["code"]: g["common"] for g in load_groups()}
    table = pd.DataFrame({
        "common": pd.Series(names).reindex(classes),
        "recordings": n_rec.reindex(classes).fillna(0).astype(int),
        "rare": n_rec.reindex(classes).fillna(0) < RARE_MAX,
        "evaluable": evaluable.reindex(classes).fillna(False),
        "underwater_share": uw_share.reindex(classes).round(2),
        "AP": ev["all"], "AP_leakage_safe": ev["leakage_safe"], "AP_air": ev.get("air"),
    }).sort_values("recordings")
    for col in ("AP", "AP_leakage_safe", "AP_air"):
        table.loc[~table["evaluable"], col] = np.nan  # no training data when tested

    # detection: any-frog score vs background
    any_frog, any_bg = P.max(1), Pb.max(1)
    win_ap = average_precision_score(np.r_[np.ones(pos.sum()), np.zeros(len(any_bg))], np.r_[any_frog[pos], any_bg])
    rec_frog = pd.Series(any_frog).groupby(lab["recording_id"].values).max()
    clip_bg = pd.Series(any_bg).groupby(bg["clip_id"].values).max()
    rec_ap = average_precision_score(np.r_[np.ones(len(rec_frog)), np.zeros(len(clip_bg))], np.r_[rec_frog, clip_bg])
    bg_fp = (any_bg >= 0.5).mean()

    ev_t = table[table["evaluable"]]
    lines = [
        f"# {encoder} multi-label head, setting={setting} (seeds={seeds}, epochs={epochs})",
        f"- classes evaluable: {int(table['evaluable'].sum())}/{len(table)} "
        f"(not evaluable: {', '.join(table.index[~table['evaluable']])} -- single site, no training data when tested)",
        f"- macro AP (evaluable): {ev_t['AP'].mean():.3f}  rare {ev_t.loc[ev_t.rare, 'AP'].mean():.3f}  "
        f"common {ev_t.loc[~ev_t.rare, 'AP'].mean():.3f}",
        f"- macro AP leakage-safe: {ev_t['AP_leakage_safe'].mean():.3f}; air-only: {ev_t['AP_air'].mean():.3f}",
        f"- detection AP (any frog vs background): window {win_ap:.3f}, recording/clip {rec_ap:.3f}; "
        f"background windows with any class >= 0.5: {bg_fp:.1%}",
        "", "```", table.round(3).to_string(), "```",
    ]
    out = RESULTS / f"{encoder}_head_{setting}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_class.csv")
    np.savez_compressed(out / "oof.npz", P=P, Pb=Pb, classes=np.array(classes))
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="birdnet_v24", choices=sorted(TRAINING_CUTOFF))
    ap.add_argument("--setting", default="audio", choices=["audio"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    run(args.encoder, args.setting, args.seeds, args.epochs)


if __name__ == "__main__":
    main()
