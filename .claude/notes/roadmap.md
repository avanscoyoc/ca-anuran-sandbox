# Roadmap: training and testing the best California anuran model

_Last updated: 2026-10-01. Tick items off and date them as they're done._

**Goal:** a model that detects and identifies every California anuran class **in ARU recordings**.
**Use case** (user): presence per site/night, plus screening clips for human review. Scores on focal recordings only matter as far as they predict that.

## Where we left off (2026-10-01, ~01:20 UTC)
- ~~Started in the background: E0 → E2 → learning curve~~ All three finished 2026-10-01 (logs: `data/results/e0_e2.log`, `data/results/learning_curve.log`). Expected to finish by ~03:00 UTC if the container stayed up.
- **Next session:**
  1. Check whether `data/results/runs/e0_focal_bg/`, `data/results/runs/e2_focal_cdfw_bg/` and `data/results/learning_curve/summary.md` exist. If not, re-run (the runs aren't resumable):
     - `pixi run experiment configs/experiments/e0_focal_bg.yaml configs/experiments/e2_focal_cdfw_bg.yaml`
     - `pixi run learning-curve configs/experiments/e0_focal_bg.yaml`
  2. `pixi run compare e0_focal_bg e2_focal_cdfw_bg`. Read E1 (E0's Tier B: `B_detection_ap`, `recall@0.5` per ARU class) for the domain-shortcut answer.
  3. Fill in the learning-curve result in `aru_data_needs.md`, then tick E0/E1/E2/LC below.
  4. Next build: E3 augmentation (mixing calls into ARU background).
- The code (ARU clip ingestion, eval harness, tests) is uncommitted unless a commit shows up in `git log`.

## How we test: one harness, frozen splits, one decision rule
Every experiment goes through `pixi run experiment configs/experiments/<x>.yaml`. No ad-hoc scripts.

**Frozen splits** (`pixi run splits`, `src/anuran/eval/splits.py`): one row per unit, written once and hash-checked.
- **Tier A, focal:** herps/iNat/XC recordings, using the manifest's recordist/site folds.
- **Tier B, ARU:** non-avian-ml clips plus ARU background.
  - ARU frog folds are contiguous runs of 50-clip proxy blocks, standing in for site/night.
  - Background keeps its category-stratified folds.
- **Tier C, locked final test:** never trained on or scored during development.
  - The last 20% of each ARU species' blocks.
  - 20% of background clips, stratified by category.
  - For common classes only: up to 15% of their focal recordings, taken from groups that are all post-2025-04-01 (leakage-safe for BirdNET and Perch). Rare classes have no Tier C.

**Metrics** (`src/anuran/eval/metrics.py`). Units are recordings or clips; a unit's score is the max over its windows.
- **Tier A:** macro AP over evaluable classes, split rare/common, plus leakage-safe and air-only; recall at precision 0.9.
- **Tier B:**
  - Per class: AP, recall at precision 0.9/0.95, recall at a 0.5 threshold, review effort (fraction of clips to listen to for 90% recall), presence AP per proxy block.
  - Per class: background false positives per hour, and how often the class fires on other species' ARU clips.
  - Overall: any-frog detection AP.
- **95% CIs:** group bootstrap that resamples whole split groups.
- **Thresholds:** per class, chosen on dev pools at precision 0.9 and saved per run (`thresholds.json`). Tier C reuses them unchanged.

**Decision rule** (`pixi run compare A B`, a paired bootstrap on identical resamples). Adopt B over A only if:
1. the paired 95% CI of Δ ARU macro AP (`B_macro_ap`) is above 0. Use Δ rare-class focal AP (`A_macro_ap_rare`) when no ARU data applies; **and**
2. no class drops more than 0.05 AP.

Change one thing at a time; each experiment builds on the last adopted model.

**Bookkeeping:**
- Every run lands in `data/results/runs/<name>/` and `data/results/runs.csv`.
- Every run also appends one line to [results_log.md](results_log.md), which is tracked.
- Run Tier C **once** per final candidate, never to choose between options.

## Experiment queue
| ID | Config / command | Question | Status |
|---|---|---|---|
| E0 | `e0_focal_bg.yaml` | Does the current MLP head reproduce through the harness? (Tier A ≈ 0.64) | done 2026-10-01: 0.652 [0.619, 0.714] |
| E1 | E0's Tier B numbers | Honest ARU baseline; does the head see ARU frogs as frogs (domain-shortcut test)? | done: ranks well (AP 0.88), but recall at 0.5 is only 7–47% for 3 of 4 species → domain shift |
| E2 | `e2_focal_cdfw_bg.yaml` | How much does adding non-avian-ml clips to training help on ARU, and does it hurt focal? | done: **ADOPTED**, ARU AP +0.116; focal unchanged; likely optimistic (proxy groups) |
| LC | `pixi run learning-curve configs/experiments/e0_focal_bg.yaml` | How many ARU clips/sites per species? (→ [aru_data_needs.md](aru_data_needs.md)) | done: plateau ~25 clips from ≥3 blocks (AP 0.846 → 0.986) |
| E3 | (to build) mix focal calls into ARU background (SNR −5…15 dB, EQ, gain), re-embed | Can augmentation replace ARU data? Test leave-species-out: train without that species' ARU clips, score its non-avian-ml clips | |
| E4 | (to build) hard-negative mining from background and confusable pairs | Fewer false positives per hour, fewer cross-species hits? | |
| E5 | Perch v2 (finish embeddings), concat, `model.type: logreg` | Best encoder and head | |
| E6 | Text ladder: ~~review attributes~~ (approved 2026-10-01) → auxiliary attribute loss (`Head.attr`) → measured acoustic features (pulse rate, duration, band, slope) → attribute zero-shot for ANMI/ANEX/RALU/RAPR | Do field-guide descriptions help rare and confusable classes? Judge on `A_macro_ap_rare` (leakage-safe) and Tier B | |
| E7 | Post-hoc CWHR range prior from recorder lat/lon (inference only; `logit + log(w)`, soft floor outside range; splits acoustic groups by range) | Fewer cross-species confusions and better site-level precision (report with and without) | ranges built 2026-10-02/03 (`pixi run ranges`, 23/28 classes incl. PACH via GAP PSHY); next: ARU site coords (user can supply), then score Tier B with and without `adjust` |
| Final | Adopted stack, evaluated once on Tier C with frozen thresholds | Reported numbers | |

**Lower priority:**
- Strong labels for the rare classes (`pixi run export-review`, then jupyter-bioacoustic). Uncertain windows count in a recording's max score.
- Fine-tune the backbone (partial unfreeze), only once there are thousands of ARU windows, and judged on Tier B.
- ~~Underwater calls out of scope~~ (corrected 2026-10-02): RABO, MYLF and RADR/RAAU call only underwater, so their ARUs are underwater recorders. No per-medium handling.

## When new ARU data arrives
1. Put the files in `data/raw/aru/<name>/` unchanged and write a data card (`configs/datasets/README.md`). Run `pixi run catalog` to see it in the inventory figure. Insist on metadata (see the checklist in [aru_data_needs.md](aru_data_needs.md)). With real site groups, replace proxy blocks (as queued for non-avian-ml in `data/non_avian_ml.py`).
2. Add the data as a new manifest source, re-run `pixi run splits --force`, and re-run the adopted model and E0. Old results aren't comparable across splits; the hash enforces this.
3. Active learning:
   - Run the model over the raw deployments.
   - Review the top-scoring and most uncertain windows in jupyter-bioacoustic.
   - Add them to training by site, keeping test sites disjoint.
4. Other sources for the rare classes (check licenses): Macaulay Library, USGS ARMI, NPS / Yosemite MYLF monitoring.

## Labeling priority (2026-10-02)
Find and label **ARU** data; don't strong-label the focal data.
- **How:** yes/no per class on model-ranked 3 s windows in jupyter-bioacoustic, not boxes, plus random windows per site-night as verified negatives.
- **How much:** ~25–50 positives per species from ≥ 3–5 site-nights, capped at ~20 per site-night.
- **Order:** RADR/RAAU, ANCA, MYLF (more sites), SPHA, PSCA, WETO, RACA, ANCN.
- **Where labels go:** `labels/<dataset>/jba_<date>_<who>.csv`. The `jba` catalog reader gets written when the first session exists.
- **Rebuild needed before the next model run (2026-10-02).** Code now says `non_avian_ml` (source, ids `non_avian_ml_<CODE>_<ID>`, split source for background), but the stored manifest, embeddings and splits still say `cdfw`/`background_aru`. Run `pixi run non-avian-ml` → `manifest` → `embed` (re-embeds the 2,290 frog clips) → `weak-labels` → `background` → `splits --force` → re-run E0/E2. Do it together with (a) below. Afterwards delete the orphaned `data/interim/cdfw_recordings.parquet` and the `cdfw_*` embedding cache files.
- **Next data step:** (a) replace the non-avian-ml proxy blocks with the real sites from the catalog (2–5 per species), and (b) bring rana_sierrae_2022 from the catalog into the manifest/window labels. (b) needs a `neg` window state and day groups with 06-26 locked. Do both in one `splits --force`. Not started (no model work yet, per the user).

## Done
- 2026-10-01: ARU clips ingested (then called CDFW: `data/cdfw.py`, source `cdfw`; renamed non_avian_ml 2026-10-02; focal folds unchanged), embedded with BirdNET; `BG_DIR` fixed; eval harness, splits, metrics, runner, compare, learning curve, tests (`tests/test_eval.py`).
