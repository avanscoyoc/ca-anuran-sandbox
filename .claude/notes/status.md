# Project status

_Last updated: 2026-10-01 (harness + CDFW ingestion added)_

## What has been built (`src/anuran/`)

| Step | Module | Output |
|---|---|---|
| Scrape californiaherps (audio + call text) | `scrape/herps.py` | `data/raw/herps/`, `data/interim/herps_recordings.parquet`, `herps_species_text.jsonl`, `data/text/species/*.txt` |
| Scrape iNaturalist (research-grade, CC licensed; 250 oldest per taxon + a "recent" pull of observations ≥ 2025-04-01) | `scrape/inat.py` | `data/raw/inat/audio/<label>/`, `inat_recordings.parquet` |
| Scrape xeno-canto (quality A/B, needs `$XC_API_KEY`) | `scrape/xenocanto.py` | `data/raw/xc/audio/`, `xc_recordings.parquet` |
| Merge, probe, dedupe, assign folds | `data/manifest.py` | `data/processed/manifest.parquet` |
| Per-class inventory | `data/inventory.py` | stdout table |
| Embed 3 s windows (1.5 s hop) | `features/embed_audio.py`, `features/encoders.py`, `data/windows.py` | `data/processed/embeddings/<enc>/{windows.parquet, embeddings.npy, cache/}` |
| Weak window labels (pos / uncertain) | `data/window_labels.py` | `window_labels.parquet` |
| Verified ARU background negatives | `notebooks/verify_background.ipynb`, `data/background.py` | `labels/background_verification.csv`, `background_{windows.parquet,embeddings.npy}` |
| Export windows for strong-label review | `data/export_review.py` | `data/processed/review/<enc>_windows.parquet` |
| Extract call attributes from field-guide text | `text/attributes.py` | `data/interim/call_attributes.yaml` |
| CDFW ARU clips as a manifest source (proxy blocks of 50 clips) | `data/cdfw.py` | `data/interim/cdfw_recordings.parquet` |
| Frozen eval splits (Tier A focal / B ARU / C locked), hash-checked | `eval/splits.py` | `data/processed/splits.{parquet,sha256}` |
| Metrics + group bootstrap CIs + paired comparison | `eval/metrics.py` | – |
| Config-driven experiments + registry | `experiment.py`, `configs/experiments/*.yaml` | `data/results/runs/<name>/`, `runs.csv`, `.claude/notes/results_log.md` |
| ARU learning curve | `learning_curve.py` | `data/results/learning_curve/` |
| Logistic-regression baseline (legacy, focal only) | `train_baseline.py` | `data/results/birdnet_v24_baseline/` |
| Multi-label MLP head (legacy script, focal only; the head itself is reused by the harness) | `models/head.py`, `train_head.py` | `data/results/birdnet_v24_head_audio/` |

### Key design choices
- **Acoustic groups.** Species whose calls can't be told apart are merged: PACH = PSRE/PSSI/PSHY, WETO = ANBB/ANBH, MYLF = RASI/RAMU. That gives 28 classes. The Coastal Tailed Frog (ASTR) is excluded because it has no call.
- **Cross-validation.** 5-fold `StratifiedGroupKFold` (seed 0). iNat/XC are grouped by `source:recordist`. Herps clips are grouped by `herps:label:county`; if no county is found, the whole species page becomes one group.
- **Weak labels.** A window is `pos` if it is the recording's only window, or if activity is ≥ 10 dB and its zero-shot score is within 4 logits of the recording's best window. Every other window is `uncertain`: never trained on, never used as a negative.
- **Secondary species** are masked out of the loss and out of AP.
- **Leakage-safe subset.** A recording counts as safe if it is from herps, or observed on or after the encoder's training cutoff (BirdNET 2023-07-01, Perch 2025-04-01).
- **Recording-level score** = max over all of the recording's windows, including uncertain ones.
- **Head** (`models/head.py`): Dropout(0.2) → Linear(1024→256) → GELU → Dropout → sigmoid outputs (one per class).
  - Loss: masked BCE with `pos_weight = clip(sqrt(neg/pos), 1, 30)`.
  - Training: AdamW (lr 1e-3, weight decay 1e-4), batch 256, 30 epochs, seeds 0/1/2.
  - Background windows are trained as all-zero targets (no explicit background class).
  - An unused auxiliary `attr` output and an `extra_losses` hook are ready for text-guided training.
- **Text attributes** (pitch, loudness, duration_s, notes_per_s, structure, rising_pitch, underwater) come from exact regex matches only, each with a verbatim evidence quote. All are `reviewed: false`, and **none are used in training yet**.

## Encoders
| Encoder | Input | Embedding | Status |
|---|---|---|---|
| BirdNET v2.4 (TFLite, `GLOBAL_AVG_POOL`) | 3 s @ 48 kHz | 1024-d | 41,419 frog windows (39,129 focal + 2,290 CDFW) + 639 background windows embedded |
| Perch v2 (Kaggle `perch_v2_cpu`) | 5 s @ 32 kHz | 1536-d | **Incomplete**: 306 windows only |

BirdNET already knows 6 of our classes (ANMI, ANCO, ANWO, ANCN, SCCO, LICA), so its zero-shot logits for them are stored as `zs_*`. Perch knows 23.

## Results (BirdNET v2.4 embeddings, pooled out-of-fold predictions)
"Rare" = ≤ 30 recordings.

| Model | Macro AP | Rare | Common | Leakage-safe | Air-only | Other |
|---|---|---|---|---|---|---|
| LogReg baseline (C=0.1, balanced) | 0.584 | 0.353 | 0.816 | 0.590 | – | top-1 recording accuracy 0.799. On the 6 classes BirdNET knows: trained 0.707 vs zero-shot 0.709 |
| MLP head + background, 3 seeds | **0.643** (26/28 classes evaluable) | 0.419 | 0.835 | 0.642 | 0.632 | detection AP: window 0.999, recording 1.000; 2.0% of background windows score ≥ 0.5 |

Per-class AP for the MLP head (all recordings / leakage-safe / air-only):

| Class | n rec | AP | Safe | Air | | Class | n rec | AP | Safe | Air |
|---|---|---|---|---|---|---|---|---|---|---|
| ANMI | 4 | 0.03 | 0.05 | 0.03 | | SPHA | 39 | 0.62 | 0.56 | 0.62 |
| ANEX | 6 | n/a | | | | SPIN | 51 | 0.71 | 0.36 | 0.75 |
| ANCN | 7 | 0.70 | 0.61 | 0.70 | | ANPU | 54 | 0.77 | 0.82 | 0.77 |
| RALU | 7 | n/a | | | | WETO | 57 | 0.81 | 0.76 | 0.81 |
| INAL | 7 | 1.00 | 1.00 | 1.00 | | PSCA | 65 | 0.89 | 0.92 | 0.89 |
| RAPR | 8 | 0.08 | 0.10 | 0.09 | | ANCO | 65 | 0.83 | 0.70 | 0.83 |
| ANCA | 9 | 0.29 | 0.26 | 0.29 | | SCCO | 153 | 0.92 | 0.95 | 0.92 |
| RACA | 10 | 0.73 | 0.78 | 0.80 | | ANWO | 161 | 0.83 | 0.86 | 0.83 |
| MYLF | 10 | 0.38 | 0.40 | 0.54 | | ELCO | 249 | 0.98 | 0.96 | 0.98 |
| XELA | 10 | 0.65 | 1.00 | 0.06 | | LICA | 257 | 0.87 | 0.87 | 0.87 |
| LIYA | 11 | 0.02 | 0.04 | 0.02 | | LIBE | 272 | 0.75 | 0.72 | 0.75 |
| RABO | 14 | 0.01 | 0.01 | 0.02 | | LISP | 287 | 0.86 | 0.87 | 0.86 |
| RADR | 17 | 0.40 | 0.44 | 0.42 | | PACH | 295 | 0.92 | 0.92 | 0.92 |
| RAAU | 29 | 0.73 | 0.80 | 0.74 | | LIPI | 330 | 0.93 | 0.89 | 0.93 |

ANEX and RALU each come from a single site, so a held-out fold never has training data for them.

**How to read these numbers:**
- **Rare-class APs are noisy.** Most rare classes have 4–17 recordings, so one recording moves AP a lot.
- **Detection AP ≈ 1.0 may be a domain shortcut.** Every negative is an ARU clip and every positive is a focal recording (see [data_inventory.md](data_inventory.md), caveat 2).
- **No result yet measures performance on ARU recordings of frogs.**

## Known bugs / loose ends
- ~~`background.py` points at the wrong folder~~ (fixed 2026-10-01: `BG_DIR = RAW / "background_aru"`).
- **Results above are pre-harness** (no CIs, no ARU tier). Current numbers are in [results_log.md](results_log.md).
- **Perch v2 embeddings are incomplete.**
- **`train_head --setting` only accepts `audio`.** The text settings are not implemented.
