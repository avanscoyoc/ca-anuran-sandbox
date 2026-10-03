# ca-anuran-sandbox

A classifier for California frog and toad calls (28 acoustic classes: native plus non-native anurans, see `configs/species.yaml`).
It is a head trained on frozen BirdNET v2.4 embeddings (Perch v2 is the alternative encoder), with an option to use field-guide call text as guidance.

## Environment
- Devcontainer + pixi. The env is at `.pixi/envs/default` (Docker volume) and is installed by `pixi install --locked`.
- Run pipeline steps with `pixi run <task>` (see `[tasks]` in `pixi.toml`). Run tests with `pixi run test`.
- `data/` is gitignored (audio is not redistributable), so results live only on the host bind mount.

## Data layout (2026-10-02)
Folders say where audio came from; tables say what is in it.
- `data/raw/focal/<dataset>/`: scraped focal recordings (herps, inat, xc). `data/raw/aru/<dataset>/`: anything from an autonomous recorder (`non_avian_ml/` = the non-avian-ml project's 3 s clips: `<CODE>/` frog folders + `background_no_frog/` + `all_3s_clips.csv`; `rana_sierrae_2022/`). Raw data is never edited.
- `configs/datasets/<dataset>.yaml`: one **data card** per dataset (provenance, label kind, reader, label map, caveats). See `configs/datasets/README.md`.
- `labels/<dataset>/`: labels **we** make (verification passes, jupyter-bioacoustic reviews). Tracked in git.
- `data/raw/range/<cwhr|gap>/`: species range maps, range only and never habitat suitability. CWHR (GeoJSON, CC-BY) is the default; USGS GAP (shapefile zip, public domain) is used only where CWHR has no map (PSHY). `configs/ranges.yaml` maps each class to its map. `pixi run ranges` downloads them and writes `data/processed/ranges.parquet` and `data/results/ranges/range_check.csv`. Used **only at inference** as a per-site prior (`anuran.data.ranges.adjust`), never as a training input.
- `data/text/`: copyrighted source text (field guide in `species/`, papers in `papers/`). `configs/call_types.yaml` holds paraphrased call-type definitions.
- `pixi run catalog`: every card becomes `data/processed/catalog/{recordings,spans}.parquet` plus the per-class inventory figure `data/results/figures/inventory.{png,csv}` (hours, 3 s windows, sites, days by label kind: weak / clip / box / negative).
- Adding a dataset: put the files in `data/raw/<focal|aru>/<name>/`, copy the closest card, add a reader in `data/catalog.py` only for a new format, then run `pixi run catalog`.
- Don't cut audio into per-species folders. Windows are views (`start_s`/`end_s`) on raw files, because window length depends on the encoder.

## Pipeline
scrape-herps / scrape-inat / scrape-xc / non-avian-ml → manifest → embed → weak-labels → background → splits (frozen) → experiment / compare / learning-curve
Experiments: `pixi run experiment configs/experiments/<x>.yaml`, then `pixi run compare <run_a> <run_b>`. Don't re-run `splits --force` casually: it invalidates every earlier result.
(outputs: `data/interim/`, `data/processed/`, `data/results/<encoder>_<model>/`)

## Notes (read these first; they survive container rebuilds)
- `.claude/notes/status.md`: what has been built, plus model results
- `.claude/notes/data_inventory.md`: clips per class/source, metadata, caveats
- `.claude/notes/roadmap.md`: plan for training and testing; **how we test (harness, tiers, decision rule) + experiment queue**
- `.claude/notes/results_log.md`: one line per experiment run (auto-appended)
- `.claude/notes/aru_data_needs.md`: how many ARU clips/site-nights are needed, and what to ask ARU partners for
- `.claude/notes/decisions.md`: append-only log of decisions and user answers

Keep these notes current: when results, data, or decisions change, update the relevant file (with dates).
