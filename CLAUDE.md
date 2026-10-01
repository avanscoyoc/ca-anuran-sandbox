# ca-anuran-sandbox

A classifier for California frog and toad calls (28 acoustic classes: native plus non-native anurans, see `configs/species.yaml`).
It is a head trained on frozen BirdNET v2.4 embeddings (Perch v2 is the alternative encoder), with an option to use field-guide call text as guidance.

## Environment
- Devcontainer + pixi. The env is at `.pixi/envs/default` (Docker volume) and is installed by `pixi install --locked`.
- Run pipeline steps with `pixi run <task>` (see `[tasks]` in `pixi.toml`). Run tests with `pixi run test`.
- `data/` is gitignored (audio is not redistributable), so results live only on the host bind mount.

## Pipeline
scrape-herps / scrape-inat / scrape-xc / cdfw → manifest → embed → weak-labels → background → splits (frozen) → experiment / compare / learning-curve
Experiments: `pixi run experiment configs/experiments/<x>.yaml`, then `pixi run compare <run_a> <run_b>`. Don't re-run `splits --force` casually: it invalidates every earlier result.
(outputs: `data/interim/`, `data/processed/`, `data/results/<encoder>_<model>/`)

## Notes (read these first; they survive container rebuilds)
- `.claude/notes/status.md`: what has been built, plus model results
- `.claude/notes/data_inventory.md`: clips per class/source, metadata, caveats
- `.claude/notes/roadmap.md`: plan for training and testing; **how we test (harness, tiers, decision rule) + experiment queue**
- `.claude/notes/results_log.md`: one line per experiment run (auto-appended)
- `.claude/notes/aru_data_needs.md`: how many ARU clips/site-nights are needed, and what to ask CDFW for
- `.claude/notes/decisions.md`: append-only log of decisions and user answers

Keep these notes current: when results, data, or decisions change, update the relevant file (with dates).
