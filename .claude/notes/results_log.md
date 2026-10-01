# Results log

`python -m anuran.experiment run` and `learning_curve` append rows here automatically. Full outputs are in `data/results/runs/<name>/` (gitignored).

**Values:** point estimate [95% CI]. The CIs come from a bootstrap that resamples whole split groups (recordist/site for focal recordings, 50-clip proxy block for CDFW clips, single clip for background).

**Metrics:**
- **A_\*** — Tier A: focal herps/iNat/XC recordings, per recording, 26 evaluable classes.
- **B_\*** — Tier B: the ARU pool (CDFW clips + background) for ANWO/LICA/PACH/RABO.
- **B_detection_ap** — any frog vs background, on ARU clips.

**Comparing runs:** use `pixi run compare <a> <b>`, a paired bootstrap that applies the decision rule in [roadmap.md](roadmap.md). Only runs with the same splits hash are comparable.

**Earlier results (before the harness):**
- `train_head` audio: macro AP 0.643 (rare 0.419, common 0.835), no CIs, no ARU tier.
- Logistic-regression baseline: macro AP 0.584.

| date | run | git | splits | A_macro_ap | A_rare | A_common | A_leakage_safe | B_macro_ap | B_detection_ap |
|---|---|---|---|---|---|---|---|---|---|
