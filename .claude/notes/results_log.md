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
| 2026-10-01 | e0_focal_bg | a2dde73 | 04caa7a1505cfb7c | 0.652 [0.619, 0.714] | 0.431 [0.355, 0.558] | 0.841 [0.808, 0.868] | 0.650 [0.612, 0.733] | 0.879 [0.798, 0.923] | 0.952 [0.927, 0.968] |

> **E1 reading (E0's Tier B, 2026-10-01):** the focal-trained head *ranks* ARU clips well (B_macro_ap 0.88, detection 0.95), but its scores are badly calibrated for ARU audio. At a 0.5 threshold it finds only 11% of PACH, 7% of RABO and 47% of LICA ARU clips (ANWO 86%). So there is a domain shift, less a hard shortcut: thresholds tuned on focal data would miss most ARU calls. The ARU pool holds only 4 species + background, so its AP is easier than a real deployment. Expect E2 (CDFW clips in training) to fix calibration mostly.
| 2026-10-01 | e2_focal_cdfw_bg | a2dde73 | 04caa7a1505cfb7c | 0.655 [0.621, 0.721] | 0.435 [0.357, 0.555] | 0.844 [0.813, 0.871] | 0.655 [0.619, 0.734] | 0.995 [0.989, 0.998] | 1.000 [1.000, 1.000] |

> **E2 vs E0 (paired bootstrap, 2026-10-01): ADOPT.** ARU macro AP 0.879 → 0.995 (Δ +0.116 [+0.073, +0.209]); recall at 0.5 is now 96–100% for all 4 CDFW species. Focal Tier A is unchanged (Δ +0.004, n.s.). Worst focal class ANCA −0.047, n.s. and inside the 0.05 rule.
> **Caveat: 0.995 is probably optimistic.** The CDFW groups are proxy blocks; train and test clips likely share sites and nights, and the ARU pool has only 4 species + background. Early learning-curve points already reach about 0.93 macro AP with 5 clips from 1 block, which also fits "test clips resemble training clips". Treat Tier B as an upper bound until CDFW site/date metadata gives real groups, or until ARU data from new sites arrives.
| learning curve | ARU macro AP with 0 CDFW clips 0.846 -> all clips 0.995; see data/results/learning_curve/summary.md |
