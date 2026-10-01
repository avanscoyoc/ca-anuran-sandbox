# Decisions & user answers (append-only, newest last)

- **2026-10-01** CDFW ARU labels are correct, per the user. Keep RABO = foothill yellow-legged frog; map PSRE → PACH, as the rest of the pipeline does.
- **2026-10-01** It's unknown whether CDFW site/date metadata exists. Use contiguous clip-ID blocks as a proxy for groups for now, and swap in real site groups if the metadata turns up.
- **2026-10-01** Project notes live in `.claude/notes/` (git-tracked), with `CLAUDE.md` pointing to them, because container rebuilds wipe `/root/.claude`.
- **2026-10-01** Use case (user): **presence per site/night + screening clips for human review**. Primary metrics are therefore Tier B (ARU) AP, recall at precision 0.9, review effort and false positives per hour, not focal AP.
- **2026-10-01** CDFW clip numbers are contiguous per species, so there are no gaps to split on. Neighbouring numbers share noise fingerprints, decaying over ~20–100 clips. Proxy group = a contiguous block of 50 clips; CDFW folds = contiguous runs of blocks; the last 20% of blocks are locked as Tier C. Results on CDFW are "proxy-grouped" until real site metadata arrives.
- **2026-10-01** Evaluation harness adopted (`anuran.experiment`, `anuran.eval`). Splits are frozen and hash-checked. Decision rule: adopt only if the paired 95% CI of Δ ARU macro AP > 0 (focal rare-class AP when no ARU data applies) and no class drops > 0.05 AP.
- **2026-10-01** Tier C focal lock: only classes with ≥ 30 recordings, only groups with all recordings ≥ 2025-04-01, ≤ 15% of a class and ≤ half of its eligible groups. Rare classes stay entirely in CV, so they have no locked test.
- **2026-10-01** The legacy `train_baseline` / `train_head` scripts are restricted to focal sources (`load(sources=FOCAL)`) so their old numbers still reproduce. New work goes through the harness.
- **2026-10-01** The user approved all extracted call attributes as-is. Recorded in `configs/attribute_review.yaml` (tracked, values only); `text/attributes.py` marks a class reviewed only while its values still match. ELCO and LISP have no description text. This unblocks E6 (text ladder) step 1.
