# How much ARU data we need, and what to ask for

_Last updated: 2026-10-01._

**Use case** (user, 2026-10-01): (1) species presence per site/night and (2) screening clips for human review.

## Count site-nights, not clips
Clips from the same recorder and night are near-duplicates: same noise floor, same chorus, often the same individual frogs. In our CDFW clips, the noise fingerprints of adjacent clip numbers are far more similar than random pairs, and the similarity fades over about 20–100 clips. So 600 clips from 2 sites carry about the information of a few dozen independent clips. Use the numbers below per **independent site-night**, and cap any one site-night at about 10–20 clips.

## Testing (sets the floor, per species)
- **Positives:** ≥ 20–30 site-nights with the species calling, from ≥ 5 sites. About 60 independent positives gives roughly ±10% on a recall of 0.8.
- **Verified absences:** about as many site-nights from habitat where the species could plausibly occur but was not calling (or is absent). These measure site-level false positives, which matter for presence mapping.
- **Full nights, not only clips:** false positives per hour and the review load can only be measured on continuous audio.

## Training (starting rule of thumb; our learning curve replaces it)
- **Published guide:** heads on frozen BirdNET/Perch embeddings reach most of their accuracy with about 16–64 examples per class (Ghani et al. 2023, few-shot transfer on bird embeddings). This hasn't been checked on frogs.
- **Our case:** we already have focal recordings, so ARU clips mostly teach what the species sounds like *through an ARU*.
- **Default target:** 50–100 ARU clips per species from ≥ 5 site-nights.
- **Confusable sets:** 100–200 clips. These are the toad trills (ANCA/ANWO/ANCO/ANPU/WETO), the leopard frogs (LIPI/LIBE/LISP/LIYA), RADR/RAAU, and MYLF vs RABO.
- **Negatives are cheap and drive false positives.** Collect hours of verified frog-free audio from every site and season. Over-sample hard negatives: crickets and katydids, wind, rain, streams, night birds, voices, vehicles.
- **The measured answer** comes from `pixi run learning-curve`: AP vs number of ARU clips and proxy sites for ANWO/LICA/PACH/RABO, in `data/results/learning_curve/summary.md`. Where the curve flattens becomes the target for the other species. Record the result here when it's done.

## Learning-curve result (2026-10-01, proxy-grouped)
5 test folds × 2 repeats, ARU pool = CDFW clips + background, macro over ANWO/LICA/PACH/RABO:
- **0 ARU clips:** AP 0.846, recall at precision 0.9 = 0.64. RABO is worst at 0.645.
- **5 clips from 1 block:** AP 0.964, recall at P0.9 = 0.89. RABO jumps to 1.00.
- **25 clips from 3 blocks:** AP 0.986, recall at P0.9 = 0.97. This is about the plateau.
- **All clips from all blocks:** AP 0.995, recall at P0.9 = 0.99.
- **Spreading clips across blocks helps a little:** 25 clips from 1 → 3 blocks gives 0.978 → 0.986, recall 0.95 → 0.97.

**Reading:** for these species, a few dozen ARU clips from ≥ 3 groups get nearly all of the gain. The gain is mostly calibration to ARU audio, not learning the species.

**Caveat:** these are proxy blocks, not real sites, and the test pool holds only 4 species, so the plateau is likely reached sooner here than it would be at new sites. **Working target stays at 50–100 clips from ≥ 5 site-nights per species**, with ~25 clips from 3+ sites as the minimum useful amount.

## Species priority for new ARU data

| Priority | Species | Why |
|---|---|---|
| 1 | RADR, RAAU, ANCA, MYLF (RASI/RAMU), SPHA, PSCA, WETO, RACA, ANCN | CA natives of management interest; ≤ 65 focal recordings, several from one site; no ARU data |
| 2 | SCCO, SPIN, ANPU, ANCO, RAPR, RALU, ANEX | Desert/peripheral natives; RAPR/RALU/ANEX are tiny and single-site |
| 3 | LIPI, LIBE, LISP | Non-natives; focal data is mostly out-of-state iNat, so we need CA ARU data from near known populations |
| More sites | ANWO, LICA, PACH, RABO | Have CDFW clips, but from unknown and probably few sites |
| Needs a hydrophone | XELA | Calls underwater; air ARUs won't help |
| Can't validate | ANMI, LIYA, INAL | Possibly extirpated in CA. Keep them in the model, flagged "no test data" |

## Checklist for CDFW (or any ARU partner)
- [ ] Site ID, recorder ID, lat/lon, deployment start/end, and recording start time for every file.
- [ ] Which site/date each of the 2,290 existing clips came from. This turns the proxy blocks into real groups and makes the current Tier B numbers trustworthy.
- [ ] Full-night recordings (or at least whole hours) for some site-nights, including nights with no frogs.
- [ ] How the clips were chosen (detector output? manual browsing?). This tells us how biased a test set they are.
- [ ] Recorder model and sample rate per deployment. RABO clips are 32 kHz and everything else is 48 kHz; is that a different recorder model or different settings?
- [ ] Which species were *checked for and absent* at each site (true absences).

## Class separability (2026-10-01)
Figure: `data/results/figures/tsne_by_class.png`; table: `data/results/figures/separability.csv`.
- **Purity:** for each species' recordings, the share of their 10 nearest neighbours that are the same class. Measured in the 1024-d BirdNET space, counting only neighbours from other recordists/sites.
  - Least separable: LIYA, RALU, ANEX, ANMI (0.00); MYLF 0.08; RAPR 0.10; ANCA 0.17; XELA 0.18; RABO 0.20; ANCN, RADR 0.21.
  - Mid-range confusions: SPIN↔SPHA (0.33/0.43), RAAU↔RADR (0.34/0.21), LIBE↔LISP (0.47/0.51).
  - Best separated: ELCO 0.94, SCCO 0.82, PACH 0.81, LIPI 0.78.
  - Caveat: classes with fewer than ~10 recordings score low partly because few same-class neighbours exist from other sites.
- **ARU clips sit apart from focal clips.** The CDFW clips form their own clusters, away from the same species' focal recordings, for all 4 species. This is the domain shift, made visible. It is why ARU clips per species matter more than more focal data.
- **Resulting data priority:**
  1. Natives that are both rare and poorly separated: RADR/RAAU, MYLF (vs RABO), ANCA, RACA, ANCN, RAPR/RALU.
  2. The confusable spadefoots SPHA/SPIN.
  3. Non-native LIBE/LISP, lower management priority.
