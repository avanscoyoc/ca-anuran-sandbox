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
_(fill in after the run: plateau clip count, gain from 1 → 3 → all groups)_

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
