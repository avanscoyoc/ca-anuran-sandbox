# Data inventory

_Last updated: 2026-10-02. Counts come from `data/processed/manifest.parquet` and from `data/raw/`. **Per-class figure + table: `pixi run catalog` → `data/results/figures/inventory.{png,csv}`** (hours, 3 s windows, sites, days by label kind). Each dataset's card is in `configs/datasets/`._

## Sources
| Source | Recordings | What it is | Dates | Metadata |
|---|---|---|---|---|
| californiaherps (`herps`) | 236 | Focal recordings from the field-guide site, plus call descriptions | Mostly undated | Caption, county/place, call types, recordist (mostly one person), underwater flag |
| iNaturalist (`inat`) | 2,204 | Research-grade, CC-licensed observation sounds. Many are from outside California, especially the non-native *Lithobates* and ELCO | 2002 – 2026-08 | Date, lat/lon, recordist, license, place |
| xeno-canto (`xc`) | 44 | Quality A/B only | 1988 – 2026-07 | Date, lat/lon, recordist, quality, "also" species |
| **non-avian-ml frog clips** (`data/raw/aru/non_avian_ml/<code>/`; formerly "CDFW ARU") | **2,290** 3 s wavs | Clips from autonomous recording units (ARUs). In the manifest as source `non_avian_ml` (the stored manifest still says `cdfw` until the rebuild) and embedded since 2026-10-01 (`data/cdfw.py`) | none | **None.** Only the species folder and a numeric clip ID (e.g. `woodhouses_toad_5910.wav`). IDs are contiguous per species; groups are proxy blocks of 50 consecutive clips |
| non-avian-ml background (`data/raw/aru/non_avian_ml/background_no_frog/`) | 645 wavs, 3 s each | Noise categories. Reviewed by ear: 692 rows, 42 had frogs (removed) → 639 frog-free windows used as negatives | none | Category only. Same non-avian-ml project as the frog clips (one clip sheet, one ID space) |
| **rana_sierrae_2022** (`data/raw/aru/rana_sierrae_2022/`) | 672 × 10 s mp3 (32 kHz) | Underwater AudioMoth soundscapes from 1 lake (device MSD-0558), exhaustively annotated in Raven for R. sierrae (MYLF) call types A–E. **In the catalog only; not yet in the manifest or splits** | 2022-06-20 – 06-26, 24 h | Raven boxes (time + frequency + call type), device, date/time; Dryad, CC0 (verify) |

Mean recording length: herps 25 s, iNat 24 s, XC 62 s. The manifest has 4,774 recordings after dedupe: 2,484 focal + 2,290 non-avian-ml.

### non-avian-ml ARU clips

| Folder | Class | Clips | Sample rate |
|---|---|---|---|
| ANWO (`woodhouses_toad_*`) | ANWO | 638 | 48 kHz |
| LICA (`american_bullfrog_*`) | LICA | 421 | 48 kHz |
| PSRE (`pacific_chorus_frog_*`) | PACH | 795 | 48 kHz |
| RABO (`yellow_legged_frog_*`) | RABO | 436 | **32 kHz** |

The user confirmed these labels are correct (2026-10-01).

**Sites and dates are known (2026-10-02).** Both the frog and background clips come from the non-avian-ml project (github.com/avanscoyoc/non-avian-ml). Its clip sheet `data/raw/aru/non_avian_ml/all_3s_clips.csv` gives each clip's source recording, from which `pixi run catalog` parses the site, date and `source_recording`.

| Class | Clips | Sites | Days | Source recordings | Sites (recorders) |
|---|---|---|---|---|---|
| ANWO | 638 | 3 | 8 | 13 | BD-29267A, BD-29267B, BD-31393A (Mojave, May 2023) |
| LICA | 421 | 2 | 8 | 10 | BD-29267B, BD-31209B |
| PACH | 795 | 5 | 8 | 16 | BD-29267A/B, BD-31209A/B, BD-32198B |
| RABO | 436 | 2 | 6 | 44 | yellow-legged-frog-1, -4 (underwater, March 2026) |
| background | 639 | 113 | 69 | 144 | many; the 52 nutria clips have no site |

So the frog clips come from **2–5 sites per species**. The 50-clip proxy blocks (8–16 per species) overstate how independent the clips are, and the Tier B/learning-curve numbers are optimistic. The manifest and splits still use proxy blocks; switching to real sites needs `splits --force` and re-runs (not done; no model work yet).

Background sample rates: 585 at 48 kHz, 54 at 32 kHz, and 6 at 24 kHz. The 24 kHz clips are dropped by `MIN_SR`.

## Recordings and call attributes per acoustic class
Counts are from the manifest. Attributes come from the Cal Herps descriptions (`configs/attribute_review.yaml`), **all approved by the user 2026-10-01**. Blank = the text doesn't say. ELCO and LISP have no description text. Structure labels name the literal words matched: `snore_rattle` = "snore"/"rattle", `peep_plink`, `chuckle_cluck`, `croak_quack_ribit`, `bleat_groan_growl`, `drone_bellow`, `click_knock`, `series_of` = "a series of …".

| Class | herps | inat | xc | non_avian_ml (ARU) | Focal total | Underwater recs | Pitch | Loudness | Call duration (s) | Notes/s | Structure | Rising pitch | Calls underwater (text) | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANMI Arizona Toad | 4 | 0 | 0 |  | 4 |  | high | loud | 5.7 |  | trill | yes |  | possibly extirpated |
| ANEX Black Toad | 6 | 0 | 0 |  | 6 |  | high | quiet |  |  | peep_plink |  |  | single site → not evaluable |
| ANCN Yosemite Toad | 5 | 2 | 0 |  | 7 |  |  | loud |  |  | trill |  |  |  |
| RALU Columbia Spotted Frog | 7 | 0 | 0 |  | 7 | 2 |  | quiet |  |  | series_of, click_knock, grunt |  | yes | single site → not evaluable |
| INAL Sonoran Desert Toad | 3 | 4 | 0 |  | 7 |  | low | quiet | 0.5–1 |  | whistle |  |  | possibly extirpated |
| RAPR Oregon Spotted Frog | 7 | 1 | 0 |  | 8 | 1 | low | quiet |  |  | series_of, click_knock |  | yes |  |
| ANCA Arroyo Toad | 5 | 4 | 0 |  | 9 |  |  |  | 10 |  | trill | yes |  |  |
| RACA Cascades Frog | 10 | 0 | 0 |  | 10 |  |  | quiet |  |  | series_of, chuckle_cluck |  | yes |  |
| MYLF Mountain yellow-legged frog complex | 10 | 0 | 0 |  | 10 | 3 |  | quiet |  |  |  | yes | yes | RASI 9 + RAMU 1; text from RASI page only |
| XELA African Clawed Frog | 6 | 1 | 3 |  | 10 | 6 |  | quiet | 0.5 |  | trill |  | yes | calls underwater; non-native |
| LIYA Lowland Leopard Frog | 7 | 4 | 0 |  | 11 |  |  | quiet |  |  | chuckle_cluck |  |  | possibly extirpated |
| RABO Foothill Yellow-legged Frog | 10 | 4 | 0 | 436 | 14 | 9 | low | quiet |  | 4–6 | series_of, snore_rattle, grunt |  | yes |  |
| RADR California Red-legged Frog | 10 | 7 | 0 |  | 17 |  |  | quiet | 1–3 |  | series_of, bleat_groan_growl |  | yes |  |
| RAAU Northern Red-legged Frog | 13 | 16 | 0 |  | 29 | 4 |  | quiet | 1–3 |  | series_of |  | yes |  |
| SPHA Western Spadefoot | 8 | 31 | 0 |  | 39 |  |  | loud | ≤1 |  | trill, snore_rattle |  |  |  |
| SPIN Great Basin Spadefoot | 5 | 46 | 0 |  | 51 |  |  | loud |  |  | snore_rattle |  |  |  |
| ANPU Red-spotted Toad | 9 | 44 | 1 |  | 54 |  | high | loud | ≤10 |  | trill |  |  |  |
| WETO Western toad | 21 | 36 | 0 |  | 57 |  | high | quiet |  |  | peep_plink |  |  | ANBB + ANBH |
| PSCA California Treefrog | 7 | 58 | 0 |  | 65 |  | low | loud |  |  | trill, croak_quack_ribit |  |  |  |
| ANCO Great Plains Toad | 8 | 55 | 2 |  | 65 |  |  | loud | 5–60 |  | trill |  |  |  |
| SCCO Couch's Spadefoot | 6 | 145 | 2 |  | 153 |  |  | loud |  |  | bleat_groan_growl | no |  |  |
| ANWO Rocky Mountain Toad | 9 | 151 | 1 | 638 | 161 |  |  | loud | 1–4 |  | snore_rattle, bleat_groan_growl |  |  |  |
| ELCO Common Coqui | 1 | 246 | 2 |  | 249 |  |  |  |  |  |  |  |  | non-native |
| LICA American Bullfrog | 11 | 246 | 0 | 421 | 257 |  | low | loud |  |  | drone_bellow |  |  | non-native |
| LIBE Rio Grande Leopard Frog | 6 | 263 | 3 |  | 272 |  | low | loud |  |  | trill, snore_rattle, chuckle_cluck |  |  | non-native |
| LISP Southern Leopard Frog | 1 | 282 | 4 |  | 287 |  |  |  |  |  |  |  |  | non-native |
| PACH Pacific chorus frog complex | 32 | 237 | 26 | 795 | 295 |  |  | loud |  |  | croak_quack_ribit | yes |  | PSRE/PSSI/PSHY |
| LIPI Northern Leopard Frog | 9 | 321 | 0 |  | 330 |  |  | moderate |  |  | snore_rattle, croak_quack_ribit, chuckle_cluck |  |  | non-native in CA |

**Weak window labels** (BirdNET, 3 s windows): 19,068 `pos` and 20,061 `uncertain`.

**`label_rank`**: 2,194 recordings are labeled to species and 290 only to a group. For example, iNat lumps the Pacific chorus treefrogs, so those clips can only be labeled PACH.

## Manifest columns
`recording_id, source, source_id, label, label_rank, acoustic_class, secondary_species, call_types, recordist, license, quality, lat, lon, date, place, caption, page_url, url, path, duration_s, md5, medium, split_group, fold`

## Text
- **Per-species field-guide text:** `data/text/species/<CODE>_<Name>.txt` (32 files) and `data/interim/herps_species_text.jsonl`.
- **Extracted call attributes:** `data/interim/call_attributes.yaml` (generated, gitignored because it quotes copyrighted text). None are reviewed yet.

## Range maps (CWHR 2026-10-02; GAP for PSHY 2026-10-03)
- 24 CWHR range maps (CDFW BIOS, CC-BY, downloaded 2026-10-02), one year-round polygon per species, clipped at the state line. Config: `configs/ranges.yaml`; check: `data/results/ranges/range_check.csv`.
- **PSHY** has no CWHR map, so it uses the USGS GAP range map aBCTRx (ScienceBase 59f5e1b7e4b063d5d307da87, doi:10.5066/F7ST7P0Z, public domain). That's HUC12 sub-watersheds dissolved to one polygon; every HUC12 is Known/extant, native and year-round (the code requires this). It isn't clipped at the state line (it reaches NV/AZ) and covers ~97,600 km² in California. Overlaps within PACH: PSHY∩PSSI ~18,400 km² (19% of PSHY's California range), PSHY∩PSRE 0, PSSI∩PSRE ~750 km².
- **Coverage:** 23 of 28 acoustic classes have a prior, including PACH. There is no map at all for ANMI, LIYA, XELA, ELCO or LISP. WETO has a species-level map only (A. boreas), so range can't split ANBB from ANBH. MYLF (RASI A070 vs RAMU A044) and PSRE vs PSSI can be split by range.
- **Check against geotagged California recordings in the catalog (n = 284, inside the union of all maps):** LICA 29/29, PSCA 45/46 (1 at 3 km), SPHA 21/23 (≤ 5 km), WETO 13/13, RABO/RADR/ANCA/SPIN all inside. PACH: before PSHY was added, 54 of 143 were outside PSRE ∪ PSSI by up to 376 km, all south of 35.7° N. With the GAP PSHY map, **143 of 143 are inside**. ANWO (1 of 3 at 104 km) and ANPU (1 of 3 at 56 km) need a look. Most rare classes have 0 California records, so they can't be calibrated from focal data and must wait for Tier B site coordinates.

## Caveats
1. **The non-avian-ml ARU clips are the only in-domain (ARU) frog data.** They were ingested 2026-10-01. The manifest still groups them by proxy blocks (real sites are now known, see above): adjacent clip numbers share noise fingerprints, so blocks of 50 consecutive clips and contiguous folds limit leakage to block boundaries. It isn't eliminated, and Tier B numbers may be somewhat optimistic until real site metadata arrives.
2. **Possible domain shortcut.** Every background negative comes from an ARU, and every frog positive is a focal or handheld recording. The near-perfect detection AP may only reflect "ARU vs not-ARU".
3. **Possible sample-rate shortcut.** Every non-avian-ml RABO clip is 32 kHz; the other frog clips and most background clips are 48 kHz. Resample everything to 48 kHz, then check that 32 kHz origin isn't predictive.
4. **The leakage-safe definition is partly an assumption.** All herps clips count as "safe", but BirdNET's training data may include some of them. Most iNat clips predate BirdNET's cutoff, so the "all recordings" AP is optimistic for the 6 classes BirdNET already knows.
5. **Class imbalance and geography.** Most data are non-native *Lithobates*, ELCO, and toads recorded outside California. The California natives that matter most for management are the rare classes.
6. **Underwater calls.** RABO, MYLF and the red-legged frogs (RADR/RAAU) call only underwater, so their ARUs are underwater recorders (user, 2026-10-02); XELA also calls underwater. This is how these species are recorded and gets no special handling. Focal `medium` flags remain from the text, for reference only.
7. **Background folds are stratified by category, not by site** (no site info). That makes them a looser split than the frog folds.

### rana_sierrae_2022 (added 2026-10-02)
Card: `configs/datasets/rana_sierrae_2022.yaml`; call types: `configs/call_types.yaml`.

| | Files | Boxes | 3 s windows |
|---|---|---|---|
| With R. sierrae calls | 434 | 1,236: A 763, E 205, C 131, D 103, B 34 | 1,678 positive |
| Negatives (no box: 195; X box only: 43) | 238 | 129 X boxes dropped (X = no call, user) | 2,993 negative in total, including the 639 noise clips |

- The 434 call files cover 7 days at 1 site, and calls occur at all hours.
- Caveats:
  - Single site, so results are within-site.
  - Aquatic-insect stridulation (Corixidae) is a frequent in-file hard negative.
  - Lossy mp3 at 32 kHz, like the non-avian-ml RABO clips.

