# Data inventory

_Last updated: 2026-10-01. Counts come from `data/processed/manifest.parquet` and from `data/raw/`._

## Sources
| Source | Recordings | What it is | Dates | Metadata |
|---|---|---|---|---|
| californiaherps (`herps`) | 236 | Focal recordings from the field-guide site, plus call descriptions | Mostly undated | Caption, county/place, call types, recordist (mostly one person), underwater flag |
| iNaturalist (`inat`) | 2,204 | Research-grade, CC-licensed observation sounds. Many are from outside California, especially the non-native *Lithobates* and ELCO | 2002 – 2026-08 | Date, lat/lon, recordist, license, place |
| xeno-canto (`xc`) | 44 | Quality A/B only | 1988 – 2026-07 | Date, lat/lon, recordist, quality, "also" species |
| **CDFW ARU** (`data/raw/cdfw_aru/audio/<code>/`) | **2,290** 3 s wavs | Clips from autonomous recording units (ARUs). In the manifest as source `cdfw` and embedded since 2026-10-01 (`data/cdfw.py`) | none | **None.** Only the species folder and a numeric clip ID (e.g. `woodhouses_toad_5910.wav`). IDs are contiguous per species; groups are proxy blocks of 50 consecutive clips |
| ARU background (`data/raw/background_aru/`) | 645 wavs, 3 s each | Noise categories. Reviewed by ear: 692 rows, 42 had frogs (removed) → 639 frog-free windows used as negatives | none | Category only. Clip IDs share a number space with the CDFW frog clips (same dataset?) |

Mean recording length: herps 25 s, iNat 24 s, XC 62 s. The manifest has 4,774 recordings after dedupe: 2,484 focal + 2,290 CDFW.

### CDFW ARU clips

| Folder | Class | Clips | Sample rate |
|---|---|---|---|
| ANWO (`woodhouses_toad_*`) | ANWO | 638 | 48 kHz |
| LICA (`american_bullfrog_*`) | LICA | 421 | 48 kHz |
| PSRE (`pacific_chorus_frog_*`) | PACH | 795 | 48 kHz |
| RABO (`yellow_legged_frog_*`) | RABO | 436 | **32 kHz** |

The user confirmed these labels are correct (2026-10-01). Nobody yet knows whether site, recorder or date metadata exists.

Background sample rates: 585 at 48 kHz, 54 at 32 kHz, and 6 at 24 kHz. The 24 kHz clips are dropped by `MIN_SR`.

## Recordings and call attributes per acoustic class
Counts are from the manifest. Attributes come from the Cal Herps descriptions (`configs/attribute_review.yaml`), **all approved by the user 2026-10-01**. Blank = the text doesn't say. ELCO and LISP have no description text. Structure labels name the literal words matched: `snore_rattle` = "snore"/"rattle", `peep_plink`, `chuckle_cluck`, `croak_quack_ribit`, `bleat_groan_growl`, `drone_bellow`, `click_knock`, `series_of` = "a series of …".

| Class | herps | inat | xc | cdfw (ARU) | Focal total | Underwater recs | Pitch | Loudness | Call duration (s) | Notes/s | Structure | Rising pitch | Calls underwater (text) | Note |
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

## Caveats
1. **The CDFW ARU clips are the only in-domain (ARU) frog data.** They were ingested 2026-10-01. Without site or date metadata, the groups are proxies: adjacent clip numbers share noise fingerprints, so blocks of 50 consecutive clips and contiguous folds limit leakage to block boundaries. It isn't eliminated, and Tier B numbers may be somewhat optimistic until real site metadata arrives.
2. **Possible domain shortcut.** Every background negative comes from an ARU, and every frog positive is a focal or handheld recording. The near-perfect detection AP may only reflect "ARU vs not-ARU".
3. **Possible sample-rate shortcut.** Every CDFW RABO clip is 32 kHz; the other CDFW frog clips and most background clips are 48 kHz. Resample everything to 48 kHz, then check that 32 kHz origin isn't predictive.
4. **The leakage-safe definition is partly an assumption.** All herps clips count as "safe", but BirdNET's training data may include some of them. Most iNat clips predate BirdNET's cutoff, so the "all recordings" AP is optimistic for the 6 classes BirdNET already knows.
5. **Class imbalance and geography.** Most data are non-native *Lithobates*, ELCO, and toads recorded outside California. The California natives that matter most for management are the rare classes.
6. **Underwater calls.** RABO (64% of its clips), XELA (60%), MYLF, RAAU, RALU and RAPR have underwater recordings, which air ARUs will mostly miss.
7. **Background folds are stratified by category, not by site** (no site info). That makes them a looser split than the frog folds.
