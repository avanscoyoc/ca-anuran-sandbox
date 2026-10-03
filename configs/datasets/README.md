# Data cards

One YAML file per dataset. A card is the record of what the data is, where it came from, how its labels
should be read, and what we decided about it. `pixi run catalog` reads every card in this folder.

Adding a dataset:
1. Put the files, unchanged, in `data/raw/focal/<name>/` (scraped focal recordings) or `data/raw/aru/<name>/` (anything from an autonomous recorder).
2. Copy the closest card and fill it in.
3. Add a reader to `src/anuran/data/catalog.py` only if the format is new.
4. Run `pixi run catalog` and check the figure.

Labels **we** make, such as reviews or verification passes, go in `labels/<name>/` (tracked), never in `data/`.

| Field | Meaning |
|---|---|
| `name` | dataset id; also the folder name under `data/raw/<kind>/` |
| `kind` | `focal` (handheld/field-guide recordings) or `aru` (autonomous recorder) |
| `raw_dir` | where the files are, relative to the repo root |
| `citation`, `url`, `license`, `redistributable` | provenance |
| `recorder`, `schedule`, `sites` | deployment metadata (`sites`: list of `{site_id, device_id, place, lat, lon}`) |
| `reader`, `reader_args` | which function in `catalog.py` turns the files into recordings + spans |
| `label_kind` | Default for the dataset; a reader may set it per recording (non_avian_ml: frog folders `clip`, background `negative`). `weak` = species somewhere in the recording; `clip` = species in this short clip; `box` = time (and frequency) box; `negative` = verified free of frogs |
| `exhaustive` | true if every call was annotated, so unannotated time is a true absence |
| `labels` | raw label → `{species, call_type}`, or `drop` |
| `grouping` | what counts as an independent unit for splits (`site_day`, `recordist`, `proxy_block`, …) |
| `caveats` | anything a reader of results must know |
