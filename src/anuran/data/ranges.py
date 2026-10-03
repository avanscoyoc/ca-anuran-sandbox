"""Species range maps -> a per-site prior on class scores. Inference only, never a training input.

Sources (configs/ranges.yaml): CWHR range maps (CDFW) by default; USGS GAP range maps only where CWHR has none (PSHY).

  pixi run ranges [--force]
    data/raw/range/cwhr/<dsN>.geojson           each map exactly as downloaded (skipped if present; --force refetches)
    data/raw/range/gap/<code>_CONUS_Range_2001v1.{zip,xml}
    data/processed/ranges.parquet               one row per mapped species: polygon in EPSG:3310 (metres), as WKB
    data/results/ranges/range_check.csv         per acoustic class: coverage, and how far geotagged California
                                                recordings in the catalog fall outside the range

Use (E7): `class_weights(lat, lon)` gives w per acoustic class at a site; `adjust(probs, classes, lat, lon)` returns
sigmoid(logit(p) + log w). w = 1 inside the range, exp(-d / scale_km) outside, never below `floor`. An acoustic group
gets the max over its members (union of ranges); a class or group with any unmapped member gets w = 1.
Config, sources and caveats: configs/ranges.yaml.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
import shapefile
import shapely
import yaml
from pyproj import CRS, Transformer

from anuran.config import ROOT, acoustic_group, vocal_species
from anuran.data.manifest import PROCESSED
from anuran.scrape.fetch import Fetcher

RANGES_YAML = ROOT / "configs" / "ranges.yaml"
TABLE = PROCESSED / "ranges.parquet"
RESULTS = ROOT / "data" / "results" / "ranges"
CKAN = "https://data.cnra.ca.gov/api/3/action/package_show"
SCIENCEBASE = "https://www.sciencebase.gov/catalog/item/"
TO_M = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)  # California Albers, equal-area metres


def load_config() -> dict:
    cfg = yaml.safe_load(RANGES_YAML.read_text())
    for code, ref in cfg["ranges"].items():
        source, ref = map_ref(ref)
        if source == "cwhr" and ("-range-" not in ref or "predicted-habitat" in ref):
            raise ValueError(f"{code}: {ref} is not a range map (habitat suitability is not used)")
    return cfg


def map_ref(entry) -> tuple[str, str]:
    """A `ranges` entry -> (source, ref): a string is a CWHR package, {gap: id} a ScienceBase item."""
    return ("cwhr", entry) if isinstance(entry, str) else next(iter(entry.items()))


def ds_id(package: str) -> str:
    return re.search(r"-(ds\d+)$", package).group(1)


def cwhr_id(package: str) -> str:
    return re.search(r"-cwhr-(a\d+[a-z]?)-", package).group(1).upper()


# ---------------------------------------------------------------- download + build
def download(cfg: dict, force: bool = False) -> None:
    fetch = Fetcher(delay=1.0)
    for source, ref in sorted({map_ref(e) for e in cfg["ranges"].values()}):
        raw = ROOT / cfg["sources"][source]["raw_dir"]
        raw.mkdir(parents=True, exist_ok=True)
        if source == "cwhr":
            dest = raw / f"{ds_id(ref)}.geojson"
            if dest.exists() and not force:
                continue
            res = fetch.get(CKAN, params={"id": ref}).json()["result"]["resources"]
            files = {dest.name: next(r["url"] for r in res if r["format"].lower() == "geojson")}
        else:
            if gap_zip(raw, ref, missing_ok=True) and not force:
                continue
            item = fetch.get(SCIENCEBASE + ref, params={"format": "json"}).json()
            if "Range Map" not in item["title"]:
                raise ValueError(f"{ref}: {item['title']!r} is not a GAP range map (habitat maps are not used)")
            files = {f["name"]: f["url"] for f in item["files"] if f["name"].endswith((".zip", ".xml"))}
        for name, url in files.items():
            (raw / name).write_bytes(fetch.get(url).content)
            print(f"downloaded {source}:{ref} -> {(raw / name).relative_to(ROOT)}")


def gap_zip(raw, item: str, missing_ok: bool = False):
    """The GAP zip for a ScienceBase item, found via its FGDC xml (which names the item id)."""
    hits = [x.with_suffix(".zip") for x in raw.glob("*_Range_*.xml") if item in x.read_text(errors="ignore")]
    if not hits and not missing_ok:
        raise FileNotFoundError(f"no GAP range map for item {item} in {raw}; run pixi run ranges")
    return hits[0] if hits else None


def to_metres(geom):
    return shapely.transform(geom, lambda xy: np.column_stack(TO_M.transform(xy[:, 0], xy[:, 1])))


def read_cwhr(raw, pkg: str) -> tuple:
    gj = json.loads((raw / f"{ds_id(pkg)}.geojson").read_text())
    props = {k.lower(): v for k, v in gj["features"][0]["properties"].items()}  # ds589 spells SHAPE_NAME
    if props["shape_name"] != cwhr_id(pkg):
        raise ValueError(f"{pkg}: file holds {props['shape_name']}, expected {cwhr_id(pkg)}")
    geom = to_metres(shapely.union_all([shapely.geometry.shape(f["geometry"]) for f in gj["features"]]))
    return geom, {"map_id": props["shape_name"], "sname": props["sname"], "season": props.get("season")}


def read_gap(raw, item: str) -> tuple:
    """GAP ships the range dissolved to one polygon, with per-HUC12 attributes in a CSV. Since the polygon can't be
    filtered, require every HUC12 to be Known/extant (no 'possibly present' or extirpated areas)."""
    path = gap_zip(raw, item)
    with zipfile.ZipFile(path) as z:
        stem = path.stem
        hucs = pd.read_csv(io.BytesIO(z.read(f"{stem}.csv")))
        src = CRS.from_wkt(z.read(f"{stem}.prj").decode())
    if set(hucs.Presence) != {"Known/extant"}:
        raise ValueError(f"{item}: presence {sorted(set(hucs.Presence))}; filter HUC12s before using this map")
    to_m = Transformer.from_crs(src, "EPSG:3310", always_xy=True)
    shapes = [shapely.geometry.shape(s.__geo_interface__) for s in shapefile.Reader(str(path)).shapes()]
    geom = shapely.transform(shapely.union_all(shapes), lambda xy: np.column_stack(to_m.transform(xy[:, 0], xy[:, 1])))
    season = "Y" if set(hucs.Season) == {"Year-round"} else ",".join(sorted(set(hucs.Season)))
    title = re.search(r"<title>([^<]*)</title>", path.with_suffix(".xml").read_text()).group(1)
    sname = re.search(r"\(([A-Z][a-z]+ [a-z]+(?: [a-z]+)?)\)", title).group(1)  # "... Treefrog (Pseudacris hypochondriaca) ..."
    return geom, {"map_id": hucs.strUC.iloc[0], "sname": sname, "season": season}


def build(cfg: dict, write: bool = True) -> pd.DataFrame:
    rows = []
    for code, entry in cfg["ranges"].items():
        source, ref = map_ref(entry)
        raw = ROOT / cfg["sources"][source]["raw_dir"]
        geom, meta = (read_cwhr if source == "cwhr" else read_gap)(raw, ref)
        rows.append({"code": code, "source": source, "ref": ref, **meta,
                     "area_km2": geom.area / 1e6, "wkb": shapely.to_wkb(geom)})
    table = pd.DataFrame(rows)
    if write:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(TABLE, index=False)
    return table


# ---------------------------------------------------------------- prior
def load_ranges(table: pd.DataFrame | None = None) -> dict:
    """Species code -> prepared polygon (EPSG:3310)."""
    table = pd.read_parquet(TABLE) if table is None else table
    out = {c: shapely.from_wkb(w) for c, w in zip(table.code, table.wkb)}
    for g in out.values():
        shapely.prepare(g)
    return out


def distance_km(ranges: dict, lat, lon) -> pd.DataFrame:
    """Rows = points, columns = mapped species: km to the range edge, 0 inside."""
    pts = shapely.points(np.column_stack(TO_M.transform(np.atleast_1d(lon), np.atleast_1d(lat))))
    return pd.DataFrame({c: shapely.distance(g, pts) / 1e3 for c, g in ranges.items()})


def weight(d_km, prior: dict):
    return np.maximum(np.exp(-np.asarray(d_km, float) / prior["scale_km"]), prior["floor"])


def class_weights(lat: float, lon: float, ranges: dict | None = None, cfg: dict | None = None) -> pd.Series:
    """w per acoustic class at one site. Groups: max over members; any unmapped member -> 1."""
    cfg = cfg or load_config()
    ranges = load_ranges() if ranges is None else ranges
    w_sp = weight(distance_km(ranges, lat, lon).iloc[0], cfg["prior"])
    w_sp = pd.Series(w_sp, index=list(ranges))
    by_class = pd.Series({s["code"]: w_sp.get(s["code"], 1.0) for s in vocal_species()}).groupby(acoustic_group()).max()
    return by_class


def adjust(probs: np.ndarray, classes: list[str], lat: float, lon: float, **kw) -> np.ndarray:
    """Range-adjusted scores: sigmoid(logit(p) + log w). Classes without a weight are unchanged."""
    w = class_weights(lat, lon, **kw).reindex(classes).fillna(1.0).to_numpy()
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + np.log(w))))


# ---------------------------------------------------------------- check
def check(table: pd.DataFrame) -> pd.DataFrame:
    """Coverage per acoustic class, and where the catalog's geotagged recordings fall relative to their range.
    'In California' = inside the union of the CWHR maps (they are clipped at the state line; GAP maps are not)."""
    ranges = load_ranges(table)
    cat = PROCESSED / "catalog"
    rec, spans = pd.read_parquet(cat / "recordings.parquet"), pd.read_parquet(cat / "spans.parquet")
    pts = (rec[rec.lat.notna()][["recording_id", "dataset", "lat", "lon"]]
           .merge(spans[~spans.dropped][["recording_id", "acoustic_class"]].drop_duplicates(), on="recording_id"))
    d = distance_km(ranges, pts.lat.to_numpy(), pts.lon.to_numpy())
    in_ca = (d[sorted(set(table.code[table.source == "cwhr"]))].min(axis=1) == 0).to_numpy()
    pts, d = pts[in_ca].reset_index(drop=True), d[in_ca].reset_index(drop=True)

    members = pd.Series(acoustic_group()).loc[[s["code"] for s in vocal_species()]]
    rows = []
    for cls, mem in members.groupby(members):
        mapped = [m for m in mem.index if m in ranges]
        unmapped = [m for m in mem.index if m not in ranges]
        row = {"class": cls, "members": " ".join(mem.index), "mapped": " ".join(mapped),
               "unmapped": " ".join(unmapped), "prior": "yes" if not unmapped else ("none" if not mapped else "partial")}
        sel = (pts.acoustic_class == cls).to_numpy()
        if mapped and sel.any():
            dist = d.loc[sel, mapped].min(axis=1)
            out = dist[dist > 0]
            row.update({"n_ca": int(sel.sum()), "frac_inside": round(float((dist == 0).mean()), 3),
                        "out_median_km": round(float(out.median()), 1) if len(out) else 0.0,
                        "out_max_km": round(float(out.max()), 1) if len(out) else 0.0})
        else:
            row["n_ca"] = int(sel.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download every map")
    args = ap.parse_args()
    cfg = load_config()
    download(cfg, force=args.force)
    table = build(cfg)
    print(table[["code", "source", "map_id", "sname", "season", "area_km2"]].round(0).to_string(index=False))
    res = check(table)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "range_check.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
