import zipfile

import numpy as np
import pytest
import shapefile
import shapely
from pyproj import CRS

from anuran.config import vocal_species
from anuran.data.ranges import (adjust, class_weights, cwhr_id, distance_km, ds_id, load_config, map_ref, read_gap,
                                to_metres, weight)

PRIOR = {"scale_km": 20, "floor": 0.02}


def test_config_covers_every_vocal_species_once():
    cfg = load_config()
    mapped, unmapped = set(cfg["ranges"]), set(cfg["unmapped"])
    assert not mapped & unmapped
    assert mapped | unmapped == {s["code"] for s in vocal_species()}


def test_config_rejects_habitat_suitability(monkeypatch, tmp_path):
    from anuran.data import ranges
    bad = tmp_path / "r.yaml"
    bad.write_text("ranges: {RAPR: oregon-spotted-frog-predicted-habitat-cwhr-a041-ds9999}\n")
    monkeypatch.setattr(ranges, "RANGES_YAML", bad)
    with pytest.raises(ValueError, match="not a range map"):
        load_config()


def test_ids_from_package_name():
    pkg = "southern-mountain-yellow-legged-frog-range-cwhr-a044-ds6131"
    assert (ds_id(pkg), cwhr_id(pkg)) == ("ds6131", "A044")


def test_weight_inside_decay_floor():
    assert weight(0.0, PRIOR) == 1.0
    assert weight(20.0, PRIOR) == pytest.approx(np.exp(-1))
    assert weight(500.0, PRIOR) == PRIOR["floor"]


@pytest.fixture
def boxes():
    """Two 1-degree boxes in central California, ~90 km apart (east-west)."""
    west, east = (to_metres(shapely.box(-122, 37, -121, 38)), to_metres(shapely.box(-120, 37, -119, 38)))
    return {"PSRE": west, "PSSI": east, "RASI": west, "RAMU": east}


def test_distance_zero_inside(boxes):
    d = distance_km(boxes, 37.5, -121.5).iloc[0]
    assert d["PSRE"] == 0 and 125 < d["PSSI"] < 140   # 1.5 degrees of longitude at 37.5 N


def test_groups_union_and_unmapped_member(boxes):
    cfg = {"prior": PRIOR}
    w = class_weights(37.5, -121.5, ranges=boxes, cfg=cfg)
    assert w["MYLF"] == 1.0               # inside RASI -> group gets the max over members
    assert w["PACH"] == 1.0               # PSHY has no map in this fixture -> no prior for the group
    assert w["LICA"] == 1.0               # unmapped species -> unchanged
    far = class_weights(37.5, -124.5, ranges=boxes, cfg=cfg)
    assert far["MYLF"] < 0.1


def test_adjust_shifts_logit(boxes):
    cfg = {"prior": {"scale_km": 20, "floor": 0.05}}
    # site ~220 km west of both MYLF boxes: MYLF gets the floor; LICA (no map in this fixture) is unchanged
    p = adjust(np.array([[0.88, 0.62]]), ["MYLF", "LICA"], 37.5, -124.5, ranges=boxes, cfg=cfg)
    logit = np.log(0.88 / 0.12) + np.log(0.05)
    assert p[0, 0] == pytest.approx(1 / (1 + np.exp(-logit)))
    assert p[0, 1] == pytest.approx(0.62)


def test_map_ref():
    assert map_ref("bullfrog-range-cwhr-a046-ds895") == ("cwhr", "bullfrog-range-cwhr-a046-ds895")
    assert map_ref({"gap": "59f5e1b7e4b063d5d307da87"}) == ("gap", "59f5e1b7e4b063d5d307da87")


def gap_fixture(raw, presence: str):
    """A tiny GAP-style download: zip (shapefile in WGS84 + HUC12 csv) next to an FGDC xml naming the item."""
    stem, item = "aXXXXx_CONUS_Range_2001v1", "abc123"
    tmp = raw / "build"
    tmp.mkdir(parents=True)
    with shapefile.Writer(str(tmp / stem), shapeType=shapefile.POLYGON) as w:
        w.field("id", "N")
        w.poly([[(-118, 34), (-118, 33), (-117, 33), (-117, 34), (-118, 34)]])
        w.record(1)
    (tmp / f"{stem}.prj").write_text(CRS.from_epsg(4326).to_wkt("WKT1_ESRI"))
    (tmp / f"{stem}.csv").write_text(f"strHUC12RNG,strUC,Presence,Season\n180701020304,aXXXXx,{presence},Year-round\n")
    with zipfile.ZipFile(raw / f"{stem}.zip", "w") as z:
        for f in tmp.iterdir():
            z.write(f, f.name)
    (raw / f"{stem}.xml").write_text(f"<title>Some Treefrog (Pseudacris testi) {stem} Range Map</title><id>{item}</id>")
    return item


def test_read_gap(tmp_path):
    geom, meta = read_gap(tmp_path, gap_fixture(tmp_path, "Known/extant"))
    assert meta == {"map_id": "aXXXXx", "sname": "Pseudacris testi", "season": "Y"}
    assert distance_km({"X": geom}, 33.5, -117.5).iloc[0, 0] == 0


def test_read_gap_rejects_uncertain_presence(tmp_path):
    with pytest.raises(ValueError, match="presence"):
        read_gap(tmp_path, gap_fixture(tmp_path, "Possibly present"))
