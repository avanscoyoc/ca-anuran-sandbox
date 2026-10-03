import numpy as np
import pytest
import soundfile as sf

from anuran.data import catalog
from anuran.data.catalog import box_hits, inventory, load_cards, read_raven, validate, windows

RAVEN_HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tannotation\n"


def test_box_rule_edges():
    ws, we = np.array([0.0]), np.array([3.0])
    assert not box_hits(ws, we, 2.81, 5.0)[0]   # 0.19 s inside, < 50% of a 2.19 s box
    assert box_hits(ws, we, 2.79, 5.0)[0]       # 0.21 s inside
    assert box_hits(ws, we, 2.9, 3.05)[0]       # short box, 0.1 s of 0.15 s (> 50%) inside
    assert not box_hits(ws, we, 3.0, 3.5)[0]    # touching, no overlap


def test_cards_in_repo_are_valid():
    names = {c["name"] for c in load_cards()}
    assert {"herps", "inat", "xc", "non_avian_ml", "rana_sierrae_2022"} <= names


def test_validate_rejects_bad_label_kind():
    card = {"name": "x", "kind": "aru", "raw_dir": "x", "reader": "raven", "label_kind": "strong", "license": "x"}
    with pytest.raises(ValueError, match="label_kind"):
        validate(card)


@pytest.fixture
def raven_dataset(tmp_path, monkeypatch):
    """Two 10 s files: one with an A box, a B box and an X box; one with only an X box."""
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    (tmp_path / "ds" / "tables").mkdir(parents=True)
    for stem, rows in {"dev1_20220620_000000": ["1\tS\t1\t1.0\t1.5\t300\t900\tA", "2\tS\t1\t6.0\t6.4\t300\t900\tB",
                                                "3\tS\t1\t8.0\t8.3\t300\t900\tX"],
                       "dev1_20220621_000000": ["1\tS\t1\t2.0\t2.5\t300\t900\tX"]}.items():
        sf.write(tmp_path / "ds" / f"{stem}.wav", np.zeros(32000 * 10, np.float32), 32000)
        (tmp_path / "ds" / "tables" / f"{stem}.txt").write_text(RAVEN_HEADER + "\n".join(rows) + "\n")
    return {"name": "ds", "kind": "aru", "raw_dir": "ds", "reader": "raven", "label_kind": "box", "license": "CC0",
            "exhaustive": True, "sites": [{"site_id": "s1", "device_id": "dev1"}],
            "reader_args": {"audio_glob": "*.wav", "table": "tables/{stem}.txt",
                            "filename": r"^(?P<device>dev\d)_(?P<date>\d{8})_"},
            "labels": {"A": {"species": "RASI", "call_type": "A"}, "B": {"species": "RASI", "call_type": "B"},
                       "X": "drop"}}


def test_raven_reader_maps_labels_and_keeps_dropped(raven_dataset):
    rec, spans = read_raven(raven_dataset)
    assert len(rec) == 2 and set(rec["site_id"]) == {"s1"} and set(rec["date"]) == {"2022-06-20", "2022-06-21"}
    live = spans[~spans["dropped"]]
    assert sorted(live["call_type"]) == ["A", "B"] and set(live["acoustic_class"]) == {"MYLF"}
    assert spans["dropped"].sum() == 2 and spans.loc[spans["dropped"], "acoustic_class"].isna().all()


def test_raven_reader_rejects_unmapped_label(raven_dataset):
    raven_dataset["labels"].pop("B")
    with pytest.raises(ValueError, match="'B' not in the card"):
        read_raven(raven_dataset)


def test_windows_and_inventory_treat_x_as_no_call(raven_dataset):
    rec, spans = catalog.build([raven_dataset], write=False)
    win = windows(rec, spans)
    pos = win[win["acoustic_class"] == "MYLF"]
    # 10 s -> windows start at 0, 1.5, 3, 4.5, 6, 7. A (1.0-1.5 s) hits 0; B (6.0-6.4 s) hits 4.5 and 6
    assert set(pos["start_s"]) == {0.0, 4.5, 6.0}
    inv = inventory(rec, spans, win).set_index(["acoustic_class", "label_kind"])
    assert inv.loc[("MYLF", "box"), "recordings"] == 1
    assert inv.loc[("negative", "negative"), "recordings"] == 1   # the X-only file is a negative
    assert inv.loc[("negative", "negative"), "windows_3s"] == 6 + 6 - 3


def test_parse_non_avian_ml_source_paths():
    from anuran.data.non_avian_ml import parse_source as p
    assert p("ExtremeSSD/Mojave/Birds/BD-33452B/20230527_200000.WAV") == \
        {"site_id": "BD-33452B", "date": "2023-05-27", "recording_id": "20230527_200000"}
    assert p("audio/samples_gun_15per_bin/UOLD4_2021_20210527_200000_12789_12804.flac")["site_id"] == "UOLD4"
    assert p("audio/samples_generator1/2021_10_26_Windmill_R13_Sound_WCenterA_20211013_065000.flac") == \
        {"site_id": "Windmill_R13_Sound_WCenterA", "date": "2021-10-13", "recording_id": "20211013_065000"}
    assert p("audio/nutria/clip_1.wav") == {"site_id": None, "date": None, "recording_id": "nutria_clip_1"}
