import pandas as pd

from anuran.config import acoustic_group
from anuran.data.manifest import assign_folds, dedupe
from anuran.scrape.xenocanto import classify


def test_acoustic_groups():
    g = acoustic_group()
    assert g["PSSI"] == g["PSRE"] == g["PSHY"] == g["PACH"] == "PACH"
    assert g["ANBB"] == g["ANBH"] == "WETO" and g["RASI"] == g["RAMU"] == "MYLF"
    assert g["PSCA"] == "PSCA" and "ASTR" not in g  # California Treefrog separate; tailed frog non-vocal


def test_xc_classify():
    assert classify("Lithobates", "catesbeianus") == ("LICA", "species")
    assert classify("Rana", "catesbeianus") == ("LICA", "species")
    assert classify("Pseudacris", "regilla") == ("PACH", "group")
    assert classify("Pseudacris", "sierra") == ("PSSI", "species")
    assert classify("Anaxyrus", "boreas") == ("WETO", "group")
    assert classify("Anaxyrus", "boreas", "halophilus") == ("ANBH", "species")
    assert classify("Anaxyrus", "woodhousii") == ("ANWO", "species")
    assert classify("Anaxyrus", "fowleri") is None


def test_dedupe_keeps_first_source_and_merges_labels():
    df = pd.DataFrame({
        "source": ["inat", "xc", "herps", "inat", "inat", "inat"],
        "label": ["LICA", "LICA", "RADR", "RADR", "LIBE", "LICA"],
        "secondary_species": [[], [], [], [], [], ["ANWO"]],
        "md5": ["a", "b", "c", "c", "d", "d"],
        "date": ["2020-05-01", "2020-05-01", None, None, "2019-01-01", "2019-01-01"],
        "lat": [38.12341, 38.1231, None, None, 30.0, 30.0], "lon": [-122.5, -122.5, None, None, -97.0, -97.0],
        "duration_s": [30.2, 29.9, 5.0, 5.0, 11.0, 11.0],
    })
    out = dedupe(df).set_index("md5")
    assert sorted(out["source"]) == ["herps", "inat", "xc"]
    assert out.loc["d", "label"] == "LIBE" and out.loc["d", "secondary_species"] == ["ANWO", "LICA"]


def test_folds_never_split_a_recordist():
    df = pd.DataFrame({
        "source": ["inat"] * 40,
        "recordist": [f"r{i // 4}" for i in range(40)],
        "acoustic_class": ["A", "B"] * 20,
    })
    df["fold"] = assign_folds(df)
    assert (df.groupby("recordist")["fold"].nunique() == 1).all()
    assert set(df["fold"]) == set(range(5))
