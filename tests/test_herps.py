import pandas as pd

from anuran.scrape.herps import PAGES, assign_labels, load_species, parse_sounds_page, strip_boilerplate

# Minimal page mirroring californiaherps.com sounds-page markup (not copied content).
PAGE = """
<html><body><table>
<tr><td>Advertisement Calls</td></tr>
<tr><td>The vocalizations of Rana testa can be described as a series of low clucks.</td></tr>
<tr><td>The following sounds were recorded at night in Lake County.</td></tr>
<tr><td><a href= "http://www.californiaherps.com/sounds/rtest1.mp3"><img src="/images/spkr.gif"></a></td>
    <td>This is a 12 second recording of two males. Courtesy of Jane Doe.</td></tr>
<tr><td><a href="/sounds/rtest2.mp3"><img src="/images/spkr.gif"></a></td>
    <td>This is a 1 minute recording of a chorus.</td></tr>
<tr><td><a href="/sounds/rtest1.mp3">again</a></td><td>duplicate link</td></tr>
<tr><td>Diphasic (two-part) Call</td></tr>
<tr><td><a href="../../sounds/rtest3.mp3"><img src="/images/spkr.gif"></a></td><td>A release call.</td></tr>
</table><img src="/frogs/sonograms/images/rtest.jpg"></body></html>
"""


def test_parse_sounds_page():
    recs, info = parse_sounds_page(PAGE, PAGES + "r.testa.sounds.html")
    assert [r.url.rsplit("/", 1)[1] for r in recs] == ["rtest1.mp3", "rtest2.mp3", "rtest3.mp3"]
    assert all(r.url.startswith("https://www.californiaherps.com/sounds/") for r in recs)
    r1, r2, r3 = recs
    assert (r1.duration_s, r1.recordist, r1.section) == (12.0, "Jane Doe", "Advertisement Calls")
    assert r1.context.startswith("The following sounds")
    assert r2.duration_s == 60.0 and set(r2.call_types) == {"chorus", "advertisement"}
    assert r3.section == "Diphasic (two-part) Call" and r3.call_types == ["release"]
    assert any("low clucks" in d for d in info["description"])
    assert info["sonograms"] == ["https://www.californiaherps.com/frogs/sonograms/images/rtest.jpg"]


def test_assign_labels_first_sentence_rule():
    df = pd.DataFrame({
        "code": ["PSRE", "PSRE", "RAPR", "PSRE", "INAL", "RADR"],
        "url": ["a", "b", "c", "d", "e", "f"],
        "caption": [
            "A 12 second recording of a male Sierran Treefrog.",
            "Trilled calls made by male Baja California Treefrogs , recorded in Riverside County.",
            "One frog calling underwater. Distant Pacific Treefrogs are heard in the background.",
            "A Pacific Treefrog and a Sierran Treefrog calling together.",
            "Two calls of a distant Sonoran Desert Toad. Great Plains Toads are heard in the foreground.",
            "One call with the growl at the end. (Compare to Rana aurora .)",
        ],
    })
    out = assign_labels(df, load_species())
    assert out["label"].tolist() == ["PSSI", "PSHY", "RAPR", "PSRE", "INAL", "RADR"]
    assert out["label_source"].tolist() == ["caption", "caption", "page", "page", "page", "page"]
    assert out.loc[2, "secondary_species"] == ["PSRE"]
    assert out.loc[3, "secondary_species"] == ["PSSI"]
    assert out.loc[4, "secondary_species"] == ["ANCO"]
    assert out.loc[5, "secondary_species"] == []
    assert out["needs_review"].tolist() == [True, True, False, False, False, False]


def test_assign_labels_override():
    df = pd.DataFrame({"code": ["RADR"], "url": ["https://x/sounds/rd.mp3"],
                       "caption": ["Two frogs in amplexus with California Toads."]})
    out = assign_labels(df, load_species(), {"rd.mp3": {"label": "RADR", "secondary": ["ANBH"]}})
    assert (out.loc[0, "label"], out.loc[0, "label_source"], out.loc[0, "needs_review"]) == ("RADR", "override", False)


def test_strip_boilerplate():
    generic = "Each species has its own unique advertisement call."
    texts = [{"description": [f"{generic} Species {i} gives a trill."]} for i in range(5)]
    strip_boilerplate(texts)
    assert texts[0]["description"] == ["Species 0 gives a trill."]
