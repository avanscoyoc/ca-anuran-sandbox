from anuran.text.attributes import extract, sentences


def _x(text):
    return extract(sentences(text))


def test_exact_matches_only_and_verbatim_evidence():
    text = "The call is a loud high-pitched trill lasting up to 10 seconds. This is a 12 second recording."
    a = _x(text)
    assert (a["pitch"], a["loudness"], a["duration_s"], a["structure"]) == ("high", "loud", [None, 10.0], ["trill"])
    assert all(e["match"] in e["sentence"] and e["sentence"] in text for e in a["evidence"])


def test_negation_and_other_senses():
    assert _x("The call is not very loud.")["loudness"] == "quiet"
    assert _x("A moderately loud snore.")["loudness"] == "moderate"
    assert _x("A loud bleating. Calls are made from quiet waters of ponds.")["loudness"] == "loud"


def test_captions_and_release_calls_ignored():
    a = _x("This is a 5 second recording of a trill. A release call is a chuckle.")
    assert a["structure"] is None and a["duration_s"] is None


def test_nothing_stated_is_null():
    a = _x("Males call at night.")
    assert all(a[k] is None for k in ("pitch", "loudness", "duration_s", "structure", "rising_pitch", "underwater"))
