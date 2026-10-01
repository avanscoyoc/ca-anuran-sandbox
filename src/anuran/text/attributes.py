"""Extract call attributes from californiaherps.com description text, verbatim only.

Every value comes from an exact keyword/regex match in the scraped text; the matched
phrase and its full sentence are stored as evidence and asserted to occur verbatim in
the source. No match -> null (masked in training). Nothing is paraphrased or inferred.

Scope: the first description block of each species page (the main call description),
minus recording captions ("This is a ...") and release-call sentences.
Groups use the union of their members' text.

  data/interim/herps_species_text.jsonl -> data/interim/call_attributes.yaml (generated, gitignored)

Human review is tracked in configs/attribute_review.yaml (values only, no quotes). A class
is `reviewed: true` only while its extracted values equal the approved ones, so changing
the rules or the text silently un-reviews exactly the classes whose values moved.
"""

from __future__ import annotations

import json
import re

import yaml

from anuran.config import INTERIM, ROOT, acoustic_group, load_groups, vocal_species

OUT = INTERIM / "call_attributes.yaml"  # contains verbatim (copyrighted) quotes: gitignored
REVIEW = ROOT / "configs" / "attribute_review.yaml"
FIELDS = ("pitch", "loudness", "duration_s", "notes_per_s", "structure", "rising_pitch", "underwater")


def approved_values() -> dict:
    return (yaml.safe_load(REVIEW.read_text()) or {}).get("classes", {}) if REVIEW.exists() else {}

# exact words/phrases -> value; the attribute value is only a label for the matched words
PITCH = [("high", r"\bhigh[- ]pitched\b"), ("low", r"\b(?:low|lo)[- ]pitched\b")]
QUIET = r"\bnot very loud\b|\blow-volume\b|\bvery little volume\b|\bnot have a lot of volume\b|" \
        r"\bnot produce much volume\b|\bquiet\b(?! waters?\b)|\bweak\b|\bfaint(?:ly)?\b"  # not "quiet waters"
MODERATE = r"\bmoderately loud\b"
LOUD = r"\b(?:very )?loud\b|\bdeafening\b"
STRUCTURE = {
    "trill": r"\btrill\w*",
    "series_of": r"\bseries of\b",
    "click_knock": r"\bclicks?\b|\bknock\w*",
    "snore_rattle": r"\bsnor\w*|\brattl\w*",
    "peep_plink": r"\bpeep\w*|\bplink\w*",
    "whistle": r"\bwhistl\w*",
    "bleat_groan_growl": r"\bbleat\w*|\bgroan\w*|\bgrowl\w*",
    "croak_quack_ribit": r"\bcroak\w*|\bquack\w*|\brib-it\b|\bkrek-ek\b",
    "chuckle_cluck": r"\bchuckl\w*|\bcluck\w*",
    "drone_bellow": r"\bdrone\b|\bbellow\b",
    "grunt": r"\bgrunt\w*",
}
NUM = r"(\d+(?:\.\d+)?|1/2|one)"
DURATION = [  # (regex, (min_group, max_group) or fixed)
    (rf"{NUM}\s*-\s*{NUM}\s*seconds?", (1, 2)),
    (rf"lasting up to {NUM} seconds?", (None, 1)),
    (rf"less than {NUM} seconds?", (None, 1)),
    (rf"(?:averaging|about) {NUM} seconds?", (1, 1)),
    (rf"{NUM} seconds? to almost a minute", (1, "60")),
]
NOTES_PER_S = rf"{NUM}\s*-\s*{NUM} notes per second"
RISING = [(True, r"\brising\b"), (False, r"\bdescending\b")]
UNDERWATER = r"\bunderwater\b|\bunder water\b"


def _num(s: str | None) -> float | None:
    return None if s is None else {"1/2": 0.5, "one": 1.0}.get(s, None) or float(s)


def sentences(text: str) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if not s or s.startswith("This is a") or re.search(r"release call", s, re.I):
            continue
        out.append(s)
    return out


def extract(sents: list[str]) -> dict:
    ev: list[dict] = []

    def find(field, pattern, sent_list=sents):
        hits = []
        for s in sent_list:
            for m in re.finditer(pattern, s, re.I):
                hits.append((m, s))
        return hits

    def note(field, m, s):
        ev.append({"field": field, "match": m.group(0), "sentence": s})

    attrs: dict = {}
    pitch = set()
    for value, pat in PITCH:
        for m, s in find("pitch", pat):
            pitch.add(value)
            note("pitch", m, s)
    attrs["pitch"] = pitch.pop() if len(pitch) == 1 else None  # conflicting -> null

    loud = set()
    for m, s in find("loudness", MODERATE):
        loud.add("moderate"); note("loudness", m, s)
    for m, s in find("loudness", QUIET):
        loud.add("quiet"); note("loudness", m, s)
    for s in sents:  # "loud" inside "not very loud"/"moderately loud" doesn't count
        rest = re.sub(f"{MODERATE}|{QUIET}", " ", s, flags=re.I)
        for m in re.finditer(LOUD, rest, re.I):
            loud.add("loud"); note("loudness", m, s)
    attrs["loudness"] = loud.pop() if len(loud) == 1 else None

    dur = None
    for pat, (a, b) in DURATION:
        hits = find("duration_s", pat)
        if hits:
            m, s = hits[0]
            lo = m.group(a) if isinstance(a, int) else a
            hi = m.group(b) if isinstance(b, int) else b
            dur = [_num(lo), _num(hi)]
            note("duration_s", m, s)
            break
    attrs["duration_s"] = dur

    hits = find("notes_per_s", NOTES_PER_S)
    attrs["notes_per_s"] = [_num(hits[0][0].group(1)), _num(hits[0][0].group(2))] if hits else None
    if hits:
        note("notes_per_s", *hits[0])

    struct = []
    for name, pat in STRUCTURE.items():
        for m, s in find("structure", pat):
            if name not in struct:
                struct.append(name)
            note("structure", m, s)
    attrs["structure"] = struct or None

    rising = set()
    for value, pat in RISING:
        for m, s in find("rising_pitch", pat):
            rising.add(value); note("rising_pitch", m, s)
    attrs["rising_pitch"] = rising.pop() if len(rising) == 1 else None

    hits = find("underwater", UNDERWATER)
    attrs["underwater"] = True if hits else None
    for m, s in hits[:1]:
        note("underwater", m, s)

    # dedupe evidence, keep order
    seen, uniq = set(), []
    for e in ev:
        k = (e["field"], e["match"].lower(), e["sentence"])
        if k not in seen:
            seen.add(k); uniq.append(e)
    attrs["evidence"] = uniq
    return attrs


def build() -> dict:
    texts = {json.loads(l)["code"]: json.loads(l) for l in open(INTERIM / "herps_species_text.jsonl")}
    groups = acoustic_group()
    members: dict[str, list[str]] = {}
    for s in vocal_species():
        members.setdefault(groups[s["code"]], []).append(s["code"])
    approved = approved_values()
    out = {}
    for cls, codes in members.items():
        source = {c: texts[c]["description"][0] for c in codes if texts.get(c, {}).get("description")}
        sents = list(dict.fromkeys(s for t in source.values() for s in sentences(t)))
        attrs = extract(sents)
        full = " ".join(source.values())
        for e in attrs["evidence"]:  # verbatim guarantee
            assert e["sentence"] in full and e["match"] in e["sentence"], (cls, e)
        reviewed = cls in approved and approved[cls] == {f: attrs[f] for f in FIELDS}
        out[cls] = {"source_pages": list(source), **attrs, "reviewed": reviewed}
    return out


def main() -> None:
    classes = build()
    names = {s["code"]: s["common"] for s in vocal_species()} | {g["code"]: g["common"] for g in load_groups()}
    header = (
        "# GENERATED by `python -m anuran.text.attributes` from californiaherps.com text.\n"
        "# Values come only from exact keyword matches; `evidence` holds the verbatim match and\n"
        "# sentence. Do not hand-edit values: fix the rules; approvals live in configs/attribute_review.yaml.\n"
        "# structure labels are named after the literal words they match (e.g. chuckle_cluck).\n"
    )
    body = yaml.safe_dump({"classes": {c: {"common": names[c], **v} for c, v in sorted(classes.items())}},
                          sort_keys=False, allow_unicode=True, width=120)
    OUT.write_text(header + body)
    filled = {f: sum(v[f] is not None for v in classes.values()) for f in FIELDS}
    stale = sorted(c for c in approved_values() if c in classes and not classes[c]["reviewed"])
    print(f"{len(classes)} classes -> {OUT}\nnon-null per field: {filled}\n"
          f"reviewed: {sum(v['reviewed'] for v in classes.values())}/{len(classes)}"
          + (f"; values changed since approval (re-review): {stale}" if stale else ""))


if __name__ == "__main__":
    main()
