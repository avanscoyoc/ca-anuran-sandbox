"""Project paths and the species / acoustic-group config."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"          # immutable, one folder per dataset, exactly as received
FOCAL_RAW = RAW / "focal"            # scraped focal recordings (weak, recording-level labels)
ARU_RAW = RAW / "aru"                # autonomous recorder data (clips, raw audio + boxes, noise)
INTERIM = ROOT / "data" / "interim"
LABELS = ROOT / "labels"             # labels we make (tracked): labels/<dataset>/...
SPECIES_YAML = ROOT / "configs" / "species.yaml"
DATASETS = ROOT / "configs" / "datasets"  # one data card per dataset


def _load(path: Path = SPECIES_YAML) -> dict:
    return yaml.safe_load(path.read_text())


def load_species(path: Path = SPECIES_YAML) -> list[dict]:
    return _load(path)["species"]


def load_groups(path: Path = SPECIES_YAML) -> list[dict]:
    return _load(path).get("groups", [])


def vocal_species(path: Path = SPECIES_YAML) -> list[dict]:
    return [s for s in load_species(path) if s.get("vocal", True)]


def acoustic_group(path: Path = SPECIES_YAML) -> dict[str, str]:
    """Label code -> acoustic class. Species map to their group (or to themselves);
    group codes map to themselves."""
    out = {s["code"]: s.get("acoustic_group", s["code"]) for s in vocal_species(path)}
    out.update({g["code"]: g["code"] for g in load_groups(path)})
    return out
