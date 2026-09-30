"""Frozen pretrained audio encoders.

BirdNET 2.4: the TFLite model fetched by the `birdnet` package, run directly with
ai-edge-litert (the package's multiprocess pipeline is built for whole files). Input
(N, 144000) float32 = 3 s at 48 kHz; embedding = GLOBAL_AVG_POOL (N, 1024); logits (N, 6522).

Perch 2.0 (CPU build): Kaggle google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu.
Loaded as a plain TF SavedModel (no perch-hoplite). Input (N, 160000) float32 = 5 s at
32 kHz; outputs 'embedding' (N, 1536) and 'label' (N, 14795) logits over assets/labels.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anuran.config import acoustic_group, load_groups, vocal_species

# Recordings observed on/after these dates cannot be in the encoder's training data.
# Perch 2.0: XC/iNat downloaded March 2025 (arXiv:2508.04665). BirdNET 2.4: released June 2023.
TRAINING_CUTOFF = {"perch_v2": "2025-04-01", "birdnet_v24": "2023-07-01"}
PERCH_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
GENUS_SYNONYMS = {"Lithobates": "Rana", "Rana": "Lithobates", "Anaxyrus": "Bufo", "Incilius": "Bufo"}
# acoustic groups whose members Perch knows only as one taxon
GROUP_TAXA = {"PACH": ["Pseudacris regilla"], "WETO": ["Anaxyrus boreas"], "MYLF": []}


def class_label_indices(labels: list[str]) -> dict[str, list[int]]:
    """Acoustic class -> indices of matching encoder labels (binomial or genus synonym)."""
    index = {name.lower(): i for i, name in enumerate(labels)}
    groups = acoustic_group()
    out: dict[str, list[int]] = {}
    for s in vocal_species():
        genus, epithet = s["scientific"].split()[:2]
        for name in (f"{genus} {epithet}", f"{GENUS_SYNONYMS.get(genus, genus)} {epithet}"):
            if name.lower() in index:
                out.setdefault(groups[s["code"]], []).append(index[name.lower()])
    for g in load_groups():
        out.setdefault(g["code"], [])
        out[g["code"]] += [index[n.lower()] for n in GROUP_TAXA.get(g["code"], []) if n.lower() in index]
    return {k: sorted(set(v)) for k, v in out.items() if v}


class Perch2:
    name = "perch_v2"
    sample_rate = 32000
    window_s = 5.0
    dim = 1536

    def __init__(self):
        import kagglehub
        import tensorflow as tf

        path = Path(kagglehub.model_download(PERCH_HANDLE))
        self._fn = tf.saved_model.load(str(path)).signatures["serving_default"]
        self._tf = tf
        self.labels = pd.read_csv(path / "assets" / "labels.csv").iloc[:, 0].tolist()
        self.class_indices = class_label_indices(self.labels)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """x: (N, 160000) -> embeddings (N, 1536), {class: max logit over its encoder labels}."""
        out = self._fn(inputs=self._tf.constant(x, dtype=self._tf.float32))
        emb, logits = out["embedding"].numpy(), out["label"].numpy()
        return emb, {c: logits[:, idx].max(axis=1) for c, idx in self.class_indices.items()}


class BirdNET24:
    name = "birdnet_v24"
    sample_rate = 48000
    window_s = 3.0
    dim = 1024
    EMBEDDING_TENSOR = "model/GLOBAL_AVG_POOL/Mean"

    def __init__(self, threads: int = 4):
        import birdnet
        from ai_edge_litert.interpreter import Interpreter

        model = birdnet.load("acoustic", "2.4", "tf")  # downloads/caches the model
        self._it = Interpreter(model_path=str(model.model_path), num_threads=threads,
                               experimental_preserve_all_tensors=True)
        self._in = self._it.get_input_details()[0]["index"]
        self._out = self._it.get_output_details()[0]["index"]
        self._emb = next(d["index"] for d in self._it.get_tensor_details() if d["name"] == self.EMBEDDING_TENSOR)
        self._n = 0
        self.labels = [s.split("_")[0] for s in model.species_list]  # "Genus species_Common" -> binomial
        self.class_indices = class_label_indices(self.labels)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if len(x) != self._n:
            self._it.resize_tensor_input(self._in, [len(x), x.shape[1]])
            self._it.allocate_tensors()
            self._n = len(x)
        self._it.set_tensor(self._in, x.astype(np.float32))
        self._it.invoke()
        emb, logits = self._it.get_tensor(self._emb).copy(), self._it.get_tensor(self._out)
        return emb, {c: logits[:, idx].max(axis=1) for c, idx in self.class_indices.items()}


ENCODERS = {"perch_v2": Perch2, "birdnet_v24": BirdNET24}
