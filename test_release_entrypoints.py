"""Synthetic integration checks for the source release's bundled CGCNN model."""

from pathlib import Path

import numpy as np
import torch

from crystal_dos_privileged import OfficialCgcnnEncoder
from crystal_dos_privileged_v3 import PrivilegedCrystalDosModelV3
from official_cgcnn_iid import build_atom_feature_lookup
from outer_target_blind_cgcnn_batch import batch_target_blind_official_graphs
from test_outer_target_blind_cgcnn_batch import _synthetic_cache


def test_bundled_official_model_missing_dos_matches_structure_only():
    root = Path(__file__).resolve().parent
    torch.manual_seed(17)
    model = PrivilegedCrystalDosModelV3(OfficialCgcnnEncoder(root / "cgcnn_model.py"))
    model.eval()
    lookup, _ = build_atom_feature_lookup(root / "atom_init.json")
    graph = batch_target_blind_official_graphs(
        _synthetic_cache(), np.array([0, 1]), lookup, device="cpu"
    )
    with torch.inference_mode():
        explicit_missing = model(graph, torch.zeros(2, 1, 576), torch.zeros(2)).prediction
        structure_only = model.predict_structure_only(graph)
    assert structure_only.shape == (2, 2)
    assert torch.isfinite(structure_only).all()
    torch.testing.assert_close(structure_only, explicit_missing, rtol=0, atol=0)
