"""Synthetic-only tests: no checkpoint, external graph or true target is opened."""

import numpy as np
import pytest
import torch

from test_outer_target_blind_cgcnn_batch import _synthetic_cache
from outer_target_blind_cgcnn_predict import predict_target_blind_cgcnn


class FixedModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def predict_structure_only(self, graph):
        assert torch.count_nonzero(graph.targets).item() == 0
        return torch.tensor(self.values[:len(graph.crystal_atom_indices)], dtype=torch.float32)


class AtomFeatureModel(torch.nn.Module):
    def predict_structure_only(self, graph):
        assert torch.count_nonzero(graph.targets).item() == 0
        values = torch.stack([graph.atom_features[ids[0], 0] for ids in graph.crystal_atom_indices])
        return torch.stack([values, -values], dim=1)


def test_exact_order_and_inverse_transform():
    cache = _synthetic_cache()
    lookup = np.arange(101 * 92, dtype=np.float32).reshape(101, 92) / 1000
    result = predict_target_blind_cgcnn(
        FixedModel([[1.0, -1.0], [2.0, -2.0]]), cache, lookup,
        np.array([50.0, 25.0]), np.array([10.0, 5.0]),
        batch_size=2, device="cpu",
    )
    np.testing.assert_array_equal(result, [[60.0, 20.0], [70.0, 15.0]])


def test_selected_original_cache_positions_preserve_requested_order():
    cache = _synthetic_cache()
    lookup = np.arange(101 * 92, dtype=np.float32).reshape(101, 92) / 1000
    result = predict_target_blind_cgcnn(
        AtomFeatureModel(), cache, lookup, [0.0, 0.0], [1.0, 1.0],
        batch_size=1, device="cpu", cache_positions=np.array([1, 0], dtype=np.int64),
    )
    np.testing.assert_allclose(result, [[0.552, -0.552], [0.092, -0.092]], atol=1e-7)


@pytest.mark.parametrize("positions", [[0, 0], [2], [-1], [0.5], []])
def test_rejects_invalid_cache_positions(positions):
    cache = _synthetic_cache()
    lookup = np.arange(101 * 92, dtype=np.float32).reshape(101, 92) / 1000
    with pytest.raises(ValueError, match="cache positions"):
        predict_target_blind_cgcnn(
            FixedModel([[0.0, 0.0]]), cache, lookup, [0.0, 0.0], [1.0, 1.0],
            batch_size=1, device="cpu", cache_positions=np.asarray(positions),
        )


def test_rejects_bad_normalizer_and_nonfinite_output():
    cache = _synthetic_cache()
    lookup = np.arange(101 * 92, dtype=np.float32).reshape(101, 92) / 1000
    with pytest.raises(ValueError, match="normalizer"):
        predict_target_blind_cgcnn(FixedModel([[0.0, 0.0]]), cache, lookup,
                                   [0.0, 0.0], [1.0, 0.0], batch_size=1, device="cpu")
    with pytest.raises(ValueError, match="nonfinite"):
        predict_target_blind_cgcnn(FixedModel([[float("nan"), 0.0]]), cache, lookup,
                                   [0.0, 0.0], [1.0, 1.0], batch_size=1, device="cpu")
