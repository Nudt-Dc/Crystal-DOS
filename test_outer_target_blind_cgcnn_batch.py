"""The outer-test graph adapter cannot receive K/G and preserves graph features."""

import numpy as np
import torch

from official_cgcnn_iid import batch_official_graphs
from outer_target_blind_cgcnn_batch import batch_target_blind_official_graphs


def _synthetic_cache():
    arrays = {
        "site_offsets.npy": np.array([0, 2, 4], dtype=np.int64),
        "graph_edge_offsets.npy": np.array([0, 4, 8], dtype=np.int64),
        "atomic_numbers.npy": np.array([1, 8, 6, 14], dtype=np.int64),
        "edge_index.npy": np.array(
            [[0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
            dtype=np.int64,
        ),
        "neighbor_counts.npy": np.array([2, 2, 2, 2], dtype=np.int64),
        "edge_distances.npy": np.array([1.0, 2.0, 1.0, 2.0, 1.5, 2.5, 1.5, 2.5]),
    }

    class Cache:
        def __init__(self):
            self.arrays = arrays

        def __len__(self):
            return 2

    return Cache()


def test_target_blind_graph_features_equal_legacy_graph_features():
    cache = _synthetic_cache()
    positions = np.array([1, 0], dtype=np.int64)
    lookup = np.arange(101 * 92, dtype=np.float32).reshape(101, 92) / 1000
    blind = batch_target_blind_official_graphs(cache, positions, lookup, device="cpu")
    legacy = batch_official_graphs(
        cache,
        positions,
        np.array([[99.0, 990.0], [11.0, 110.0]], dtype=np.float32),
        lookup,
        target_index=0,
        device="cpu",
    )
    for field in ("atom_features", "neighbor_features", "neighbor_indices"):
        torch.testing.assert_close(getattr(blind, field), getattr(legacy, field))
    for left, right in zip(blind.crystal_atom_indices, legacy.crystal_atom_indices):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(blind.targets, torch.zeros((2, 1)))
    torch.testing.assert_close(legacy.targets, torch.tensor([[11.0], [99.0]]))
