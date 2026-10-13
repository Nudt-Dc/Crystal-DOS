"""Build frozen CGCNN graph features without accepting external target values.

This is only a preprocessing helper. It does not load a model, make predictions,
open candidate K/G records, or authorize an outer test. The legacy graph batch
container has a required ``targets`` field, so a constant zero placeholder is
used. Callers must use only structure features and never treat that placeholder
as truth.
"""

from __future__ import annotations

import numpy as np
import torch

from crystal_cgcnn_smoke import CrystalGraphCache
from official_cgcnn_iid import OfficialGraphBatch, batch_official_graphs


def batch_target_blind_official_graphs(
    cache: CrystalGraphCache,
    positions: np.ndarray,
    atom_lookup: np.ndarray,
    *,
    device: str | torch.device,
) -> OfficialGraphBatch:
    """Return training-identical crystal features with no target input argument."""

    placeholder = np.zeros((len(cache), 2), dtype=np.float32)
    batch = batch_official_graphs(
        cache, positions, placeholder, atom_lookup, target_index=0, device=device
    )
    if torch.count_nonzero(batch.targets).item() != 0:
        raise AssertionError("target-blind batch placeholder changed")
    return batch
