"""Pure crystal-only CGCNN prediction kernel for a future sealed outer test.

This module does not load checkpoints, labels or external data. The caller must
enforce release authorization and seal the full output before any target reveal.
"""

from __future__ import annotations

import numpy as np
import torch

from outer_target_blind_cgcnn_batch import batch_target_blind_official_graphs


def predict_target_blind_cgcnn(
    model: torch.nn.Module,
    cache,
    atom_lookup: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    *,
    batch_size: int,
    device: str | torch.device,
    cache_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return K/G GPa predictions in selected original-cache position order."""

    selection = (np.arange(len(cache), dtype=np.int64) if cache_positions is None
                 else np.asarray(cache_positions))
    if (selection.ndim != 1 or selection.dtype.kind not in "iu" or len(selection) < 1
            or np.any(selection < 0) or np.any(selection >= len(cache))
            or len(np.unique(selection)) != len(selection)):
        raise ValueError("cache positions must be unique in-range integer indices")
    count = len(selection)
    mean = np.asarray(target_mean, dtype=np.float32)
    scale = np.asarray(target_scale, dtype=np.float32)
    if count < 1 or batch_size < 1:
        raise ValueError("nonempty cache and positive batch size required")
    if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("frozen two-target normalizer is invalid")
    outputs = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, count, batch_size):
            positions = selection[first:min(first + batch_size, count)]
            graph = batch_target_blind_official_graphs(cache, positions, atom_lookup, device=device)
            standardized = model.predict_structure_only(graph)
            if standardized.shape != (len(positions), 2) or not torch.isfinite(standardized).all():
                raise ValueError("model emitted incomplete or nonfinite predictions")
            outputs.append(standardized.detach().cpu().numpy().astype(np.float32, copy=False))
    predictions = np.concatenate(outputs, axis=0) * scale + mean
    if predictions.shape != (count, 2) or not np.isfinite(predictions).all():
        raise ValueError("inverse-transformed predictions are invalid")
    return predictions.astype(np.float32, copy=False)
