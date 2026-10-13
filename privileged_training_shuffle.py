"""Deterministic, train-partition-only DOS assignment counterfactual."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from diagnostic_baselines import DiagnosticData
from experiment_controls import derive_seed


SHUFFLE_SPEC_VERSION = "privileged_loco_training_shuffle_v1"
VARIANT_CONFIG_V4: dict[str, dict[str, Any]] = {
    "distill_train_shuffled": {
        "base_variant": "distill_decoupled",
        "training_dos_policy": "within_train_observed_sattolo",
        "shuffle_repeat": 0,
    }
}
LOCO_VARIANTS_V4 = tuple(VARIANT_CONFIG_V4)


@dataclass(frozen=True)
class TrainingDosAssignment:
    paired_train_positions: np.ndarray
    original_dos_rows: np.ndarray
    assigned_dos_rows: np.ndarray
    full_dos_rows: np.ndarray
    seed: int


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_loco_unit(value: str) -> str:
    if value not in {f"loco_{index:02d}" for index in range(10)}:
        raise ValueError("evaluation_unit must be loco_00 through loco_09")
    return value


def _sattolo_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("Sattolo shuffle requires at least two observed training DOS rows")
    order = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap = int(rng.integers(0, index))
        order[index], order[swap] = order[swap], order[index]
    return order


def build_training_dos_assignment(
    data: DiagnosticData,
    *,
    evaluation_unit: str,
    base_seed: int,
) -> TrainingDosAssignment:
    """Shuffle DOS only among observed training materials, preserving its multiset."""
    evaluation_unit = _require_loco_unit(evaluation_unit)
    if data.evaluation_unit != evaluation_unit:
        raise ValueError("loaded data evaluation_unit differs from requested LOCO fold")
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    paired = train[np.asarray(data.observed[train], dtype=bool)]
    original = np.asarray(data.dos_rows[paired], dtype=np.int64)
    if len(paired) < 2 or np.any(original < 0) or len(np.unique(original)) != len(original):
        raise ValueError("observed training DOS rows must be unique and contain at least two rows")
    seed = derive_seed(
        spec_version=SHUFFLE_SPEC_VERSION,
        base_seed=base_seed,
        evaluation_unit=evaluation_unit,
        control_family="train_dos_shuffle",
        repeat=0,
    )
    order = _sattolo_indices(len(original), np.random.Generator(np.random.PCG64(seed)))
    assigned = original[order]
    if np.any(assigned == original) or not np.array_equal(
        np.sort(assigned), np.sort(original)
    ):
        raise RuntimeError("training DOS assignment is not a fixed-point-free permutation")
    full = np.asarray(data.dos_rows, dtype=np.int64).copy()
    full[paired] = assigned
    outside = np.ones(len(full), dtype=bool)
    outside[paired] = False
    if not np.array_equal(full[outside], np.asarray(data.dos_rows)[outside]):
        raise RuntimeError("training DOS assignment changed rows outside paired training data")
    return TrainingDosAssignment(paired, original, assigned, full, seed)
