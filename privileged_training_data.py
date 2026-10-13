"""Frozen split-conditioned batching rules for formal Crystal+DOS training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from diagnostic_baselines import DiagnosticData
from experiment_controls import derive_seed


SPEC_VERSION = "crystal_dos_privileged_training_v1"
VARIANT_CONFIG: Mapping[str, Mapping[str, float]] = {
    "base": {"alignment_weight": 0.0, "distillation_weight": 0.0, "modality_dropout": 0.0},
    "align": {"alignment_weight": 0.1, "distillation_weight": 0.0, "modality_dropout": 0.0},
    "distill": {"alignment_weight": 0.0, "distillation_weight": 0.1, "modality_dropout": 0.0},
    "align_distill": {"alignment_weight": 0.1, "distillation_weight": 0.1, "modality_dropout": 0.0},
    "full": {"alignment_weight": 0.1, "distillation_weight": 0.1, "modality_dropout": 0.5},
}


@dataclass(frozen=True)
class TrainNormalizers:
    dos_mean: np.ndarray
    dos_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray


@dataclass(frozen=True)
class EpochPlan:
    batches: tuple[np.ndarray, ...]
    teacher_mask: np.ndarray
    input_mask: np.ndarray
    seed: int


def compute_train_normalizers(data: DiagnosticData) -> TrainNormalizers:
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    paired = np.asarray(data.paired_partitions["train"], dtype=np.int64)
    dos = np.asarray(data.dos_features[data.dos_rows[paired]], dtype=np.float64)
    targets = np.asarray(data.targets[train], dtype=np.float64)
    dos_mean, dos_scale = dos.mean(axis=0), dos.std(axis=0, ddof=1)
    target_mean, target_scale = targets.mean(axis=0), targets.std(axis=0, ddof=1)
    dos_scale[dos_scale < 1e-8] = 1.0
    if not all(np.isfinite(value).all() for value in (dos_mean, dos_scale, target_mean, target_scale)):
        raise ValueError("training-only normalizers must be finite")
    if np.any(dos_scale <= 0) or np.any(target_scale <= 0):
        raise ValueError("training-only scales must be positive")
    return TrainNormalizers(
        dos_mean.astype(np.float32), dos_scale.astype(np.float32),
        target_mean.astype(np.float32), target_scale.astype(np.float32),
    )


def build_epoch_plan(
    train_positions: np.ndarray, observed: np.ndarray, *, variant: str,
    batch_size: int, base_seed: int, evaluation_unit: str, epoch: int,
) -> EpochPlan:
    positions = np.asarray(train_positions, dtype=np.int64)
    observed_all = np.asarray(observed, dtype=bool)
    if variant not in VARIANT_CONFIG:
        raise ValueError(f"unknown variant: {variant}")
    if positions.ndim != 1 or len(positions) < 2 or len(set(positions.tolist())) != len(positions):
        raise ValueError("train_positions must be unique and nonempty")
    if batch_size < 2 or np.any(positions < 0) or np.any(positions >= len(observed_all)):
        raise ValueError("invalid batch size or train position")
    seed = derive_seed(
        spec_version=SPEC_VERSION, base_seed=base_seed,
        evaluation_unit=evaluation_unit, control_family="batch_order", repeat=epoch,
    )
    ordered = np.random.Generator(np.random.PCG64(seed)).permutation(positions)
    batches = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    if len(batches[-1]) == 1:
        batches[-2] = np.concatenate([batches[-2], batches[-1]])
        batches.pop()
    teacher = observed_all.astype(np.uint8)
    input_mask = teacher.copy()
    dropout_rate = float(VARIANT_CONFIG[variant]["modality_dropout"])
    paired_train = positions[teacher[positions] == 1]
    drop_count = int(np.floor(dropout_rate * len(paired_train) + 0.5))
    if drop_count:
        dropout_seed = derive_seed(
            spec_version=SPEC_VERSION, base_seed=base_seed,
            evaluation_unit=evaluation_unit, control_family="modality_dropout", repeat=epoch,
        )
        drop = np.random.Generator(np.random.PCG64(dropout_seed)).permutation(paired_train)[:drop_count]
        input_mask[drop] = 0
    return EpochPlan(tuple(batches), teacher, input_mask, seed)


def normalized_batch_arrays(
    data: DiagnosticData, positions: np.ndarray, normalizers: TrainNormalizers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(positions, dtype=np.int64)
    teacher = data.observed[selected].astype(np.float32)
    dos = np.zeros((len(selected), 1, data.dos_features.shape[2]), dtype=np.float32)
    available = teacher == 1
    if available.any():
        raw = np.asarray(data.dos_features[data.dos_rows[selected[available]]], dtype=np.float32)
        dos[available] = (raw - normalizers.dos_mean) / normalizers.dos_scale
    targets = (
        np.asarray(data.targets[selected], dtype=np.float32) - normalizers.target_mean
    ) / normalizers.target_scale
    if not np.isfinite(dos).all() or not np.isfinite(targets).all():
        raise ValueError("normalized batch contains non-finite values")
    return dos, targets, teacher
