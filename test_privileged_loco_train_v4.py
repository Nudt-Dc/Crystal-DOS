import numpy as np

from diagnostic_baselines import DiagnosticData
from privileged_training_shuffle import (
    LOCO_VARIANTS_V4,
    SHUFFLE_SPEC_VERSION,
    build_training_dos_assignment,
)


def _data(unit: str = "loco_00") -> DiagnosticData:
    material_ids = np.asarray([f"mp-{index}" for index in range(12)])
    observed = np.asarray([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    dos_rows = np.full(12, -1, dtype=np.int64)
    dos_rows[observed] = np.arange(int(observed.sum()), dtype=np.int64)
    partitions = {
        "train": np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        "validation": np.asarray([8, 9], dtype=np.int64),
        "test": np.asarray([10, 11], dtype=np.int64),
    }
    paired = {name: rows[observed[rows]] for name, rows in partitions.items()}
    return DiagnosticData(
        evaluation_unit=unit,
        material_ids=material_ids,
        targets=np.zeros((12, 2), dtype=np.float64),
        soap_features=np.zeros((12, 2), dtype=np.float32),
        dos_features=np.zeros((int(observed.sum()), 1, 4), dtype=np.float32),
        dos_rows=dos_rows,
        observed=observed,
        partitions=partitions,
        paired_partitions=paired,
        target_names=("k_vrh_gpa", "g_vrh_gpa"),
        source_sha256={"source": "fixed"},
    )


def test_v4_matrix_is_one_predeclared_counterfactual():
    assert LOCO_VARIANTS_V4 == ("distill_train_shuffled",)
    assert SHUFFLE_SPEC_VERSION == "privileged_loco_training_shuffle_v1"


def test_training_shuffle_is_deterministic_derangement_within_train_only():
    data = _data()
    first = build_training_dos_assignment(data, evaluation_unit="loco_00", base_seed=20260910)
    second = build_training_dos_assignment(data, evaluation_unit="loco_00", base_seed=20260910)
    assert np.array_equal(first.assigned_dos_rows, second.assigned_dos_rows)
    assert np.array_equal(np.sort(first.assigned_dos_rows), np.sort(first.original_dos_rows))
    assert np.all(first.assigned_dos_rows != first.original_dos_rows)
    outside = np.ones(len(data.material_ids), dtype=bool)
    outside[first.paired_train_positions] = False
    assert np.array_equal(first.full_dos_rows[outside], data.dos_rows[outside])
    assert np.all(first.full_dos_rows[~data.observed] == -1)


def test_training_shuffle_changes_with_seed_and_fold_namespace():
    data = _data()
    baseline = build_training_dos_assignment(data, evaluation_unit="loco_00", base_seed=20260910)
    other_seed = build_training_dos_assignment(data, evaluation_unit="loco_00", base_seed=20260911)
    other_fold = build_training_dos_assignment(_data("loco_01"), evaluation_unit="loco_01", base_seed=20260910)
    assert not np.array_equal(baseline.assigned_dos_rows, other_seed.assigned_dos_rows)
    assert not np.array_equal(baseline.assigned_dos_rows, other_fold.assigned_dos_rows)


def test_training_shuffle_rejects_fold_mismatch():
    data = _data()
    try:
        build_training_dos_assignment(data, evaluation_unit="loco_01", base_seed=20260910)
    except ValueError as exc:
        assert "evaluation_unit" in str(exc)
    else:
        raise AssertionError("fold mismatch was accepted")
