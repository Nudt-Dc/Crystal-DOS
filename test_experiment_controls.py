import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import experiment_controls as controls

from experiment_controls import (
    ControlSpec,
    build_control_unit,
    build_low_label_masks,
    build_missingness_masks,
    build_shuffled_dos_source_rows,
    build_stratified_order,
    derive_seed,
    fraction_key,
    load_control_inputs,
    materialize_experiment_controls,
    rate_key,
    write_control_unit,
)


@pytest.fixture()
def synthetic_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    master_dir = tmp_path / "master"
    dos_dir = tmp_path / "dos"
    split_dir = tmp_path / "splits"
    for directory in (master_dir, dos_dir, split_dir):
        directory.mkdir()

    n_materials = 30
    material_ids = np.asarray([f"mp-{index:02d}" for index in range(n_materials)], dtype=str)
    observed = np.ones(n_materials, dtype=np.uint8)
    observed[-1] = 0
    dos_row = np.full(n_materials, np.nan)
    dos_row[:-1] = np.arange(n_materials - 1)
    master = pd.DataFrame(
        {
            "material_id": material_ids,
            "crystal_row": np.arange(n_materials, dtype=np.int64),
            "dos_row": pd.array(dos_row, dtype="Int64"),
            "dos_observed": observed,
            "k_vrh_gpa": np.linspace(10.0, 39.0, n_materials),
            "g_vrh_gpa": np.linspace(20.0, 49.0, n_materials),
        }
    )
    master.to_parquet(master_dir / "master_table.parquet", index=False)
    np.save(dos_dir / "material_ids.npy", material_ids[:-1], allow_pickle=False)
    np.save(
        dos_dir / "dos_features.npy",
        np.arange((n_materials - 1) * 4, dtype=np.float32).reshape(n_materials - 1, 1, 4),
        allow_pickle=False,
    )
    (dos_dir / "preprocessing_config.json").write_text(
        json.dumps({"schema_version": "dos_v1", "point_count": 4}) + "\n", encoding="utf-8"
    )

    stable_keys = [f"{n_materials - index:03d}" for index in range(n_materials)]
    strata = [f"stratum-{index % 3}" for index in range(n_materials)]
    iid = pd.DataFrame(
        {
            "material_id": material_ids,
            "crystal_row": master["crystal_row"],
            "dos_row": master["dos_row"],
            "split": ["train"] * 15 + ["validation"] * 6 + ["test"] * 9,
            "stratum": strata,
            "stable_order_key": stable_keys,
        }
    )
    iid.to_parquet(split_dir / "iid_split.parquet", index=False)
    loco_rows: list[dict[str, object]] = []
    for fold_number in range(10):
        fold_name = f"loco_{fold_number:02d}"
        held_out = set(range(fold_number * 3, fold_number * 3 + 3))
        remaining = [index for index in range(n_materials) if index not in held_out]
        validation = set(remaining[:3])
        for index in range(n_materials):
            split = "test" if index in held_out else ("validation" if index in validation else "train")
            loco_rows.append(
                {
                    "fold": fold_name,
                    "material_id": material_ids[index],
                    "crystal_row": index,
                    "dos_row": None if not observed[index] else index,
                    "split": split,
                    "stratum": strata[index],
                    "stable_order_key": stable_keys[index],
                }
            )
    pd.DataFrame(loco_rows).to_parquet(split_dir / "soap_loco_folds.parquet", index=False)
    (split_dir / "split_metadata.json").write_text(
        json.dumps({"schema_version": "splits_v1", "fold_count": 10}) + "\n", encoding="utf-8"
    )
    return master_dir, dos_dir, split_dir


def test_load_control_inputs_reads_strict_sources_and_hashes_all_direct_files(
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    inputs = load_control_inputs(master_dir, dos_dir, split_dir)
    assert inputs.master["material_id"].map(type).eq(str).all()
    assert inputs.dos_feature_row_count == 29
    np.testing.assert_array_equal(inputs.dos_material_ids, [f"mp-{index:02d}" for index in range(29)])
    assert inputs.iid["material_id"].nunique() == 30
    assert inputs.loco["fold"].nunique() == 10
    assert inputs.loco_fold_count == 10
    assert set(inputs.source_sha256) == {
        "master/master_table.parquet",
        "dos/material_ids.npy",
        "dos/dos_features.npy",
        "dos/preprocessing_config.json",
        "split/iid_split.parquet",
        "split/soap_loco_folds.parquet",
        "split/split_metadata.json",
    }
    for key, digest in inputs.source_sha256.items():
        role, filename = key.split("/", 1)
        path = {"master": master_dir, "dos": dos_dir, "split": split_dir}[role] / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not hasattr(inputs, "dos_features")


def test_loader_accepts_unique_noncontiguous_dos_rows(
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    master = pd.read_parquet(master_dir / "master_table.parquet")
    observed = master["dos_observed"].astype(bool)
    master.loc[observed, "dos_row"] = master.loc[observed, "dos_row"].astype(np.int64) * 2
    master.to_parquet(master_dir / "master_table.parquet", index=False)
    mapping = master.set_index("material_id")["dos_row"]
    for filename in ("iid_split.parquet", "soap_loco_folds.parquet"):
        frame = pd.read_parquet(split_dir / filename)
        frame["dos_row"] = frame["material_id"].map(mapping)
        frame.to_parquet(split_dir / filename, index=False)
    source_ids = np.asarray([f"unused-{index:02d}" for index in range(57)], dtype=str)
    material_ids = master.loc[observed, "material_id"].astype(str).to_numpy()
    source_ids[np.arange(len(material_ids)) * 2] = material_ids
    np.save(dos_dir / "material_ids.npy", source_ids, allow_pickle=False)
    np.save(dos_dir / "dos_features.npy", np.zeros((57, 1, 4), dtype=np.float32), allow_pickle=False)
    inputs = load_control_inputs(master_dir, dos_dir, split_dir)
    assert inputs.dos_feature_row_count == 57
    np.testing.assert_array_equal(inputs.dos_material_ids[np.arange(29) * 2], material_ids)


def test_materialize_accepts_metadata_bound_nine_fold_loco(
    synthetic_sources: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    master = pd.read_parquet(master_dir / "master_table.parquet")
    iid = pd.read_parquet(split_dir / "iid_split.parquet")
    rows: list[dict[str, object]] = []
    for fold_number in range(9):
        held_out = set(master.index[master.index % 9 == fold_number].tolist())
        remaining = [index for index in master.index if index not in held_out]
        validation = set(remaining[:3])
        for index in master.index:
            rows.append({
                "fold": f"loco_{fold_number:02d}",
                "material_id": master.at[index, "material_id"],
                "crystal_row": master.at[index, "crystal_row"],
                "dos_row": master.at[index, "dos_row"],
                "split": "test" if index in held_out else (
                    "validation" if index in validation else "train"
                ),
                "stratum": iid.at[index, "stratum"],
                "stable_order_key": iid.at[index, "stable_order_key"],
            })
    pd.DataFrame(rows).to_parquet(split_dir / "soap_loco_folds.parquet", index=False)
    (split_dir / "split_metadata.json").write_text(
        json.dumps({"schema_version": "amended_v2", "loco_fold_count": 9}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "controls-nine"
    receipt = materialize_experiment_controls(master_dir, dos_dir, split_dir, output)
    assert receipt["evaluation_unit_count"] == 10
    assert sorted(path.name for path in (output / "loco").iterdir()) == [
        f"loco_{index:02d}" for index in range(9)
    ]


def test_loader_opens_dos_features_with_read_only_memory_mapping(
    synthetic_sources: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_load = np.load
    observed_kwargs: list[dict[str, object]] = []

    def spy_load(file: object, *args: object, **kwargs: object) -> object:
        if Path(file).name == "dos_features.npy":
            observed_kwargs.append(dict(kwargs))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy_load)
    load_control_inputs(*synthetic_sources)
    assert observed_kwargs == [{"allow_pickle": False, "mmap_mode": "r"}]


@pytest.mark.parametrize("shape", [(29, 4), (29, 2, 4), (29, 1, 4, 1)])
def test_loader_rejects_dos_feature_rank_or_channel_shape(
    synthetic_sources: tuple[Path, Path, Path], shape: tuple[int, ...]
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    np.save(dos_dir / "dos_features.npy", np.zeros(shape, dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match=r"dos_features\.npy.*shape"):
        load_control_inputs(master_dir, dos_dir, split_dir)


def test_build_control_unit_uses_frozen_iid_membership_and_aligned_controls(
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    inputs = load_control_inputs(*synthetic_sources)
    unit = build_control_unit(inputs.iid, inputs, ControlSpec(), evaluation_unit="iid")
    expected_test = inputs.iid.loc[inputs.iid["split"] == "test", "material_id"].tolist()
    expected_test.sort(key=lambda value: (inputs.iid.loc[inputs.iid["material_id"] == value, "stable_order_key"].iloc[0], value))
    np.testing.assert_array_equal(unit.full_test_ids, expected_test)
    assert set(unit.paired_test_ids) < set(unit.full_test_ids)
    assert set(unit.train_ids).isdisjoint(set(unit.full_test_ids))
    assert unit.paired_test_dos_rows.dtype == np.int64
    assert unit.train_strata.shape == unit.train_ids.shape
    expected_train = inputs.iid.loc[inputs.iid["split"] == "train", ["material_id", "stratum", "stable_order_key"]].copy()
    expected_train = expected_train.sort_values(["stable_order_key", "material_id"], kind="mergesort")
    np.testing.assert_array_equal(unit.train_ids, expected_train["material_id"].to_numpy(dtype=str))
    np.testing.assert_array_equal(unit.train_strata, expected_train["stratum"].to_numpy(dtype=str))
    assert set(unit.train_ids).isdisjoint(inputs.iid.loc[inputs.iid["split"] == "validation", "material_id"])
    assert set(unit.train_ids).isdisjoint(inputs.iid.loc[inputs.iid["split"] == "test", "material_id"])
    for mask in unit.low_label_masks.values():
        assert mask.shape[1] == len(unit.train_ids)
    assert unit.summary["evaluation_unit"] == "iid"
    expected_counts = {
        "total": len(inputs.master),
        "train": int((inputs.iid["split"] == "train").sum()),
        "validation": int((inputs.iid["split"] == "validation").sum()),
        "test": int((inputs.iid["split"] == "test").sum()),
    }
    expected_paired = int(
        inputs.master.set_index("material_id")
        .loc[inputs.iid.loc[inputs.iid["split"] == "test", "material_id"], "dos_observed"]
        .astype(bool)
        .sum()
    )
    assert unit.summary["counts"] == expected_counts
    assert unit.summary["total_count"] == expected_counts["total"]
    assert unit.summary["train_count"] == expected_counts["train"]
    assert unit.summary["validation_count"] == expected_counts["validation"]
    assert unit.summary["test_count"] == expected_counts["test"]
    assert unit.summary["paired_test_count"] == expected_paired
    assert unit.summary["dos_state_counts"] == {
        "observed": expected_paired,
        "missing": expected_counts["test"] - expected_paired,
    }
    assert set(unit.summary["missing_counts"]) == {rate_key(rate) for rate in ControlSpec().missing_rates}
    assert all(len(counts) == 5 for counts in unit.summary["missing_counts"].values())
    assert unit.summary["shuffle_repeat_count"] == 5
    assert set(unit.summary["low_label_counts"]) == {fraction_key(fraction) for fraction in ControlSpec().label_fractions}
    assert all(len(counts) == 5 for counts in unit.summary["low_label_counts"].values())
    json.dumps(unit.summary)


def test_build_control_unit_accepts_one_loco_fold_and_preserves_cluster_membership(
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    inputs = load_control_inputs(*synthetic_sources)
    fold = inputs.loco.loc[inputs.loco["fold"] == "loco_03"].copy()
    unit = build_control_unit(fold, inputs, ControlSpec(), evaluation_unit="loco_03")
    assert set(unit.full_test_ids) == {"mp-09", "mp-10", "mp-11"}
    assert len(unit.paired_test_ids) == 3
    assert int(unit.summary["counts"]["test"]) == 3


def test_build_control_unit_rejects_multiple_loco_folds(
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    inputs = load_control_inputs(*synthetic_sources)
    with pytest.raises(ValueError, match="loco_00|evaluation unit|one fold"):
        build_control_unit(inputs.loco, inputs, ControlSpec(), evaluation_unit="loco")


def _rewrite_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def test_loader_rejects_loco_outer_test_partition_duplicate_and_omission(
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    """A complete ten-fold source must hold every master material out exactly once."""
    master_dir, dos_dir, split_dir = synthetic_sources
    path = split_dir / "soap_loco_folds.parquet"
    frame = pd.read_parquet(path)
    duplicate = (frame["fold"] == "loco_01") & (frame["material_id"] == "mp-00")
    omitted = (frame["fold"] == "loco_01") & (frame["material_id"] == "mp-03")
    assert frame.loc[duplicate, "split"].iloc[0] == "validation"
    assert frame.loc[omitted, "split"].iloc[0] == "test"
    frame.loc[duplicate, "split"] = "test"
    frame.loc[omitted, "split"] = "validation"
    _rewrite_parquet(path, frame)

    with pytest.raises(ValueError, match="LOCO outer-test partition invariant"):
        load_control_inputs(master_dir, dos_dir, split_dir)


def test_loader_rejects_reordered_dos_ids(synthetic_sources: tuple[Path, Path, Path]) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    ids = np.load(dos_dir / "material_ids.npy", allow_pickle=False)
    np.save(dos_dir / "material_ids.npy", ids[::-1], allow_pickle=False)
    with pytest.raises(ValueError, match="DOS.*IDs|material_ids.npy|order"):
        load_control_inputs(master_dir, dos_dir, split_dir)


@pytest.mark.parametrize("mutation,filename,invariant", [
    ("duplicate_master", "master_table.parquet", "duplicate material IDs"),
    ("bad_dos_row", "iid_split.parquet", "dos_row"),
    ("feature_rows", "dos_features.npy", "row count"),
    ("missing_column", "master_table.parquet", "required columns"),
    ("nonfinite", "master_table.parquet", "must be finite"),
    ("missing_fold_row", "soap_loco_folds.parquet", "every master ID"),
    ("duplicate_fold_row", "soap_loco_folds.parquet", "duplicate fold/material"),
    ("unknown_split", "iid_split.parquet", "unknown split labels"),
    ("mapping", "iid_split.parquet", "mapping disagrees"),
])
def test_loader_rejects_corrupt_sources(
    synthetic_sources: tuple[Path, Path, Path], mutation: str, filename: str, invariant: str
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    master_path = master_dir / "master_table.parquet"
    iid_path = split_dir / "iid_split.parquet"
    loco_path = split_dir / "soap_loco_folds.parquet"
    if mutation == "duplicate_master":
        frame = pd.read_parquet(master_path)
        frame.loc[1, "material_id"] = frame.loc[0, "material_id"]
        _rewrite_parquet(master_path, frame)
    elif mutation == "bad_dos_row":
        frame = pd.read_parquet(master_path)
        frame.loc[1, "dos_row"] = 99
        _rewrite_parquet(master_path, frame)
    elif mutation == "feature_rows":
        np.save(dos_dir / "dos_features.npy", np.zeros((28, 1, 4), dtype=np.float32), allow_pickle=False)
    elif mutation == "missing_column":
        frame = pd.read_parquet(master_path).drop(columns=["g_vrh_gpa"])
        _rewrite_parquet(master_path, frame)
    elif mutation == "nonfinite":
        frame = pd.read_parquet(master_path)
        frame.loc[0, "k_vrh_gpa"] = np.inf
        _rewrite_parquet(master_path, frame)
    elif mutation == "missing_fold_row":
        frame = pd.read_parquet(loco_path)
        _rewrite_parquet(loco_path, frame.iloc[1:])
    elif mutation == "duplicate_fold_row":
        frame = pd.read_parquet(loco_path)
        _rewrite_parquet(loco_path, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    elif mutation == "unknown_split":
        frame = pd.read_parquet(iid_path)
        frame.loc[0, "split"] = "holdout"
        _rewrite_parquet(iid_path, frame)
    elif mutation == "mapping":
        frame = pd.read_parquet(iid_path)
        frame.loc[0, "crystal_row"] = 1
        _rewrite_parquet(iid_path, frame)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError) as caught:
        load_control_inputs(master_dir, dos_dir, split_dir)
    message = str(caught.value)
    assert filename in message
    assert invariant in message


@pytest.mark.parametrize(
    "role,filename,invariant",
    [
        ("master", "master_table.parquet", "could not be read"),
        ("split", "iid_split.parquet", "could not be read"),
        ("split", "soap_loco_folds.parquet", "could not be read"),
    ],
)
def test_loader_reports_corrupt_parquet_with_exact_file_and_read_invariant(
    synthetic_sources: tuple[Path, Path, Path], role: str, filename: str, invariant: str
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    root = {"master": master_dir, "dos": dos_dir, "split": split_dir}[role]
    (root / filename).write_bytes(b"not a parquet file")
    with pytest.raises(ValueError) as caught:
        load_control_inputs(master_dir, dos_dir, split_dir)
    message = str(caught.value)
    assert filename in message
    assert invariant in message


@pytest.mark.parametrize("filename", ["preprocessing_config.json", "split_metadata.json"])
def test_loader_reports_invalid_utf8_metadata_with_file_and_json_invariant(
    synthetic_sources: tuple[Path, Path, Path], filename: str
) -> None:
    master_dir, dos_dir, split_dir = synthetic_sources
    root = dos_dir if filename.startswith("preprocessing") else split_dir
    (root / filename).write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError) as caught:
        load_control_inputs(master_dir, dos_dir, split_dir)
    message = str(caught.value)
    assert filename in message
    assert "UTF-8" in message


def test_control_spec_has_formal_constants() -> None:
    spec = ControlSpec()
    assert spec.spec_version == "experiment_controls_v1"
    assert spec.base_seed == 20260909
    assert spec.missing_rates == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert spec.label_fractions == (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
    assert spec.repeats == 5


def test_derive_seed_matches_sha256_namespace() -> None:
    args = dict(
        spec_version="experiment_controls_v1",
        base_seed=20260909,
        evaluation_unit="test",
        control_family="missingness",
        repeat=0,
    )
    payload = "experiment_controls_v1|20260909|test|missingness|0".encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert derive_seed(**args) == expected
    assert derive_seed(**args) == derive_seed(**args)
    assert derive_seed(**{**args, "repeat": 1}) != derive_seed(**args)
    assert derive_seed(**{**args, "control_family": "other"}) != derive_seed(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spec_version": "", "base_seed": 1, "evaluation_unit": "u", "control_family": "f", "repeat": 0},
        {"spec_version": "v", "base_seed": -1, "evaluation_unit": "u", "control_family": "f", "repeat": 0},
        {"spec_version": "v", "base_seed": True, "evaluation_unit": "u", "control_family": "f", "repeat": 0},
        {"spec_version": "v", "base_seed": 1, "evaluation_unit": "", "control_family": "f", "repeat": 0},
        {"spec_version": "v", "base_seed": 1, "evaluation_unit": "u", "control_family": "", "repeat": 0},
        {"spec_version": "v", "base_seed": 1, "evaluation_unit": "u", "control_family": "f", "repeat": -1},
        {"spec_version": "v", "base_seed": 1, "evaluation_unit": "u", "control_family": "f", "repeat": True},
        {"spec_version": "v", "base_seed": 1, "evaluation_unit": "u", "control_family": "f", "repeat": 0.5},
    ],
)
def test_derive_seed_rejects_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_seed(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("spec_version", ""),
        ("spec_version", "   "),
        ("base_seed", -1),
        ("base_seed", True),
        ("base_seed", 1.0),
        ("missing_rates", (0.0, 0.25, 0.25)),
        ("missing_rates", (float("nan"),)),
        ("missing_rates", (float("inf"),)),
        ("missing_rates", (-0.1,)),
        ("missing_rates", (1.1,)),
        ("label_fractions", (0.1, 0.1)),
        ("label_fractions", (float("nan"),)),
        ("label_fractions", (-0.1,)),
        ("label_fractions", (1.1,)),
        ("repeats", 4),
        ("repeats", 0),
        ("repeats", True),
    ],
)
def test_control_spec_rejects_invalid_values(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError), match=field):
        ControlSpec(**kwargs)


@pytest.mark.parametrize("rate,expected", [(0.0, "missing_rate_000"), (0.25, "missing_rate_025"), (1.0, "missing_rate_100")])
def test_rate_key_formats_percentages(rate: float, expected: str) -> None:
    assert rate_key(rate) == expected


@pytest.mark.parametrize("rate", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_rate_key_rejects_invalid_rates(rate: object) -> None:
    with pytest.raises((TypeError, ValueError), match="rate"):
        rate_key(rate)  # type: ignore[arg-type]


def test_missingness_masks_use_rounded_counts_and_are_binary_nested() -> None:
    spec = ControlSpec()
    masks = build_missingness_masks(7, spec=spec, evaluation_unit="eval")
    assert list(masks) == ["missing_rate_000", "missing_rate_025", "missing_rate_050", "missing_rate_075", "missing_rate_100"]
    for mask in masks.values():
        assert mask.shape == (5, 7)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}
    expected_missing = [0, 2, 4, 5, 7]
    for repeat in range(5):
        missing_sets = []
        for mask, count in zip(masks.values(), expected_missing):
            assert int((mask[repeat] == 0).sum()) == count
            missing_sets.append(set(np.flatnonzero(mask[repeat] == 0)))
        assert all(low <= high for low, high in zip(missing_sets, missing_sets[1:]))


def test_missingness_masks_are_deterministic_and_repeat_specific() -> None:
    spec = ControlSpec()
    first = build_missingness_masks(11, spec=spec, evaluation_unit="eval")
    second = build_missingness_masks(11, spec=spec, evaluation_unit="eval")
    assert all(np.array_equal(first[key], second[key]) for key in first)
    other = build_missingness_masks(11, spec=spec, evaluation_unit="other")
    assert any(not np.array_equal(first[key], other[key]) for key in first)


@pytest.mark.parametrize("material_count", [0, -1, True, 1.5])
def test_missingness_masks_require_positive_integer_material_count(material_count: object) -> None:
    with pytest.raises((TypeError, ValueError), match="material_count"):
        build_missingness_masks(material_count, spec=ControlSpec(), evaluation_unit="eval")  # type: ignore[arg-type]


def test_missingness_masks_require_nonempty_evaluation_unit() -> None:
    with pytest.raises((TypeError, ValueError), match="evaluation_unit"):
        build_missingness_masks(3, spec=ControlSpec(), evaluation_unit=" ")


def test_missingness_masks_reject_duplicate_rounded_rate_keys() -> None:
    spec = ControlSpec(missing_rates=(0.251, 0.252, 0.5))
    with pytest.raises(ValueError, match="rate key"):
        build_missingness_masks(7, spec=spec, evaluation_unit="eval")


def test_shuffled_dos_source_rows_are_deterministic_derangements() -> None:
    dos_rows = np.array([8, 2, 11, 4, 19], dtype=np.int64)
    first = build_shuffled_dos_source_rows(dos_rows, spec=ControlSpec(), evaluation_unit="eval")
    second = build_shuffled_dos_source_rows(dos_rows, spec=ControlSpec(), evaluation_unit="eval")

    assert first.shape == (5, 5)
    assert first.dtype == np.int64
    assert first.tobytes() == second.tobytes()
    for shuffled_rows in first:
        assert np.array_equal(np.sort(shuffled_rows), np.sort(dos_rows))
        assert np.all(shuffled_rows != dos_rows)


@pytest.mark.parametrize(
    "dos_rows",
    [
        np.array([8, 2, 8], dtype=np.int64),
        np.array([True, False], dtype=bool),
        np.array([8.0, 2.0], dtype=float),
        np.array([[8, 2]], dtype=np.int64),
        np.array([8], dtype=np.int64),
    ],
)
def test_shuffled_dos_source_rows_reject_invalid_rows(dos_rows: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="dos_rows"):
        build_shuffled_dos_source_rows(dos_rows, spec=ControlSpec(), evaluation_unit="eval")


@pytest.mark.parametrize(
    "dos_rows",
    [
        np.array([0, 2**63 - 1], dtype=np.uint64),
        np.array([0, 2**63], dtype=np.uint64),
        np.array([2**63, 2**64 - 1], dtype=np.uint64),
    ],
)
def test_shuffled_dos_source_rows_require_lossless_int64_values(dos_rows: np.ndarray) -> None:
    if int(dos_rows.max()) <= np.iinfo(np.int64).max:
        result = build_shuffled_dos_source_rows(dos_rows, spec=ControlSpec(), evaluation_unit="eval")
        assert result.dtype == np.int64
    else:
        with pytest.raises(ValueError, match="int64"):
            build_shuffled_dos_source_rows(dos_rows, spec=ControlSpec(), evaluation_unit="eval")


def _low_label_inputs() -> tuple[np.ndarray, np.ndarray]:
    material_ids = np.array([f"material-{index:02d}" for index in range(40)])
    strata = np.array(["small"] * 12 + ["large"] * 28)
    return material_ids, strata


def test_low_label_masks_have_expected_nested_budgets_and_strata() -> None:
    material_ids, strata = _low_label_inputs()
    masks = build_low_label_masks(material_ids, strata, spec=ControlSpec(), evaluation_unit="eval")

    assert list(masks) == [
        "label_fraction_001",
        "label_fraction_005",
        "label_fraction_010",
        "label_fraction_025",
        "label_fraction_050",
        "label_fraction_100",
    ]
    expected_budgets = [0, 2, 4, 10, 20, 40]
    for mask, expected_budget in zip(masks.values(), expected_budgets):
        assert mask.shape == (5, 40)
        assert mask.dtype == np.uint8
        for repeat_mask in mask:
            budget = int(repeat_mask.sum())
            assert budget == expected_budget
            for stratum in ("small", "large"):
                selected = int(repeat_mask[strata == stratum].sum())
                ideal = budget * int((strata == stratum).sum()) / material_ids.size
                assert abs(selected - ideal) <= 1
    for repeat in range(5):
        selected_sets = [set(np.flatnonzero(mask[repeat])) for mask in masks.values()]
        assert all(low <= high for low, high in zip(selected_sets, selected_sets[1:]))


def test_low_label_masks_keep_imbalanced_strata_within_one_of_ideal() -> None:
    material_ids = np.array([f"material-{index:02d}" for index in range(12)])
    strata = np.array(["singleton"] + ["large"] * 11)
    spec = ControlSpec()
    masks = build_low_label_masks(material_ids, strata, spec=spec, evaluation_unit="eval")

    for fraction, mask in zip(spec.label_fractions, masks.values()):
        for repeat_mask in mask:
            budget = int(repeat_mask.sum())
            expected_budget = int(math.floor(fraction * material_ids.size + 0.5))
            assert budget == expected_budget
            for stratum in ("singleton", "large"):
                selected = int(repeat_mask[strata == stratum].sum())
                ideal = budget * int((strata == stratum).sum()) / material_ids.size
                assert abs(selected - ideal) <= 1


def test_low_label_masks_keep_custom_64_percent_checkpoint_balanced() -> None:
    sizes = (1, 1, 1, 11, 11)
    material_ids = np.array([f"material-{index:02d}" for index in range(sum(sizes))])
    strata = np.concatenate(
        [np.full(size, f"stratum-{index}") for index, size in enumerate(sizes)]
    )
    spec = ControlSpec(label_fractions=(0.64,))
    masks = build_low_label_masks(material_ids, strata, spec=spec, evaluation_unit="eval")
    mask = masks["label_fraction_064"]

    for repeat_mask in mask:
        assert int(repeat_mask.sum()) == 16
        for index, size in enumerate(sizes):
            selected = int(repeat_mask[strata == f"stratum-{index}"].sum())
            assert abs(selected - 16 * size / 25) <= 1


@pytest.mark.parametrize(
    "sizes,fractions",
    [
        ((1, 11), (0.02, 0.09, 0.33, 0.67, 1.0)),
        ((1, 1, 1, 11, 11), (0.04, 0.20, 0.64, 0.84)),
        ((2, 3, 7, 13), (0.08, 0.27, 0.55, 0.91, 1.0)),
    ],
)
def test_low_label_masks_custom_checkpoints_are_balanced_nested_and_deterministic(
    sizes: tuple[int, ...], fractions: tuple[float, ...]
) -> None:
    material_ids = np.array([f"material-{index:02d}" for index in range(sum(sizes))])
    strata = np.concatenate(
        [np.full(size, f"stratum-{index}") for index, size in enumerate(sizes)]
    )
    spec = ControlSpec(label_fractions=fractions)
    first = build_low_label_masks(material_ids, strata, spec=spec, evaluation_unit="eval")
    second = build_low_label_masks(material_ids, strata, spec=spec, evaluation_unit="eval")

    assert all(first[key].tobytes() == second[key].tobytes() for key in first)
    ordered_masks = sorted(
        ((fraction, first[fraction_key(fraction)]) for fraction in fractions), key=lambda item: item[0]
    )
    for repeat in range(spec.repeats):
        selected_sets = []
        for fraction, mask in ordered_masks:
            budget = int(math.floor(fraction * material_ids.size + 0.5))
            repeat_mask = mask[repeat]
            assert int(repeat_mask.sum()) == budget
            selected_sets.append(set(np.flatnonzero(repeat_mask)))
            for index, size in enumerate(sizes):
                selected = int(repeat_mask[strata == f"stratum-{index}"].sum())
                assert abs(selected - budget * size / material_ids.size) <= 1
        assert all(low <= high for low, high in zip(selected_sets, selected_sets[1:]))


def test_stratified_order_is_deterministic_per_namespace_and_varies_by_repeat() -> None:
    material_ids, strata = _low_label_inputs()
    spec = ControlSpec()
    first = build_stratified_order(material_ids, strata, spec=spec, evaluation_unit="eval", repeat=0)
    second = build_stratified_order(material_ids, strata, spec=spec, evaluation_unit="eval", repeat=0)
    other_repeat = build_stratified_order(material_ids, strata, spec=spec, evaluation_unit="eval", repeat=1)

    assert first.dtype == np.int64
    assert np.array_equal(np.sort(first), np.arange(40, dtype=np.int64))
    assert first.tobytes() == second.tobytes()
    assert not np.array_equal(first, other_repeat)


@pytest.mark.parametrize(
    "fraction,expected",
    [
        (0.01, "label_fraction_001"),
        (0.05, "label_fraction_005"),
        (0.10, "label_fraction_010"),
        (0.25, "label_fraction_025"),
        (0.50, "label_fraction_050"),
        (1.0, "label_fraction_100"),
    ],
)
def test_fraction_key_formats_label_fractions(fraction: float, expected: str) -> None:
    assert fraction_key(fraction) == expected


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_fraction_key_rejects_invalid_fractions(fraction: object) -> None:
    with pytest.raises((TypeError, ValueError), match="fraction"):
        fraction_key(fraction)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "material_ids,strata",
    [
        (np.array(["a", "a"]), np.array(["x", "y"])),
        (np.array(["a", "b"]), np.array(["x"])),
        (np.array([], dtype=str), np.array([], dtype=str)),
        (np.array(["a", "b"]), np.array(["", "x"])),
        (np.array(["a", "b"]), np.array([None, "x"], dtype=object)),
        (np.array(["a", "b"]), np.array([np.nan, "x"], dtype=object)),
    ],
)
def test_stratified_order_rejects_invalid_inputs(
    material_ids: np.ndarray, strata: np.ndarray
) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_stratified_order(
            material_ids, strata, spec=ControlSpec(), evaluation_unit="eval", repeat=0
        )


def test_low_label_masks_reject_duplicate_rounded_fraction_keys() -> None:
    material_ids, strata = _low_label_inputs()
    spec = ControlSpec(label_fractions=(0.051, 0.052, 1.0))
    with pytest.raises(ValueError, match="fraction key"):
        build_low_label_masks(material_ids, strata, spec=spec, evaluation_unit="eval")


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_materialize_writes_exact_all_unit_layout_and_loadable_contracts(
    synthetic_sources: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Would fail if a unit is missing, extra, or serializes a wrong array contract."""
    master_dir, dos_dir, split_dir = synthetic_sources
    output_dir = tmp_path / "formal-controls"

    receipt = materialize_experiment_controls(master_dir, dos_dir, split_dir, output_dir)

    assert receipt["complete"] is True
    assert receipt["evaluation_unit_count"] == 11
    assert sorted(path.name for path in output_dir.iterdir() if path.is_dir()) == ["iid", "loco"]
    assert sorted(path.name for path in (output_dir / "loco").iterdir() if path.is_dir()) == [
        f"loco_{index:02d}" for index in range(10)
    ]
    assert not any((output_dir / f"loco_{index:02d}").exists() for index in range(10))
    assert {path.name for path in output_dir.iterdir() if path.is_file()} == {"controls_metadata.json"}
    required = {
        "full_test_material_ids.npy", "paired_test_material_ids.npy", "paired_test_dos_rows.npy",
        "evaluation_missingness_masks.npz", "shuffled_dos_source_rows.npy", "train_material_ids.npy",
        "low_label_masks.npz", "control_summary.json",
    }
    for unit_name in ["iid", *[f"loco_{index:02d}" for index in range(10)]]:
        unit_dir = output_dir / ("iid" if unit_name == "iid" else Path("loco") / unit_name)
        assert {path.name for path in unit_dir.iterdir()} == required
        full = np.load(unit_dir / "full_test_material_ids.npy", allow_pickle=False)
        paired = np.load(unit_dir / "paired_test_material_ids.npy", allow_pickle=False)
        rows = np.load(unit_dir / "paired_test_dos_rows.npy", allow_pickle=False)
        train = np.load(unit_dir / "train_material_ids.npy", allow_pickle=False)
        shuffled = np.load(unit_dir / "shuffled_dos_source_rows.npy", allow_pickle=False)
        assert full.ndim == paired.ndim == rows.ndim == train.ndim == 1
        assert full.dtype.kind == paired.dtype.kind == train.dtype.kind == "U"
        assert rows.dtype == shuffled.dtype == np.dtype("int64")
        assert rows.shape == paired.shape
        assert shuffled.shape == (5, paired.size)
        with np.load(unit_dir / "evaluation_missingness_masks.npz", allow_pickle=False) as masks:
            assert masks.files == ["missing_rate_000", "missing_rate_025", "missing_rate_050", "missing_rate_075", "missing_rate_100"]
            assert all(masks[key].dtype == np.dtype("uint8") and masks[key].shape == (5, paired.size) for key in masks.files)
        with np.load(unit_dir / "low_label_masks.npz", allow_pickle=False) as masks:
            assert masks.files == ["label_fraction_001", "label_fraction_005", "label_fraction_010", "label_fraction_025", "label_fraction_050", "label_fraction_100"]
            assert all(masks[key].dtype == np.dtype("uint8") and masks[key].shape == (5, train.size) for key in masks.files)


def test_materialize_records_hashes_and_is_byte_reproducible_without_paths_or_timestamps(
    synthetic_sources: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Would fail if formal bytes depend on destination/staging details or recorded hashes lie."""
    master_dir, dos_dir, split_dir = synthetic_sources
    first, second = tmp_path / "first", tmp_path / "second"
    materialize_experiment_controls(master_dir, dos_dir, split_dir, first)
    materialize_experiment_controls(master_dir, dos_dir, split_dir, second)
    assert _relative_hashes(first) == _relative_hashes(second)
    metadata = json.loads((first / "controls_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["unit_file_sha256"]) == {
        f"iid/{name}" for name in {
            "full_test_material_ids.npy", "paired_test_material_ids.npy", "paired_test_dos_rows.npy",
            "evaluation_missingness_masks.npz", "shuffled_dos_source_rows.npy", "train_material_ids.npy",
            "low_label_masks.npz", "control_summary.json",
        }
    } | {
        f"loco/loco_{index:02d}/{name}"
        for index in range(10)
        for name in {
            "full_test_material_ids.npy", "paired_test_material_ids.npy", "paired_test_dos_rows.npy",
            "evaluation_missingness_masks.npz", "shuffled_dos_source_rows.npy", "train_material_ids.npy",
            "low_label_masks.npz", "control_summary.json",
        }
    }
    for relative, digest in metadata["unit_file_sha256"].items():
        assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == digest
    for unit_name in metadata["evaluation_units"]:
        unit_dir = first / ("iid" if unit_name == "iid" else Path("loco") / unit_name)
        summary = json.loads((unit_dir / "control_summary.json").read_text(encoding="utf-8"))
        assert set(summary["output_file_sha256"]) == {
            "full_test_material_ids.npy", "paired_test_material_ids.npy", "paired_test_dos_rows.npy",
            "evaluation_missingness_masks.npz", "shuffled_dos_source_rows.npy", "train_material_ids.npy", "low_label_masks.npz",
        }
        for name, digest in summary["output_file_sha256"].items():
            assert hashlib.sha256((unit_dir / name).read_bytes()).hexdigest() == digest
    for json_path in first.rglob("*.json"):
        text = json_path.read_text(encoding="utf-8")
        assert str(first) not in text and str(second) not in text and ".staging-" not in text
        assert "timestamp" not in text.lower()


def test_materialize_reuses_exact_output_and_preserves_conflicting_destination(
    synthetic_sources: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Would fail if an existing formal output is overwritten instead of compared byte-for-byte."""
    master_dir, dos_dir, split_dir = synthetic_sources
    output_dir = tmp_path / "controls"
    materialize_experiment_controls(master_dir, dos_dir, split_dir, output_dir)
    before = _relative_hashes(output_dir)
    receipt = materialize_experiment_controls(master_dir, dos_dir, split_dir, output_dir)
    assert receipt["complete"] is True and receipt["reused"] is True
    assert _relative_hashes(output_dir) == before
    (output_dir / "loco" / "loco_00" / "train_material_ids.npy").write_bytes(b"conflict")
    conflicting_before = _relative_hashes(output_dir)
    with pytest.raises(FileExistsError):
        materialize_experiment_controls(master_dir, dos_dir, split_dir, output_dir)
    assert _relative_hashes(output_dir) == conflicting_before


@pytest.mark.parametrize("source_index", [0, 1, 2])
def test_materialize_rejects_source_nested_output_and_cleans_failed_staging(
    synthetic_sources: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    source_index: int,
) -> None:
    """Would fail if a failed build publishes partial data or staging cleanup escapes its exact target."""
    master_dir, dos_dir, split_dir = synthetic_sources
    with pytest.raises(ValueError, match="output"):
        materialize_experiment_controls(
            master_dir, dos_dir, split_dir, (master_dir, dos_dir, split_dir)[source_index] / "nested"
        )
    output_dir = tmp_path / "never-published"
    real_builder = controls.build_control_unit
    calls = 0

    def fail_midway(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected builder failure")
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(controls, "build_control_unit", fail_midway)
    with pytest.raises(RuntimeError, match="injected"):
        materialize_experiment_controls(master_dir, dos_dir, split_dir, output_dir)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".never-published.staging-*"))


def test_materialize_diagnostic_flag_is_recorded_without_changing_controls(
    synthetic_sources: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Would fail if diagnostic mode changes formal control rates, repeats, RNG, or unit outputs."""
    master_dir, dos_dir, split_dir = synthetic_sources
    normal, diagnostic = tmp_path / "normal", tmp_path / "diagnostic"
    materialize_experiment_controls(master_dir, dos_dir, split_dir, normal)
    materialize_experiment_controls(master_dir, dos_dir, split_dir, diagnostic, diagnostic_smoke=True)
    normal_metadata = json.loads((normal / "controls_metadata.json").read_text(encoding="utf-8"))
    diagnostic_metadata = json.loads((diagnostic / "controls_metadata.json").read_text(encoding="utf-8"))
    assert normal_metadata["diagnostic_smoke"] is False
    assert diagnostic_metadata["diagnostic_smoke"] is True
    for key in ("spec", "rng", "evaluation_units", "evaluation_unit_count", "unit_counts", "unit_file_sha256"):
        assert diagnostic_metadata[key] == normal_metadata[key]
