"""Deterministic primitives shared by Experiment Controls v1."""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


UNIT_FILES = {
    "full_test_material_ids.npy",
    "paired_test_material_ids.npy",
    "paired_test_dos_rows.npy",
    "evaluation_missingness_masks.npz",
    "shuffled_dos_source_rows.npy",
    "train_material_ids.npy",
    "low_label_masks.npz",
    "control_summary.json",
}


@dataclass(frozen=True)
class ControlInputs:
    master: pd.DataFrame
    iid: pd.DataFrame
    loco: pd.DataFrame
    dos_material_ids: np.ndarray
    dos_feature_row_count: int
    source_sha256: Mapping[str, str]
    loco_fold_count: int


@dataclass(frozen=True)
class ControlUnit:
    evaluation_unit: str
    full_test_ids: np.ndarray
    paired_test_ids: np.ndarray
    paired_test_dos_rows: np.ndarray
    train_ids: np.ndarray
    train_strata: np.ndarray
    missingness_masks: Mapping[str, np.ndarray]
    shuffled_rows: np.ndarray
    low_label_masks: Mapping[str, np.ndarray]
    summary: Mapping[str, Any]


def _require_name(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_integer(name: str, value: object, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_rates(name: str, values: object) -> tuple[float, ...]:
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of rates") from exc
    if not raw_values:
        raise ValueError(f"{name} must not be empty")
    normalized: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{index}] must be a real number")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError(f"{name}[{index}] must be finite and in [0, 1]")
        normalized.append(converted)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    return tuple(normalized)


@dataclass(frozen=True)
class ControlSpec:
    spec_version: str = "experiment_controls_v1"
    base_seed: int = 20260909
    missing_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    label_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
    repeats: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_version", _require_name("spec_version", self.spec_version))
        object.__setattr__(self, "base_seed", _require_integer("base_seed", self.base_seed))
        object.__setattr__(self, "missing_rates", _validate_rates("missing_rates", self.missing_rates))
        object.__setattr__(self, "label_fractions", _validate_rates("label_fractions", self.label_fractions))
        repeats = _require_integer("repeats", self.repeats)
        if repeats != 5:
            raise ValueError("repeats must be exactly five")
        object.__setattr__(self, "repeats", repeats)


def derive_seed(
    *,
    spec_version: str,
    base_seed: int,
    evaluation_unit: str,
    control_family: str,
    repeat: int,
) -> int:
    """Derive a stable 64-bit seed from the complete control namespace."""
    spec_version = _require_name("spec_version", spec_version)
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)
    control_family = _require_name("control_family", control_family)
    base_seed = _require_integer("base_seed", base_seed)
    repeat = _require_integer("repeat", repeat)
    payload = (
        f"{spec_version}|{base_seed}|{evaluation_unit}|"
        f"{control_family}|{repeat}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validated_rate(rate: object) -> float:
    if isinstance(rate, bool) or not isinstance(rate, Real):
        raise TypeError("rate must be a real number")
    converted = float(rate)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("rate must be finite and in [0, 1]")
    return converted


def rate_key(rate: float) -> str:
    """Return the stable key for a missingness rate."""
    percentage = int(round(_validated_rate(rate) * 100))
    return f"missing_rate_{percentage:03d}"


def _sattolo_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    order = np.arange(size, dtype=np.int64)
    for i in range(size - 1, 0, -1):
        j = int(rng.integers(0, i))
        order[i], order[j] = order[j], order[i]
    return order


def build_shuffled_dos_source_rows(
    dos_rows: np.ndarray,
    *,
    spec: ControlSpec,
    evaluation_unit: str,
) -> np.ndarray:
    """Build repeat-specific Sattolo permutations for shuffled-DOS controls."""
    if not isinstance(dos_rows, np.ndarray):
        raise TypeError("dos_rows must be a NumPy array")
    if dos_rows.ndim != 1:
        raise ValueError("dos_rows must be one-dimensional")
    if dos_rows.dtype.kind not in "iu":
        raise TypeError("dos_rows must contain integers")
    if dos_rows.dtype.kind == "u" and np.any(dos_rows > np.iinfo(np.int64).max):
        raise ValueError("dos_rows must be representable as int64")
    if dos_rows.size < 2:
        raise ValueError("dos_rows must contain at least two rows")
    if np.unique(dos_rows).size != dos_rows.size:
        raise ValueError("dos_rows must be unique")
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec must be a ControlSpec")
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)

    shuffled_rows = np.empty((spec.repeats, dos_rows.size), dtype=np.int64)
    for repeat in range(spec.repeats):
        seed = derive_seed(
            spec_version=spec.spec_version,
            base_seed=spec.base_seed,
            evaluation_unit=evaluation_unit,
            control_family="shuffle",
            repeat=repeat,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        shuffled_rows[repeat] = dos_rows[_sattolo_indices(dos_rows.size, rng)]
    return shuffled_rows


def _validated_fraction(fraction: object) -> float:
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise TypeError("fraction must be a real number")
    converted = float(fraction)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("fraction must be finite and in [0, 1]")
    return converted


def fraction_key(fraction: float) -> str:
    """Return the stable key for a configured label fraction."""
    percentage = int(round(_validated_fraction(fraction) * 100))
    return f"label_fraction_{percentage:03d}"


def _stable_scalar_token(value: object, *, name: str, reject_empty: bool) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError(f"{name} must not contain missing values")
    if isinstance(value, str):
        if reject_empty and not value.strip():
            raise ValueError(f"{name} must not contain empty values")
        return f"str:{value}"
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, Integral):
        return f"int:{int(value)}"
    if isinstance(value, Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must not contain missing values")
        return f"float:{converted.hex()}"
    raise TypeError(f"{name} must contain stable scalar values")


def _sha256_key(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def _validate_stratified_inputs(
    material_ids: np.ndarray, strata: np.ndarray
) -> tuple[list[str], list[str]]:
    if not isinstance(material_ids, np.ndarray) or not isinstance(strata, np.ndarray):
        raise TypeError("material_ids and strata must be NumPy arrays")
    if material_ids.ndim != 1 or strata.ndim != 1:
        raise ValueError("material_ids and strata must be one-dimensional")
    if material_ids.size == 0 or strata.size == 0:
        raise ValueError("material_ids and strata must not be empty")
    if material_ids.size != strata.size:
        raise ValueError("material_ids and strata must have matching lengths")

    material_tokens = [
        _stable_scalar_token(value, name="material_ids", reject_empty=False)
        for value in material_ids
    ]
    if len(set(material_tokens)) != len(material_tokens):
        raise ValueError("material_ids must be unique")
    stratum_tokens = [
        _stable_scalar_token(value, name="strata", reject_empty=True) for value in strata
    ]
    return material_tokens, stratum_tokens


def build_stratified_order(
    material_ids: np.ndarray,
    strata: np.ndarray,
    *,
    spec: ControlSpec,
    evaluation_unit: str,
    repeat: int,
) -> np.ndarray:
    """Build one deterministic, proportionally interleaved training order."""
    material_tokens, stratum_tokens = _validate_stratified_inputs(material_ids, strata)
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec must be a ControlSpec")
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)
    repeat = _require_integer("repeat", repeat)
    if repeat >= spec.repeats:
        raise ValueError("repeat must be within the configured repeats")

    members_by_stratum: dict[str, list[int]] = {}
    for position, stratum in enumerate(stratum_tokens):
        members_by_stratum.setdefault(stratum, []).append(position)

    namespace = (spec.spec_version, str(spec.base_seed), evaluation_unit, str(repeat))
    for stratum, members in members_by_stratum.items():
        members.sort(
            key=lambda position: (
                _sha256_key(
                    *namespace, "low_label_member", stratum, material_tokens[position]
                ),
                material_tokens[position],
            )
        )

    strata_keys = tuple(members_by_stratum)
    tie_keys = {
        stratum: _sha256_key(*namespace, "low_label_stratum", stratum)
        for stratum in strata_keys
    }
    total_size = material_ids.size
    checkpoint_budgets = {
        total_size
        if fraction == 1.0
        else int(math.floor(fraction * total_size + 0.5))
        for fraction in spec.label_fractions
    }
    checkpoint_budgets.add(total_size)

    previous_quotas = {stratum: 0 for stratum in strata_keys}
    checkpoints: list[tuple[int, dict[str, int]]] = []
    for budget in sorted(checkpoint_budgets):
        quotas = {
            stratum: max(
                previous_quotas[stratum],
                int(math.floor(budget * len(members_by_stratum[stratum]) / total_size)),
            )
            for stratum in strata_keys
        }
        upper_bounds = {
            stratum: min(
                len(members_by_stratum[stratum]),
                int(math.floor(budget * len(members_by_stratum[stratum]) / total_size + 1)),
            )
            for stratum in strata_keys
        }
        minimum_total = sum(quotas.values())
        if minimum_total > budget:
            raise ValueError(
                f"infeasible low-label checkpoint budget {budget}: "
                f"monotone quotas require {minimum_total} selections"
            )
        while minimum_total < budget:
            available = [
                stratum for stratum in strata_keys if quotas[stratum] < upper_bounds[stratum]
            ]
            if not available:
                raise ValueError(
                    f"infeasible low-label checkpoint budget {budget}: "
                    "no stratum can receive another selection"
                )
            selected_stratum = max(
                available,
                key=lambda stratum: (
                    budget * len(members_by_stratum[stratum]) / total_size
                    - math.floor(budget * len(members_by_stratum[stratum]) / total_size),
                    tie_keys[stratum],
                ),
            )
            quotas[selected_stratum] += 1
            minimum_total += 1
        checkpoints.append((budget, quotas))
        previous_quotas = quotas

    emitted_counts = {stratum: 0 for stratum in strata_keys}
    order: list[int] = []
    for _, quotas in checkpoints:
        for stratum in sorted(strata_keys, key=lambda value: tie_keys[value]):
            start = emitted_counts[stratum]
            end = quotas[stratum]
            order.extend(members_by_stratum[stratum][start:end])
            emitted_counts[stratum] = end
    return np.asarray(order, dtype=np.int64)


def build_low_label_masks(
    material_ids: np.ndarray,
    strata: np.ndarray,
    *,
    spec: ControlSpec,
    evaluation_unit: str,
) -> dict[str, np.ndarray]:
    """Build nested low-label masks using deterministic stratified prefixes."""
    _validate_stratified_inputs(material_ids, strata)
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec must be a ControlSpec")
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)

    keys = [fraction_key(fraction) for fraction in spec.label_fractions]
    if len(set(keys)) != len(keys):
        raise ValueError("label fraction key collision: rounded percentages are not unique")

    material_count = material_ids.size
    masks = {
        key: np.zeros((spec.repeats, material_count), dtype=np.uint8) for key in keys
    }
    counts = [
        material_count
        if fraction == 1.0
        else int(math.floor(fraction * material_count + 0.5))
        for fraction in spec.label_fractions
    ]
    for repeat in range(spec.repeats):
        order = build_stratified_order(
            material_ids, strata, spec=spec, evaluation_unit=evaluation_unit, repeat=repeat
        )
        for key, count in zip(keys, counts):
            masks[key][repeat, order[:count]] = 1
    return masks


def build_missingness_masks(
    material_count: int,
    *,
    spec: ControlSpec,
    evaluation_unit: str,
) -> dict[str, np.ndarray]:
    """Build deterministic observed/missing masks for each configured rate."""
    material_count = _require_integer("material_count", material_count)
    if material_count <= 0:
        raise ValueError("material_count must be positive")
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec must be a ControlSpec")
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)

    keys = [rate_key(rate) for rate in spec.missing_rates]
    if len(set(keys)) != len(keys):
        raise ValueError("missing rate key collision: rounded percentages are not unique")

    masks = {
        key: np.ones((spec.repeats, material_count), dtype=np.uint8)
        for key in keys
    }
    counts = [int(math.floor(rate * material_count + 0.5)) for rate in spec.missing_rates]
    for repeat in range(spec.repeats):
        seed = derive_seed(
            spec_version=spec.spec_version,
            base_seed=spec.base_seed,
            evaluation_unit=evaluation_unit,
            control_family="missingness",
            repeat=repeat,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        permutation = rng.permutation(material_count)
        for key, missing_count in zip(keys, counts):
            masks[key][repeat, permutation[:missing_count]] = 0
    return masks


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_columns(frame: pd.DataFrame, required: set[str], domain: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{domain} is missing required columns: {missing}")


def _normalise_frame_ids(frame: pd.DataFrame, domain: str, *, unique: bool = True) -> pd.DataFrame:
    result = frame.copy()
    if result["material_id"].isna().any():
        raise ValueError(f"{domain} material_id contains missing IDs")
    result["material_id"] = result["material_id"].map(str)
    if result["material_id"].str.strip().eq("").any():
        raise ValueError(f"{domain} material_id contains empty IDs")
    if unique and result["material_id"].duplicated().any():
        raise ValueError(f"{domain} contains duplicate material IDs")
    return result


def _scalar_is_null(value: object) -> bool:
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _integer_value(value: object, *, domain: str, allow_null: bool) -> int | None:
    if _scalar_is_null(value):
        if allow_null:
            return None
        raise ValueError(f"{domain} contains a missing integer")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{domain} must contain integer values")
    converted = float(value)
    if not math.isfinite(converted) or not converted.is_integer():
        raise ValueError(f"{domain} must contain integer values")
    return int(converted)


def _validate_master(master: pd.DataFrame, path: Path) -> pd.DataFrame:
    domain = f"master file {path}"
    _required_columns(
        master,
        {"material_id", "crystal_row", "dos_row", "dos_observed", "k_vrh_gpa", "g_vrh_gpa"},
        domain,
    )
    result = _normalise_frame_ids(master, domain)
    n_materials = len(result)
    crystal_rows = [_integer_value(value, domain=f"{domain} crystal_row", allow_null=False) for value in result["crystal_row"]]
    if sorted(crystal_rows) != list(range(n_materials)) or len(set(crystal_rows)) != n_materials:
        raise ValueError(f"{domain} crystal_row must be unique and contiguous 0..N-1")
    for column in ("k_vrh_gpa", "g_vrh_gpa"):
        numeric = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{domain} {column} must be finite")
        result[column] = numeric
    observed_values: list[bool] = []
    dos_values: list[int | None] = []
    for observed, row in zip(result["dos_observed"], result["dos_row"]):
        if isinstance(observed, (bool, np.bool_)):
            is_observed = bool(observed)
        elif isinstance(observed, Real) and not isinstance(observed, (bool, np.bool_)):
            numeric_observed = float(observed)
            if not math.isfinite(numeric_observed) or numeric_observed not in (0.0, 1.0):
                raise ValueError(f"{domain} dos_observed must be boolean or 0/1")
            is_observed = bool(int(numeric_observed))
        else:
            raise ValueError(f"{domain} dos_observed must be boolean or 0/1")
        dos_value = _integer_value(row, domain=f"{domain} dos_row", allow_null=not is_observed)
        if not is_observed and dos_value is not None:
            raise ValueError(f"{domain} unobserved rows must have null dos_row")
        observed_values.append(is_observed)
        dos_values.append(dos_value)
    observed_rows = [value for value in dos_values if value is not None]
    if any(value < 0 for value in observed_rows) or len(set(observed_rows)) != len(observed_rows):
        raise ValueError(f"{domain} observed dos_row must be unique and non-negative")
    result["crystal_row"] = np.asarray(crystal_rows, dtype=np.int64)
    result["dos_observed"] = np.asarray(observed_values, dtype=np.uint8)
    result["dos_row"] = pd.array(dos_values, dtype="Int64")
    return result


def _validate_split_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    master: pd.DataFrame,
    loco: bool,
    require_all_folds: bool = True,
    expected_fold_count: int = 10,
) -> pd.DataFrame:
    domain = f"{'LOCO' if loco else 'IID'} split file {path}"
    required = {"material_id", "crystal_row", "dos_row", "split", "stratum", "stable_order_key"}
    if loco:
        required.add("fold")
    _required_columns(frame, required, domain)
    result = _normalise_frame_ids(frame, domain, unique=not loco)
    labels = set(result["split"].dropna().astype(str))
    if labels - {"train", "validation", "test"}:
        raise ValueError(f"{domain} has unknown split labels: {sorted(labels - {'train', 'validation', 'test'})}")
    if result["split"].isna().any():
        raise ValueError(f"{domain} split labels must be train/validation/test")
    if result["stratum"].isna().any() or result["stable_order_key"].isna().any():
        raise ValueError(f"{domain} stratum and stable_order_key must be non-null")
    result["split"] = result["split"].astype(str)
    result["stratum"] = result["stratum"].map(str)
    result["stable_order_key"] = result["stable_order_key"].map(str)
    if result["stratum"].str.strip().eq("").any() or result["stable_order_key"].str.strip().eq("").any():
        raise ValueError(f"{domain} stratum and stable_order_key must be non-empty")
    master_ids = set(master["material_id"])
    if loco:
        if result["fold"].isna().any() or result["fold"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{domain} fold labels must be non-empty")
        result["fold"] = result["fold"].map(str)
        if result.duplicated(subset=["fold", "material_id"]).any():
            raise ValueError(f"{domain} contains duplicate fold/material rows")
        folds = sorted(result["fold"].unique())
        if require_all_folds and len(folds) != expected_fold_count:
            raise ValueError(f"{domain} must contain exactly {expected_fold_count} folds")
        for fold in folds:
            fold_frame = result[result["fold"] == fold]
            if len(fold_frame) != len(master) or set(fold_frame["material_id"]) != master_ids:
                raise ValueError(f"{domain} fold {fold} must contain every master ID exactly once")
            counts = fold_frame["split"].value_counts()
            if any(int(counts.get(label, 0)) == 0 for label in ("train", "validation", "test")):
                raise ValueError(f"{domain} fold {fold} must have nonempty train/validation/test")
        if require_all_folds:
            outer_test_ids = result.loc[result["split"] == "test", "material_id"]
            outer_test_counts = outer_test_ids.value_counts()
            if (
                len(outer_test_ids) != len(master_ids)
                or set(outer_test_ids) != master_ids
                or any(int(outer_test_counts.get(material_id, 0)) != 1 for material_id in master_ids)
            ):
                raise ValueError(
                    f"{domain} LOCO outer-test partition invariant requires every master ID "
                    f"exactly once as test across {expected_fold_count} folds"
                )
    else:
        if len(result) != len(master) or set(result["material_id"]) != master_ids:
            raise ValueError(f"{domain} must contain every master ID exactly once")
        if result["material_id"].duplicated().any():
            raise ValueError(f"{domain} contains duplicate material IDs")
        counts = result["split"].value_counts()
        if any(int(counts.get(label, 0)) == 0 for label in ("train", "validation", "test")):
            raise ValueError(f"{domain} must have nonempty train/validation/test")
    master_index = master.set_index("material_id")
    for _, row in result.iterrows():
        expected = master_index.loc[row["material_id"]]
        crystal = _integer_value(row["crystal_row"], domain=f"{domain} crystal_row", allow_null=False)
        dos = _integer_value(row["dos_row"], domain=f"{domain} dos_row", allow_null=True)
        expected_crystal = int(expected["crystal_row"])
        expected_dos = _integer_value(expected["dos_row"], domain=f"{domain} master dos_row", allow_null=True)
        if crystal != expected_crystal or dos != expected_dos:
            raise ValueError(f"{domain} crystal_row/dos_row mapping disagrees with master for {row['material_id']}")
    return result.reset_index(drop=True)


def load_control_inputs(
    master_dir: Path | str,
    dos_dir: Path | str,
    split_dir: Path | str,
) -> ControlInputs:
    master_root, dos_root, split_root = Path(master_dir), Path(dos_dir), Path(split_dir)
    files = {
        "master/master_table.parquet": master_root / "master_table.parquet",
        "dos/material_ids.npy": dos_root / "material_ids.npy",
        "dos/dos_features.npy": dos_root / "dos_features.npy",
        "dos/preprocessing_config.json": dos_root / "preprocessing_config.json",
        "split/iid_split.parquet": split_root / "iid_split.parquet",
        "split/soap_loco_folds.parquet": split_root / "soap_loco_folds.parquet",
        "split/split_metadata.json": split_root / "split_metadata.json",
    }
    for key, path in files.items():
        if not path.is_file():
            raise ValueError(f"missing source file {key}: {path}")
    source_sha256 = {key: _source_sha256(path) for key, path in files.items()}
    metadata_values: dict[str, dict[str, Any]] = {}
    for key in ("dos/preprocessing_config.json", "split/split_metadata.json"):
        try:
            with files[key].open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"invalid metadata source {key}: JSON root must be an object")
            metadata_values[key] = value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid metadata source {key} (UTF-8/JSON invariant): {exc}") from exc
    split_metadata = metadata_values["split/split_metadata.json"]
    raw_fold_count = split_metadata.get("loco_fold_count", split_metadata.get("fold_count"))
    try:
        fold_count = int(raw_fold_count)
        numeric_fold_count = float(raw_fold_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("split metadata must declare an integer LOCO fold count") from exc
    if (
        isinstance(raw_fold_count, bool)
        or not math.isfinite(numeric_fold_count)
        or numeric_fold_count != fold_count
        or fold_count < 2
    ):
        raise ValueError("split metadata LOCO fold count must be an integer of at least two")

    master_path = files["master/master_table.parquet"]
    try:
        master_raw = pd.read_parquet(master_path)
    except Exception as exc:
        raise ValueError(f"master_table.parquet source {master_path} could not be read: {exc}") from exc
    try:
        master = _validate_master(master_raw, master_path)
    except Exception as exc:
        raise ValueError(f"master_table.parquet source validation failed: {exc}") from exc

    iid_path = files["split/iid_split.parquet"]
    try:
        iid_raw = pd.read_parquet(iid_path)
    except Exception as exc:
        raise ValueError(f"iid_split.parquet source {iid_path} could not be read: {exc}") from exc
    try:
        iid = _validate_split_frame(iid_raw, path=iid_path, master=master, loco=False)
    except Exception as exc:
        raise ValueError(f"iid_split.parquet source validation failed: {exc}") from exc

    loco_path = files["split/soap_loco_folds.parquet"]
    try:
        loco_raw = pd.read_parquet(loco_path)
    except Exception as exc:
        raise ValueError(f"soap_loco_folds.parquet source {loco_path} could not be read: {exc}") from exc
    try:
        loco = _validate_split_frame(
            loco_raw,
            path=loco_path,
            master=master,
            loco=True,
            expected_fold_count=fold_count,
        )
    except Exception as exc:
        raise ValueError(f"soap_loco_folds.parquet source validation failed: {exc}") from exc
    try:
        dos_ids_raw = np.load(files["dos/material_ids.npy"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"DOS ID source material_ids.npy is not loadable without pickle: {exc}") from exc
    if dos_ids_raw.ndim != 1 or dos_ids_raw.size == 0:
        raise ValueError("DOS ID source material_ids.npy must be one-dimensional and nonempty")
    if any(_scalar_is_null(value) for value in dos_ids_raw):
        raise ValueError("DOS ID source material_ids.npy contains missing IDs")
    dos_ids = np.asarray([str(value) for value in dos_ids_raw], dtype=str)
    if np.any(np.char.str_len(np.char.strip(dos_ids)) == 0) or np.unique(dos_ids).size != dos_ids.size:
        raise ValueError("DOS ID source material_ids.npy must contain unique nonempty IDs")
    observed_master = master[master["dos_observed"].astype(bool)].sort_values("dos_row", kind="mergesort")
    observed_rows = observed_master["dos_row"].astype(np.int64).to_numpy()
    if observed_rows.size and int(observed_rows.max()) >= dos_ids.size:
        raise ValueError("master dos_row indexes exceed the DOS ID source row count")
    expected_ids = observed_master["material_id"].to_numpy(dtype=str)
    if not np.array_equal(dos_ids[observed_rows], expected_ids):
        raise ValueError(
            "DOS ID source material_ids.npy does not match observed master IDs at dos_row indexes"
        )
    try:
        features = np.load(files["dos/dos_features.npy"], allow_pickle=False, mmap_mode="r")
        feature_shape = features.shape
        if len(feature_shape) != 3 or feature_shape[1] != 1:
            raise ValueError(f"DOS feature source dos_features.npy must have shape (N_observed, 1, point_count), got {feature_shape}")
        if int(feature_shape[0]) != dos_ids.size:
            raise ValueError(f"DOS feature source dos_features.npy row count {feature_shape[0]} disagrees with DOS IDs {dos_ids.size}")
        feature_rows = int(feature_shape[0])
        del features
    except (OSError, ValueError) as exc:
        raise ValueError(f"DOS feature source dos_features.npy validation failed: {exc}") from exc
    return ControlInputs(master, iid, loco, dos_ids, feature_rows, source_sha256, fold_count)


def _same_unit_rows(actual: pd.DataFrame, expected: pd.DataFrame, *, domain: str) -> None:
    columns = ["material_id", "crystal_row", "dos_row", "split", "stratum", "stable_order_key"]
    if len(actual) != len(expected):
        raise ValueError(f"{domain} does not contain exactly one complete evaluation unit")
    left = actual[columns].copy().sort_values("material_id", kind="mergesort").reset_index(drop=True)
    right = expected[columns].copy().sort_values("material_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(left["material_id"].to_numpy(dtype=str), right["material_id"].to_numpy(dtype=str)):
        raise ValueError(f"{domain} membership disagrees with frozen source")
    for column in ("crystal_row", "split", "stratum", "stable_order_key"):
        if not np.array_equal(left[column].astype(str), right[column].astype(str)):
            raise ValueError(f"{domain} {column} disagrees with frozen source")
    for frame in (left, right):
        nulls = frame["dos_row"].isna().to_numpy()
        values = pd.to_numeric(frame["dos_row"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        if frame is left:
            left_dos = (nulls, values)
        else:
            right_dos = (nulls, values)
    if not np.array_equal(left_dos[0], right_dos[0]) or not np.array_equal(left_dos[1], right_dos[1]):
        raise ValueError(f"{domain} dos_row disagrees with frozen source")


def build_control_unit(
    split_frame: pd.DataFrame,
    inputs: ControlInputs,
    spec: ControlSpec,
    evaluation_unit: str,
) -> ControlUnit:
    if not isinstance(inputs, ControlInputs):
        raise TypeError("inputs must be a ControlInputs")
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec must be a ControlSpec")
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)
    if not isinstance(split_frame, pd.DataFrame):
        raise TypeError(f"evaluation unit {evaluation_unit} must be a pandas DataFrame")
    if "fold" in split_frame.columns:
        folds = split_frame["fold"].dropna().astype(str).unique()
        if len(folds) != 1:
            raise ValueError(f"evaluation unit {evaluation_unit} must contain exactly one LOCO fold")
        fold = str(folds[0])
        if evaluation_unit != fold:
            raise ValueError(f"evaluation unit {evaluation_unit} does not match LOCO fold {fold}")
        expected = inputs.loco[inputs.loco["fold"] == fold]
        actual = _validate_split_frame(split_frame, path=Path(f"<evaluation unit {evaluation_unit}>"), master=inputs.master, loco=True, require_all_folds=False)
        if actual["fold"].nunique() != 1:
            raise ValueError(f"evaluation unit {evaluation_unit} must contain exactly one LOCO fold")
    else:
        if evaluation_unit != "iid":
            raise ValueError(f"evaluation unit {evaluation_unit} must be IID or one filtered LOCO fold")
        expected = inputs.iid
        actual = _validate_split_frame(split_frame, path=Path("<evaluation unit iid>"), master=inputs.master, loco=False)
    _same_unit_rows(actual, expected, domain=f"evaluation unit {evaluation_unit}")
    test = actual[actual["split"] == "test"].sort_values(["stable_order_key", "material_id"], kind="mergesort")
    train = actual[actual["split"] == "train"].sort_values(["stable_order_key", "material_id"], kind="mergesort")
    master_index = inputs.master.set_index("material_id")
    full_test_ids = test["material_id"].to_numpy(dtype=str)
    paired_mask = np.asarray([bool(master_index.loc[mid, "dos_observed"]) for mid in full_test_ids], dtype=bool)
    paired_test_ids = full_test_ids[paired_mask]
    paired_test_dos_rows = np.asarray(
        [int(master_index.loc[mid, "dos_row"]) for mid in paired_test_ids], dtype=np.int64
    )
    if paired_test_dos_rows.size < 2:
        raise ValueError(f"evaluation unit {evaluation_unit} requires at least two paired test DOS rows")
    train_ids = train["material_id"].to_numpy(dtype=str)
    train_strata = train["stratum"].to_numpy(dtype=str)
    missingness_masks = build_missingness_masks(
        len(paired_test_ids), spec=spec, evaluation_unit=evaluation_unit
    )
    shuffled_rows = build_shuffled_dos_source_rows(
        paired_test_dos_rows, spec=spec, evaluation_unit=evaluation_unit
    )
    low_label_masks = build_low_label_masks(
        train_ids, train_strata, spec=spec, evaluation_unit=evaluation_unit
    )
    missing_counts = {key: [int((mask[repeat] == 0).sum()) for repeat in range(spec.repeats)] for key, mask in missingness_masks.items()}
    low_label_counts = {key: [int(mask[repeat].sum()) for repeat in range(spec.repeats)] for key, mask in low_label_masks.items()}
    summary = {
        "evaluation_unit": evaluation_unit,
        "counts": {"total": int(len(actual)), "train": int((actual["split"] == "train").sum()), "validation": int((actual["split"] == "validation").sum()), "test": int((actual["split"] == "test").sum())},
        "total_count": int(len(actual)), "train_count": int(len(train_ids)),
        "validation_count": int((actual["split"] == "validation").sum()), "test_count": int(len(full_test_ids)),
        "paired_test_count": int(len(paired_test_ids)),
        "dos_state_counts": {"observed": int(paired_mask.sum()), "missing": int((~paired_mask).sum())},
        "missing_counts": missing_counts, "shuffle_repeat_count": int(shuffled_rows.shape[0]),
        "low_label_counts": low_label_counts,
    }
    return ControlUnit(evaluation_unit, full_test_ids, paired_test_ids, paired_test_dos_rows, train_ids, train_strata, missingness_masks, shuffled_rows, low_label_masks, summary)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _require_non_object_array(name: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype.hasobject:
        raise TypeError(f"{name} must not have object dtype")
    return value


def _write_npy(path: Path, array: np.ndarray) -> None:
    _require_non_object_array(path.name, array)
    np.save(path, array, allow_pickle=False)


def _npy_bytes(array: np.ndarray) -> bytes:
    _require_non_object_array("NPZ member", array)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write ordered NPZ members with fixed ZIP metadata and NPY payloads."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            if not isinstance(key, str) or not key:
                raise ValueError("NPZ keys must be nonempty strings")
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, _npy_bytes(array))


def _sha256_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _source_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _relative_tree_paths(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _unit_relative_path(evaluation_unit: str) -> Path:
    """Map the logical IID/LOCO unit name to its fixed formal tree location."""
    evaluation_unit = _require_name("evaluation_unit", evaluation_unit)
    if evaluation_unit == "iid":
        return Path("iid")
    return Path("loco") / evaluation_unit


def _unit_output_paths(destination: Path) -> dict[str, Path]:
    return {
        "full_test_material_ids.npy": destination / "full_test_material_ids.npy",
        "paired_test_material_ids.npy": destination / "paired_test_material_ids.npy",
        "paired_test_dos_rows.npy": destination / "paired_test_dos_rows.npy",
        "evaluation_missingness_masks.npz": destination / "evaluation_missingness_masks.npz",
        "shuffled_dos_source_rows.npy": destination / "shuffled_dos_source_rows.npy",
        "train_material_ids.npy": destination / "train_material_ids.npy",
        "low_label_masks.npz": destination / "low_label_masks.npz",
    }


def write_control_unit(unit: ControlUnit, destination: Path | str) -> dict[str, str]:
    """Serialize one already-built control unit without nondeterministic metadata."""
    if not isinstance(unit, ControlUnit):
        raise TypeError("unit must be a ControlUnit")
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"unit destination already exists: {destination_path}")
    destination_path.mkdir(parents=False)
    paths = _unit_output_paths(destination_path)
    _write_npy(paths["full_test_material_ids.npy"], unit.full_test_ids)
    _write_npy(paths["paired_test_material_ids.npy"], unit.paired_test_ids)
    _write_npy(paths["paired_test_dos_rows.npy"], unit.paired_test_dos_rows)
    _write_deterministic_npz(paths["evaluation_missingness_masks.npz"], unit.missingness_masks)
    _write_npy(paths["shuffled_dos_source_rows.npy"], unit.shuffled_rows)
    _write_npy(paths["train_material_ids.npy"], unit.train_ids)
    _write_deterministic_npz(paths["low_label_masks.npz"], unit.low_label_masks)
    hashes = {name: _source_sha256(path) for name, path in paths.items()}
    summary = dict(unit.summary)
    summary["output_file_sha256"] = hashes
    (destination_path / "control_summary.json").write_bytes(_canonical_json_bytes(summary))
    return {name: _source_sha256(destination_path / name) for name in sorted(UNIT_FILES)}


def _assert_array_matches(path: Path, expected: np.ndarray) -> None:
    actual = np.load(path, allow_pickle=False)
    if not isinstance(actual, np.ndarray) or actual.dtype.hasobject:
        raise ValueError(f"serialized array {path.name} is not a non-object array")
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"serialized array {path.name} does not match its control unit")


def _assert_npz_matches(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != list(expected):
            raise ValueError(f"serialized NPZ {path.name} has wrong key order")
        for key, expected_array in expected.items():
            actual = archive[key]
            if actual.dtype.hasobject or actual.dtype != expected_array.dtype or actual.shape != expected_array.shape or not np.array_equal(actual, expected_array):
                raise ValueError(f"serialized NPZ {path.name} member {key} does not match its control unit")


def _validate_written_unit(destination: Path, unit: ControlUnit) -> dict[str, str]:
    if not destination.is_dir() or {path.name for path in destination.iterdir()} != UNIT_FILES:
        raise ValueError(f"serialized unit {unit.evaluation_unit} has an invalid file inventory")
    _assert_array_matches(destination / "full_test_material_ids.npy", unit.full_test_ids)
    _assert_array_matches(destination / "paired_test_material_ids.npy", unit.paired_test_ids)
    _assert_array_matches(destination / "paired_test_dos_rows.npy", unit.paired_test_dos_rows)
    _assert_npz_matches(destination / "evaluation_missingness_masks.npz", unit.missingness_masks)
    _assert_array_matches(destination / "shuffled_dos_source_rows.npy", unit.shuffled_rows)
    _assert_array_matches(destination / "train_material_ids.npy", unit.train_ids)
    _assert_npz_matches(destination / "low_label_masks.npz", unit.low_label_masks)
    hashes = {name: _source_sha256(destination / name) for name in sorted(UNIT_FILES)}
    try:
        summary = json.loads((destination / "control_summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"serialized unit {unit.evaluation_unit} has invalid JSON") from exc
    expected_hashes = {name: hashes[name] for name in hashes if name != "control_summary.json"}
    expected_summary = dict(unit.summary)
    expected_summary["output_file_sha256"] = expected_hashes
    if summary != expected_summary:
        raise ValueError(f"serialized unit {unit.evaluation_unit} summary does not match")
    return hashes


def _resolved_directory(path: Path | str, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory: {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _cleanup_staging(staging: Path, output_parent: Path) -> None:
    resolved_staging = staging.resolve()
    if resolved_staging.parent != output_parent:
        raise RuntimeError("refusing to clean staging outside the resolved output parent")
    if resolved_staging.exists():
        shutil.rmtree(resolved_staging)


def _metadata_payload(
    inputs: ControlInputs,
    units: Mapping[str, ControlUnit],
    unit_hashes: Mapping[str, Mapping[str, str]],
    spec: ControlSpec,
    diagnostic_smoke: bool,
) -> dict[str, Any]:
    unit_names = list(units)
    return {
        "schema_version": spec.spec_version,
        "spec": {
            "base_seed": spec.base_seed,
            "missing_rates": list(spec.missing_rates),
            "label_fractions": list(spec.label_fractions),
            "repeats": spec.repeats,
        },
        "rng": {"bit_generator": "PCG64", "seed_derivation": "sha256-namespace-v1"},
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
        "diagnostic_smoke": bool(diagnostic_smoke),
        "source_sha256": dict(inputs.source_sha256),
        "evaluation_units": unit_names,
        "evaluation_unit_count": len(unit_names),
        "unit_counts": {name: dict(units[name].summary["counts"]) for name in unit_names},
        "unit_file_sha256": {
            f"{_unit_relative_path(name).as_posix()}/{filename}": digest
            for name in unit_names
            for filename, digest in unit_hashes[name].items()
        },
    }


def _validate_staging(
    staging: Path,
    inputs: ControlInputs,
    units: Mapping[str, ControlUnit],
    spec: ControlSpec,
    diagnostic_smoke: bool,
) -> None:
    unit_hashes = {
        name: _validate_written_unit(staging / _unit_relative_path(name), unit)
        for name, unit in units.items()
    }
    expected_metadata = _metadata_payload(inputs, units, unit_hashes, spec, diagnostic_smoke)
    metadata_path = staging / "controls_metadata.json"
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("controls metadata is not valid JSON") from exc
    if actual_metadata != expected_metadata:
        raise ValueError("controls metadata does not match staged controls")
    expected_inventory = {"controls_metadata.json"}
    expected_inventory.update(
        f"{_unit_relative_path(name).as_posix()}/{filename}"
        for name in units
        for filename in UNIT_FILES
    )
    if set(_sha256_manifest(staging)) != expected_inventory:
        raise ValueError("staging contains temporary or unexpected files")
    expected_paths = set(expected_inventory) | {"loco"} | {
        _unit_relative_path(name).as_posix() for name in units
    }
    if _relative_tree_paths(staging) != expected_paths:
        raise ValueError("staging contains temporary or unexpected directories")


def materialize_experiment_controls(
    master_dir: Path | str,
    dos_dir: Path | str,
    split_dir: Path | str,
    output_dir: Path | str,
    *,
    spec: ControlSpec = ControlSpec(),
    diagnostic_smoke: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish IID plus every reviewed LOCO control unit."""
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec must be a ControlSpec")
    if not isinstance(diagnostic_smoke, bool):
        raise TypeError("diagnostic_smoke must be a bool")
    master_root = _resolved_directory(master_dir, "master_dir")
    dos_root = _resolved_directory(dos_dir, "dos_dir")
    split_root = _resolved_directory(split_dir, "split_dir")
    resolved_output = Path(output_dir).resolve()
    output_parent = resolved_output.parent
    if not output_parent.is_dir():
        raise ValueError(f"output parent must be an existing directory: {output_parent}")
    if any(_is_within(resolved_output, source) for source in (master_root, dos_root, split_root)):
        raise ValueError("output_dir must not equal or be inside a source directory")

    staging = output_parent / f".{resolved_output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    staging = staging.resolve()
    if staging.parent != output_parent:
        raise RuntimeError("staging path is not directly under the resolved output parent")
    published = False
    try:
        inputs = load_control_inputs(master_root, dos_root, split_root)
        expected_loco = [f"loco_{index:02d}" for index in range(inputs.loco_fold_count)]
        actual_loco = sorted(inputs.loco["fold"].unique().tolist())
        if actual_loco != expected_loco:
            raise ValueError(
                "LOCO source folds must be contiguous from loco_00 through "
                f"loco_{inputs.loco_fold_count - 1:02d}"
            )
        units: dict[str, ControlUnit] = {
            "iid": build_control_unit(inputs.iid, inputs, spec, evaluation_unit="iid")
        }
        for fold in expected_loco:
            units[fold] = build_control_unit(
                inputs.loco[inputs.loco["fold"] == fold], inputs, spec, evaluation_unit=fold
            )
        (staging / "loco").mkdir()
        for name, unit in units.items():
            write_control_unit(unit, staging / _unit_relative_path(name))
        unit_hashes = {
            name: _validate_written_unit(staging / _unit_relative_path(name), unit)
            for name, unit in units.items()
        }
        metadata = _metadata_payload(inputs, units, unit_hashes, spec, diagnostic_smoke)
        (staging / "controls_metadata.json").write_bytes(_canonical_json_bytes(metadata))
        _validate_staging(staging, inputs, units, spec, diagnostic_smoke)

        if resolved_output.exists():
            if (
                resolved_output.is_dir()
                and _relative_tree_paths(resolved_output) == _relative_tree_paths(staging)
                and _sha256_manifest(resolved_output) == _sha256_manifest(staging)
            ):
                _cleanup_staging(staging, output_parent)
                return {
                    "status": "complete", "complete": True,
                    "evaluation_unit_count": 1 + inputs.loco_fold_count,
                    "output_dir": str(resolved_output), "reused": True,
                    "source_sha256": dict(inputs.source_sha256),
                }
            raise FileExistsError(f"formal output already exists with different content: {resolved_output}")
        staging.rename(resolved_output)
        published = True
        return {
            "status": "complete", "complete": True,
            "evaluation_unit_count": 1 + inputs.loco_fold_count,
            "output_dir": str(resolved_output), "reused": False,
            "source_sha256": dict(inputs.source_sha256),
        }
    finally:
        if not published and staging.exists():
            _cleanup_staging(staging, output_parent)
