"""Leakage-safe, framework-neutral linear diagnostics for Crystal+DOS."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd


TARGET_NAMES = ("k_vrh_gpa", "g_vrh_gpa")
MODEL_NAMES = (
    "soap_ridge_all",
    "soap_ridge_paired",
    "dos_ridge_paired",
    "soap_dos_ridge_paired",
)
DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def _require_complete(metadata: Mapping[str, Any], path: Path, schema: str) -> None:
    if metadata.get("status") != "complete" or metadata.get("complete") is not True:
        raise ValueError(f"metadata is not complete: {path}")
    if metadata.get("schema_version") != schema:
        raise ValueError(f"unexpected schema_version in {path}")


def _verify_recorded_hash(path: Path, metadata: Mapping[str, Any], key: str) -> str:
    recorded = metadata.get("output_file_sha256")
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get(key), str):
        raise ValueError(f"missing recorded SHA-256 for {key} in metadata")
    actual = _sha256(path)
    if actual != recorded[key]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return actual


def _load_unicode_vector(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1 or value.dtype.kind != "U":
        raise ValueError(f"expected one-dimensional Unicode IDs: {path}")
    if len(set(value.tolist())) != len(value):
        raise ValueError(f"duplicate material IDs: {path}")
    return value


def _unit_path(root: Path, evaluation_unit: str) -> Path:
    if evaluation_unit == "iid":
        return root / "iid"
    if (
        evaluation_unit.startswith("loco_")
        and len(evaluation_unit) == 7
        and evaluation_unit[5:].isdigit()
    ):
        return root / "loco" / evaluation_unit
    raise ValueError("evaluation_unit must be iid or a two-digit loco_NN label")


@dataclass(frozen=True)
class DiagnosticData:
    """Validated arrays and fixed memberships for one evaluation unit."""

    evaluation_unit: str
    material_ids: np.ndarray
    targets: np.ndarray
    soap_features: np.ndarray
    dos_features: np.ndarray
    dos_rows: np.ndarray
    observed: np.ndarray
    partitions: Mapping[str, np.ndarray]
    paired_partitions: Mapping[str, np.ndarray]
    target_names: tuple[str, str]
    source_sha256: Mapping[str, str]

    def _positions(self, split: str, paired: bool) -> np.ndarray:
        source = self.paired_partitions if paired else self.partitions
        if split not in source:
            raise ValueError("split must be train, validation, or test")
        return source[split]

    def ids(self, split: str, *, paired: bool = False) -> np.ndarray:
        return self.material_ids[self._positions(split, paired)].copy()

    def targets_for(self, split: str, *, paired: bool = False) -> np.ndarray:
        return self.targets[self._positions(split, paired)].copy()

    def features(self, model: str, split: str) -> np.ndarray:
        if model not in MODEL_NAMES:
            raise ValueError(f"unknown diagnostic model: {model}")
        paired = model != "soap_ridge_all"
        positions = self._positions(split, paired)
        soap = np.asarray(self.soap_features[positions], dtype=np.float64)
        if model in {"soap_ridge_all", "soap_ridge_paired"}:
            return soap
        rows = self.dos_rows[positions]
        if np.any(rows < 0):
            raise ValueError(f"{model} received a material without observed DOS")
        dos = np.asarray(self.dos_features[rows], dtype=np.float64).reshape(len(rows), -1)
        return dos if model == "dos_ridge_paired" else np.concatenate([soap, dos], axis=1)


def load_diagnostic_data(
    master_dir: Path | str,
    crystal_cache_dir: Path | str,
    dos_dir: Path | str,
    split_dir: Path | str,
    controls_dir: Path | str,
    *,
    evaluation_unit: str,
    soap_dir: Path | str | None = None,
) -> DiagnosticData:
    """Load one fixed split and fail closed on any row or control misalignment."""
    master_root = Path(master_dir)
    cache_root = Path(crystal_cache_dir)
    dos_root = Path(dos_dir)
    split_root = Path(split_dir)
    soap_root = Path(soap_dir) if soap_dir is not None else split_root
    controls_root = Path(controls_dir)

    master_meta = _read_json(master_root / "master_metadata.json")
    cache_meta = _read_json(cache_root / "cache_metadata.json")
    dos_meta_path = (
        dos_root / "quality_summary.json"
        if (dos_root / "quality_summary.json").is_file()
        else dos_root / "preprocessing_config.json"
    )
    dos_meta = _read_json(dos_meta_path)
    split_meta = _read_json(split_root / "split_metadata.json")
    soap_meta = (
        split_meta
        if soap_root.resolve() == split_root.resolve()
        else _read_json(soap_root / "split_metadata.json")
    )
    controls_meta = _read_json(controls_root / "controls_metadata.json")
    if (
        master_meta.get("complete") is not True
        or master_meta.get("schema_version") not in {
            "crystal_dos_master_v1", "aflow_confirmatory_master_v1"
        }
    ):
        raise ValueError("master metadata is incomplete or unsupported")
    if (
        cache_meta.get("complete") is not True
        or cache_meta.get("schema_version") not in {
            "crystal_cache_v1", "aflow_confirmatory_crystal_cache_v1"
        }
    ):
        raise ValueError("crystal cache metadata is incomplete or unsupported")
    if (
        split_meta.get("complete") is not True
        or split_meta.get("schema_version") not in {
            "experiment_splits_v1", "aflow_confirmatory_split_amendment_v2"
        }
    ):
        raise ValueError("split metadata is incomplete or unsupported")
    if soap_root.resolve() != split_root.resolve():
        soap_metadata_path = soap_root / "split_metadata.json"
        if (
            soap_meta.get("schema_version") != "experiment_splits_v1"
            or soap_meta.get("row_count") != split_meta.get("row_count")
            or split_meta.get("source_sha256", {}).get("split_metadata.json")
            != _sha256(soap_metadata_path)
        ):
            raise ValueError("amended split is not bound to the SOAP embedding source")
    if dos_meta.get("complete") is not True and dos_meta.get("status") != "complete":
        raise ValueError("DOS metadata is not complete")
    if controls_meta.get("schema_version") != "experiment_controls_v1":
        raise ValueError("unexpected controls schema_version")
    units = controls_meta.get("evaluation_units")
    if not isinstance(units, list) or evaluation_unit not in units:
        raise ValueError(f"controls metadata does not contain {evaluation_unit}")
    unit_root = _unit_path(controls_root, evaluation_unit)

    master_path = master_root / "master_table.parquet"
    cache_ids_path = cache_root / "material_ids.npy"
    dos_ids_path = dos_root / "material_ids.npy"
    dos_features_path = dos_root / "dos_features.npy"
    soap_path = soap_root / "soap_svd_256.npy"
    split_path = split_root / (
        "iid_split.parquet" if evaluation_unit == "iid" else "soap_loco_folds.parquet"
    )
    source_hashes = {
        "master/master_table.parquet": _verify_recorded_hash(
            master_path, master_meta, "master_table.parquet"
        ),
        "crystal_cache/material_ids.npy": _verify_recorded_hash(
            cache_ids_path, cache_meta, "material_ids.npy"
        ),
        "dos/material_ids.npy": _verify_recorded_hash(
            dos_ids_path, dos_meta, "material_ids.npy"
        ),
        "dos/dos_features.npy": _verify_recorded_hash(
            dos_features_path, dos_meta, "dos_features.npy"
        ),
        "split/soap_svd_256.npy": _verify_recorded_hash(
            soap_path, soap_meta, "soap_svd_256.npy"
        ),
        f"split/{split_path.name}": _verify_recorded_hash(
            split_path, split_meta, split_path.name
        ),
    }

    master = pd.read_parquet(master_path)
    required = {
        "material_id", "crystal_row", "dos_row", "dos_observed", *TARGET_NAMES
    }
    if not required <= set(master.columns):
        raise ValueError(f"master table missing columns: {sorted(required - set(master.columns))}")
    ids = master["material_id"].astype(str).to_numpy()
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("master material IDs must be nonempty and unique")
    targets = master[list(TARGET_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("K/G targets must be finite")
    crystal_rows = pd.to_numeric(master["crystal_row"], errors="raise").to_numpy(dtype=np.int64)
    cache_ids = _load_unicode_vector(cache_ids_path)
    if len(cache_ids) != len(ids) or np.any(crystal_rows < 0) or np.any(crystal_rows >= len(cache_ids)):
        raise ValueError("crystal cache rows do not cover master material IDs")
    if not np.array_equal(cache_ids[crystal_rows], ids):
        raise ValueError("crystal cache material ID mapping disagrees with master crystal_row")
    cache_hash = source_hashes["crystal_cache/material_ids.npy"]
    recorded_soap_ids = soap_meta.get("source_sha256", {})
    if not isinstance(recorded_soap_ids, Mapping) or recorded_soap_ids.get("soap_material_ids") != cache_hash:
        raise ValueError("SOAP source material IDs do not match the crystal cache")

    soap = np.load(soap_path, mmap_mode="r", allow_pickle=False)
    if soap.ndim != 2 or soap.shape[0] != len(ids) or soap.dtype.kind != "f":
        raise ValueError("SOAP features must be a floating matrix aligned to master rows")
    dos_ids = _load_unicode_vector(dos_ids_path)
    dos_features = np.load(dos_features_path, mmap_mode="r", allow_pickle=False)
    if dos_features.ndim != 3 or dos_features.shape[0] != len(dos_ids) or dos_features.shape[1] != 1:
        raise ValueError("DOS features must have shape (N_observed, 1, points)")
    expected_shape = dos_meta.get("feature_shape")
    if expected_shape is None and {
        "row_count", "point_count"
    } <= set(dos_meta):
        expected_shape = [int(dos_meta["row_count"]), 1, int(dos_meta["point_count"])]
    if expected_shape != list(dos_features.shape):
        raise ValueError("DOS feature shape disagrees with quality summary")
    observed = master["dos_observed"].astype(bool).to_numpy()
    numeric_rows = pd.to_numeric(master["dos_row"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if np.any(observed & ~np.isfinite(numeric_rows)) or np.any(~observed & np.isfinite(numeric_rows)):
        raise ValueError("dos_observed and dos_row nullability disagree")
    dos_rows = np.full(len(ids), -1, dtype=np.int64)
    dos_rows[observed] = numeric_rows[observed].astype(np.int64)
    if np.any(dos_rows[observed] < 0) or np.any(dos_rows[observed] >= len(dos_ids)):
        raise ValueError("observed DOS row is out of bounds")
    if not np.array_equal(dos_ids[dos_rows[observed]], ids[observed]):
        raise ValueError("DOS material IDs disagree with master dos_row")

    split_frame = pd.read_parquet(split_path)
    if evaluation_unit != "iid":
        if "fold" not in split_frame:
            raise ValueError("LOCO split is missing fold")
        split_frame = split_frame.loc[split_frame["fold"].astype(str) == evaluation_unit].copy()
    split_required = {"material_id", "split", "stable_order_key", "crystal_row", "dos_row"}
    if not split_required <= set(split_frame.columns):
        raise ValueError("split file is missing required columns")
    split_frame["material_id"] = split_frame["material_id"].astype(str)
    if len(split_frame) != len(ids) or split_frame["material_id"].duplicated().any() or set(split_frame["material_id"]) != set(ids):
        raise ValueError("fixed split must contain each master material exactly once")
    labels = set(split_frame["split"].astype(str))
    if labels != {"train", "validation", "test"}:
        raise ValueError("fixed split must contain train, validation, and test only")
    master_index = {value: index for index, value in enumerate(ids)}
    mapped_rows = split_frame["material_id"].map(master_index).to_numpy(dtype=np.int64)
    if not np.array_equal(
        pd.to_numeric(split_frame["crystal_row"], errors="raise").to_numpy(dtype=np.int64),
        crystal_rows[mapped_rows],
    ):
        raise ValueError("split crystal_row mapping disagrees with master")
    split_dos_rows = pd.to_numeric(split_frame["dos_row"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not np.array_equal(split_dos_rows, numeric_rows[mapped_rows], equal_nan=True):
        raise ValueError("split dos_row mapping disagrees with master")
    split_frame = split_frame.sort_values(
        ["stable_order_key", "material_id"], kind="mergesort"
    )
    partitions: dict[str, np.ndarray] = {}
    paired_partitions: dict[str, np.ndarray] = {}
    for label in ("train", "validation", "test"):
        label_ids = split_frame.loc[split_frame["split"] == label, "material_id"]
        positions = np.asarray([master_index[value] for value in label_ids], dtype=np.int64)
        if len(positions) == 0:
            raise ValueError(f"{evaluation_unit} has empty {label} partition")
        partitions[label] = positions
        paired_partitions[label] = positions[observed[positions]]
        if len(paired_partitions[label]) == 0:
            raise ValueError(f"{evaluation_unit} has empty paired {label} partition")

    control_hashes = controls_meta.get("unit_file_sha256")
    if not isinstance(control_hashes, Mapping):
        raise ValueError("controls metadata is missing unit_file_sha256")
    prefix = "iid" if evaluation_unit == "iid" else f"loco/{evaluation_unit}"
    control_names = (
        "train_material_ids.npy", "full_test_material_ids.npy",
        "paired_test_material_ids.npy", "paired_test_dos_rows.npy",
    )
    controls: dict[str, np.ndarray] = {}
    for name in control_names:
        path = unit_root / name
        key = f"{prefix}/{name}"
        if not isinstance(control_hashes.get(key), str) or _sha256(path) != control_hashes[key]:
            raise ValueError(f"control SHA-256 mismatch: {key}")
        controls[name] = np.load(path, allow_pickle=False)
    expected_train = ids[partitions["train"]]
    expected_test = ids[partitions["test"]]
    expected_paired = ids[paired_partitions["test"]]
    expected_paired_rows = dos_rows[paired_partitions["test"]]
    if not np.array_equal(controls["train_material_ids.npy"], expected_train):
        raise ValueError("control train membership differs from the fixed split")
    if not np.array_equal(controls["full_test_material_ids.npy"], expected_test):
        raise ValueError("control full-test membership differs from the fixed split")
    if not np.array_equal(controls["paired_test_material_ids.npy"], expected_paired):
        raise ValueError("paired control membership must equal observed test materials")
    if not np.array_equal(controls["paired_test_dos_rows.npy"], expected_paired_rows):
        raise ValueError("paired control DOS rows disagree with master")

    return DiagnosticData(
        evaluation_unit=evaluation_unit,
        material_ids=ids,
        targets=targets,
        soap_features=soap,
        dos_features=dos_features,
        dos_rows=dos_rows,
        observed=observed,
        partitions=partitions,
        paired_partitions=paired_partitions,
        target_names=TARGET_NAMES,
        source_sha256=source_hashes,
    )


@dataclass(frozen=True)
class RidgeFit:
    selected_alpha: float
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    validation_score: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("prediction features have the wrong shape")
        standardized = (values - self.feature_mean) / self.feature_scale
        return (standardized @ self.weights) * self.target_scale + self.target_mean


def _finite_matrix(value: np.ndarray, name: str, rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{name} row count differs")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def select_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> RidgeFit:
    """Fit train-only scalers and select alpha using validation standardized MAE."""
    train_x = _finite_matrix(x_train, "x_train")
    train_y = _finite_matrix(y_train, "y_train", len(train_x))
    validation_x = _finite_matrix(x_validation, "x_validation")
    validation_y = _finite_matrix(y_validation, "y_validation", len(validation_x))
    if validation_x.shape[1] != train_x.shape[1] or validation_y.shape[1] != train_y.shape[1]:
        raise ValueError("train and validation dimensions differ")
    candidates = tuple(float(value) for value in alphas)
    if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("alphas must be nonempty, finite, and positive")

    feature_mean = train_x.mean(axis=0)
    feature_scale = train_x.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    target_mean = train_y.mean(axis=0)
    target_scale = train_y.std(axis=0)
    target_scale[target_scale == 0] = 1.0
    normalized_x = (train_x - feature_mean) / feature_scale
    normalized_y = (train_y - target_mean) / target_scale
    validation_xn = (validation_x - feature_mean) / feature_scale
    validation_yn = (validation_y - target_mean) / target_scale
    gram = normalized_x.T @ normalized_x / len(normalized_x)
    rhs = normalized_x.T @ normalized_y / len(normalized_x)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ rhs

    best: tuple[float, float, np.ndarray] | None = None
    for alpha in candidates:
        weights = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        score = float(np.mean(np.abs(validation_xn @ weights - validation_yn)))
        candidate = (score, alpha, weights)
        if best is None or (score, alpha) < (best[0], best[1]):
            best = candidate
    assert best is not None
    return RidgeFit(
        selected_alpha=best[1], weights=best[2], feature_mean=feature_mean,
        feature_scale=feature_scale, target_mean=target_mean,
        target_scale=target_scale, validation_score=best[0],
    )


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    actual = _finite_matrix(truth, "truth")
    predicted = _finite_matrix(prediction, "prediction", len(actual))
    if actual.shape != predicted.shape:
        raise ValueError("truth and prediction shapes differ")
    residual = predicted - actual
    mae = np.mean(np.abs(residual), axis=0)
    mse = np.mean(residual ** 2, axis=0)
    centered = actual - actual.mean(axis=0)
    denominator = np.sum(centered ** 2, axis=0)
    numerator = np.sum(residual ** 2, axis=0)
    variable = denominator > 0
    r2 = np.empty_like(denominator)
    r2[variable] = 1.0 - numerator[variable] / denominator[variable]
    r2[~variable] = np.where(numerator[~variable] == 0, 1.0, 0.0)
    return {"mae": mae, "rmse": np.sqrt(mse), "r2": r2}


def run_diagnostic_models(
    data: DiagnosticData,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """Run four leakage-safe diagnostics for one fixed evaluation unit."""
    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        paired = model != "soap_ridge_all"
        train_x = data.features(model, "train")
        validation_x = data.features(model, "validation")
        train_y = data.targets_for("train", paired=paired)
        validation_y = data.targets_for("validation", paired=paired)
        fitted = select_ridge_alpha(
            train_x, train_y, validation_x, validation_y, alphas=alphas
        )
        sample_counts = {
            split: int(len(data.ids(split, paired=paired)))
            for split in ("train", "validation", "test")
        }
        models[model] = {
            "cohort": "paired" if paired else "all",
            "selected_alpha": fitted.selected_alpha,
            "validation_standardized_mae": fitted.validation_score,
            "sample_counts": sample_counts,
            "feature_count": int(train_x.shape[1]),
        }
        for split in ("validation", "test"):
            truth = data.targets_for(split, paired=paired)
            metrics = regression_metrics(truth, fitted.predict(data.features(model, split)))
            for target_index, target in enumerate(data.target_names):
                for metric, values in metrics.items():
                    rows.append(
                        {
                            "model": model,
                            "cohort": "paired" if paired else "all",
                            "split": split,
                            "target": target,
                            "metric": metric,
                            "value": float(values[target_index]),
                            "n_samples": sample_counts[split],
                            "selected_alpha": fitted.selected_alpha,
                        }
                    )
    return {
        "schema_version": "diagnostic_baselines_v1",
        "evaluation_unit": data.evaluation_unit,
        "target_names": list(data.target_names),
        "models": models,
        "metrics": rows,
        "source_sha256": dict(data.source_sha256),
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _metrics_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "model", "cohort", "split", "target", "metric", "value",
        "n_samples", "selected_alpha",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def render_diagnostic_report(result: Mapping[str, Any]) -> str:
    models = result.get("models", {})
    metrics = result.get("metrics", [])
    lookup = {
        (row["model"], row["split"], row["target"], row["metric"]): row["value"]
        for row in metrics
    }
    lines = [
        "# Crystal+DOS 线性诊断基线报告",
        "",
        f"总体状态：**{'完成' if result.get('complete', True) else '未完成'}**。",
        "",
        f"评估单元：`{result.get('evaluation_unit', 'unknown')}`。",
        "",
        "| 模型 | 总体 | 训练/验证/测试 | alpha | Test K MAE (GPa) | Test G MAE (GPa) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        item = models.get(name, {})
        counts = item.get("sample_counts", {})
        lines.append(
            f"| `{name}` | {item.get('cohort', '')} | "
            f"{counts.get('train', 0)}/{counts.get('validation', 0)}/{counts.get('test', 0)} | "
            f"{item.get('selected_alpha', '')} | "
            f"{lookup.get((name, 'test', 'k_vrh_gpa', 'mae'), float('nan')):.6g} | "
            f"{lookup.get((name, 'test', 'g_vrh_gpa', 'mae'), float('nan')):.6g} |"
        )
    lines.extend(
        [
            "",
            "结论边界：这是固定划分与数据对齐的**诊断结果，非论文最终基线**；不能替代 PotNet、CGCNN、ALIGNN、DOS 1D CNN 或特权信息蒸馏模型。",
            "",
            "未进行深度模型训练；未据此声称准确性、鲁棒性或泛化提升。",
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_diagnostic_results(
    result: Mapping[str, Any], output_dir: Path | str
) -> dict[str, Any]:
    """Atomically publish deterministic diagnostic metrics and reuse exact outputs."""
    destination = Path(output_dir).resolve()
    metrics_bytes = _metrics_csv_bytes(result["metrics"])
    report_bytes = render_diagnostic_report(result).encode("utf-8")
    summary = {
        "schema_version": "diagnostic_baselines_v1",
        "status": "complete",
        "complete": True,
        "evaluation_unit": result["evaluation_unit"],
        "target_names": result["target_names"],
        "models": result["models"],
        "metrics_row_count": len(result["metrics"]),
        "source_sha256": result["source_sha256"],
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
        "output_file_sha256": {
            "metrics.csv": hashlib.sha256(metrics_bytes).hexdigest(),
            "quality_report.md": hashlib.sha256(report_bytes).hexdigest(),
        },
    }
    payloads = {
        "metrics.csv": metrics_bytes,
        "quality_report.md": report_bytes,
        "run_summary.json": _canonical_json_bytes(summary),
    }
    if destination.exists():
        if not destination.is_dir() or set(path.name for path in destination.iterdir()) != set(payloads):
            raise FileExistsError(f"existing diagnostic output has a different inventory: {destination}")
        if any((destination / name).read_bytes() != payload for name, payload in payloads.items()):
            raise FileExistsError(f"existing diagnostic output differs: {destination}")
        return {"status": "complete", "complete": True, "evaluation_unit": result["evaluation_unit"],
                "model_count": len(result["models"]), "output_dir": str(destination), "reused": True}

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        for name, payload in payloads.items():
            with (staging / name).open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"status": "complete", "complete": True, "evaluation_unit": result["evaluation_unit"],
            "model_count": len(result["models"]), "output_dir": str(destination), "reused": False}
