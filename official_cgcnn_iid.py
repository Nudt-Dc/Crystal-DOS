"""Official-CGCNN IID training on the audited periodic graph cache."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import nn

from crystal_cgcnn_smoke import CrystalGraphCache


OFFICIAL_COMMIT = "f42ab233c4ee0c416879d6bc2d22a264418413ad"
OFFICIAL_MODEL_SHA256 = "0a773283d83fae13d274cacbf67153adcf3a4875cb64075b558fbdd6e998198f"
OFFICIAL_ATOM_INIT_SHA256 = "cb90bd2257dbf8cfc4e6b2b6d4348d4616e8e9680f1339b1461137032e0f1894"
TARGET_NAMES = ("k_vrh_gpa", "g_vrh_gpa")
OUTPUT_INVENTORY = {
    "epoch_history.csv", "test_predictions.csv", "model_k.pt", "model_g.pt",
    "quality_report.md", "run_summary.json",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def signature_sha256(signature: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(signature)).hexdigest()


def build_atom_feature_lookup(path: Path | str) -> tuple[np.ndarray, dict[str, int]]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("atom initialization JSON must be a nonempty object")
    parsed = {int(key): np.asarray(features, dtype=np.float32) for key, features in value.items()}
    feature_counts = {array.shape for array in parsed.values()}
    if set(parsed) != set(range(1, 101)) or feature_counts != {(92,)}:
        raise ValueError("official atom initialization contract must be elements 1-100 with 92 features")
    lookup = np.zeros((101, 92), dtype=np.float32)
    for atomic_number, features in parsed.items():
        lookup[atomic_number] = features
    if not np.isfinite(lookup[1:]).all():
        raise ValueError("atom initialization contains non-finite values")
    return lookup, {"element_count": 100, "feature_count": 92}


def load_official_cgcnn_class(path: Path | str):
    source = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"cgcnn_official_model_{sha256_file(source)[:16]}", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official CGCNN model: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, "CrystalGraphConvNet", None)
    if model_class is None:
        raise ImportError("official model file lacks CrystalGraphConvNet")
    return model_class


@dataclass(frozen=True)
class OfficialGraphBatch:
    atom_features: torch.Tensor
    neighbor_features: torch.Tensor
    neighbor_indices: torch.Tensor
    crystal_atom_indices: list[torch.Tensor]
    targets: torch.Tensor


def batch_official_graphs(
    cache: CrystalGraphCache,
    positions: np.ndarray,
    targets: np.ndarray,
    atom_lookup: np.ndarray,
    *,
    target_index: int,
    device: str | torch.device,
) -> OfficialGraphBatch:
    indices = np.asarray(positions)
    target_matrix = np.asarray(targets, dtype=np.float32)
    lookup = np.asarray(atom_lookup, dtype=np.float32)
    if (
        indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 1
        or np.any(indices < 0) or np.any(indices >= len(cache))
        or len(set(indices.tolist())) != len(indices)
    ):
        raise ValueError("positions must be unique in-range integer indices")
    if target_matrix.shape != (len(cache), 2) or not np.isfinite(target_matrix).all():
        raise ValueError("targets must be a finite [material_count, 2] matrix")
    if target_index not in (0, 1) or lookup.ndim != 2 or lookup.shape[0] < 101:
        raise ValueError("target index or atom lookup is invalid")
    arrays = cache.arrays
    site_offsets = arrays["site_offsets.npy"]
    edge_offsets = arrays["graph_edge_offsets.npy"]
    atom_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []
    distance_blocks: list[np.ndarray] = []
    count_blocks: list[np.ndarray] = []
    crystal_indices: list[np.ndarray] = []
    base = 0
    for position in indices.tolist():
        node_start, node_stop = int(site_offsets[position]), int(site_offsets[position + 1])
        edge_start, edge_stop = int(edge_offsets[position]), int(edge_offsets[position + 1])
        atomic_numbers = np.asarray(arrays["atomic_numbers.npy"][node_start:node_stop], dtype=np.int64)
        if np.any(atomic_numbers >= len(lookup)):
            raise ValueError("atomic number is not covered by the official initialization")
        local_edges = np.asarray(arrays["edge_index.npy"][:, edge_start:edge_stop], dtype=np.int64) - node_start
        counts = np.asarray(arrays["neighbor_counts.npy"][node_start:node_stop], dtype=np.int64)
        expected_targets = np.repeat(np.arange(node_stop - node_start, dtype=np.int64), counts)
        if not np.array_equal(local_edges[1], expected_targets):
            raise ValueError("cache edges are not grouped by center atom")
        atom_blocks.append(lookup[atomic_numbers])
        source_blocks.append(local_edges[0] + base)
        distance_blocks.append(np.asarray(arrays["edge_distances.npy"][edge_start:edge_stop]))
        count_blocks.append(counts)
        crystal_indices.append(np.arange(base, base + node_stop - node_start, dtype=np.int64))
        base += node_stop - node_start
    counts_all = np.concatenate(count_blocks)
    unique_counts = np.unique(counts_all)
    if len(unique_counts) != 1 or unique_counts[0] < 1:
        raise ValueError("official dense CGCNN batch requires one positive neighbor count for every atom")
    neighbor_count = int(unique_counts[0])
    sources = np.concatenate(source_blocks).reshape(base, neighbor_count)
    distances = np.concatenate(distance_blocks).reshape(base, neighbor_count)
    gaussian_centers = np.arange(0.0, 8.0 + 0.2, 0.2)
    neighbor_features = np.exp(-((distances[..., None] - gaussian_centers) ** 2) / (0.2 ** 2))
    selected_device = torch.device(device)
    return OfficialGraphBatch(
        atom_features=torch.as_tensor(
            np.concatenate(atom_blocks), dtype=torch.float32, device=selected_device
        ),
        neighbor_features=torch.as_tensor(
            neighbor_features, dtype=torch.float32, device=selected_device
        ),
        neighbor_indices=torch.as_tensor(sources, dtype=torch.long, device=selected_device),
        crystal_atom_indices=[
            torch.as_tensor(values, dtype=torch.long, device=selected_device)
            for values in crystal_indices
        ],
        targets=torch.as_tensor(
            target_matrix[indices, target_index, None], dtype=torch.float32, device=selected_device
        ),
    )


def regression_metrics_1d(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(truth, dtype=np.float64).reshape(-1)
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or len(actual) < 1 or not (
        np.isfinite(actual).all() and np.isfinite(predicted).all()
    ):
        raise ValueError("truth and prediction must be equal finite nonempty vectors")
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    numerator = float(np.sum(residual ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - numerator / denominator) if denominator > 0 else (1.0 if numerator == 0 else 0.0),
    }


def _position_batches(
    positions: np.ndarray, batch_size: int, *, seed: int | None = None,
    avoid_singleton: bool = False,
) -> list[np.ndarray]:
    values = np.asarray(positions, dtype=np.int64).copy()
    if seed is not None:
        values = np.random.Generator(np.random.PCG64(seed)).permutation(values)
    batches = [values[start:start + batch_size] for start in range(0, len(values), batch_size)]
    if avoid_singleton and batches and len(batches[-1]) == 1:
        if len(batches) == 1:
            raise ValueError("training partition cannot contain only one sample")
        batches[-2] = np.concatenate([batches[-2], batches[-1]])
        batches.pop()
    return batches


def _forward(model: nn.Module, batch: OfficialGraphBatch) -> torch.Tensor:
    return model(
        batch.atom_features, batch.neighbor_features,
        batch.neighbor_indices, batch.crystal_atom_indices,
    )


def _evaluate(
    model: nn.Module,
    cache: CrystalGraphCache,
    positions: np.ndarray,
    targets: np.ndarray,
    atom_lookup: np.ndarray,
    *,
    target_index: int,
    target_mean: float,
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_positions in _position_batches(positions, batch_size):
            batch = batch_official_graphs(
                cache, batch_positions, targets, atom_lookup,
                target_index=target_index, device=device,
            )
            physical = _forward(model, batch) * target_scale + target_mean
            predictions.append(physical.reshape(-1).cpu().numpy())
    joined = np.concatenate(predictions)
    return regression_metrics_1d(targets[positions, target_index], joined), joined


@dataclass
class OfficialIIDFit:
    result: dict[str, Any]
    history: list[dict[str, Any]]
    test_positions: np.ndarray
    test_predictions: np.ndarray
    model_states: dict[str, dict[str, torch.Tensor]]
    normalizers: dict[str, dict[str, float]]


def train_official_iid(
    cache: CrystalGraphCache,
    material_ids: np.ndarray,
    targets: np.ndarray,
    partitions: Mapping[str, np.ndarray],
    atom_lookup: np.ndarray,
    official_model_path: Path | str,
    *,
    device: str = "cuda:0",
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    seed: int = 20260910,
    evaluation_unit: str = "iid",
) -> OfficialIIDFit:
    if epochs < 1 or batch_size < 2 or learning_rate <= 0 or momentum < 0 or weight_decay < 0:
        raise ValueError("training configuration is invalid")
    ids = np.asarray(material_ids)
    values = np.asarray(targets, dtype=np.float32)
    if ids.shape != (len(cache),) or values.shape != (len(cache), 2):
        raise ValueError("material IDs or targets are not aligned to graph cache")
    valid_units = {"iid", *(f"loco_{index:02d}" for index in range(10))}
    if evaluation_unit not in valid_units:
        raise ValueError("evaluation_unit must be iid or loco_00 through loco_09")
    required = {"train", "validation", "test"}
    if set(partitions) != required:
        raise ValueError("partitions must contain train, validation, and test")
    combined = np.concatenate([np.asarray(partitions[name], dtype=np.int64) for name in required])
    if len(combined) != len(cache) or len(np.unique(combined)) != len(cache) or set(combined.tolist()) != set(range(len(cache))):
        raise ValueError("partitions must form an exact material partition")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_class = load_official_cgcnn_class(official_model_path)
    train_positions = np.asarray(partitions["train"], dtype=np.int64)
    validation_positions = np.asarray(partitions["validation"], dtype=np.int64)
    test_positions = np.asarray(partitions["test"], dtype=np.int64)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    history: list[dict[str, Any]] = []
    predictions = np.empty((len(test_positions), 2), dtype=np.float32)
    states: dict[str, dict[str, torch.Tensor]] = {}
    normalizers: dict[str, dict[str, float]] = {}
    target_results: dict[str, Any] = {}
    try:
        for target_index, target_name in enumerate(TARGET_NAMES):
            model_seed = seed + target_index
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_seed)
            model = model_class(
                92, 41, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                classification=False,
            ).to(selected_device)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
            target_mean = float(values[train_positions, target_index].mean())
            target_scale = float(values[train_positions, target_index].std(ddof=1))
            if not math.isfinite(target_scale) or target_scale <= 0:
                raise ValueError(f"invalid train-only target scale: {target_name}")
            best_validation_mae = math.inf
            best_epoch = -1
            best_state: dict[str, torch.Tensor] | None = None
            for epoch in range(epochs):
                model.train()
                sum_loss = 0.0
                sample_count = 0
                for batch_positions in _position_batches(
                    train_positions, batch_size, seed=model_seed + epoch,
                    avoid_singleton=True,
                ):
                    batch = batch_official_graphs(
                        cache, batch_positions, values, atom_lookup,
                        target_index=target_index, device=selected_device,
                    )
                    normalized_target = (batch.targets - target_mean) / target_scale
                    optimizer.zero_grad(set_to_none=True)
                    output = _forward(model, batch)
                    loss = nn.functional.mse_loss(output, normalized_target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite training loss for {target_name}")
                    loss.backward()
                    if not all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    ):
                        raise RuntimeError(f"non-finite gradient for {target_name}")
                    optimizer.step()
                    sum_loss += float(loss.item()) * len(batch_positions)
                    sample_count += len(batch_positions)
                validation_metrics, _ = _evaluate(
                    model, cache, validation_positions, values, atom_lookup,
                    target_index=target_index, target_mean=target_mean,
                    target_scale=target_scale, batch_size=batch_size,
                    device=selected_device,
                )
                current_lr = float(optimizer.param_groups[0]["lr"])
                history.append({
                    "target": target_name, "epoch": epoch + 1,
                    "train_normalized_mse": sum_loss / sample_count,
                    "validation_mae_gpa": validation_metrics["mae"],
                    "learning_rate": current_lr,
                })
                if validation_metrics["mae"] < best_validation_mae:
                    best_validation_mae = validation_metrics["mae"]
                    best_epoch = epoch + 1
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                scheduler.step()
            if best_state is None:
                raise RuntimeError(f"no valid checkpoint selected for {target_name}")
            model.load_state_dict(best_state)
            test_metrics, target_predictions = _evaluate(
                model, cache, test_positions, values, atom_lookup,
                target_index=target_index, target_mean=target_mean,
                target_scale=target_scale, batch_size=batch_size,
                device=selected_device,
            )
            predictions[:, target_index] = target_predictions.astype(np.float32)
            states[target_name] = best_state
            normalizers[target_name] = {"mean_gpa": target_mean, "scale_gpa": target_scale}
            target_results[target_name] = {
                "seed": model_seed,
                "best_epoch": best_epoch,
                "best_validation_mae_gpa": best_validation_mae,
                "test_metrics": test_metrics,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            }
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cudnn.benchmark = previous_benchmark
    complete = all(
        math.isfinite(value)
        for target in target_results.values()
        for value in [target["best_validation_mae_gpa"], *target["test_metrics"].values()]
    )
    result = {
        "schema_version": (
            "official_cgcnn_iid_v1" if evaluation_unit == "iid"
            else "official_cgcnn_loco_v1"
        ),
        "status": "complete" if complete else "failed",
        "complete": bool(complete),
        "evaluation_unit": evaluation_unit,
        "targets": target_results,
        "sample_counts": {name: int(len(partitions[name])) for name in ("train", "validation", "test")},
        "config": {
            "epochs": epochs, "batch_size": batch_size, "optimizer": "SGD",
            "learning_rate": learning_rate, "momentum": momentum,
            "weight_decay": weight_decay, "lr_milestones": [100],
            "atom_feature_count": 92, "distance_feature_count": 41,
            "atom_fea_len": 64, "n_conv": 3, "h_fea_len": 128, "n_h": 1,
            "base_seed": seed, "device": str(selected_device),
            "target_normalization": "full_train_mean_and_sample_std",
        },
    }
    return OfficialIIDFit(
        result=result, history=history, test_positions=test_positions,
        test_predictions=predictions, model_states=states, normalizers=normalizers,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def validate_existing_output(output_dir: Path | str, signature: Mapping[str, Any]) -> dict[str, Any] | None:
    destination = Path(output_dir).resolve()
    if not destination.exists():
        return None
    if not destination.is_dir() or {path.name for path in destination.iterdir()} != OUTPUT_INVENTORY:
        raise FileExistsError("existing official CGCNN output inventory differs")
    summary = json.loads((destination / "run_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("complete") is not True
        or summary.get("run_signature_sha256") != signature_sha256(signature)
    ):
        raise FileExistsError("existing official CGCNN run signature differs")
    recorded = summary.get("output_file_sha256")
    if not isinstance(recorded, Mapping):
        raise FileExistsError("existing official CGCNN output hashes are missing")
    for name in OUTPUT_INVENTORY - {"run_summary.json"}:
        if recorded.get(name) != sha256_file(destination / name):
            raise FileExistsError(f"existing official CGCNN output differs: {name}")
    return {
        "status": "complete", "complete": True, "reused": True,
        "output_dir": str(destination),
    }


def materialize_official_iid(
    fit: OfficialIIDFit,
    output_dir: Path | str,
    *,
    material_ids: np.ndarray,
    targets: np.ndarray,
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    if destination.exists():
        existing = validate_existing_output(destination, signature)
        assert existing is not None
        return existing
    ids = np.asarray(material_ids).astype(str)
    truth = np.asarray(targets, dtype=np.float64)[fit.test_positions]
    prediction_rows = [
        {
            "material_id": ids[position],
            "k_true_gpa": truth[row, 0], "k_pred_gpa": fit.test_predictions[row, 0],
            "g_true_gpa": truth[row, 1], "g_pred_gpa": fit.test_predictions[row, 1],
        }
        for row, position in enumerate(fit.test_positions.tolist())
    ]
    history_bytes = _csv_bytes(
        fit.history,
        ("target", "epoch", "train_normalized_mse", "validation_mae_gpa", "learning_rate"),
    )
    predictions_bytes = _csv_bytes(
        prediction_rows,
        ("material_id", "k_true_gpa", "k_pred_gpa", "g_true_gpa", "g_pred_gpa"),
    )
    evaluation_unit = str(fit.result["evaluation_unit"])
    report_lines = [
        f"# Crystal-only 官方 CGCNN {evaluation_unit} 基线 v1", "",
        "状态：**完成**。", "",
        "| 目标 | 最佳 epoch | Validation MAE (GPa) | Test MAE (GPa) | Test RMSE (GPa) | Test R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_name in TARGET_NAMES:
        item = fit.result["targets"][target_name]
        report_lines.append(
            f"| `{target_name}` | {item['best_epoch']} | {item['best_validation_mae_gpa']:.6g} | "
            f"{item['test_metrics']['mae']:.6g} | {item['test_metrics']['rmse']:.6g} | "
            f"{item['test_metrics']['r2']:.6g} |"
        )
    boundary = (
        "这是单随机种子 IID 基线；多种子与 SOAP-LOCO 未完成前，不作为最终论文统计。"
        if evaluation_unit == "iid"
        else "这是单个 SOAP-LOCO 外层折、单随机种子结果；必须汇总全部十折与多随机种子后解释结构域外泛化。"
    )
    report_lines.extend([
        "", "模型依据固定官方 CGCNN 提交；K/G 分别训练单目标回归器。",
        "", f"边界：{boundary}",
    ])
    report_bytes = ("\n".join(report_lines) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        (staging / "epoch_history.csv").write_bytes(history_bytes)
        (staging / "test_predictions.csv").write_bytes(predictions_bytes)
        (staging / "quality_report.md").write_bytes(report_bytes)
        for target_name, filename in zip(TARGET_NAMES, ("model_k.pt", "model_g.pt")):
            torch.save({
                "schema_version": "official_cgcnn_checkpoint_v1",
                "target": target_name,
                "state_dict": fit.model_states[target_name],
                "normalizer": fit.normalizers[target_name],
                "config": fit.result["config"],
                "run_signature_sha256": signature_sha256(signature),
            }, staging / filename)
        hashed_names = OUTPUT_INVENTORY - {"run_summary.json"}
        summary = {
            **fit.result,
            "official_upstream": {
                "repository": "https://github.com/txie-93/cgcnn",
                "commit": OFFICIAL_COMMIT,
                "model_sha256": OFFICIAL_MODEL_SHA256,
                "atom_init_sha256": OFFICIAL_ATOM_INIT_SHA256,
            },
            "run_signature": dict(signature),
            "run_signature_sha256": signature_sha256(signature),
            "output_file_sha256": {
                name: sha256_file(staging / name) for name in sorted(hashed_names)
            },
        }
        (staging / "run_summary.json").write_bytes(canonical_json_bytes(summary))
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "status": fit.result["status"], "complete": fit.result["complete"],
        "reused": False, "output_dir": str(destination),
    }
