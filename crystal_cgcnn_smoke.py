"""Deterministic CGCNN-style graph smoke test using the audited graph cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GRAPH_ARRAYS = (
    "material_ids.npy", "atomic_numbers.npy", "site_offsets.npy", "edge_index.npy",
    "edge_distances.npy", "edge_image_vectors.npy", "graph_edge_offsets.npy",
    "neighbor_counts.npy",
)
GRAPH_INVENTORY = {*GRAPH_ARRAYS, "graph_audit.json", "graph_metadata.json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _close_memmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


class CrystalGraphCache:
    """Read-only, hash-verified graph cache aligned one-to-one with a master table."""

    def __init__(self, root: Path | str, *, expected_material_ids: np.ndarray):
        self.root = Path(root).resolve()
        if not self.root.is_dir() or {path.name for path in self.root.iterdir()} != GRAPH_INVENTORY:
            raise ValueError("graph cache inventory differs from the v1 contract")
        self.metadata = _read_json(self.root / "graph_metadata.json")
        if (
            self.metadata.get("schema_version") != "crystal_graph_cache_v1"
            or self.metadata.get("complete") is not True
            or self.metadata.get("status") != "complete"
            or self.metadata.get("edge_direction") != "neighbor_to_center"
        ):
            raise ValueError("graph cache metadata is not complete")
        recorded = self.metadata.get("output_file_sha256")
        if not isinstance(recorded, Mapping):
            raise ValueError("graph cache output hashes are missing")
        for name in [*GRAPH_ARRAYS, "graph_audit.json"]:
            if recorded.get(name) != _sha256(self.root / name):
                raise ValueError(f"graph cache SHA-256 mismatch: {name}")
        self.arrays = {
            name: np.load(self.root / name, mmap_mode="r", allow_pickle=False)
            for name in GRAPH_ARRAYS
        }
        ids = self.arrays["material_ids.npy"]
        expected = np.asarray(expected_material_ids)
        if (
            ids.ndim != 1 or ids.dtype.kind != "U" or expected.ndim != 1
            or expected.dtype.kind not in "UO" or not np.array_equal(ids, expected.astype(str))
        ):
            self.close()
            raise ValueError("graph cache material ID order differs from the master table")
        numbers = self.arrays["atomic_numbers.npy"]
        offsets = self.arrays["site_offsets.npy"]
        edges = self.arrays["edge_index.npy"]
        distances = self.arrays["edge_distances.npy"]
        graph_offsets = self.arrays["graph_edge_offsets.npy"]
        counts = self.arrays["neighbor_counts.npy"]
        valid = (
            numbers.ndim == 1 and np.all((numbers >= 1) & (numbers <= 118))
            and offsets.shape == (len(ids) + 1,) and offsets[0] == 0
            and offsets[-1] == len(numbers) and np.all(np.diff(offsets) > 0)
            and edges.shape == (2, len(distances))
            and graph_offsets.shape == (len(ids) + 1,) and graph_offsets[0] == 0
            and graph_offsets[-1] == len(distances) and np.all(np.diff(graph_offsets) >= 0)
            and counts.shape == (len(numbers),)
            and np.array_equal(np.bincount(edges[1], minlength=len(numbers)), counts)
            and self.metadata.get("material_count") == len(ids)
            and self.metadata.get("total_sites") == len(numbers)
            and self.metadata.get("total_edges") == len(distances)
        )
        if not bool(valid):
            self.close()
            raise ValueError("graph cache array contract failed")

    def __len__(self) -> int:
        return len(self.arrays["material_ids.npy"])

    def close(self) -> None:
        for value in getattr(self, "arrays", {}).values():
            _close_memmap(value)


@dataclass(frozen=True)
class GraphBatch:
    atomic_numbers: torch.Tensor
    edge_index: torch.Tensor
    edge_distances: torch.Tensor
    neighbor_counts: torch.Tensor
    node_counts: torch.Tensor
    targets: torch.Tensor


def batch_graphs(
    cache: CrystalGraphCache,
    positions: np.ndarray,
    targets: np.ndarray,
    *,
    device: str | torch.device,
) -> GraphBatch:
    indices = np.asarray(positions)
    target_matrix = np.asarray(targets, dtype=np.float32)
    if (
        indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 1
        or np.any(indices < 0) or np.any(indices >= len(cache))
        or len(set(indices.tolist())) != len(indices)
    ):
        raise ValueError("positions must be unique, in-range integer graph indices")
    if target_matrix.shape != (len(cache), 2) or not np.isfinite(target_matrix).all():
        raise ValueError("targets must be a finite [material_count, 2] matrix")
    arrays = cache.arrays
    site_offsets = arrays["site_offsets.npy"]
    graph_offsets = arrays["graph_edge_offsets.npy"]
    number_blocks: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []
    distance_blocks: list[np.ndarray] = []
    count_blocks: list[np.ndarray] = []
    node_counts: list[int] = []
    batch_node_start = 0
    for position in indices.tolist():
        node_start, node_stop = int(site_offsets[position]), int(site_offsets[position + 1])
        edge_start, edge_stop = int(graph_offsets[position]), int(graph_offsets[position + 1])
        node_count = node_stop - node_start
        local_edges = np.asarray(arrays["edge_index.npy"][:, edge_start:edge_stop]) - node_start
        if local_edges.size and not np.all((local_edges >= 0) & (local_edges < node_count)):
            raise ValueError("graph edge crosses material boundary")
        number_blocks.append(np.asarray(arrays["atomic_numbers.npy"][node_start:node_stop]))
        edge_blocks.append(local_edges + batch_node_start)
        distance_blocks.append(np.asarray(arrays["edge_distances.npy"][edge_start:edge_stop]))
        count_blocks.append(np.asarray(arrays["neighbor_counts.npy"][node_start:node_stop]))
        node_counts.append(node_count)
        batch_node_start += node_count
    selected_device = torch.device(device)
    return GraphBatch(
        atomic_numbers=torch.as_tensor(np.concatenate(number_blocks), dtype=torch.long, device=selected_device),
        edge_index=torch.as_tensor(np.concatenate(edge_blocks, axis=1), dtype=torch.long, device=selected_device),
        edge_distances=torch.as_tensor(
            np.concatenate(distance_blocks), dtype=torch.float32, device=selected_device
        ),
        neighbor_counts=torch.as_tensor(
            np.concatenate(count_blocks).astype(np.int64), dtype=torch.long, device=selected_device
        ),
        node_counts=torch.as_tensor(node_counts, dtype=torch.long, device=selected_device),
        targets=torch.as_tensor(target_matrix[indices], dtype=torch.float32, device=selected_device),
    )


class GaussianDistance(nn.Module):
    def __init__(self, cutoff: float = 8.0, bins: int = 41):
        super().__init__()
        if cutoff <= 0 or bins < 2:
            raise ValueError("cutoff and bins are invalid")
        centers = torch.linspace(0.0, float(cutoff), int(bins))
        self.register_buffer("centers", centers)
        self.gamma = float(1.0 / (centers[1] - centers[0]).item() ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances[:, None] - self.centers[None, :]) ** 2)


class CrystalGraphConvolution(nn.Module):
    def __init__(self, hidden_dim: int, distance_bins: int):
        super().__init__()
        self.message = nn.Linear(2 * hidden_dim + distance_bins, 2 * hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atoms: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_counts: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index[0], edge_index[1]
        joined = torch.cat([atoms[target], atoms[source], edge_features], dim=1)
        gate, core = self.message(joined).chunk(2, dim=1)
        messages = torch.sigmoid(gate) * F.softplus(core)
        aggregated = torch.segment_reduce(messages, "sum", lengths=neighbor_counts)
        return F.softplus(atoms + self.normalization(aggregated))


class CrystalRegressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        distance_bins: int = 41,
        convolution_count: int = 3,
        cutoff: float = 8.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.distance = GaussianDistance(cutoff, distance_bins)
        self.convolutions = nn.ModuleList([
            CrystalGraphConvolution(hidden_dim, distance_bins)
            for _ in range(convolution_count)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        atoms = self.embedding(batch.atomic_numbers)
        edge_features = self.distance(batch.edge_distances)
        for convolution in self.convolutions:
            atoms = convolution(
                atoms, batch.edge_index, edge_features, batch.neighbor_counts
            )
        pooled = torch.segment_reduce(atoms, "mean", lengths=batch.node_counts)
        return self.head(pooled)


def _fit_once(
    batch: GraphBatch,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    hidden_dim: int,
    cutoff: float,
) -> tuple[float, float, np.ndarray, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = CrystalRegressor(hidden_dim=hidden_dim, cutoff=cutoff).to(batch.targets.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial = float(F.mse_loss(model(batch), batch.targets).item())
    finite_gradients = True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(batch), batch.targets)
        if not torch.isfinite(loss):
            finite_gradients = False
            break
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            finite_gradients = False
            break
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(batch)
        final = float(F.mse_loss(prediction, batch.targets).item())
    return initial, final, prediction.detach().cpu().numpy(), finite_gradients


def run_graph_overfit_smoke(
    cache: CrystalGraphCache,
    positions: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int = 400,
    learning_rate: float = 5e-3,
    seed: int = 20260910,
    device: str = "cuda:0",
    hidden_dim: int = 64,
) -> dict[str, Any]:
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or hidden_dim < 2:
        raise ValueError("training configuration is invalid")
    indices = np.asarray(positions)
    values = np.asarray(targets, dtype=np.float32)
    selected = values[indices]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = np.zeros_like(values)
    normalized[indices] = (selected - mean) / scale
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    batch = batch_graphs(cache, indices, normalized, device=selected_device)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        first = _fit_once(
            batch, steps=steps, learning_rate=learning_rate, seed=seed,
            hidden_dim=hidden_dim, cutoff=float(cache.metadata["cutoff_angstrom"]),
        )
        second = _fit_once(
            batch, steps=steps, learning_rate=learning_rate, seed=seed,
            hidden_dim=hidden_dim, cutoff=float(cache.metadata["cutoff_angstrom"]),
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cudnn.benchmark = previous_benchmark
    repeat_equal = first[0] == second[0] and first[1] == second[1] and np.array_equal(first[2], second[2])
    ratio = first[1] / first[0]
    complete = bool(
        all(math.isfinite(value) for value in (first[0], first[1], ratio))
        and first[3] and second[3] and repeat_equal and ratio < 0.25
    )
    return {
        "schema_version": "crystal_only_cgcnn_smoke_v1",
        "status": "complete" if complete else "failed",
        "complete": complete,
        "sample_count": int(len(indices)),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(selected_device),
        "hidden_dim": int(hidden_dim),
        "convolution_count": 3,
        "distance_bins": 41,
        "initial_objective": first[0],
        "final_objective": first[1],
        "objective_ratio": ratio,
        "finite_gradients": bool(first[3] and second[3]),
        "deterministic_repeat_equal": bool(repeat_equal),
        "target_mean_gpa": mean.astype(float).tolist(),
        "target_scale_gpa": scale.astype(float).tolist(),
    }


def subset_sha256(material_ids: np.ndarray) -> str:
    values = np.asarray(material_ids).astype(str).tolist()
    if any("\n" in value or "\r" in value for value in values):
        raise ValueError("material IDs cannot contain line breaks")
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def materialize_graph_smoke_receipt(result: Mapping[str, Any], output_dir: Path | str) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    summary = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report = (
        "# Crystal-only CGCNN-style 烟雾测试 v1\n\n"
        f"状态：**{'完成' if result.get('complete') else '失败'}**。\n\n"
        f"固定 IID-train 子集：{result.get('sample_count')} 图；设备：`{result.get('device')}`。\n\n"
        f"初始/最终标准化 MSE：{result.get('initial_objective'):.8g} / "
        f"{result.get('final_objective'):.8g}；比率：{result.get('objective_ratio'):.8g}。\n\n"
        f"梯度有限：{result.get('finite_gradients')}；确定性重复一致："
        f"{result.get('deterministic_repeat_equal')}。\n\n"
        "边界：这是训练集小样本过拟合门禁，不是 validation/test 结果，也不是论文最终 CGCNN 基线。\n"
    ).encode("utf-8")
    payloads = {"run_summary.json": summary, "quality_report.md": report}
    if destination.exists():
        if not destination.is_dir() or {path.name for path in destination.iterdir()} != set(payloads):
            raise FileExistsError("existing smoke output inventory differs")
        if any((destination / name).read_bytes() != payload for name, payload in payloads.items()):
            raise FileExistsError("existing smoke output differs")
        return {"status": result["status"], "complete": result["complete"], "reused": True,
                "output_dir": str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        for name, payload in payloads.items():
            path = staging / name
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"status": result["status"], "complete": result["complete"], "reused": False,
            "output_dir": str(destination)}

