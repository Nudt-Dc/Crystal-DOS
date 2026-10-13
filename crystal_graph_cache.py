"""Build a deterministic, framework-independent periodic crystal graph cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np
from pymatgen.core import Lattice, Structure


SCHEMA_VERSION = "crystal_graph_cache_v1"
SOURCE_SCHEMA_VERSIONS = {
    "crystal_cache_v1",
    "aflow_confirmatory_crystal_cache_v1",
}
SOURCE_ARRAYS = (
    "material_ids.npy",
    "lattice_matrices.npy",
    "fractional_coordinates.npy",
    "atomic_numbers.npy",
    "site_offsets.npy",
)
OUTPUT_ARRAYS = (
    "material_ids.npy",
    "atomic_numbers.npy",
    "site_offsets.npy",
    "edge_index.npy",
    "edge_distances.npy",
    "edge_image_vectors.npy",
    "graph_edge_offsets.npy",
    "neighbor_counts.npy",
)
OUTPUT_INVENTORY = {*OUTPUT_ARRAYS, "graph_audit.json", "graph_metadata.json"}


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
        raise ValueError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _close_memmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def build_periodic_edges(
    lattice_matrix: np.ndarray,
    fractional_coordinates: np.ndarray,
    *,
    cutoff: float,
    max_neighbors: int,
) -> dict[str, np.ndarray]:
    """Return capped directed periodic edges, sorted independently per center atom."""
    lattice = np.asarray(lattice_matrix, dtype=np.float64)
    frac = np.asarray(fractional_coordinates, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) <= 0:
        raise ValueError("lattice_matrix must be finite, nonsingular, and 3x3")
    if frac.ndim != 2 or frac.shape[1] != 3 or len(frac) < 1 or not np.isfinite(frac).all():
        raise ValueError("fractional_coordinates must be a nonempty finite Nx3 matrix")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if isinstance(max_neighbors, bool) or not isinstance(max_neighbors, int) or max_neighbors < 1:
        raise ValueError("max_neighbors must be a positive integer")

    structure = Structure(
        Lattice(lattice),
        np.ones(len(frac), dtype=np.int16),
        frac,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    centers, neighbors, images_raw, distances = structure.get_neighbor_list(
        r=float(cutoff), numerical_tol=1e-8, exclude_self=True
    )
    centers = np.asarray(centers, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    images_float = np.asarray(images_raw, dtype=np.float64)
    images = np.rint(images_float).astype(np.int64)
    if images_float.shape != images.shape or not np.allclose(images_float, images, rtol=0, atol=1e-8):
        raise ValueError("pymatgen returned non-integral periodic image vectors")
    valid = (
        np.isfinite(distances)
        & (distances > 1e-8)
        & (distances <= float(cutoff) + 1e-8)
    )
    centers, neighbors, images, distances = (
        value[valid] for value in (centers, neighbors, images, distances)
    )

    selected_indices: list[np.ndarray] = []
    counts = np.zeros(len(frac), dtype=np.int16)
    for center in range(len(frac)):
        candidates = np.flatnonzero(centers == center)
        if len(candidates):
            order = np.lexsort(
                (
                    images[candidates, 2],
                    images[candidates, 1],
                    images[candidates, 0],
                    neighbors[candidates],
                    distances[candidates],
                )
            )
            chosen = candidates[order[:max_neighbors]]
        else:
            chosen = np.empty(0, dtype=np.int64)
        selected_indices.append(chosen)
        counts[center] = len(chosen)
    selected = np.concatenate(selected_indices) if selected_indices else np.empty(0, dtype=np.int64)
    if len(selected):
        edge_index = np.stack([neighbors[selected], centers[selected]]).astype(np.int64)
        edge_distances = distances[selected].astype(np.float32)
        edge_images_int64 = images[selected]
        if np.any(edge_images_int64 < np.iinfo(np.int16).min) or np.any(
            edge_images_int64 > np.iinfo(np.int16).max
        ):
            raise ValueError("periodic image vector exceeds int16 range")
        edge_images = edge_images_int64.astype(np.int16)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_distances = np.empty(0, dtype=np.float32)
        edge_images = np.empty((0, 3), dtype=np.int16)
    return {
        "edge_index": edge_index,
        "edge_distances": edge_distances,
        "edge_image_vectors": edge_images,
        "neighbor_counts": counts,
    }


def _source_state(source_root: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, np.ndarray]]:
    metadata_path = source_root / "cache_metadata.json"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("schema_version") not in SOURCE_SCHEMA_VERSIONS
        or metadata.get("status") != "complete"
        or metadata.get("complete") is not True
    ):
        raise ValueError("source crystal cache metadata is not complete")
    recorded = metadata.get("output_file_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError("source crystal cache lacks output hashes")
    source_hashes = {"cache_metadata.json": _sha256(metadata_path)}
    for name in SOURCE_ARRAYS:
        path = source_root / name
        actual = _sha256(path)
        if recorded.get(name) != actual:
            raise ValueError(f"source crystal cache SHA-256 mismatch: {name}")
        source_hashes[name] = actual
    arrays = {
        name: np.load(source_root / name, mmap_mode="r", allow_pickle=False)
        for name in SOURCE_ARRAYS
    }
    ids = arrays["material_ids.npy"]
    lattices = arrays["lattice_matrices.npy"]
    frac = arrays["fractional_coordinates.npy"]
    numbers = arrays["atomic_numbers.npy"]
    offsets = arrays["site_offsets.npy"]
    count = len(ids)
    valid = (
        ids.ndim == 1
        and ids.dtype.kind == "U"
        and len(set(ids.tolist())) == count
        and lattices.shape == (count, 3, 3)
        and frac.ndim == 2
        and frac.shape[1] == 3
        and numbers.ndim == 1
        and offsets.shape == (count + 1,)
        and count > 0
        and offsets[0] == 0
        and np.all(np.diff(offsets) > 0)
        and offsets[-1] == len(frac) == len(numbers)
        and np.isfinite(lattices).all()
        and np.isfinite(frac).all()
        and np.all((numbers >= 1) & (numbers <= 118))
    )
    if not bool(valid):
        for value in arrays.values():
            _close_memmap(value)
        raise ValueError("source crystal cache array contract failed")
    if metadata.get("material_count") != count or metadata.get("total_sites") != int(offsets[-1]):
        for value in arrays.values():
            _close_memmap(value)
        raise ValueError("source crystal cache counts disagree with arrays")
    return metadata, source_hashes, arrays


def _audit_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    cutoff: float,
    max_neighbors: int,
    material_count: int,
) -> dict[str, Any]:
    ids = arrays["material_ids.npy"]
    numbers = arrays["atomic_numbers.npy"]
    site_offsets = arrays["site_offsets.npy"]
    edge_index = arrays["edge_index.npy"]
    distances = arrays["edge_distances.npy"]
    images = arrays["edge_image_vectors.npy"]
    graph_offsets = arrays["graph_edge_offsets.npy"]
    counts = arrays["neighbor_counts.npy"]
    total_sites = len(numbers)
    total_edges = len(distances)
    checks = {
        "material_ids": ids.ndim == 1 and ids.dtype.kind == "U" and len(ids) == material_count,
        "atomic_numbers": numbers.ndim == 1 and bool(np.all((numbers >= 1) & (numbers <= 118))),
        "site_offsets": site_offsets.shape == (material_count + 1,)
        and site_offsets[0] == 0 and site_offsets[-1] == total_sites
        and bool(np.all(np.diff(site_offsets) > 0)),
        "edge_shapes": edge_index.shape == (2, total_edges)
        and images.shape == (total_edges, 3),
        "edge_distances": distances.ndim == 1 and bool(
            np.isfinite(distances).all() and np.all(distances > 0)
            and np.all(distances <= cutoff + 1e-6)
        ),
        "edge_indices": total_edges > 0 and bool(
            np.all(edge_index >= 0) and np.all(edge_index < total_sites)
        ),
        "graph_edge_offsets": graph_offsets.shape == (material_count + 1,)
        and graph_offsets[0] == 0 and graph_offsets[-1] == total_edges
        and bool(np.all(np.diff(graph_offsets) >= 0)),
        "neighbor_counts": counts.shape == (total_sites,)
        and bool(np.all(counts >= 0) and np.all(counts <= max_neighbors)),
        "no_zero_self_edges": bool(
            np.all((edge_index[0] != edge_index[1]) | np.any(images != 0, axis=1))
        ),
    }
    graph_boundaries = True
    grouped_by_center = True
    for graph in range(material_count):
        node_start, node_stop = int(site_offsets[graph]), int(site_offsets[graph + 1])
        edge_start, edge_stop = int(graph_offsets[graph]), int(graph_offsets[graph + 1])
        block = edge_index[:, edge_start:edge_stop]
        if block.size and not bool(np.all((block >= node_start) & (block < node_stop))):
            graph_boundaries = False
            break
        targets = block[1]
        if len(targets) > 1 and not bool(np.all(np.diff(targets) >= 0)):
            grouped_by_center = False
            break
    checks["graph_boundaries"] = graph_boundaries
    checks["grouped_by_center"] = grouped_by_center
    if checks["edge_indices"] and checks["neighbor_counts"]:
        observed = np.bincount(edge_index[1], minlength=total_sites)
        checks["neighbor_count_consistency"] = bool(np.array_equal(observed, counts))
    else:
        checks["neighbor_count_consistency"] = False
    complete = all(bool(value) for value in checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "failed",
        "complete": complete,
        "cutoff_angstrom": float(cutoff),
        "max_neighbors": int(max_neighbors),
        "material_count": int(material_count),
        "total_sites": int(total_sites),
        "total_edges": int(total_edges),
        "minimum_neighbors": int(counts.min()) if len(counts) else 0,
        "maximum_neighbors": int(counts.max()) if len(counts) else 0,
        "insufficient_neighbor_site_count": int(np.count_nonzero(counts < max_neighbors)),
        "zero_neighbor_site_count": int(np.count_nonzero(counts == 0)),
        "checks": checks,
    }


def _validate_existing(
    destination: Path,
    *,
    cutoff: float,
    max_neighbors: int,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not destination.is_dir() or {path.name for path in destination.iterdir()} != OUTPUT_INVENTORY:
        raise FileExistsError(f"existing graph cache inventory differs: {destination}")
    metadata = _read_json(destination / "graph_metadata.json")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("complete") is not True
        or metadata.get("status") != "complete"
        or metadata.get("cutoff_angstrom") != float(cutoff)
        or metadata.get("max_neighbors") != int(max_neighbors)
        or metadata.get("source_sha256") != dict(source_hashes)
    ):
        raise FileExistsError(f"existing graph cache metadata differs: {destination}")
    recorded = metadata.get("output_file_sha256")
    if not isinstance(recorded, Mapping):
        raise FileExistsError(f"existing graph cache hashes differ: {destination}")
    for name in [*OUTPUT_ARRAYS, "graph_audit.json"]:
        if recorded.get(name) != _sha256(destination / name):
            raise FileExistsError(f"existing graph cache differs: {destination}")
    audit = _read_json(destination / "graph_audit.json")
    if audit.get("complete") is not True or audit.get("total_edges") != metadata.get("total_edges"):
        raise FileExistsError(f"existing graph cache audit differs: {destination}")
    return {
        "status": "complete",
        "complete": True,
        "material_count": metadata["material_count"],
        "total_sites": metadata["total_sites"],
        "total_edges": metadata["total_edges"],
        "output_dir": str(destination),
        "reused": True,
    }


def materialize_graph_cache(
    source_cache_dir: Path | str,
    output_dir: Path | str,
    *,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
) -> dict[str, Any]:
    source = Path(source_cache_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("output directory must be outside source cache")
    if not math.isfinite(cutoff) or cutoff <= 0 or isinstance(max_neighbors, bool) \
            or not isinstance(max_neighbors, int) or max_neighbors < 1:
        raise ValueError("invalid graph configuration")
    _, source_hashes, source_arrays = _source_state(source)
    try:
        if destination.exists():
            return _validate_existing(
                destination,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                source_hashes=source_hashes,
            )
        ids = source_arrays["material_ids.npy"]
        lattices = source_arrays["lattice_matrices.npy"]
        frac = source_arrays["fractional_coordinates.npy"]
        numbers = source_arrays["atomic_numbers.npy"]
        site_offsets = source_arrays["site_offsets.npy"]
        edge_blocks: list[np.ndarray] = []
        distance_blocks: list[np.ndarray] = []
        image_blocks: list[np.ndarray] = []
        count_blocks: list[np.ndarray] = []
        graph_edge_offsets = [0]
        for graph in range(len(ids)):
            node_start, node_stop = int(site_offsets[graph]), int(site_offsets[graph + 1])
            built = build_periodic_edges(
                lattices[graph],
                frac[node_start:node_stop],
                cutoff=cutoff,
                max_neighbors=max_neighbors,
            )
            edge_blocks.append(built["edge_index"] + node_start)
            distance_blocks.append(built["edge_distances"])
            image_blocks.append(built["edge_image_vectors"])
            count_blocks.append(built["neighbor_counts"])
            graph_edge_offsets.append(graph_edge_offsets[-1] + built["edge_index"].shape[1])
        output_arrays = {
            "material_ids.npy": np.asarray(ids),
            "atomic_numbers.npy": np.asarray(numbers),
            "site_offsets.npy": np.asarray(site_offsets),
            "edge_index.npy": np.concatenate(edge_blocks, axis=1),
            "edge_distances.npy": np.concatenate(distance_blocks),
            "edge_image_vectors.npy": np.concatenate(image_blocks, axis=0),
            "graph_edge_offsets.npy": np.asarray(graph_edge_offsets, dtype=np.int64),
            "neighbor_counts.npy": np.concatenate(count_blocks).astype(np.int16),
        }
        audit = _audit_arrays(
            output_arrays,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            material_count=len(ids),
        )
        if not audit["complete"]:
            raise ValueError(f"graph cache audit failed: {audit}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(
            f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            staging.mkdir()
            for name, value in output_arrays.items():
                _save_array(staging / name, value)
            _write_bytes(staging / "graph_audit.json", _json_bytes(audit))
            hashed_names = [*OUTPUT_ARRAYS, "graph_audit.json"]
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "complete": True,
                "cutoff_angstrom": float(cutoff),
                "max_neighbors": int(max_neighbors),
                "edge_direction": "neighbor_to_center",
                "edge_sort_key": [
                    "distance", "neighbor_local_index", "image_x", "image_y", "image_z"
                ],
                "material_count": int(len(ids)),
                "total_sites": int(len(numbers)),
                "total_edges": int(output_arrays["edge_distances.npy"].shape[0]),
                "insufficient_neighbor_site_count": audit["insufficient_neighbor_site_count"],
                "zero_neighbor_site_count": audit["zero_neighbor_site_count"],
                "source_sha256": source_hashes,
                "output_file_sha256": {
                    name: _sha256(staging / name) for name in hashed_names
                },
            }
            _write_bytes(staging / "graph_metadata.json", _json_bytes(metadata))
            os.replace(staging, destination)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return {
            "status": "complete",
            "complete": True,
            "material_count": int(len(ids)),
            "total_sites": int(len(numbers)),
            "total_edges": int(output_arrays["edge_distances.npy"].shape[0]),
            "output_dir": str(destination),
            "reused": False,
        }
    finally:
        for value in source_arrays.values():
            _close_memmap(value)
