"""Pack validated crystal structures into framework-independent arrays."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd
from pymatgen.core import Structure


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from crystal_dos_master import _sha256_file, _write_json


CACHE_SCHEMA_VERSION = "crystal_cache_v1"


def _save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _stream_selected_records(
    snapshot_dir: Path,
    wanted_ids: set[str],
) -> dict[str, tuple[dict[str, Any], str]]:
    selected: dict[str, tuple[dict[str, Any], str]] = {}
    for shard in sorted((snapshot_dir / "shards").glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                material_id = str(record.get("material_id", ""))
                if material_id not in wanted_ids:
                    continue
                if material_id in selected:
                    raise ValueError(f"duplicate structure for {material_id}")
                selected[material_id] = (record, shard.name)
    missing = sorted(wanted_ids - set(selected))
    if missing:
        raise ValueError(f"missing structure for master IDs: {missing[:5]}")
    return selected


def load_cached_structure(cache_dir: Path | str, row: int) -> dict[str, np.ndarray]:
    """Load one packed structure without retaining memory-map file handles."""
    root = Path(cache_dir)
    lattices = np.load(root / "lattice_matrices.npy", mmap_mode="r", allow_pickle=False)
    frac = np.load(root / "fractional_coordinates.npy", mmap_mode="r", allow_pickle=False)
    numbers = np.load(root / "atomic_numbers.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / "site_offsets.npy", mmap_mode="r", allow_pickle=False)
    if row < 0 or row >= len(lattices):
        raise IndexError(row)
    start, stop = int(offsets[row]), int(offsets[row + 1])
    result = {
        "lattice": np.array(lattices[row], copy=True),
        "fractional_coordinates": np.array(frac[start:stop], copy=True),
        "atomic_numbers": np.array(numbers[start:stop], copy=True),
    }
    for array in (lattices, frac, numbers, offsets):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return result


def _validate_disk_cache(
    root: Path,
    expected_ids: list[str],
    source_structures: list[Structure],
) -> dict[str, Any]:
    ids = np.load(root / "material_ids.npy", allow_pickle=False).astype(str)
    lattices = np.load(root / "lattice_matrices.npy", mmap_mode="r", allow_pickle=False)
    frac = np.load(root / "fractional_coordinates.npy", mmap_mode="r", allow_pickle=False)
    numbers = np.load(root / "atomic_numbers.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / "site_offsets.npy", mmap_mode="r", allow_pickle=False)
    checks = {
        "id_order": ids.tolist() == expected_ids,
        "lattice_shape": lattices.shape == (len(expected_ids), 3, 3),
        "offset_shape": offsets.shape == (len(expected_ids) + 1,),
        "offset_bounds": bool(
            len(offsets) > 0
            and offsets[0] == 0
            and np.all(np.diff(offsets) > 0)
            and offsets[-1] == len(frac) == len(numbers)
        ),
        "finite_geometry": bool(np.isfinite(lattices).all() and np.isfinite(frac).all()),
        "atomic_numbers": bool(np.all((numbers >= 1) & (numbers <= 118))),
        "positive_lattices": bool(np.all(np.abs(np.linalg.det(lattices)) > 0)),
    }
    reconstructed = 0
    if all(checks.values()):
        for index, source in enumerate(source_structures):
            cached = load_cached_structure(root, index)
            rebuilt = Structure(
                cached["lattice"],
                cached["atomic_numbers"],
                cached["fractional_coordinates"],
            )
            if len(rebuilt) != len(source):
                break
            if rebuilt.composition != source.composition:
                break
            if not np.isclose(rebuilt.volume, source.volume, rtol=1e-10, atol=1e-10):
                break
            reconstructed += 1
    complete = all(checks.values()) and reconstructed == len(expected_ids)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "material_count": len(expected_ids),
        "total_sites": int(len(numbers)),
        "reconstructed_count": reconstructed,
        "checks": checks,
    }


def pack_structure_cache(
    snapshot_dir: Path | str,
    master_table: pd.DataFrame,
    output_dir: Path | str,
    *,
    structure_loader: Callable[[dict[str, Any]], Structure] = Structure.from_dict,
) -> dict[str, Any]:
    """Pack selected snapshot structures and atomically publish the cache."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if "material_id" not in master_table:
        raise ValueError("master table is missing material_id")
    expected_ids = master_table["material_id"].astype(str).tolist()
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("master material IDs must be unique")
    if expected_ids != sorted(expected_ids):
        raise ValueError("master material IDs must be sorted")

    records = _stream_selected_records(Path(snapshot_dir), set(expected_ids))
    source_structures: list[Structure] = []
    lattices: list[np.ndarray] = []
    frac_blocks: list[np.ndarray] = []
    number_blocks: list[np.ndarray] = []
    offsets = [0]
    manifest_rows: list[dict[str, Any]] = []
    for row, material_id in enumerate(expected_ids):
        record, shard_name = records[material_id]
        structure = structure_loader(record["structure"])
        if not structure.is_ordered:
            raise ValueError(f"unordered structure is unsupported: {material_id}")
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
        frac = np.asarray(structure.frac_coords, dtype=np.float64)
        numbers = np.asarray([site.specie.Z for site in structure], dtype=np.int16)
        if lattice.shape != (3, 3) or frac.shape != (len(structure), 3):
            raise ValueError(f"invalid structure geometry: {material_id}")
        source_structures.append(structure)
        lattices.append(lattice)
        frac_blocks.append(frac)
        number_blocks.append(numbers)
        start = offsets[-1]
        offsets.append(start + len(structure))
        manifest_rows.append(
            {
                "material_id": material_id,
                "cache_row": row,
                "site_start": start,
                "site_stop": offsets[-1],
                "nsites": len(structure),
                "formula_pretty": structure.composition.reduced_formula,
                "volume": float(structure.volume),
                "source_shard": shard_name,
                "source_batch_index": int(record.get("batch_index", -1)),
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        arrays = {
            "material_ids.npy": np.asarray(expected_ids),
            "lattice_matrices.npy": np.stack(lattices).astype(np.float64),
            "fractional_coordinates.npy": np.concatenate(frac_blocks, axis=0).astype(np.float64),
            "atomic_numbers.npy": np.concatenate(number_blocks).astype(np.int16),
            "site_offsets.npy": np.asarray(offsets, dtype=np.int64),
        }
        for name, array in arrays.items():
            _save_npy(staging / name, array)
        pd.DataFrame(manifest_rows).to_parquet(staging / "cache_manifest.parquet", index=False)
        audit = _validate_disk_cache(staging, expected_ids, source_structures)
        if not audit["complete"]:
            raise ValueError(f"cache disk validation failed: {audit}")
        _write_json(staging / "cache_audit.json", audit)
        file_names = [*arrays, "cache_manifest.parquet", "cache_audit.json"]
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "complete": True,
            "material_count": len(expected_ids),
            "total_sites": int(offsets[-1]),
            "output_file_sha256": {
                name: _sha256_file(staging / name) for name in file_names
            },
        }
        _write_json(staging / "cache_metadata.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "status": "complete",
        "complete": True,
        "material_count": len(expected_ids),
        "total_sites": int(offsets[-1]),
        "output_dir": str(destination.resolve()),
    }
