"""Pack the independent AFLOW relaxed structures into framework-neutral arrays."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from aflow_confirmatory_inventory import sha256_file
from aflow_confirmatory_overlap import aflow_structure


SCHEMA_VERSION = "aflow_confirmatory_crystal_cache_v1"


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_cache(
    master_dir: Path | str,
    cohort_dir: Path | str,
    output_dir: Path | str,
) -> dict:
    master_root = Path(master_dir).resolve()
    cohort_root = Path(cohort_dir).resolve()
    destination = Path(output_dir).resolve()
    metadata_file = master_root / "master_metadata.json"
    table_file = master_root / "master_table.parquet"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != "aflow_confirmatory_master_v1"
        or metadata.get("complete") is not True
        or metadata.get("output_file_sha256", {}).get("master_table.parquet")
        != sha256_file(table_file)
    ):
        raise ValueError("AFLOW master binding is invalid")
    table = pd.read_parquet(table_file)
    ids = table["material_id"].astype(str).tolist()
    if len(ids) != int(metadata["row_count"]) or ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("AFLOW master row order is invalid")
    lattices: list[np.ndarray] = []
    frac_blocks: list[np.ndarray] = []
    number_blocks: list[np.ndarray] = []
    offsets = [0]
    manifest: list[dict] = []
    species: set[int] = set()
    for row_index, row in table.iterrows():
        path = cohort_root / str(row["cohort_record_dir"]) / str(row["structure_filename"])
        if sha256_file(path) != str(row["structure_sha256"]):
            raise ValueError(f"AFLOW structure hash drift for {row['material_id']}")
        structure = aflow_structure(path)
        if not structure.is_ordered:
            raise ValueError(f"unordered AFLOW structure: {row['material_id']}")
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
        frac = np.asarray(structure.frac_coords, dtype=np.float64)
        numbers = np.asarray([site.specie.Z for site in structure], dtype=np.int16)
        if (
            lattice.shape != (3, 3)
            or frac.shape != (len(structure), 3)
            or not np.isfinite(lattice).all()
            or not np.isfinite(frac).all()
            or abs(float(np.linalg.det(lattice))) <= 1e-10
        ):
            raise ValueError(f"invalid AFLOW geometry for {row['material_id']}")
        start = offsets[-1]
        offsets.append(start + len(structure))
        lattices.append(lattice)
        frac_blocks.append(frac)
        number_blocks.append(numbers)
        species.update(int(value) for value in numbers)
        manifest.append({
            "material_id": str(row["material_id"]),
            "cache_row": row_index,
            "site_start": start,
            "site_stop": offsets[-1],
            "nsites": len(structure),
            "reduced_formula": structure.composition.reduced_formula,
            "volume": float(structure.volume),
            "structure_sha256": str(row["structure_sha256"]),
        })
    arrays = {
        "material_ids.npy": np.asarray(ids),
        "lattice_matrices.npy": np.stack(lattices).astype(np.float64),
        "fractional_coordinates.npy": np.concatenate(frac_blocks).astype(np.float64),
        "atomic_numbers.npy": np.concatenate(number_blocks).astype(np.int16),
        "site_offsets.npy": np.asarray(offsets, dtype=np.int64),
    }
    if (
        arrays["lattice_matrices.npy"].shape != (len(ids), 3, 3)
        or arrays["site_offsets.npy"].shape != (len(ids) + 1,)
        or arrays["site_offsets.npy"][0] != 0
        or not np.all(np.diff(arrays["site_offsets.npy"]) > 0)
        or arrays["site_offsets.npy"][-1] != len(arrays["fractional_coordinates.npy"])
        or len(arrays["fractional_coordinates.npy"]) != len(arrays["atomic_numbers.npy"])
    ):
        raise ValueError("AFLOW packed-cache contract failed")
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite AFLOW cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        for name, value in arrays.items():
            _save(staging / name, value)
        manifest_file = staging / "cache_manifest.parquet"
        pd.DataFrame(manifest).to_parquet(manifest_file, index=False)
        cache_metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "material_count": len(ids),
            "total_sites": int(offsets[-1]),
            "species_atomic_numbers": sorted(species),
            "source_sha256": {
                "master_metadata.json": sha256_file(metadata_file),
                "master_table.parquet": sha256_file(table_file),
            },
            "output_file_sha256": {
                **{name: sha256_file(staging / name) for name in arrays},
                "cache_manifest.parquet": sha256_file(manifest_file),
            },
            "model_training_performed": False,
            "model_inference_performed": False,
        }
        (staging / "cache_metadata.json").write_text(
            json.dumps(cache_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return cache_metadata

