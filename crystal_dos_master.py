"""Build the strict-clean Crystal + DOS experiment master table."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd


MASTER_SCHEMA_VERSION = "crystal_dos_master_v1"
DOS_REASONS = ("observed", "natural_missing", "window_insufficient")


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _has_warning(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "[]", "{}", "None", "nan", "NaN"}
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _require_unique_ids(frame: pd.DataFrame, *, source: str) -> None:
    if "material_id" not in frame.columns:
        raise ValueError(f"{source} is missing material_id")
    if frame["material_id"].isna().any():
        raise ValueError(f"{source} material_id contains null values")
    ids = frame["material_id"].astype(str)
    if (ids.str.strip() == "").any():
        raise ValueError(f"{source} material_id contains empty values")
    if not ids.is_unique:
        raise ValueError(f"{source} material_id must be unique")


def build_master_table(
    inventory_path: Path | str,
    structure_manifest_path: Path | str,
    dos_ids_path: Path | str,
) -> pd.DataFrame:
    """Return the lexically ordered strict-clean master population."""
    inventory = pd.read_parquet(inventory_path)
    required_inventory = {
        "material_id",
        "formula_pretty",
        "k_vrh_gpa",
        "g_vrh_gpa",
        "has_crystal",
        "deprecated",
        "elasticity_warnings",
        "has_dos",
    }
    missing = sorted(required_inventory - set(inventory.columns))
    if missing:
        raise ValueError(f"inventory is missing columns: {missing}")
    _require_unique_ids(inventory, source="inventory")
    inventory = inventory.copy()
    inventory["material_id"] = inventory["material_id"].astype(str)

    structures = pd.read_parquet(structure_manifest_path)
    required_structures = {
        "material_id",
        "formula_pretty",
        "chemsys",
        "spacegroup_symbol",
        "spacegroup_number",
        "nsites",
        "volume",
        "batch_index",
        "validation_status",
    }
    missing = sorted(required_structures - set(structures.columns))
    if missing:
        raise ValueError(f"structure manifest is missing columns: {missing}")
    _require_unique_ids(structures, source="structure manifest")
    structures = structures.copy()
    structures["material_id"] = structures["material_id"].astype(str)

    dos_ids = np.load(dos_ids_path, allow_pickle=False).astype(str)
    if len(set(dos_ids.tolist())) != len(dos_ids):
        raise ValueError("DOS material IDs must be unique")
    dos_row_by_id = {value: index for index, value in enumerate(dos_ids.tolist())}

    strict_mask = (
        ~inventory["deprecated"].fillna(False).astype(bool)
        & ~inventory["elasticity_warnings"].map(_has_warning)
        & inventory["k_vrh_gpa"].notna()
        & inventory["g_vrh_gpa"].notna()
        & inventory["has_crystal"].fillna(False).astype(bool)
    )
    strict = inventory.loc[strict_mask].copy()
    strict_ids = set(strict["material_id"])
    outside = sorted(set(dos_row_by_id) - strict_ids)
    if outside:
        raise ValueError(f"DOS IDs outside strict population: {outside[:5]}")

    valid_structures = structures.loc[structures["validation_status"] == "valid"].copy()
    structure_ids = set(valid_structures["material_id"])
    absent = sorted(strict_ids - structure_ids)
    if absent:
        raise ValueError(f"strict IDs missing valid structures: {absent[:5]}")

    structure_columns = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "spacegroup_symbol",
        "spacegroup_number",
        "nsites",
        "volume",
        "batch_index",
    ]
    master = strict.drop(columns=["formula_pretty"]).merge(
        valid_structures[structure_columns],
        on="material_id",
        how="left",
        validate="one_to_one",
    )
    master = master.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    master.insert(1, "crystal_row", np.arange(len(master), dtype=np.int64))
    observed = master["material_id"].isin(dos_row_by_id)
    master["dos_observed"] = observed.astype(bool)
    master["dos_row"] = pd.array(
        [dos_row_by_id.get(value) for value in master["material_id"]],
        dtype="Int64",
    )
    master["dos_missing_reason"] = np.where(
        observed,
        "observed",
        np.where(master["has_dos"].fillna(False).astype(bool), "window_insufficient", "natural_missing"),
    )
    master["crystal_batch_index"] = master["batch_index"].astype(np.int64)
    master["quality_tier"] = "strict_clean"

    columns = [
        "material_id",
        "crystal_row",
        "k_vrh_gpa",
        "g_vrh_gpa",
        "dos_observed",
        "dos_row",
        "dos_missing_reason",
        "formula_pretty",
        "chemsys",
        "spacegroup_symbol",
        "spacegroup_number",
        "nsites",
        "volume",
        "crystal_batch_index",
        "quality_tier",
    ]
    return master[columns]


def validate_master_table(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
    expected_dos_counts: Mapping[str, int] | None = None,
    require_dense_dos_rows: bool = True,
) -> dict[str, Any]:
    """Validate master invariants and return a machine-readable gate."""
    required = {
        "material_id",
        "crystal_row",
        "k_vrh_gpa",
        "g_vrh_gpa",
        "dos_observed",
        "dos_row",
        "dos_missing_reason",
        "nsites",
        "volume",
        "quality_tier",
    }
    missing_columns = sorted(required - set(frame.columns))
    ids = frame["material_id"].astype(str) if "material_id" in frame else pd.Series(dtype=str)
    category_counts = (
        frame["dos_missing_reason"].value_counts().to_dict()
        if "dos_missing_reason" in frame
        else {}
    )
    finite_targets = (
        np.isfinite(frame[["k_vrh_gpa", "g_vrh_gpa"]].to_numpy(dtype=float)).all(axis=1)
        if {"k_vrh_gpa", "g_vrh_gpa"} <= set(frame.columns)
        else np.zeros(len(frame), dtype=bool)
    )
    observed_mask = (
        frame["dos_observed"].astype(bool).to_numpy()
        if "dos_observed" in frame
        else np.zeros(len(frame), dtype=bool)
    )
    observed_rows = (
        sorted(frame.loc[observed_mask, "dos_row"].dropna().astype(int).tolist())
        if "dos_row" in frame
        else []
    )
    missing_dos_rows_are_null = bool(
        "dos_row" in frame and frame.loc[~observed_mask, "dos_row"].isna().all()
    )
    crystal_rows_match = bool(
        "crystal_row" in frame
        and frame["crystal_row"].astype(int).tolist() == list(range(len(frame)))
    )
    expected_rows = list(range(int(np.sum(observed_mask))))
    if require_dense_dos_rows:
        dos_rows_valid = observed_rows == expected_rows
    else:
        dos_rows_valid = (
            len(observed_rows) == len(set(observed_rows))
            and all(value >= 0 for value in observed_rows)
        )
    checks = {
        "required_columns": not missing_columns,
        "count": expected_count is None or len(frame) == expected_count,
        "unique_nonempty_ids": len(ids) == len(set(ids)) and bool((ids.str.strip() != "").all()),
        "sorted_ids": ids.tolist() == sorted(ids.tolist()),
        "finite_targets": int(np.sum(finite_targets)) == len(frame),
        "category_values": set(category_counts) <= set(DOS_REASONS),
        "category_counts": expected_dos_counts is None or category_counts == dict(expected_dos_counts),
        "dos_rows": dos_rows_valid and missing_dos_rows_are_null,
        "crystal_rows": crystal_rows_match,
        "strict_tier": "quality_tier" in frame and bool((frame["quality_tier"] == "strict_clean").all()),
    }
    complete = all(checks.values())
    return {
        "schema_version": MASTER_SCHEMA_VERSION,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "row_count": int(len(frame)),
        "unique_id_count": int(ids.nunique()),
        "finite_target_count": int(np.sum(finite_targets)),
        "dos_category_counts": {str(key): int(value) for key, value in category_counts.items()},
        "missing_columns": missing_columns,
        "checks": checks,
    }


def materialize_master_table(
    inventory_path: Path | str,
    structure_manifest_path: Path | str,
    dos_ids_path: Path | str,
    output_dir: Path | str,
    *,
    expected_count: int = 10682,
    expected_dos_counts: Mapping[str, int] | None = None,
    max_materials: int | None = None,
) -> dict[str, Any]:
    """Build, revalidate, and atomically publish a master-table directory."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if max_materials is not None and max_materials <= 0:
        raise ValueError("max_materials must be positive")

    full_master = build_master_table(
        inventory_path,
        structure_manifest_path,
        dos_ids_path,
    )
    full_audit = validate_master_table(
        full_master,
        expected_count=expected_count,
        expected_dos_counts=expected_dos_counts,
    )
    if not full_audit["complete"]:
        raise ValueError(f"full master validation failed: {full_audit['checks']}")
    master = (
        full_master.iloc[:max_materials].copy()
        if max_materials is not None
        else full_master
    )
    master["crystal_row"] = np.arange(len(master), dtype=np.int64)
    audit = validate_master_table(
        master,
        expected_count=len(master),
        require_dense_dos_rows=max_materials is None,
    )
    if not audit["complete"]:
        raise ValueError(f"selected master validation failed: {audit['checks']}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        parquet_path = staging / "master_table.parquet"
        csv_path = staging / "master_table.csv"
        audit_path = staging / "master_audit.json"
        master.to_parquet(parquet_path, index=False)
        master.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")

        parquet = pd.read_parquet(parquet_path)
        csv = pd.read_csv(csv_path)
        for column in ("material_id", "dos_missing_reason"):
            if parquet[column].astype(str).tolist() != csv[column].astype(str).tolist():
                raise ValueError(f"CSV/Parquet mismatch in {column}")
        for column in ("k_vrh_gpa", "g_vrh_gpa"):
            if not np.allclose(
                parquet[column].to_numpy(dtype=float),
                csv[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"CSV/Parquet mismatch in {column}")
        disk_audit = validate_master_table(
            parquet,
            expected_count=len(master),
            require_dense_dos_rows=max_materials is None,
        )
        if not disk_audit["complete"]:
            raise ValueError("published Parquet failed master validation")
        _write_json(audit_path, audit)
        output_hashes = {
            name: _sha256_file(staging / name)
            for name in ("master_table.parquet", "master_table.csv", "master_audit.json")
        }
        metadata = {
            "schema_version": MASTER_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "complete": True,
            "full_input_count": int(len(full_master)),
            "row_count": int(len(master)),
            "max_materials": max_materials,
            "source_sha256": {
                "inventory": _sha256_file(inventory_path),
                "structure_manifest": _sha256_file(structure_manifest_path),
                "dos_material_ids": _sha256_file(dos_ids_path),
            },
            "column_dtypes": {name: str(dtype) for name, dtype in master.dtypes.items()},
            "output_file_sha256": output_hashes,
        }
        _write_json(staging / "master_metadata.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "status": "complete",
        "complete": True,
        "row_count": int(len(master)),
        "full_input_count": int(len(full_master)),
        "output_dir": str(destination.resolve()),
    }
