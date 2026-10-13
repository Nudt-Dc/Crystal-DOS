"""Publish the structurally independent AFLOW confirmatory master table."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from aflow_confirmatory_inventory import sha256_file


SCHEMA_VERSION = "aflow_confirmatory_master_v1"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("retained AFLOW record file is invalid or empty")
    return rows


def materialize_master(
    overlap_dir: Path | str,
    preprocessed_dir: Path | str,
    output_dir: Path | str,
) -> dict:
    overlap_root = Path(overlap_dir).resolve()
    preprocessed_root = Path(preprocessed_dir).resolve()
    destination = Path(output_dir).resolve()
    overlap_summary_file = overlap_root / "overlap_summary.json"
    retained_file = overlap_root / "external_confirmation_records.jsonl.gz"
    overlap = _json(overlap_summary_file)
    if (
        overlap.get("schema_version") != "aflow_confirmatory_structure_overlap_v1"
        or overlap.get("complete") is not True
        or overlap.get("output_file_sha256", {}).get(retained_file.name)
        != sha256_file(retained_file)
        or overlap.get("target_values_used_for_filtering") is not False
        or overlap.get("model_predictions_used_for_filtering") is not False
    ):
        raise ValueError("AFLOW overlap binding is invalid")
    preprocessing_summary_file = preprocessed_root / "preprocessing_summary.json"
    preprocessing_manifest_file = preprocessed_root / "preprocessing_manifest.csv"
    auids_file = preprocessed_root / "auids.npy"
    targets_file = preprocessed_root / "targets_gpa.npy"
    preprocessing = _json(preprocessing_summary_file)
    for name, path in (
        ("preprocessing_manifest.csv", preprocessing_manifest_file),
        ("auids.npy", auids_file),
        ("targets_gpa.npy", targets_file),
    ):
        if preprocessing.get("output_file_sha256", {}).get(name) != sha256_file(path):
            raise ValueError(f"AFLOW preprocessing hash drift: {name}")
    if preprocessing.get("complete") is not True or int(preprocessing.get("retained_count", -1)) != 5564:
        raise ValueError("AFLOW preprocessing is incomplete")
    all_auids = np.load(auids_file, allow_pickle=False).astype(str)
    targets = np.load(targets_file, allow_pickle=False)
    manifest = pd.read_csv(preprocessing_manifest_file)
    if (
        targets.shape != (len(all_auids), 2)
        or len(manifest) != len(all_auids)
        or manifest["auid"].astype(str).tolist() != all_auids.tolist()
        or len(set(all_auids.tolist())) != len(all_auids)
    ):
        raise ValueError("AFLOW preprocessing row contract is invalid")
    dos_row_by_auid = {auid: row for row, auid in enumerate(all_auids.tolist())}
    manifest_by_auid = manifest.set_index(manifest["auid"].astype(str), drop=False)
    retained = _jsonl_gz(retained_file)
    if len(retained) != int(overlap["external_confirmation_count"]):
        raise ValueError("AFLOW retained record count mismatch")
    retained_auids = [str(row["auid"]) for row in retained]
    if len(set(retained_auids)) != len(retained_auids) or not set(retained_auids) <= set(all_auids):
        raise ValueError("AFLOW retained AUID membership is invalid")
    rows: list[dict] = []
    for crystal_row, record in enumerate(sorted(retained, key=lambda row: str(row["auid"]))):
        auid = str(record["auid"])
        dos_row = dos_row_by_auid[auid]
        source = manifest_by_auid.loc[auid]
        source_target = np.asarray(
            [float(record["ael_bulk_modulus_vrh"]), float(record["ael_shear_modulus_vrh"])],
            dtype=targets.dtype,
        )
        if not np.array_equal(targets[dos_row], source_target):
            raise ValueError(f"AFLOW target mismatch for {auid}")
        rows.append({
            "material_id": auid,
            "auid": auid,
            "crystal_row": crystal_row,
            "dos_row": dos_row,
            "dos_observed": True,
            "k_vrh_gpa": float(targets[dos_row, 0]),
            "g_vrh_gpa": float(targets[dos_row, 1]),
            "compound": str(record.get("compound", "")),
            "reduced_formula": str(record["reduced_formula"]),
            "spacegroup_relax": int(record.get("spacegroup_relax", -1)),
            "pearson_symbol_relax": str(record.get("Pearson_symbol_relax", "")),
            "cohort_record_dir": str(record["cohort_record_dir"]),
            "structure_filename": str(record["structure_filename"]),
            "dos_filename": str(record["dos_filename"]),
            "structure_sha256": str(source["structure_sha256"]),
            "dos_sha256": str(source["dos_sha256"]),
            "quality_tier": "aflow_confirmatory_overlap_clean",
        })
    table = pd.DataFrame(rows)
    if table["material_id"].tolist() != sorted(table["material_id"].tolist()):
        raise ValueError("AFLOW master IDs are not sorted")
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite AFLOW master: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        parquet_path = staging / "master_table.parquet"
        csv_path = staging / "master_table.csv"
        table.to_parquet(parquet_path, index=False)
        table.to_csv(csv_path, index=False, lineterminator="\n")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "row_count": len(table),
            "unique_auid_count": int(table["auid"].nunique()),
            "dos_observed_count": int(table["dos_observed"].sum()),
            "source_sha256": {
                "overlap_summary.json": sha256_file(overlap_summary_file),
                "external_confirmation_records.jsonl.gz": sha256_file(retained_file),
                "preprocessing_summary.json": sha256_file(preprocessing_summary_file),
                "preprocessing_manifest.csv": sha256_file(preprocessing_manifest_file),
                "auids.npy": sha256_file(auids_file),
                "targets_gpa.npy": sha256_file(targets_file),
            },
            "output_file_sha256": {
                "master_table.parquet": sha256_file(parquet_path),
                "master_table.csv": sha256_file(csv_path),
            },
            "target_values_used_for_structure_grouping": False,
            "model_training_performed": False,
            "model_inference_performed": False,
        }
        (staging / "master_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata
