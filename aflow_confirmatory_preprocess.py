"""Deterministic AFLOW total-DOS preprocessing on the frozen MP energy grid."""

from __future__ import annotations

import csv
import json
import lzma
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from aflow_confirmatory_inventory import sha256_file
from audit_aflow_confirmatory_cohort import (
    MAX_GLOBAL_NEGATIVE_AREA_FRACTION,
    MAX_SPIN_CHANNEL_SIGN_ARTIFACT_FRACTION,
    MAX_WINDOW_NEGATIVE_AREA_FRACTION,
    ordered_receipts,
)


SCHEMA_VERSION = "aflow_confirmatory_dos_preprocessed_v1"
FIELDS = (
    "position", "auid", "compound", "k_vrh_gpa", "g_vrh_gpa",
    "dos_spin_schema", "source_grid", "aligned_energy_min_ev",
    "aligned_energy_max_ev", "pre_norm_area", "record_dir",
    "structure_filename", "dos_filename", "structure_sha256", "dos_sha256",
)


def _jsonl(path: Path) -> list[dict]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"non-object JSONL row in {path}")
    return values


def _physical_total_dos(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    total = payload["tDOS_data"]
    energy = np.asarray(total["energy"], dtype=np.float64)
    fields = set(total)
    if fields == {"energy", "tDOS", "sum"}:
        density = np.asarray(total["tDOS"], dtype=np.float64)
        schema = "unpolarized"
    elif fields == {
        "energy", "spin_majority", "spin_minority",
        "sum_spin_majority", "sum_spin_minority",
    }:
        majority = np.asarray(total["spin_majority"], dtype=np.float64)
        minority = np.asarray(total["spin_minority"], dtype=np.float64)
        density = majority - minority
        schema = "spin_polarized"
    else:
        raise ValueError(f"unsupported AFLOW tDOS schema: {sorted(fields)}")
    if (
        energy.ndim != 1
        or density.shape != energy.shape
        or len(energy) < 3
        or not np.isfinite(energy).all()
        or not np.isfinite(density).all()
        or not np.all(np.diff(energy) > 0)
    ):
        raise ValueError("invalid AFLOW physical total DOS")
    positive = np.clip(density, 0.0, None)
    negative = np.clip(-density, 0.0, None)
    global_positive = float(np.trapezoid(positive, energy))
    global_negative = float(np.trapezoid(negative, energy))
    global_fraction = global_negative / (global_positive + global_negative)
    window = (energy >= -10.0) & (energy <= 8.0)
    if int(window.sum()) < 3:
        raise ValueError("frozen energy window contains fewer than three source points")
    if schema == "spin_polarized":
        def sign_artifact_fraction(channel: np.ndarray, *, expected_positive: bool, mask: np.ndarray) -> float:
            selected = channel[mask]
            selected_energy = energy[mask]
            good = np.clip(selected if expected_positive else -selected, 0.0, None)
            bad = np.clip(-selected if expected_positive else selected, 0.0, None)
            good_area = float(np.trapezoid(good, selected_energy))
            bad_area = float(np.trapezoid(bad, selected_energy))
            return bad_area / (good_area + bad_area)
        full = np.ones(len(energy), dtype=bool)
        channel_fractions = (
            sign_artifact_fraction(majority, expected_positive=True, mask=full),
            sign_artifact_fraction(majority, expected_positive=True, mask=window),
            sign_artifact_fraction(minority, expected_positive=False, mask=full),
            sign_artifact_fraction(minority, expected_positive=False, mask=window),
        )
        if max(channel_fractions) > MAX_SPIN_CHANNEL_SIGN_ARTIFACT_FRACTION:
            raise ValueError("AFLOW spin channel exceeds the frozen sign-artifact quality gate")
        density = np.clip(majority, 0.0, None) - np.clip(minority, None, 0.0)
        positive = density
        negative = np.zeros_like(density)
        global_positive = float(np.trapezoid(positive, energy))
        global_negative = 0.0
        global_fraction = 0.0
    window_positive = float(np.trapezoid(positive[window], energy[window]))
    window_negative = float(np.trapezoid(negative[window], energy[window]))
    window_fraction = window_negative / (window_positive + window_negative)
    if (
        global_fraction > MAX_GLOBAL_NEGATIVE_AREA_FRACTION
        or window_fraction > MAX_WINDOW_NEGATIVE_AREA_FRACTION
    ):
        raise ValueError("AFLOW DOS exceeds the frozen negative-area quality gate")
    return energy, positive, schema


def materialize_preprocessed(
    records_path: Path | str,
    cohort_dir: Path | str,
    cohort_audit_dir: Path | str,
    output_dir: Path | str,
    *,
    expected_count: int = 5564,
    energy_min_ev: float = -10.0,
    energy_max_ev: float = 8.0,
    point_count: int = 576,
) -> dict:
    records_file = Path(records_path).resolve()
    cohort_root = Path(cohort_dir).resolve()
    audit_root = Path(cohort_audit_dir).resolve()
    destination = Path(output_dir).resolve()
    if not energy_min_ev < energy_max_ev or point_count < 3:
        raise ValueError("invalid target energy grid")
    receipts_file = cohort_root / "receipts.jsonl"
    audit_summary_file = audit_root / "audit_summary.json"
    audit_manifest_file = audit_root / "audit_manifest.csv"
    audit = json.loads(audit_summary_file.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != "aflow_confirmatory_cohort_audit_v1"
        or audit.get("complete") is not True
        or int(audit.get("audited_count", -1)) != expected_count
        or audit.get("input_sha256", {}).get("candidate_records.jsonl")
        != sha256_file(records_file)
        or audit.get("input_sha256", {}).get("receipts.jsonl")
        != sha256_file(receipts_file)
        or audit.get("output_file_sha256", {}).get("audit_manifest.csv")
        != sha256_file(audit_manifest_file)
    ):
        raise ValueError("AFLOW cohort audit binding is invalid")
    records = _jsonl(records_file)
    raw_receipts = _jsonl(receipts_file)
    if len(records) != expected_count or len(raw_receipts) != expected_count:
        raise ValueError("AFLOW record count mismatch")
    receipts = ordered_receipts(raw_receipts, expected_count)
    if [row["auid"] for row in records] != [row["auid"] for row in receipts]:
        raise ValueError("AFLOW receipt order mismatch")
    energy_grid = np.linspace(energy_min_ev, energy_max_ev, point_count, dtype=np.float64)
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite preprocessing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    excluded: list[dict] = []
    features: list[np.ndarray] = []
    targets: list[list[float]] = []
    auids: list[str] = []
    try:
        staging.mkdir()
        for position, (record, receipt) in enumerate(zip(records, receipts), 1):
            record_dir = cohort_root / receipt["record_dir"]
            dos_info = receipt["files"]["dos_json"]
            structure_info = receipt["files"]["structure"]
            dos_path = record_dir / dos_info["filename"]
            energy, density, schema = _physical_total_dos(dos_path)
            if energy[0] > energy_min_ev or energy[-1] < energy_max_ev:
                excluded.append({
                    "position": position,
                    "auid": record["auid"],
                    "reason": "source_dos_does_not_cover_frozen_window",
                    "aligned_energy_min_ev": float(energy[0]),
                    "aligned_energy_max_ev": float(energy[-1]),
                })
                continue
            interpolated = np.interp(energy_grid, energy, density)
            area = float(np.trapezoid(interpolated, energy_grid))
            if not np.isfinite(area) or area <= 0:
                raise ValueError(f"nonpositive interpolated DOS area for {record['auid']}")
            normalized = (interpolated / area).astype(np.float32)
            k_value = float(record["ael_bulk_modulus_vrh"])
            g_value = float(record["ael_shear_modulus_vrh"])
            if not np.isfinite([k_value, g_value]).all() or min(k_value, g_value) <= 0:
                raise ValueError(f"invalid K/G targets for {record['auid']}")
            auids.append(record["auid"])
            features.append(normalized[None, :])
            targets.append([k_value, g_value])
            rows.append({
                "position": position,
                "auid": record["auid"],
                "compound": record.get("compound", ""),
                "k_vrh_gpa": k_value,
                "g_vrh_gpa": g_value,
                "dos_spin_schema": schema,
                "source_grid": len(energy),
                "aligned_energy_min_ev": float(energy[0]),
                "aligned_energy_max_ev": float(energy[-1]),
                "pre_norm_area": area,
                "record_dir": receipt["record_dir"],
                "structure_filename": structure_info["filename"],
                "dos_filename": dos_info["filename"],
                "structure_sha256": structure_info["sha256"],
                "dos_sha256": dos_info["sha256"],
            })
        if not rows:
            raise ValueError("no AFLOW records cover the frozen energy window")
        feature_array = np.stack(features).astype(np.float32, copy=False)
        target_array = np.asarray(targets, dtype=np.float32)
        np.save(staging / "auids.npy", np.asarray(auids))
        np.save(staging / "energy_grid_ev.npy", energy_grid.astype(np.float32))
        np.save(staging / "dos_features.npy", feature_array)
        np.save(staging / "targets_gpa.npy", target_array)
        with (staging / "preprocessing_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with (staging / "exclusions.csv").open("w", encoding="utf-8", newline="") as handle:
            exclusion_fields = (
                "position", "auid", "reason", "aligned_energy_min_ev",
                "aligned_energy_max_ev",
            )
            writer = csv.DictWriter(handle, fieldnames=exclusion_fields)
            writer.writeheader()
            writer.writerows(excluded)
        integrals = np.trapezoid(feature_array[:, 0, :], energy_grid, axis=1)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "input_count": expected_count,
            "retained_count": len(rows),
            "excluded_window_count": len(excluded),
            "energy_min_ev": energy_min_ev,
            "energy_max_ev": energy_max_ev,
            "point_count": point_count,
            "feature_shape": list(feature_array.shape),
            "target_shape": list(target_array.shape),
            "integral_max_abs_error": float(np.max(np.abs(integrals - 1.0))),
            "spin_schema_counts": {
                name: sum(row["dos_spin_schema"] == name for row in rows)
                for name in ("unpolarized", "spin_polarized")
            },
            "input_sha256": {
                "candidate_records.jsonl": sha256_file(records_file),
                "receipts.jsonl": sha256_file(receipts_file),
                "cohort_audit_summary.json": sha256_file(audit_summary_file),
                "cohort_audit_manifest.csv": sha256_file(audit_manifest_file),
            },
            "fermi_alignment": "AFLOW tDOS_data.energy already equals raw energy minus Efermi",
            "spin_combination": "spin_majority - spin_minority",
            "negative_area_quality_gate": {
                "global_max_fraction": MAX_GLOBAL_NEGATIVE_AREA_FRACTION,
                "window_minus10_plus8_max_fraction": MAX_WINDOW_NEGATIVE_AREA_FRACTION,
                "spin_channel_sign_artifact_max_fraction": MAX_SPIN_CHANNEL_SIGN_ARTIFACT_FRACTION,
                "accepted_values_clipped_to_zero": True,
            },
            "normalization": "trapezoidal unit-area normalization after linear interpolation",
            "model_training_performed": False,
            "model_inference_performed": False,
        }
        output_names = (
            "auids.npy", "energy_grid_ev.npy", "dos_features.npy", "targets_gpa.npy",
            "preprocessing_manifest.csv", "exclusions.csv",
        )
        summary["output_file_sha256"] = {
            name: sha256_file(staging / name) for name in output_names
        }
        (staging / "preprocessing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary
