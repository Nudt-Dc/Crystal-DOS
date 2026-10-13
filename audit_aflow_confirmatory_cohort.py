"""Independent disk audit for an AFLOW confirmatory structure + DOS cohort."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import lzma
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from aflow_confirmatory_inventory import sha256_file, validate_data_use_basis


SCHEMA_VERSION = "aflow_confirmatory_cohort_audit_v1"
AUDIT_FIELDS = (
    "position", "auid", "compound", "atom_count", "dos_spin_schema",
    "dos_grid", "aligned_energy_min_ev", "aligned_energy_max_ev",
    "covers_minus10_plus8", "negative_point_count",
    "global_negative_area_fraction", "window_negative_area_fraction",
    "majority_sign_artifact_global_fraction", "majority_sign_artifact_window_fraction",
    "minority_sign_artifact_global_fraction", "minority_sign_artifact_window_fraction",
    "integrated_total_dos_after_clipping", "structure_sha256", "dos_sha256",
)

MAX_GLOBAL_NEGATIVE_AREA_FRACTION = 0.05
MAX_WINDOW_NEGATIVE_AREA_FRACTION = 0.02
MAX_SPIN_CHANNEL_SIGN_ARTIFACT_FRACTION = 0.02


def _jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def ordered_receipts(receipts: list[dict], expected_count: int) -> list[dict]:
    """Rebuild frozen candidate order after append-only retry receipts."""
    try:
        positions = [int(receipt["position"]) for receipt in receipts]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("receipt position is missing or invalid") from exc
    if len(positions) != expected_count or set(positions) != set(range(1, expected_count + 1)):
        raise ValueError("receipt positions are not a unique complete 1-based sequence")
    return sorted(receipts, key=lambda receipt: int(receipt["position"]))


def _finite_vector(value: object, name: str, expected: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {expected}")
    return array


def _audit_dos(path: Path) -> dict:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"DOS_grid", "Efermi", "Emin", "Emax", "species", "tDOS_data"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("DOS payload is missing required fields")
    grid = int(payload["DOS_grid"])
    if grid < 3:
        raise ValueError("DOS grid must contain at least three points")
    total = payload["tDOS_data"]
    if not isinstance(total, dict):
        raise ValueError("tDOS_data must be an object")
    fields = set(total)
    if fields == {"energy", "tDOS", "sum"}:
        schema = "unpolarized"
        density = _finite_vector(total["tDOS"], "tDOS", grid)
        _finite_vector(total["sum"], "sum", grid)
        majority = None
        minority = None
    elif fields == {
        "energy", "spin_majority", "spin_minority",
        "sum_spin_majority", "sum_spin_minority",
    }:
        schema = "spin_polarized"
        majority = _finite_vector(total["spin_majority"], "spin_majority", grid)
        minority = _finite_vector(total["spin_minority"], "spin_minority", grid)
        _finite_vector(total["sum_spin_majority"], "sum_spin_majority", grid)
        _finite_vector(total["sum_spin_minority"], "sum_spin_minority", grid)
        density = majority - minority
    else:
        raise ValueError(f"unsupported AFLOW tDOS schema: {sorted(fields)}")
    energy = _finite_vector(total["energy"], "energy", grid)
    if not np.all(np.diff(energy) > 0):
        raise ValueError("DOS energy grid is not strictly increasing")
    positive = np.clip(density, 0.0, None)
    negative = np.clip(-density, 0.0, None)
    positive_area = float(np.trapezoid(positive, energy))
    negative_area = float(np.trapezoid(negative, energy))
    global_fraction = negative_area / (positive_area + negative_area)
    window_mask = (energy >= -10.0) & (energy <= 8.0)
    if int(window_mask.sum()) < 3:
        raise ValueError("frozen DOS window contains fewer than three source points")
    channel_fractions = {
        "majority_sign_artifact_global_fraction": 0.0,
        "majority_sign_artifact_window_fraction": 0.0,
        "minority_sign_artifact_global_fraction": 0.0,
        "minority_sign_artifact_window_fraction": 0.0,
    }
    if majority is not None and minority is not None:
        def sign_artifact_fractions(channel: np.ndarray, *, expected_positive: bool) -> tuple[float, float]:
            good = np.clip(channel if expected_positive else -channel, 0.0, None)
            bad = np.clip(-channel if expected_positive else channel, 0.0, None)
            global_good = float(np.trapezoid(good, energy))
            global_bad = float(np.trapezoid(bad, energy))
            window_good = float(np.trapezoid(good[window_mask], energy[window_mask]))
            window_bad = float(np.trapezoid(bad[window_mask], energy[window_mask]))
            return (
                global_bad / (global_good + global_bad),
                window_bad / (window_good + window_bad),
            )
        majority_fractions = sign_artifact_fractions(majority, expected_positive=True)
        minority_fractions = sign_artifact_fractions(minority, expected_positive=False)
        channel_fractions = {
            "majority_sign_artifact_global_fraction": majority_fractions[0],
            "majority_sign_artifact_window_fraction": majority_fractions[1],
            "minority_sign_artifact_global_fraction": minority_fractions[0],
            "minority_sign_artifact_window_fraction": minority_fractions[1],
        }
        if max(channel_fractions.values()) > MAX_SPIN_CHANNEL_SIGN_ARTIFACT_FRACTION:
            raise ValueError("spin-channel sign artifact exceeds the frozen 2% quality gate")
    window_positive_area = float(np.trapezoid(positive[window_mask], energy[window_mask]))
    window_negative_area = float(np.trapezoid(negative[window_mask], energy[window_mask]))
    window_fraction = window_negative_area / (window_positive_area + window_negative_area)
    if global_fraction > MAX_GLOBAL_NEGATIVE_AREA_FRACTION:
        raise ValueError("global negative DOS area exceeds the frozen 5% quality gate")
    if window_fraction > MAX_WINDOW_NEGATIVE_AREA_FRACTION:
        raise ValueError("window negative DOS area exceeds the frozen 2% quality gate")
    density = positive
    integral = float(np.trapezoid(density, energy))
    if not math.isfinite(integral) or integral <= 0:
        raise ValueError("physical total DOS integral is not positive and finite")
    efermi = float(payload["Efermi"])
    emin = float(payload["Emin"])
    emax = float(payload["Emax"])
    if not all(math.isfinite(value) for value in (efermi, emin, emax)):
        raise ValueError("DOS energy metadata is not finite")
    # AFLOW generated DOS stores tDOS_data.energy already relative to E_F.
    step_tolerance = max(1e-4, 2.0 * float(np.median(np.diff(energy))))
    if abs(float(energy[0]) - (emin - efermi)) > step_tolerance:
        raise ValueError("DOS energy minimum is not aligned as Emin - Efermi")
    if abs(float(energy[-1]) - (emax - efermi)) > step_tolerance:
        raise ValueError("DOS energy maximum is not aligned as Emax - Efermi")
    return {
        "dos_spin_schema": schema,
        "dos_grid": grid,
        "aligned_energy_min_ev": float(energy[0]),
        "aligned_energy_max_ev": float(energy[-1]),
        "covers_minus10_plus8": bool(energy[0] <= -10.0 and energy[-1] >= 8.0),
        "negative_point_count": int(np.sum(negative > 0.0)),
        "global_negative_area_fraction": global_fraction,
        "window_negative_area_fraction": window_fraction,
        **channel_fractions,
        "integrated_total_dos_after_clipping": integral,
    }


def _audit_structure(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not {"lattice", "species", "atoms"}.issubset(payload):
        raise ValueError("structure payload is missing required fields")
    lattice = np.asarray(payload["lattice"], dtype=np.float64)
    atoms = payload["atoms"]
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("structure lattice is invalid")
    if abs(float(np.linalg.det(lattice))) <= 1e-10:
        raise ValueError("structure lattice is singular")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("structure atoms are invalid")
    for atom in atoms:
        if not isinstance(atom, dict) or not {"name", "position"}.issubset(atom):
            raise ValueError("structure atom is invalid")
        position = np.asarray(atom["position"], dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("structure atom position is invalid")
    return {"atom_count": len(atoms)}


def audit_cohort(
    records_path: Path | str,
    data_use_record: Path | str,
    cohort_dir: Path | str,
    output_dir: Path | str,
    *,
    expected_count: int = 5564,
) -> dict:
    records_file = Path(records_path).resolve()
    basis_file = Path(data_use_record).resolve()
    cohort_root = Path(cohort_dir).resolve()
    destination = Path(output_dir).resolve()
    basis = validate_data_use_basis(basis_file)
    records = _jsonl(records_file)
    receipts_file = cohort_root / "receipts.jsonl"
    raw_receipts = _jsonl(receipts_file)
    if len(records) != expected_count or len(raw_receipts) != expected_count:
        raise ValueError("candidate or receipt count does not match expected_count")
    receipts = ordered_receipts(raw_receipts, expected_count)
    candidate_auids = [str(row.get("auid", "")) for row in records]
    receipt_auids = [str(row.get("auid", "")) for row in receipts]
    if len(set(candidate_auids)) != expected_count or len(set(receipt_auids)) != expected_count:
        raise ValueError("candidate or receipt AUIDs are not unique")
    if receipt_auids != candidate_auids:
        raise ValueError("receipt AUID order differs from frozen candidate order")
    failures_file = cohort_root / "failures.jsonl"
    historical_failures = _jsonl(failures_file) if failures_file.exists() else []
    unresolved_failures = sorted(
        {str(row.get("auid", "")) for row in historical_failures} - set(receipt_auids)
    )
    if unresolved_failures:
        raise ValueError(f"unresolved failed AUIDs remain: {unresolved_failures[:5]}")

    rows: list[dict] = []
    spin_counts = {"unpolarized": 0, "spin_polarized": 0}
    grid_counts: dict[str, int] = {}
    for position, (record, receipt) in enumerate(zip(records, receipts), 1):
        if int(receipt.get("position", -1)) != position:
            raise ValueError(f"receipt position mismatch for {record['auid']}")
        if receipt.get("aurl") != record.get("aurl"):
            raise ValueError(f"receipt AURL mismatch for {record['auid']}")
        record_dir = cohort_root / str(receipt.get("record_dir", ""))
        file_info = receipt.get("files")
        if not isinstance(file_info, dict) or set(file_info) != {"structure", "dos_json"}:
            raise ValueError(f"receipt file map is invalid for {record['auid']}")
        resolved: dict[str, Path] = {}
        for kind, info in file_info.items():
            if not isinstance(info, dict) or not {"filename", "sha256", "bytes", "url"}.issubset(info):
                raise ValueError(f"receipt file entry is invalid for {record['auid']}")
            path = record_dir / str(info["filename"])
            if not path.is_file() or path.stat().st_size != int(info["bytes"]):
                raise ValueError(f"file size mismatch for {record['auid']} {kind}")
            if sha256_file(path) != info["sha256"]:
                raise ValueError(f"file hash mismatch for {record['auid']} {kind}")
            resolved[kind] = path
        structure = _audit_structure(resolved["structure"])
        if structure["atom_count"] != len(record.get("positions_fractional", [])):
            raise ValueError(f"inventory/structure atom-count mismatch for {record['auid']}")
        dos = _audit_dos(resolved["dos_json"])
        stored_format_audit = receipt.get("format_audit", {})
        if stored_format_audit.get("dos_spin_schema") != dos["dos_spin_schema"]:
            raise ValueError(f"stored/recomputed DOS schema mismatch for {record['auid']}")
        if int(stored_format_audit.get("dos_grid", -1)) != dos["dos_grid"]:
            raise ValueError(f"stored/recomputed DOS grid mismatch for {record['auid']}")
        spin_counts[dos["dos_spin_schema"]] += 1
        grid_counts[str(dos["dos_grid"])] = grid_counts.get(str(dos["dos_grid"]), 0) + 1
        rows.append({
            "position": position,
            "auid": record["auid"],
            "compound": record.get("compound", ""),
            "atom_count": structure["atom_count"],
            **dos,
            "structure_sha256": file_info["structure"]["sha256"],
            "dos_sha256": file_info["dos_json"]["sha256"],
        })

    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        manifest_path = staging / "audit_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        coverage_count = sum(bool(row["covers_minus10_plus8"]) for row in rows)
        energy_minima = [float(row["aligned_energy_min_ev"]) for row in rows]
        energy_maxima = [float(row["aligned_energy_max_ev"]) for row in rows]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_count": expected_count,
            "audited_count": len(rows),
            "unique_auid_count": len(set(receipt_auids)),
            "historical_failure_count": len(historical_failures),
            "unresolved_failure_count": 0,
            "data_use_basis_type": basis["basis_type"],
            "spin_schema_counts": spin_counts,
            "dos_grid_counts": grid_counts,
            "strict_window_minus10_plus8_count": coverage_count,
            "strict_window_minus10_plus8_fraction": coverage_count / len(rows),
            "negative_point_record_count": sum(row["negative_point_count"] > 0 for row in rows),
            "maximum_global_negative_area_fraction": max(
                float(row["global_negative_area_fraction"]) for row in rows
            ),
            "maximum_window_negative_area_fraction": max(
                float(row["window_negative_area_fraction"]) for row in rows
            ),
            "maximum_spin_channel_sign_artifact_fraction": max(
                float(row[key])
                for row in rows
                for key in (
                    "majority_sign_artifact_global_fraction",
                    "majority_sign_artifact_window_fraction",
                    "minority_sign_artifact_global_fraction",
                    "minority_sign_artifact_window_fraction",
                )
            ),
            "negative_area_quality_gate": {
                "global_max_fraction": MAX_GLOBAL_NEGATIVE_AREA_FRACTION,
                "window_minus10_plus8_max_fraction": MAX_WINDOW_NEGATIVE_AREA_FRACTION,
                "spin_channel_sign_artifact_max_fraction": MAX_SPIN_CHANNEL_SIGN_ARTIFACT_FRACTION,
                "accepted_values_clipped_to_zero_for_preprocessing": True,
            },
            "aligned_energy_min_ev_range": [min(energy_minima), max(energy_minima)],
            "aligned_energy_max_ev_range": [min(energy_maxima), max(energy_maxima)],
            "input_sha256": {
                "candidate_records.jsonl": sha256_file(records_file),
                "data_use_record": sha256_file(basis_file),
                "receipts.jsonl": sha256_file(receipts_file),
            },
            "raw_vasp_dos_downloaded": False,
            "model_training_performed": False,
            "model_inference_performed": False,
        }
        report = (
            "# AFLOW confirmatory cohort independent audit\n\n"
            f"Status: complete. Recomputed hashes and parsed all {len(rows)} relaxed structures "
            f"and generated total-DOS files. Unresolved failures: 0.\n\n"
            f"Spin schemas: {spin_counts}. DOS grids: {grid_counts}. The frozen "
            f"[-10,+8] eV window is fully covered by {coverage_count}/{len(rows)} records.\n\n"
            "The AFLOW tDOS energy array was verified as already Fermi-aligned "
            "(Emin - Efermi through Emax - Efermi). Spin-polarized physical total DOS uses "
            "spin_majority - spin_minority because AFLOW stores the minority channel negative. "
            "Generated-DOS numerical oscillations pass only when negative spectral weight is at "
            "most 5% globally and 2% inside [-10,+8] eV; accepted negative values are clipped to zero. "
            "Each spin channel additionally has a 2% sign-artifact gate before channel-wise clipping. "
            "No model training or inference was performed.\n"
        )
        report_path = staging / "quality_report.md"
        report_path.write_text(report, encoding="utf-8", newline="\n")
        summary["output_file_sha256"] = {
            "audit_manifest.csv": sha256_file(manifest_path),
            "quality_report.md": sha256_file(report_path),
        }
        (staging / "audit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary
