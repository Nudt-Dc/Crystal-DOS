"""Frozen structure-overlap exclusion for the AFLOW confirmatory cohort."""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from pymatgen.core import Lattice, Structure

from aflow_confirmatory_inventory import sha256_file
from audit_aflow_confirmatory_cohort import ordered_receipts
from jarvis_mp_overlap_audit import (
    default_matcher,
    jarvis_structure,
    load_candidate_records as load_jarvis_candidates,
    load_mp_structures,
)


SCHEMA_VERSION = "aflow_confirmatory_structure_overlap_v1"
FIELDS = (
    "position", "auid", "reduced_formula", "site_count",
    "same_formula_mp_count", "same_formula_jarvis_count",
    "matched_mp_material_ids", "matched_jarvis_jids",
    "matched_aflow_representative_auid", "external_structure_overlap",
    "within_aflow_duplicate", "eligible_external_confirmation",
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return values


def aflow_structure(path: Path | str) -> Structure:
    payload = _json(Path(path))
    if str(payload.get("coordinates_type", "")).lower() not in {"direct", "fractional"}:
        raise ValueError("AFLOW relaxed structure is not in direct coordinates")
    scale = float(payload.get("scale", 1.0))
    lattice = np.asarray(payload.get("lattice"), dtype=np.float64) * scale
    atoms = payload.get("atoms")
    if (
        not np.isfinite(scale)
        or scale <= 0
        or lattice.shape != (3, 3)
        or not np.isfinite(lattice).all()
        or abs(float(np.linalg.det(lattice))) <= 1e-10
        or not isinstance(atoms, list)
        or not atoms
    ):
        raise ValueError("AFLOW relaxed structure geometry is invalid")
    species: list[str] = []
    positions: list[list[float]] = []
    for atom in atoms:
        if float(atom.get("occupancy", 1.0)) != 1.0:
            raise ValueError("disordered AFLOW structure is outside the frozen matcher contract")
        position = np.asarray(atom.get("position"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("AFLOW atom position is invalid")
        species.append(str(atom["name"]))
        positions.append(position.tolist())
    return Structure(
        Lattice(lattice), species, positions,
        coords_are_cartesian=False, to_unit_cell=True,
    )


def _group_structures(values: list[tuple[str, Structure]]) -> dict[str, list[tuple[str, Structure]]]:
    grouped: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    for identifier, structure in values:
        grouped[structure.composition.reduced_formula].append((identifier, structure))
    for group in grouped.values():
        group.sort(key=lambda item: item[0])
    return grouped


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def materialize_overlap(
    records_path: Path | str,
    cohort_dir: Path | str,
    cohort_audit_dir: Path | str,
    mp_master_dir: Path | str,
    mp_snapshot_dir: Path | str,
    jarvis_candidates_path: Path | str,
    frozen_config_path: Path | str,
    output_dir: Path | str,
    *,
    expected_count: int = 5564,
) -> dict:
    records_file = Path(records_path).resolve()
    cohort_root = Path(cohort_dir).resolve()
    audit_root = Path(cohort_audit_dir).resolve()
    master_root = Path(mp_master_dir).resolve()
    snapshot_root = Path(mp_snapshot_dir).resolve()
    jarvis_file = Path(jarvis_candidates_path).resolve()
    config_file = Path(frozen_config_path).resolve()
    destination = Path(output_dir).resolve()
    config = _json(config_file)
    matcher_cfg = config.get("matcher", {})
    expected_matcher = {
        "implementation": "pymatgen.analysis.structure_matcher.StructureMatcher",
        "comparator": "ElementComparator",
        "ltol": 0.2,
        "stol": 0.3,
        "angle_tol_degrees": 5.0,
        "primitive_cell": True,
        "scale": True,
        "attempt_supercell": False,
        "composition_block": "reduced_formula",
    }
    if config.get("schema_version") != "aflow_confirmatory_overlap_reference_v1" or matcher_cfg != expected_matcher:
        raise ValueError("frozen AFLOW overlap matcher configuration drift")
    references = config["reference_corpora"]
    master_metadata_file = master_root / "master_metadata.json"
    master_table_file = master_root / "master_table.csv"
    snapshot_metadata_file = snapshot_root / "snapshot_metadata.json"
    if sha256_file(master_metadata_file) != references["materials_project"]["master_metadata_sha256"]:
        raise ValueError("MP master metadata hash drift")
    if sha256_file(master_table_file) != references["materials_project"]["master_table_sha256"]:
        raise ValueError("MP master table hash drift")
    if sha256_file(snapshot_metadata_file) != references["materials_project"]["snapshot_metadata_sha256"]:
        raise ValueError("MP snapshot metadata hash drift")
    if sha256_file(jarvis_file) != references["jarvis"]["candidate_records_sha256"]:
        raise ValueError("JARVIS pre-overlap candidate hash drift")
    audit_summary_file = audit_root / "audit_summary.json"
    audit_manifest_file = audit_root / "audit_manifest.csv"
    receipts_file = cohort_root / "receipts.jsonl"
    audit = _json(audit_summary_file)
    if (
        audit.get("complete") is not True
        or int(audit.get("audited_count", -1)) != expected_count
        or audit.get("input_sha256", {}).get("candidate_records.jsonl") != sha256_file(records_file)
        or audit.get("input_sha256", {}).get("receipts.jsonl") != sha256_file(receipts_file)
        or audit.get("output_file_sha256", {}).get("audit_manifest.csv") != sha256_file(audit_manifest_file)
    ):
        raise ValueError("AFLOW cohort audit binding is invalid")
    records = _jsonl(records_file)
    raw_receipts = _jsonl(receipts_file)
    if len(records) != expected_count or len(raw_receipts) != expected_count:
        raise ValueError("AFLOW cohort count mismatch")
    receipts = ordered_receipts(raw_receipts, expected_count)
    if [row["auid"] for row in records] != [row["auid"] for row in receipts]:
        raise ValueError("AFLOW receipt order mismatch")
    with master_table_file.open("r", encoding="utf-8", newline="") as handle:
        mp_ids = {str(row["material_id"]) for row in csv.DictReader(handle)}
    if len(mp_ids) != int(references["materials_project"]["material_count"]):
        raise ValueError("MP reference count mismatch")
    mp_structures = load_mp_structures(snapshot_root, mp_ids)
    jarvis_candidates = load_jarvis_candidates(jarvis_file)
    if len(jarvis_candidates) != int(references["jarvis"]["pre_overlap_candidate_count"]):
        raise ValueError("JARVIS reference count mismatch")
    jarvis_structures = [(str(row["jid"]), jarvis_structure(row["atoms"])) for row in jarvis_candidates]
    mp_grouped = _group_structures(list(mp_structures.items()))
    jarvis_grouped = _group_structures(jarvis_structures)
    aflow_values: list[tuple[dict, dict, Structure]] = []
    for record, receipt in zip(records, receipts):
        structure_info = receipt["files"]["structure"]
        structure_path = cohort_root / receipt["record_dir"] / structure_info["filename"]
        if sha256_file(structure_path) != structure_info["sha256"]:
            raise ValueError(f"AFLOW structure hash drift for {record['auid']}")
        aflow_values.append((record, receipt, aflow_structure(structure_path)))

    matcher = default_matcher()
    audit_rows: list[dict] = []
    retained_records: list[dict] = []
    representatives: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    mp_overlap_count = 0
    jarvis_overlap_count = 0
    aflow_duplicate_count = 0
    comparison_counts = {"mp": 0, "jarvis": 0, "within_aflow": 0}
    for position, (record, receipt, structure) in enumerate(aflow_values, 1):
        formula = structure.composition.reduced_formula
        mp_candidates = mp_grouped.get(formula, [])
        jarvis_candidates_same = jarvis_grouped.get(formula, [])
        comparison_counts["mp"] += len(mp_candidates)
        comparison_counts["jarvis"] += len(jarvis_candidates_same)
        matched_mp = [identifier for identifier, other in mp_candidates if matcher.fit(other, structure)]
        matched_jarvis = [identifier for identifier, other in jarvis_candidates_same if matcher.fit(other, structure)]
        external_overlap = bool(matched_mp or matched_jarvis)
        if matched_mp:
            mp_overlap_count += 1
        if matched_jarvis:
            jarvis_overlap_count += 1
        matched_aflow = ""
        if not external_overlap:
            prior = representatives[formula]
            comparison_counts["within_aflow"] += len(prior)
            for representative_auid, representative_structure in prior:
                if matcher.fit(representative_structure, structure):
                    matched_aflow = representative_auid
                    break
            if matched_aflow:
                aflow_duplicate_count += 1
            else:
                prior.append((record["auid"], structure))
        eligible = not external_overlap and not matched_aflow
        audit_rows.append({
            "position": position,
            "auid": record["auid"],
            "reduced_formula": formula,
            "site_count": len(structure),
            "same_formula_mp_count": len(mp_candidates),
            "same_formula_jarvis_count": len(jarvis_candidates_same),
            "matched_mp_material_ids": ";".join(matched_mp),
            "matched_jarvis_jids": ";".join(matched_jarvis),
            "matched_aflow_representative_auid": matched_aflow,
            "external_structure_overlap": external_overlap,
            "within_aflow_duplicate": bool(matched_aflow),
            "eligible_external_confirmation": eligible,
        })
        if eligible:
            retained_records.append({
                **record,
                "cohort_record_dir": receipt["record_dir"],
                "structure_filename": receipt["files"]["structure"]["filename"],
                "dos_filename": receipt["files"]["dos_json"]["filename"],
                "reduced_formula": formula,
            })

    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite overlap output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        manifest_path = staging / "overlap_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(audit_rows)
        retained_path = staging / "external_confirmation_records.jsonl.gz"
        _write_jsonl_gz(retained_path, retained_records)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "candidate_count": len(records),
            "mp_reference_count": len(mp_structures),
            "jarvis_reference_count": len(jarvis_structures),
            "mp_overlap_count": mp_overlap_count,
            "jarvis_overlap_count": jarvis_overlap_count,
            "external_union_overlap_count": sum(bool(row["external_structure_overlap"]) for row in audit_rows),
            "within_aflow_duplicate_count": aflow_duplicate_count,
            "external_confirmation_count": len(retained_records),
            "structure_comparison_counts": comparison_counts,
            "matcher": matcher_cfg,
            "representative_policy": "frozen candidate-record order within reduced formula after external exclusion",
            "target_values_used_for_filtering": False,
            "model_predictions_used_for_filtering": False,
            "model_training_performed": False,
            "model_inference_performed": False,
            "input_sha256": {
                "candidate_records.jsonl": sha256_file(records_file),
                "receipts.jsonl": sha256_file(receipts_file),
                "cohort_audit_summary.json": sha256_file(audit_summary_file),
                "cohort_audit_manifest.csv": sha256_file(audit_manifest_file),
                "frozen_overlap_config.json": sha256_file(config_file),
                "mp_master_metadata.json": sha256_file(master_metadata_file),
                "mp_master_table.csv": sha256_file(master_table_file),
                "mp_snapshot_metadata.json": sha256_file(snapshot_metadata_file),
                "jarvis_candidate_records.jsonl.gz": sha256_file(jarvis_file),
            },
            "mp_snapshot_tree_content_sha256": _json(snapshot_metadata_file).get("tree_content_sha256"),
        }
        report = (
            "# AFLOW external-confirmation structure-overlap audit\n\n"
            f"Compared {len(records)} AFLOW candidates against {len(mp_structures)} Materials "
            f"Project structures and all {len(jarvis_structures)} pre-overlap JARVIS candidates. "
            f"External union overlap excluded {summary['external_union_overlap_count']} records; "
            f"within-AFLOW structure deduplication excluded {aflow_duplicate_count}; "
            f"{len(retained_records)} independent records remain.\n\n"
            "All comparisons used the pre-frozen reduced-formula blocking and StructureMatcher "
            "configuration. No targets or model predictions were used for filtering.\n"
        )
        report_path = staging / "quality_report.md"
        report_path.write_text(report, encoding="utf-8", newline="\n")
        summary["output_file_sha256"] = {
            "overlap_manifest.csv": sha256_file(manifest_path),
            "external_confirmation_records.jsonl.gz": sha256_file(retained_path),
            "quality_report.md": sha256_file(report_path),
        }
        (staging / "overlap_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary
