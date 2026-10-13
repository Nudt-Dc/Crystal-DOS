"""Remove Materials Project structural overlap from JARVIS candidates."""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Lattice, Structure


SCHEMA_VERSION = "jarvis_mp_structure_overlap_audit_v1"
OVERLAP_FIELDS = (
    "jid", "reference", "reduced_formula", "site_count",
    "same_formula_mp_count", "structure_comparison_count",
    "mp_structure_overlap", "matched_mp_material_ids",
    "k_vrh_gpa", "g_vrh_gpa", "eligible_external_confirmation",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_candidate_records(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid candidate JSONL line {line_number}") from exc
            if not isinstance(value, dict) or not str(value.get("jid", "")):
                raise ValueError(f"invalid candidate record at line {line_number}")
            rows.append(value)
    identifiers = [str(row["jid"]) for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate JIDs must be nonempty and unique")
    return rows


def jarvis_structure(atoms: dict) -> Structure:
    required = {"lattice_mat", "coords", "elements", "cartesian"}
    if not isinstance(atoms, dict) or not required.issubset(atoms):
        raise ValueError("invalid JARVIS atoms payload")
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    coords = np.asarray(atoms["coords"], dtype=np.float64)
    elements = [str(value) for value in atoms["elements"]]
    if (
        lattice.shape != (3, 3)
        or not np.isfinite(lattice).all()
        or abs(np.linalg.det(lattice)) <= 1e-12
        or coords.shape != (len(elements), 3)
        or not elements
        or not np.isfinite(coords).all()
        or not isinstance(atoms["cartesian"], bool)
    ):
        raise ValueError("JARVIS atoms geometry is invalid")
    return Structure(
        Lattice(lattice), elements, coords,
        coords_are_cartesian=bool(atoms["cartesian"]), to_unit_cell=True,
    )


def load_mp_structures(
    snapshot_dir: Path | str, wanted_ids: set[str]
) -> dict[str, Structure]:
    root = Path(snapshot_dir)
    result: dict[str, Structure] = {}
    for shard in sorted((root / "shards").glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                material_id = str(row.get("material_id", ""))
                if material_id not in wanted_ids:
                    continue
                if material_id in result:
                    raise ValueError(f"duplicate MP structure: {material_id}")
                result[material_id] = Structure.from_dict(row["structure"])
    missing = sorted(wanted_ids - set(result))
    if missing:
        raise ValueError(f"missing MP structures: {missing[:5]}")
    return result


def default_matcher() -> StructureMatcher:
    return StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        comparator=ElementComparator(),
    )


def audit_overlap(
    candidates: list[dict],
    mp_structures: dict[str, Structure],
    *,
    matcher: StructureMatcher | None = None,
) -> tuple[list[dict], list[dict]]:
    selected_matcher = matcher or default_matcher()
    grouped: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    for material_id, structure in mp_structures.items():
        grouped[structure.composition.reduced_formula].append((material_id, structure))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    audit_rows: list[dict] = []
    retained: list[dict] = []
    for record in sorted(candidates, key=lambda row: str(row["jid"])):
        structure = jarvis_structure(record["atoms"])
        formula = structure.composition.reduced_formula
        comparisons = grouped.get(formula, [])
        matched = [
            material_id
            for material_id, mp_structure in comparisons
            if selected_matcher.fit(mp_structure, structure)
        ]
        overlaps = bool(matched)
        audit_rows.append({
            "jid": record["jid"],
            "reference": record.get("reference", ""),
            "reduced_formula": formula,
            "site_count": len(structure),
            "same_formula_mp_count": len(comparisons),
            "structure_comparison_count": len(comparisons),
            "mp_structure_overlap": overlaps,
            "matched_mp_material_ids": ";".join(matched),
            "k_vrh_gpa": record["k_vrh_gpa"],
            "g_vrh_gpa": record["g_vrh_gpa"],
            "eligible_external_confirmation": not overlaps,
        })
        if not overlaps:
            retained.append(record)
    return audit_rows, retained


def _write_deterministic_jsonl_gz(path: Path, records: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def materialize_overlap_audit(
    inventory_dir: Path | str,
    mp_master_dir: Path | str,
    mp_snapshot_dir: Path | str,
    output_dir: Path | str,
) -> dict:
    inventory_root = Path(inventory_dir).resolve()
    master_root = Path(mp_master_dir).resolve()
    snapshot_root = Path(mp_snapshot_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite overlap audit: {destination}")
    inventory_summary_path = inventory_root / "inventory_summary.json"
    candidate_path = inventory_root / "candidate_records.jsonl.gz"
    inventory = load_json(inventory_summary_path)
    if (
        inventory.get("schema_version") != "jarvis_external_inventory_v1"
        or inventory.get("complete") is not True
        or inventory.get("model_inference_performed") is not False
        or inventory.get("output_file_sha256", {}).get(candidate_path.name)
        != sha256_file(candidate_path)
    ):
        raise ValueError("candidate inventory binding is invalid")
    master_metadata_path = master_root / "master_metadata.json"
    master_table_path = master_root / "master_table.csv"
    master = load_json(master_metadata_path)
    if (
        master.get("schema_version") != "crystal_dos_master_v1"
        or master.get("complete") is not True
        or master.get("output_file_sha256", {}).get(master_table_path.name)
        != sha256_file(master_table_path)
    ):
        raise ValueError("MP master binding is invalid")
    snapshot_metadata_path = snapshot_root / "snapshot_metadata.json"
    snapshot = load_json(snapshot_metadata_path)
    if snapshot.get("complete") is not True or snapshot.get("status") != "complete":
        raise ValueError("MP structure snapshot is incomplete")
    with master_table_path.open("r", encoding="utf-8", newline="") as handle:
        wanted_ids = {str(row["material_id"]) for row in csv.DictReader(handle)}
    if len(wanted_ids) != int(master["row_count"]):
        raise ValueError("MP master material IDs are not unique")
    candidates = load_candidate_records(candidate_path)
    if len(candidates) != int(inventory["eligible_pre_overlap_count"]):
        raise ValueError("candidate count does not match inventory summary")
    mp_structures = load_mp_structures(snapshot_root, wanted_ids)
    audit_rows, retained = audit_overlap(candidates, mp_structures)
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        with (staging / "overlap_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=OVERLAP_FIELDS)
            writer.writeheader()
            writer.writerows(audit_rows)
        _write_deterministic_jsonl_gz(
            staging / "external_confirmation_records.jsonl.gz", retained
        )
        overlap_count = len(audit_rows) - len(retained)
        report = (
            "# JARVIS–MP structural-overlap audit\n\n"
            f"Compared {len(audit_rows)} pre-overlap JARVIS candidates against all "
            f"{len(mp_structures)} structures in the exact Materials Project development corpus. "
            f"StructureMatcher excluded {overlap_count} equivalent structures and retained "
            f"{len(retained)} external-confirmation candidates.\n\n"
            "Matching used reduced-composition blocking followed by Pymatgen StructureMatcher "
            "with ltol=0.2, stol=0.3, angle_tol=5 degrees, primitive-cell reduction, scaling, "
            "and element identity. No target values or model predictions were used for filtering.\n"
        )
        (staging / "quality_report.md").write_text(report, encoding="utf-8")
        output_hashes = {
            name: sha256_file(staging / name)
            for name in (
                "overlap_manifest.csv",
                "external_confirmation_records.jsonl.gz",
                "quality_report.md",
            )
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "candidate_count": len(audit_rows),
            "mp_development_structure_count": len(mp_structures),
            "mp_structure_overlap_count": overlap_count,
            "external_confirmation_count": len(retained),
            "matcher": {
                "implementation": "pymatgen.analysis.structure_matcher.StructureMatcher",
                "ltol": 0.2,
                "stol": 0.3,
                "angle_tol_degrees": 5.0,
                "primitive_cell": True,
                "scale": True,
                "attempt_supercell": False,
                "comparator": "ElementComparator",
            },
            "input_sha256": {
                "inventory_summary.json": sha256_file(inventory_summary_path),
                "candidate_records.jsonl.gz": sha256_file(candidate_path),
                "master_metadata.json": sha256_file(master_metadata_path),
                "master_table.csv": sha256_file(master_table_path),
                "snapshot_metadata.json": sha256_file(snapshot_metadata_path),
            },
            "mp_snapshot_tree_content_sha256": snapshot.get("tree_content_sha256"),
            "output_file_sha256": output_hashes,
            "target_values_used_for_filtering": False,
            "model_inference_performed": False,
        }
        (staging / "overlap_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary

