"""Materialize label-quality variants from coverage-valid DOS features."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dos_preprocess import _file_sha256, build_evaluation_masks


SOURCE_SPEC_VERSION = "dos-full-preprocessing-v2-strict-window"


def _has_warning(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "[]"}
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def materialize_labeled_dos_dataset(
    preprocessed_dir: Path | str,
    labels_csv: Path | str,
    output_dir: Path | str,
    *,
    policy: str,
    spec_version: str,
    missing_rates: Sequence[float],
    repeats: int,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an ordered labeled DOS subset and explicit exclusion manifest."""
    if policy not in {"strict_clean", "broad_labeled"}:
        raise ValueError("policy must be 'strict_clean' or 'broad_labeled'")
    source = Path(preprocessed_dir)
    labels_path = Path(labels_csv)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")

    source_summary = json.loads(
        (source / "quality_summary.json").read_text(encoding="utf-8")
    )
    source_config = json.loads(
        (source / "preprocessing_config.json").read_text(encoding="utf-8")
    )
    if source_summary.get("status") != "complete":
        raise ValueError("source preprocessed dataset must be complete")
    if source_config.get("spec_version") != SOURCE_SPEC_VERSION:
        raise ValueError("unsupported source preprocessing specification")

    material_ids = np.load(source / "material_ids.npy", allow_pickle=False).astype(str)
    features = np.load(
        source / "dos_features.npy", mmap_mode="r", allow_pickle=False
    )
    energy_grid = np.load(source / "energy_grid_ev.npy", allow_pickle=False)
    if features.shape != (len(material_ids), 1, len(energy_grid)):
        raise ValueError("source DOS feature shape is inconsistent")
    if len(set(material_ids.tolist())) != len(material_ids):
        raise ValueError("source material_ids must be unique")

    labels = pd.read_csv(labels_path)
    required = {
        "material_id",
        "k_vrh_gpa",
        "g_vrh_gpa",
        "deprecated",
        "elasticity_warnings",
    }
    missing_columns = sorted(required - set(labels.columns))
    if missing_columns:
        raise ValueError(f"label table is missing columns: {missing_columns}")
    labels["material_id"] = labels["material_id"].astype(str)
    if not labels["material_id"].is_unique:
        raise ValueError("label table material_id must be unique")
    aligned = labels.set_index("material_id").reindex(material_ids)

    records: list[dict[str, Any]] = []
    included_indices: list[int] = []
    for source_index, material_id in enumerate(material_ids):
        row = aligned.iloc[source_index]
        reasons: list[str] = []
        if pd.isna(row["k_vrh_gpa"]) or pd.isna(row["g_vrh_gpa"]):
            reasons.append("missing_k_or_g")
        if policy == "strict_clean":
            if bool(row["deprecated"]) if not pd.isna(row["deprecated"]) else False:
                reasons.append("deprecated")
            if _has_warning(row["elasticity_warnings"]):
                reasons.append("elasticity_warning")
        included = not reasons
        if included:
            included_indices.append(source_index)
        records.append(
            {
                "source_row": source_index,
                "output_row": len(included_indices) - 1 if included else None,
                "material_id": str(material_id),
                "k_vrh_gpa": row["k_vrh_gpa"],
                "g_vrh_gpa": row["g_vrh_gpa"],
                "deprecated": row["deprecated"],
                "has_elasticity_warning": _has_warning(
                    row["elasticity_warnings"]
                ),
                "status": "included" if included else "excluded",
                "exclusion_reasons": ";".join(reasons) if reasons else None,
            }
        )

    manifest = pd.DataFrame(records)
    included_features = np.asarray(features[included_indices], dtype=np.float32)
    included_ids = material_ids[included_indices]
    masks = build_evaluation_masks(
        len(included_ids), missing_rates, repeats, base_seed
    )

    destination.mkdir(parents=True)
    np.save(destination / "energy_grid_ev.npy", energy_grid.astype(np.float32))
    np.save(destination / "dos_features.npy", included_features)
    np.save(destination / "dos_observed_mask.npy", np.ones(len(included_ids), dtype=np.uint8))
    np.save(destination / "material_ids.npy", included_ids)
    np.savez_compressed(destination / "evaluation_missingness_masks.npz", **masks)
    manifest.to_csv(destination / "selection_manifest.csv", index=False, encoding="utf-8")
    manifest.loc[manifest["status"] == "excluded"].to_csv(
        destination / "excluded_materials.csv", index=False, encoding="utf-8"
    )
    config = {
        "spec_version": spec_version,
        "selection_policy": policy,
        "point_count": int(len(energy_grid)),
        "energy_min_ev": float(energy_grid[0]),
        "energy_max_ev": float(energy_grid[-1]),
        "missing_rates": [float(value) for value in missing_rates],
        "repeats": repeats,
        "base_seed": base_seed,
        "source_spec_version": source_config["spec_version"],
        "source_material_ids_sha256": _file_sha256(source / "material_ids.npy"),
        "labels_sha256": _file_sha256(labels_path),
    }
    _write_json(destination / "preprocessing_config.json", config)

    output_names = [
        "energy_grid_ev.npy",
        "dos_features.npy",
        "dos_observed_mask.npy",
        "material_ids.npy",
        "evaluation_missingness_masks.npz",
        "selection_manifest.csv",
        "excluded_materials.csv",
        "preprocessing_config.json",
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_count": int(len(material_ids)),
        "included_count": int(len(included_ids)),
        "excluded_count": int(len(material_ids) - len(included_ids)),
        "feature_shape": list(included_features.shape),
        "feature_dtype": str(included_features.dtype),
        "missing_counts": {
            key: np.sum(value == 0, axis=1).astype(int).tolist()
            for key, value in masks.items()
        },
        "output_file_sha256": {
            name: _file_sha256(destination / name) for name in output_names
        },
    }
    _write_json(destination / "quality_summary.json", summary)
    return manifest, summary
