"""Deterministic total-DOS preprocessing for model-ready fixed-grid inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DosPreprocessConfig:
    """Numerical contract for one versioned total-DOS representation."""

    energy_min_ev: float = -10.0
    energy_max_ev: float = 8.0
    point_count: int = 576
    negative_tolerance: float = 1e-8
    min_area: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite([self.energy_min_ev, self.energy_max_ev]).all():
            raise ValueError("energy window endpoints must be finite")
        if self.energy_min_ev >= self.energy_max_ev:
            raise ValueError("energy_min_ev must be below energy_max_ev")
        if self.point_count < 2:
            raise ValueError("point_count must be at least 2")
        if self.negative_tolerance < 0 or self.min_area < 0:
            raise ValueError("tolerances must be nonnegative")


def _rate_key(rate: float) -> str:
    return f"missing_rate_{int(round(rate * 100)):03d}"


def build_evaluation_masks(
    material_count: int,
    missing_rates: Sequence[float],
    repeats: int,
    base_seed: int,
) -> dict[str, np.ndarray]:
    """Build reproducible nested observed masks for evaluation missing rates."""
    if material_count <= 0:
        raise ValueError("material_count must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rates = tuple(float(rate) for rate in missing_rates)
    if any(not np.isfinite(rate) or rate < 0 or rate > 1 for rate in rates):
        raise ValueError("missing rates must be finite values in [0, 1]")
    keys = [_rate_key(rate) for rate in rates]
    if len(set(keys)) != len(keys):
        raise ValueError("missing rates produce duplicate output keys")

    masks = {
        key: np.ones((repeats, material_count), dtype=np.uint8)
        for key in keys
    }
    for repeat in range(repeats):
        permutation = np.random.default_rng(base_seed + repeat).permutation(
            material_count
        )
        for rate, key in zip(rates, keys, strict=True):
            missing_count = int(np.floor(rate * material_count + 0.5))
            masks[key][repeat, permutation[:missing_count]] = 0
    return masks


def preprocess_dos_arrays(
    arrays: Mapping[str, Any], config: DosPreprocessConfig
) -> dict[str, Any]:
    """Validate and transform one total DOS into an area-normalized grid."""
    required = {
        "energies_ev",
        "efermi_ev",
        "energy_minus_efermi_ev",
        "density_up",
        "density_down",
        "density_total",
        "spin_up_present",
        "spin_down_present",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"missing required DOS arrays: {missing}")

    aligned = np.asarray(arrays["energy_minus_efermi_ev"], dtype=np.float64)
    raw_energy = np.asarray(arrays["energies_ev"], dtype=np.float64)
    efermi = float(arrays["efermi_ev"])
    density_up = np.asarray(arrays["density_up"], dtype=np.float64)
    density_down = np.asarray(arrays["density_down"], dtype=np.float64)
    density_total = np.asarray(arrays["density_total"], dtype=np.float64)
    vectors = (aligned, raw_energy, density_up, density_down, density_total)

    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("DOS arrays must be one-dimensional")
    if not all(vector.shape == aligned.shape for vector in vectors):
        raise ValueError("DOS arrays must have identical shapes")
    if len(aligned) < 3 or not all(np.isfinite(vector).all() for vector in vectors):
        raise ValueError("DOS arrays must have at least 3 finite points")
    if not np.isfinite(efermi):
        raise ValueError("efermi must be finite")
    if not np.all(np.diff(aligned) > 0):
        raise ValueError("aligned energy must be strictly increasing")
    if not np.allclose(raw_energy - efermi, aligned, rtol=0, atol=1e-10):
        raise ValueError("raw and aligned energy arrays are inconsistent")
    if not np.allclose(
        density_total, density_up + density_down, rtol=0, atol=1e-10
    ):
        raise ValueError("density_total does not match the spin sum")
    if min(
        float(density_up.min()),
        float(density_down.min()),
        float(density_total.min()),
    ) < -config.negative_tolerance:
        raise ValueError("DOS contains materially negative density")
    if aligned[0] > config.energy_min_ev or aligned[-1] < config.energy_max_ev:
        raise ValueError("source DOS does not cover the configured energy window")

    energy_grid = np.linspace(
        config.energy_min_ev,
        config.energy_max_ev,
        config.point_count,
        dtype=np.float64,
    )
    clipped_total = np.maximum(density_total, 0.0)
    interpolated = np.interp(energy_grid, aligned, clipped_total)
    pre_norm_area = float(np.trapezoid(interpolated, energy_grid))
    if not np.isfinite(pre_norm_area) or pre_norm_area <= config.min_area:
        raise ValueError("interpolated DOS area is not positive and finite")

    feature = (interpolated / pre_norm_area).astype(np.float32)[None, :]
    return {
        "energy_grid_ev": energy_grid.astype(np.float32),
        "feature": feature,
        "pre_norm_area": pre_norm_area,
        "original_point_count": int(len(aligned)),
        "aligned_energy_min_ev": float(aligned[0]),
        "aligned_energy_max_ev": float(aligned[-1]),
        "spin_up_present": bool(arrays["spin_up_present"]),
        "spin_down_present": bool(arrays["spin_down_present"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def materialize_preprocessed_dataset(
    sample_dir: Path | str,
    output_dir: Path | str,
    *,
    config: DosPreprocessConfig,
    missing_rates: Sequence[float],
    repeats: int,
    base_seed: int,
    source_manifest_name: str = "dos_sample_manifest.csv",
    coverage_policy: str = "fail",
    spec_version: str = "dos-preprocessing-v1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize an ordered, auditable fixed-grid DOS dataset."""
    if Path(source_manifest_name).name != source_manifest_name:
        raise ValueError("source_manifest_name must be a file name")
    if coverage_policy not in {"fail", "exclude"}:
        raise ValueError("coverage_policy must be 'fail' or 'exclude'")
    if not spec_version.strip():
        raise ValueError("spec_version must be nonempty")
    source = Path(sample_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)

    selected_path = source / "selected_materials.csv"
    source_manifest_path = source / source_manifest_name
    selected = pd.read_csv(selected_path)
    source_manifest = pd.read_csv(source_manifest_path)
    selected_ids = selected["material_id"].astype(str).tolist()
    manifest_ids = source_manifest["material_id"].astype(str).tolist()
    if (
        not selected["material_id"].is_unique
        or not source_manifest["material_id"].is_unique
    ):
        raise ValueError("material_id must be unique in both source tables")
    if selected_ids != manifest_ids:
        raise ValueError("selected and source manifest material order must match")
    if not source_manifest["status"].isin(["success", "existing"]).all():
        raise ValueError("source manifest contains unsuccessful rows")

    rates = tuple(float(rate) for rate in missing_rates)
    config_payload = {
        "spec_version": spec_version,
        **asdict(config),
        "coverage_policy": coverage_policy,
        "source_manifest_name": source_manifest_name,
        "missing_rates": list(rates),
        "repeats": repeats,
        "base_seed": base_seed,
        "selected_materials_sha256": _file_sha256(selected_path),
        "source_manifest_sha256": _file_sha256(source_manifest_path),
    }
    _write_json(destination / "preprocessing_config.json", config_payload)

    source_by_id = source_manifest.set_index("material_id")
    records: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    included_ids: list[str] = []
    energy_grid: np.ndarray | None = None
    for material_id in selected_ids:
        source_path = source / "files" / f"{material_id}.npz"
        expected_sha256 = str(source_by_id.loc[material_id, "sha256"])
        record: dict[str, Any] = {
            "output_row": None,
            "material_id": material_id,
            "source_file": f"files/{material_id}.npz",
            "expected_source_sha256": expected_sha256,
        }
        actual_sha256: str | None = None
        try:
            actual_sha256 = _file_sha256(source_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "source SHA-256 does not match the download manifest"
                )
            with np.load(source_path, allow_pickle=False) as saved:
                transformed = preprocess_dos_arrays(
                    {name: saved[name] for name in saved.files}, config
                )
            if energy_grid is not None and not np.array_equal(
                energy_grid, transformed["energy_grid_ev"]
            ):
                raise ValueError("material produced an inconsistent fixed grid")
            energy_grid = transformed["energy_grid_ev"]
            features.append(transformed["feature"])
            included_ids.append(material_id)
            record.update(
                {
                    "output_row": len(included_ids) - 1,
                    "source_sha256": actual_sha256,
                    "source_sha256_matches": True,
                    "status": "success",
                    "original_point_count": transformed["original_point_count"],
                    "aligned_energy_min_ev": transformed[
                        "aligned_energy_min_ev"
                    ],
                    "aligned_energy_max_ev": transformed[
                        "aligned_energy_max_ev"
                    ],
                    "pre_norm_area": transformed["pre_norm_area"],
                    "spin_up_present": transformed["spin_up_present"],
                    "spin_down_present": transformed["spin_down_present"],
                    "error_type": None,
                    "error_message": None,
                }
            )
        except (OSError, KeyError, ValueError) as error:
            is_coverage_exclusion = (
                coverage_policy == "exclude"
                and isinstance(error, ValueError)
                and str(error)
                == "source DOS does not cover the configured energy window"
            )
            record.update(
                {
                    "source_sha256": actual_sha256,
                    "source_sha256_matches": actual_sha256 == expected_sha256,
                    "status": "excluded" if is_coverage_exclusion else "failed",
                    "exclusion_reason": (
                        "insufficient_energy_coverage"
                        if is_coverage_exclusion
                        else None
                    ),
                    "original_point_count": None,
                    "aligned_energy_min_ev": None,
                    "aligned_energy_max_ev": None,
                    "pre_norm_area": None,
                    "spin_up_present": None,
                    "spin_down_present": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error).replace("\n", " ")[:500],
                }
            )
        records.append(record)

    preprocessing_manifest = pd.DataFrame(records)
    manifest_path = destination / "preprocessing_manifest.csv"
    preprocessing_manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    failed_count = int((preprocessing_manifest["status"] == "failed").sum())
    excluded_count = int((preprocessing_manifest["status"] == "excluded").sum())
    success_count = int((preprocessing_manifest["status"] == "success").sum())
    exclusion_path = destination / "coverage_exclusions.csv"
    if coverage_policy == "exclude":
        preprocessing_manifest.loc[
            preprocessing_manifest["status"] == "excluded"
        ].to_csv(exclusion_path, index=False, encoding="utf-8")
    if failed_count:
        summary = {
            "status": "partial",
            "source_count": len(preprocessing_manifest),
            "sample_size": success_count,
            "success_count": success_count,
            "excluded_count": excluded_count,
            "failed_count": failed_count,
            "feature_shape": None,
            "output_file_sha256": {
                "preprocessing_config.json": _file_sha256(
                    destination / "preprocessing_config.json"
                ),
                "preprocessing_manifest.csv": _file_sha256(manifest_path),
            },
        }
        _write_json(destination / "quality_summary.json", summary)
        return preprocessing_manifest, summary

    feature_array = np.stack(features, axis=0).astype(np.float32)
    material_ids = np.asarray(included_ids, dtype=str)
    observed_mask = np.ones(len(included_ids), dtype=np.uint8)
    evaluation_masks = build_evaluation_masks(
        len(included_ids), rates, repeats, base_seed
    )
    assert energy_grid is not None
    np.save(destination / "energy_grid_ev.npy", energy_grid)
    np.save(destination / "dos_features.npy", feature_array)
    np.save(destination / "dos_observed_mask.npy", observed_mask)
    np.save(destination / "material_ids.npy", material_ids)
    np.savez_compressed(
        destination / "evaluation_missingness_masks.npz", **evaluation_masks
    )

    integrals = np.trapezoid(feature_array[:, 0, :], energy_grid, axis=1)
    output_names = [
        "energy_grid_ev.npy",
        "dos_features.npy",
        "dos_observed_mask.npy",
        "material_ids.npy",
        "evaluation_missingness_masks.npz",
        "preprocessing_config.json",
        "preprocessing_manifest.csv",
    ]
    if coverage_policy == "exclude":
        output_names.append("coverage_exclusions.csv")
    summary = {
        "status": "complete",
        "source_count": len(selected_ids),
        "sample_size": len(included_ids),
        "success_count": success_count,
        "excluded_count": excluded_count,
        "failed_count": 0,
        "feature_shape": list(feature_array.shape),
        "feature_dtype": str(feature_array.dtype),
        "maximum_integral_error": float(np.max(np.abs(integrals - 1.0))),
        "missing_counts": {
            key: np.sum(value == 0, axis=1).astype(int).tolist()
            for key, value in evaluation_masks.items()
        },
        "output_file_sha256": {
            name: _file_sha256(destination / name) for name in output_names
        },
    }
    _write_json(destination / "quality_summary.json", summary)
    return preprocessing_manifest, summary
