"""Publish the pre-registered AFLOW v2 split amendment after the v1 size gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from aflow_confirmatory_inventory import sha256_file
from experiment_splits import build_strata, make_iid_split, make_loco_folds


SCHEMA_VERSION = "aflow_confirmatory_split_amendment_v2"
FROZEN_COARSENING = {
    "method": "deterministic_nearest_centroid_merge",
    "minimum_total_per_cluster": 75,
    "minimum_observed_dos_per_cluster": 75,
    "violating_source_order": ["ascending_count", "ascending_minimum_material_id"],
    "merge_target_order": [
        "ascending_centroid_euclidean_distance", "ascending_minimum_material_id"
    ],
    "centroid_update": "arithmetic_mean_of_member_svd_embeddings",
    "repeat_until_all_clusters_pass": True,
    "canonical_label_order": "ascending_minimum_material_id",
    "expected_final_cluster_count": 9,
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _canonicalize(labels: np.ndarray, material_ids: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    unique = sorted(int(value) for value in np.unique(labels))
    ordered = sorted(
        unique,
        key=lambda label: min(material_ids[labels == label].tolist()),
    )
    mapping = {old: new for new, old in enumerate(ordered)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64), mapping


def merge_undersized_clusters(
    embedding: np.ndarray,
    cluster_ids: np.ndarray,
    material_ids: np.ndarray,
    *,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Deterministically merge each smallest invalid cluster into its nearest centroid."""
    features = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(cluster_ids, dtype=np.int64).copy()
    ids = np.asarray(material_ids, dtype=str)
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(ids):
        raise ValueError("embedding, cluster IDs, and material IDs must align")
    if minimum_count <= 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("invalid coarsening inputs")
    if not np.isfinite(features).all():
        raise ValueError("embedding must be finite")
    history: list[dict] = []
    while True:
        unique = sorted(int(value) for value in np.unique(labels))
        counts = {label: int((labels == label).sum()) for label in unique}
        invalid = [label for label in unique if counts[label] < minimum_count]
        if not invalid:
            break
        if len(unique) <= 2:
            raise ValueError("cannot satisfy the cluster minimum without collapsing LOCO")
        source = min(
            invalid,
            key=lambda label: (counts[label], min(ids[labels == label].tolist())),
        )
        centroids = {
            label: np.asarray(features[labels == label].mean(axis=0), dtype=np.float64)
            for label in unique
        }
        candidates = [label for label in unique if label != source]
        target = min(
            candidates,
            key=lambda label: (
                float(np.linalg.norm(centroids[source] - centroids[label])),
                min(ids[labels == label].tolist()),
            ),
        )
        distance = float(np.linalg.norm(centroids[source] - centroids[target]))
        history.append({
            "source_cluster": source,
            "source_count": counts[source],
            "target_cluster": target,
            "target_count_before": counts[target],
            "combined_count": counts[source] + counts[target],
            "centroid_euclidean_distance": distance,
        })
        labels[labels == source] = target
    canonical, mapping = _canonicalize(labels, ids)
    final_centers = np.stack([
        features[canonical == cluster_id].mean(axis=0)
        for cluster_id in range(len(np.unique(canonical)))
    ]).astype(np.float32)
    for item in history:
        item["final_canonical_mapping"] = {
            str(old): int(new) for old, new in sorted(mapping.items())
        }
    return canonical, final_centers, history


def materialize_amended_splits(
    master_path: Path | str,
    v1_split_dir: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
) -> dict:
    master_file = Path(master_path).resolve()
    v1_root = Path(v1_split_dir).resolve()
    config_file = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite amended split: {destination}")
    config = _json(config_file)
    v1_metadata_file = v1_root / "split_metadata.json"
    v1_metadata = _json(v1_metadata_file)
    assignments_file = v1_root / "soap_cluster_assignments.parquet"
    embedding_file = v1_root / "soap_svd_256.npy"
    centers_file = v1_root / "soap_cluster_centers.npy"
    expected_hashes = config.get("source_v1_sha256", {})
    source_files = {
        "split_metadata.json": v1_metadata_file,
        "soap_cluster_assignments.parquet": assignments_file,
        "soap_svd_256.npy": embedding_file,
        "soap_cluster_centers.npy": centers_file,
    }
    if (
        config.get("schema_version") != "aflow_confirmatory_split_amendment_config_v2"
        or config.get("reason") != "v1_minimum_cluster_gate_failure"
        or config.get("population_count") != 2560
        or config.get("coarsening") != FROZEN_COARSENING
        or config.get("iid_sizes") != {"train": 2048, "validation": 256, "test": 256}
        or config.get("loco_validation_fraction_of_non_test") != 0.1
        or config.get("seed") != 20260908
        or any(expected_hashes.get(name) != sha256_file(path) for name, path in source_files.items())
        or v1_metadata.get("status") != "partial"
        or v1_metadata.get("complete") is not False
        or v1_metadata.get("cluster_gate", {}).get("minimum_total") != 75
        or v1_metadata.get("cluster_gate", {}).get("minimum_observed") != 75
        or v1_metadata.get("cluster_gate", {}).get("failures") != [{
            "cluster_id": 7,
            "minimum_observed": 75,
            "minimum_total": 75,
            "observed": 62,
            "total": 62,
        }]
        or config.get("target_values_used_for_coarsening") is not False
        or config.get("model_predictions_used") is not False
        or config.get("model_training_previously_performed") is not False
        or config.get("post_amendment_parameter_changes_forbidden") is not True
    ):
        raise ValueError("AFLOW v1 failure/amendment binding is invalid")
    master = pd.read_parquet(master_file).reset_index(drop=True)
    assignments = pd.read_parquet(assignments_file).reset_index(drop=True)
    embedding = np.load(embedding_file, allow_pickle=False)
    ids = master["material_id"].astype(str).to_numpy()
    if (
        len(master) != config.get("population_count")
        or not np.array_equal(assignments["material_id"].astype(str).to_numpy(), ids)
        or embedding.shape != (len(master), 256)
    ):
        raise ValueError("AFLOW v1 rows do not align with the master table")
    minimum = int(config["coarsening"]["minimum_total_per_cluster"])
    labels, centers, history = merge_undersized_clusters(
        embedding,
        assignments["cluster_id"].to_numpy(dtype=np.int64),
        ids,
        minimum_count=minimum,
    )
    expected_clusters = int(config["coarsening"]["expected_final_cluster_count"])
    counts = {str(i): int((labels == i).sum()) for i in range(expected_clusters)}
    if (
        len(np.unique(labels)) != expected_clusters
        or min(counts.values()) < minimum
        or len(history) != 1
    ):
        raise ValueError("AFLOW amended cluster gate failed")
    strata, strata_metadata = build_strata(master)
    iid = make_iid_split(master, sizes=(2048, 256, 256), seed=20260908)
    loco = make_loco_folds(
        master,
        labels,
        strata,
        seed=20260908,
        minimum_total=minimum,
        minimum_observed=minimum,
        expected_cluster_count=expected_clusters,
    )
    amended_assignments = assignments.copy()
    amended_assignments["v1_cluster_id"] = amended_assignments["cluster_id"].astype(np.int64)
    amended_assignments["cluster_id"] = labels
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        amended_assignments.to_parquet(staging / "soap_cluster_assignments.parquet", index=False)
        np.save(staging / "soap_cluster_centers.npy", centers, allow_pickle=False)
        iid.to_parquet(staging / "iid_split.parquet", index=False)
        loco.to_parquet(staging / "soap_loco_folds.parquet", index=False)
        output_names = (
            "soap_cluster_assignments.parquet", "soap_cluster_centers.npy",
            "iid_split.parquet", "soap_loco_folds.parquet",
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "complete": True,
            "row_count": len(master),
            "seed": 20260908,
            "v1_status_preserved": "partial",
            "coarsening_method": config["coarsening"],
            "merge_history": history,
            "cluster_counts": counts,
            "cluster_observed_counts": {
                str(i): int(master.loc[labels == i, "dos_observed"].astype(bool).sum())
                for i in range(expected_clusters)
            },
            "iid_counts": {str(k): int(v) for k, v in iid["split"].value_counts().items()},
            "loco_fold_count": int(loco["fold"].nunique()),
            "stratification": strata_metadata,
            "target_values_used_for_coarsening": False,
            "model_predictions_used": False,
            "model_training_performed": False,
            "source_sha256": {
                "master_table.parquet": sha256_file(master_file),
                **{name: sha256_file(path) for name, path in source_files.items()},
                "amendment_config.json": sha256_file(config_file),
            },
            "output_file_sha256": {
                name: sha256_file(staging / name) for name in output_names
            },
        }
        (staging / "split_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "complete": True,
        "row_count": len(master),
        "cluster_count": expected_clusters,
        "output_dir": str(destination),
    }
