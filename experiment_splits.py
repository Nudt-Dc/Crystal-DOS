"""Create deterministic IID and SOAP-LOCO experiment splits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
import uuid

import numpy as np
import pandas as pd


SPLIT_SCHEMA_VERSION = "experiment_splits_v1"
DEFAULT_SEED = 20260908
SPLIT_ORDER = ("train", "validation", "test")


def stable_order_key(material_id: str, *, seed: int = DEFAULT_SEED) -> str:
    return hashlib.sha256(f"{seed}:{material_id}".encode("utf-8")).hexdigest()


def _qcut_codes(values: pd.Series) -> tuple[pd.Series, list[float]]:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("stratification labels must be finite")
    codes, edges = pd.qcut(numeric, q=5, labels=False, retbins=True, duplicates="drop")
    if pd.isna(codes).all():
        code_series = pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)
    else:
        code_series = pd.Series(codes, index=values.index).fillna(0).astype(np.int64)
    return code_series, [float(value) for value in np.asarray(edges)]


def build_strata(master: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Build the approved DOS/K/G joint stratum for every master row."""
    required = {"material_id", "k_vrh_gpa", "g_vrh_gpa", "dos_observed"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"master is missing columns: {missing}")
    if master["material_id"].astype(str).duplicated().any():
        raise ValueError("master material IDs must be unique")
    k_codes, k_edges = _qcut_codes(master["k_vrh_gpa"])
    g_codes, g_edges = _qcut_codes(master["g_vrh_gpa"])
    observed = master["dos_observed"].astype(bool).astype(np.int8)
    strata = pd.Series(
        [f"d{d}_k{k}_g{g}" for d, k, g in zip(observed, k_codes, g_codes)],
        index=master.index,
        name="stratum",
        dtype=str,
    )
    return strata, {
        "method": "pandas.qcut",
        "q": 5,
        "duplicates": "drop",
        "k_bin_edges": k_edges,
        "g_bin_edges": g_edges,
    }


def _initial_assignments(
    working: pd.DataFrame,
    targets: dict[str, int],
) -> pd.Series:
    total = len(working)
    ratios = {name: targets[name] / total for name in SPLIT_ORDER}
    assignments = pd.Series(index=working.index, dtype=object)
    for stratum in sorted(working["stratum"].unique()):
        indexes = working.index[working["stratum"] == stratum].tolist()
        indexes.sort(key=lambda idx: (working.at[idx, "stable_order_key"], working.at[idx, "material_id"]))
        count = len(indexes)
        ideal = {name: count * ratios[name] for name in SPLIT_ORDER}
        quotas = {name: int(np.floor(ideal[name])) for name in SPLIT_ORDER}
        remaining = count - sum(quotas.values())
        priority = sorted(
            SPLIT_ORDER,
            key=lambda name: (-(ideal[name] - quotas[name]), SPLIT_ORDER.index(name)),
        )
        for name in priority[:remaining]:
            quotas[name] += 1
        cursor = 0
        for name in SPLIT_ORDER:
            chosen = indexes[cursor:cursor + quotas[name]]
            assignments.loc[chosen] = name
            cursor += quotas[name]
    return assignments


def _reconcile_exact_targets(
    working: pd.DataFrame,
    assignments: pd.Series,
    targets: dict[str, int],
) -> pd.Series:
    ratios = {name: targets[name] / len(working) for name in SPLIT_ORDER}
    while True:
        counts = assignments.value_counts().to_dict()
        deficits = {name: targets[name] - int(counts.get(name, 0)) for name in SPLIT_ORDER}
        if all(value == 0 for value in deficits.values()):
            return assignments
        under = max(
            (name for name in SPLIT_ORDER if deficits[name] > 0),
            key=lambda name: (deficits[name], -SPLIT_ORDER.index(name)),
        )
        over = max(
            (name for name in SPLIT_ORDER if deficits[name] < 0),
            key=lambda name: (-deficits[name], -SPLIT_ORDER.index(name)),
        )
        candidates = []
        for index in assignments.index[assignments == over]:
            stratum = working.at[index, "stratum"]
            member_mask = working["stratum"] == stratum
            group_size = int(member_mask.sum())
            group_assignments = assignments.loc[member_mask]
            current_under = int((group_assignments == under).sum())
            current_over = int((group_assignments == over).sum())
            before = abs(current_under - group_size * ratios[under]) + abs(current_over - group_size * ratios[over])
            after = abs(current_under + 1 - group_size * ratios[under]) + abs(current_over - 1 - group_size * ratios[over])
            candidates.append(
                (
                    after - before,
                    working.at[index, "stable_order_key"],
                    working.at[index, "material_id"],
                    index,
                )
            )
        if not candidates:
            raise ValueError("cannot reconcile exact split targets")
        assignments.at[min(candidates)[-1]] = under


def make_iid_split(
    master: pd.DataFrame,
    *,
    sizes: Sequence[int] = (8546, 1068, 1068),
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Create one exact, deterministic, jointly stratified IID split."""
    if len(sizes) != 3 or any(int(value) < 0 for value in sizes):
        raise ValueError("sizes must contain three non-negative values")
    if sum(int(value) for value in sizes) != len(master):
        raise ValueError("split sizes must sum to the master count")
    working = master.copy()
    working["material_id"] = working["material_id"].astype(str)
    if working["material_id"].duplicated().any():
        raise ValueError("master material IDs must be unique")
    strata, _ = build_strata(working)
    working["stratum"] = strata
    working["stable_order_key"] = [
        stable_order_key(value, seed=seed) for value in working["material_id"]
    ]
    targets = dict(zip(SPLIT_ORDER, (int(value) for value in sizes)))
    assignments = _initial_assignments(working, targets)
    assignments = _reconcile_exact_targets(working, assignments, targets)
    working["split"] = assignments
    result_columns = [
        name
        for name in ("material_id", "crystal_row", "dos_row", "split", "stratum", "stable_order_key")
        if name in working.columns
    ]
    result = working[result_columns].sort_values("material_id", kind="mergesort").reset_index(drop=True)
    if result["split"].value_counts().to_dict() != {
        name: targets[name] for name in SPLIT_ORDER if targets[name] > 0
    }:
        raise ValueError("exact split target reconciliation failed")
    return result


def fit_structure_embedding(
    soap_matrix: Any,
    *,
    svd_factory: Any | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit the frozen 256-component SVD and return L2-normalized rows."""
    if len(soap_matrix.shape) != 2:
        raise ValueError("SOAP matrix must be two-dimensional")
    if int(soap_matrix.shape[1]) < 256:
        raise ValueError("SOAP matrix must have at least 256 features")

    if svd_factory is None:
        from sklearn.decomposition import TruncatedSVD

        svd_factory = TruncatedSVD

    from scipy import sparse

    if sparse.issparse(soap_matrix):
        normalized_input = soap_matrix.tocsr().astype(np.float32, copy=True)
        squared_norms = np.asarray(normalized_input.multiply(normalized_input).sum(axis=1)).ravel()
        if not np.isfinite(squared_norms).all() or np.any(squared_norms <= 0):
            raise ValueError("SOAP matrix rows must have finite non-zero norms")
        normalized_input = sparse.diags(1.0 / np.sqrt(squared_norms)) @ normalized_input
    else:
        normalized_input = np.asarray(soap_matrix, dtype=np.float32)
        if not np.isfinite(normalized_input).all():
            raise ValueError("SOAP matrix must contain only finite values")
        input_norms = np.linalg.norm(normalized_input, axis=1)
        if np.any(input_norms <= 0):
            raise ValueError("SOAP matrix rows must have finite non-zero norms")
        normalized_input = normalized_input / input_norms[:, None]

    model = svd_factory(
        n_components=256,
        algorithm="randomized",
        random_state=DEFAULT_SEED,
    )
    embedding = np.asarray(model.fit_transform(normalized_input), dtype=np.float32)
    expected_shape = (int(soap_matrix.shape[0]), 256)
    if embedding.shape != expected_shape:
        raise ValueError(f"SVD returned shape {embedding.shape}, expected {expected_shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("SVD embedding must contain only finite values")
    output_norms = np.linalg.norm(embedding, axis=1)
    if np.any(output_norms <= 0):
        raise ValueError("SVD embedding rows must have non-zero norms")
    embedding = np.asarray(embedding / output_norms[:, None], dtype=np.float32)

    state = {
        "components": np.asarray(model.components_),
        "explained_variance": np.asarray(model.explained_variance_),
        "explained_variance_ratio": np.asarray(model.explained_variance_ratio_),
        "singular_values": np.asarray(model.singular_values_),
    }
    return embedding, state


def canonicalize_clusters(
    labels: Sequence[int],
    material_ids: Sequence[str],
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Rename raw cluster labels by each cluster's minimum material ID."""
    raw_labels = np.asarray(labels, dtype=np.int64)
    ids = np.asarray([str(value) for value in material_ids], dtype=object)
    raw_centers = np.asarray(centers)
    if raw_labels.ndim != 1 or len(raw_labels) != len(ids):
        raise ValueError("cluster labels and material IDs must have equal lengths")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("material IDs must be unique")
    unique_labels = sorted(int(value) for value in np.unique(raw_labels))
    if unique_labels != list(range(len(unique_labels))):
        raise ValueError("raw cluster labels must be contiguous from zero")
    if raw_centers.ndim != 2 or raw_centers.shape[0] != len(unique_labels):
        raise ValueError("cluster centers must align with raw labels")

    ordered_raw = sorted(
        unique_labels,
        key=lambda label: min(ids[raw_labels == label].tolist()),
    )
    mapping = {raw_label: canonical for canonical, raw_label in enumerate(ordered_raw)}
    canonical = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int64)
    ordered_centers = np.asarray(raw_centers[ordered_raw]).copy()
    return canonical, ordered_centers, mapping


def fit_canonical_kmeans(
    embedding: np.ndarray,
    material_ids: Sequence[str],
    *,
    kmeans_factory: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[str, Any]]:
    """Fit the frozen KMeans contract and canonicalize its numeric labels."""
    features = np.asarray(embedding, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(material_ids):
        raise ValueError("embedding and material IDs must align")
    if features.shape[0] < 10:
        raise ValueError("KMeans requires at least 10 materials")
    if not np.isfinite(features).all():
        raise ValueError("embedding must contain only finite values")
    if kmeans_factory is None:
        from sklearn.cluster import KMeans

        kmeans_factory = KMeans
    model = kmeans_factory(
        n_clusters=10,
        init="k-means++",
        n_init=20,
        max_iter=500,
        algorithm="lloyd",
        random_state=DEFAULT_SEED,
    )
    raw_labels = np.asarray(model.fit_predict(features), dtype=np.int64)
    labels, centers, mapping = canonicalize_clusters(
        raw_labels,
        material_ids,
        np.asarray(model.cluster_centers_),
    )
    state = {
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
    }
    return labels, centers, mapping, state


def _validation_indexes(
    working: pd.DataFrame,
    *,
    validation_count: int,
) -> set[Any]:
    """Choose an exact deterministic 10% validation set with stratum quotas."""
    if validation_count < 0 or validation_count > len(working):
        raise ValueError("invalid validation count")
    ratio = validation_count / len(working) if len(working) else 0.0
    groups: dict[str, list[Any]] = {}
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for stratum in sorted(working["stratum"].astype(str).unique()):
        indexes = working.index[working["stratum"].astype(str) == stratum].tolist()
        indexes.sort(key=lambda idx: (working.at[idx, "stable_order_key"], working.at[idx, "material_id"]))
        groups[stratum] = indexes
        ideal = len(indexes) * ratio
        quotas[stratum] = int(np.floor(ideal))
        fractions.append((ideal - quotas[stratum], stratum))
    deficit = validation_count - sum(quotas.values())
    for _, stratum in sorted(fractions, key=lambda item: (-item[0], item[1]))[:deficit]:
        quotas[stratum] += 1
    return {
        index
        for stratum, indexes in groups.items()
        for index in indexes[:quotas[stratum]]
    }


def make_loco_folds(
    master: pd.DataFrame,
    cluster_ids: Sequence[int],
    strata: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    minimum_total: int = 300,
    minimum_observed: int = 150,
    expected_cluster_count: int = 10,
) -> pd.DataFrame:
    """Create deterministic leave-one-cluster-out folds with a 10% validation set."""
    if len(master) != len(cluster_ids) or len(master) != len(strata):
        raise ValueError("master, cluster IDs, and strata must have equal lengths")
    required = {"material_id", "dos_observed"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"master is missing columns: {missing}")
    working = master.copy().reset_index(drop=True)
    working["material_id"] = working["material_id"].astype(str)
    if working["material_id"].duplicated().any():
        raise ValueError("master material IDs must be unique")
    working["cluster_id"] = np.asarray(cluster_ids, dtype=np.int64)
    working["stratum"] = np.asarray(strata, dtype=str)
    working["stable_order_key"] = [
        stable_order_key(value, seed=seed) for value in working["material_id"]
    ]
    if expected_cluster_count < 2:
        raise ValueError("LOCO requires at least two clusters")
    unique_clusters = sorted(int(value) for value in working["cluster_id"].unique())
    if unique_clusters != list(range(expected_cluster_count)):
        raise ValueError(
            f"LOCO requires exactly {expected_cluster_count} canonical clusters "
            f"numbered 0 through {expected_cluster_count - 1}"
        )

    for cluster_id in unique_clusters:
        cluster = working[working["cluster_id"] == cluster_id]
        total = len(cluster)
        observed = int(cluster["dos_observed"].astype(bool).sum())
        if total < minimum_total or observed < minimum_observed:
            raise ValueError(
                f"cluster {cluster_id} violates minimum gate: "
                f"total={total} (minimum {minimum_total}), "
                f"observed={observed} (minimum {minimum_observed})"
            )

    fold_frames = []
    for held_out in unique_clusters:
        remaining = working[working["cluster_id"] != held_out]
        validation_count = int(round(0.10 * len(remaining)))
        validation_indexes = _validation_indexes(
            remaining,
            validation_count=validation_count,
        )
        fold = working.copy()
        fold["fold"] = f"loco_{held_out:02d}"
        fold["split"] = "train"
        fold.loc[list(validation_indexes), "split"] = "validation"
        fold.loc[fold["cluster_id"] == held_out, "split"] = "test"
        fold_frames.append(fold)

    combined = pd.concat(fold_frames, ignore_index=True)
    result_columns = [
        name
        for name in (
            "fold",
            "material_id",
            "crystal_row",
            "dos_row",
            "cluster_id",
            "split",
            "stratum",
            "stable_order_key",
        )
        if name in combined.columns
    ]
    return combined[result_columns].sort_values(
        ["fold", "material_id"], kind="mergesort"
    ).reset_index(drop=True)


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cluster_gate_failures(
    master: pd.DataFrame,
    cluster_ids: np.ndarray,
    *,
    minimum_total: int,
    minimum_observed: int,
) -> list[dict[str, int]]:
    failures = []
    observed = master["dos_observed"].astype(bool).to_numpy()
    for cluster_id in range(10):
        member_mask = cluster_ids == cluster_id
        total = int(member_mask.sum())
        observed_count = int(observed[member_mask].sum())
        if total < minimum_total or observed_count < minimum_observed:
            failures.append({
                "cluster_id": cluster_id,
                "total": total,
                "observed": observed_count,
                "minimum_total": int(minimum_total),
                "minimum_observed": int(minimum_observed),
            })
    return failures


def materialize_experiment_splits(
    master_path: Path | str,
    soap_dir: Path | str,
    output_dir: Path | str,
    *,
    iid_sizes: Sequence[int] = (8546, 1068, 1068),
    seed: int = DEFAULT_SEED,
    minimum_total: int = 300,
    minimum_observed: int = 150,
    diagnostic_smoke: bool = False,
    svd_factory: Any | None = None,
    kmeans_factory: Any | None = None,
) -> dict[str, Any]:
    """Fit structure-only models and atomically publish frozen split artifacts."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if minimum_total <= 0 or minimum_observed < 0:
        raise ValueError("cluster minimum gates must be non-negative and total must be positive")

    master_file = Path(master_path)
    soap_root = Path(soap_dir)
    features_path = soap_root / "soap_features.npz"
    ids_path = soap_root / "soap_material_ids.npy"
    master = pd.read_parquet(master_file).reset_index(drop=True)
    required = {"material_id", "dos_observed", "k_vrh_gpa", "g_vrh_gpa"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"master is missing columns: {missing}")
    master["material_id"] = master["material_id"].astype(str)
    if master["material_id"].duplicated().any():
        raise ValueError("master material IDs must be unique")

    from scipy import sparse

    soap = sparse.load_npz(features_path).tocsr()
    soap_ids = np.load(ids_path, allow_pickle=False).astype(str)
    expected_ids = master["material_id"].to_numpy(dtype=str)
    if soap.shape[0] != len(master) or not np.array_equal(soap_ids, expected_ids):
        raise ValueError("SOAP rows and material IDs must exactly match the master table")

    embedding, svd_state = fit_structure_embedding(soap, svd_factory=svd_factory)
    cluster_ids, centers, mapping, kmeans_state = fit_canonical_kmeans(
        embedding,
        expected_ids,
        kmeans_factory=kmeans_factory,
    )
    strata, strata_metadata = build_strata(master)
    cluster_assignments = pd.DataFrame({
        "material_id": expected_ids,
        "cluster_id": cluster_ids,
    })
    for column in ("crystal_row", "dos_row", "dos_observed"):
        if column in master:
            cluster_assignments[column] = master[column].to_numpy()

    gate_failures = _cluster_gate_failures(
        master,
        cluster_ids,
        minimum_total=minimum_total,
        minimum_observed=minimum_observed,
    )
    iid_split = None
    loco_folds = None
    if not gate_failures:
        iid_split = make_iid_split(master, sizes=iid_sizes, seed=seed)
        loco_folds = make_loco_folds(
            master,
            cluster_ids,
            strata,
            seed=seed,
            minimum_total=minimum_total,
            minimum_observed=minimum_observed,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        array_files = {
            "soap_svd_256.npy": embedding,
            "svd_components.npy": svd_state["components"],
            "svd_explained_variance.npy": svd_state["explained_variance"],
            "svd_explained_variance_ratio.npy": svd_state["explained_variance_ratio"],
            "svd_singular_values.npy": svd_state["singular_values"],
            "soap_cluster_centers.npy": centers,
        }
        for filename, values in array_files.items():
            np.save(staging / filename, values, allow_pickle=False)
        cluster_assignments.to_parquet(
            staging / "soap_cluster_assignments.parquet", index=False
        )
        if iid_split is not None and loco_folds is not None:
            iid_split.to_parquet(staging / "iid_split.parquet", index=False)
            loco_folds.to_parquet(staging / "soap_loco_folds.parquet", index=False)

        artifact_names = sorted(
            path.name for path in staging.iterdir() if path.is_file()
        )
        output_hashes = {
            name: _sha256_file(staging / name) for name in artifact_names
        }
        complete = not gate_failures
        metadata = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "partial",
            "complete": complete,
            "row_count": int(len(master)),
            "diagnostic_smoke": bool(diagnostic_smoke),
            "seed": int(seed),
            "svd": {
                "n_components": 256,
                "algorithm": "randomized",
                "random_state": DEFAULT_SEED,
            },
            "kmeans": {
                "n_clusters": 10,
                "init": "k-means++",
                "n_init": 20,
                "max_iter": 500,
                "algorithm": "lloyd",
                "random_state": DEFAULT_SEED,
                **kmeans_state,
            },
            "canonical_cluster_mapping": {
                str(raw): int(canonical) for raw, canonical in mapping.items()
            },
            "cluster_counts": {
                str(cluster_id): int((cluster_ids == cluster_id).sum())
                for cluster_id in range(10)
            },
            "cluster_observed_counts": {
                str(cluster_id): int(
                    master.loc[cluster_ids == cluster_id, "dos_observed"].astype(bool).sum()
                )
                for cluster_id in range(10)
            },
            "cluster_gate": {
                "minimum_total": int(minimum_total),
                "minimum_observed": int(minimum_observed),
                "failures": gate_failures,
            },
            "stratification": strata_metadata,
            "iid_counts": (
                {str(key): int(value) for key, value in iid_split["split"].value_counts().items()}
                if iid_split is not None
                else {}
            ),
            "loco_fold_count": int(loco_folds["fold"].nunique()) if loco_folds is not None else 0,
            "source_sha256": {
                "master_table": _sha256_file(master_file),
                "soap_features": _sha256_file(features_path),
                "soap_material_ids": _sha256_file(ids_path),
            },
            "output_file_sha256": output_hashes,
        }
        _write_json(staging / "split_metadata.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "status": "complete" if complete else "partial",
        "complete": complete,
        "row_count": int(len(master)),
        "cluster_gate_failure_count": int(len(gate_failures)),
        "output_dir": str(destination.resolve()),
    }
