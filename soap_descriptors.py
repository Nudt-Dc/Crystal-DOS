"""Compute approved periodic SOAP descriptors from the packed cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy import sparse as scipy_sparse


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SOURCE_DIR))

from crystal_dos_master import _sha256_file


APPROVED_SPECIES = (
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91,
    92, 93, 94,
)


class SoapSnapshotConflict(RuntimeError):
    """Raised when a formal SOAP artifact cannot be safely reused."""


@dataclass(frozen=True)
class SoapShardReceipt:
    batch_index: int
    row_start: int
    row_stop: int
    material_ids_sha256: str
    rows: int
    columns: int
    nnz: int
    file_bytes: int
    file_sha256: str
    status: str = "complete"


@dataclass(frozen=True)
class SoapConfig:
    periodic: bool = True
    r_cut: float = 5.0
    n_max: int = 6
    l_max: int = 4
    sigma: float = 0.5
    rbf: str = "gto"
    average: str = "inner"
    compression: dict[str, str] = field(
        default_factory=lambda: {"mode": "mu1nu1"}
    )
    sparse: bool = True
    dtype: str = "float32"
    species: tuple[int, ...] = APPROVED_SPECIES
    random_state: int = 20260908

    def descriptor_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("random_state")
        values["species"] = list(self.species)
        return values


def _config_payload(config: SoapConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["species"] = list(config.species)
    return payload


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _csr_content_sha256(matrix: scipy_sparse.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.data, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_sparse_atomic(path: Path, matrix: scipy_sparse.csr_matrix) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    scipy_sparse.save_npz(temporary, matrix, compressed=True)
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _default_atoms_factory(**kwargs: Any) -> Any:
    from ase import Atoms

    return Atoms(**kwargs)


def _default_soap_factory(**kwargs: Any) -> Any:
    from dscribe.descriptors import SOAP

    return SOAP(**kwargs)


def build_ase_systems(
    cache_dir: Path | str,
    row_indexes: Sequence[int],
    *,
    atoms_factory: Callable[..., Any] | None = None,
) -> list[Any]:
    """Convert packed cache rows to periodic ASE-compatible objects."""
    factory = atoms_factory or _default_atoms_factory
    root = Path(cache_dir)
    lattices = np.load(root / "lattice_matrices.npy", allow_pickle=False)
    frac = np.load(root / "fractional_coordinates.npy", allow_pickle=False)
    numbers = np.load(root / "atomic_numbers.npy", allow_pickle=False)
    offsets = np.load(root / "site_offsets.npy", allow_pickle=False)
    indexes = [int(value) for value in row_indexes]
    if len(indexes) != len(set(indexes)):
        raise ValueError("row indexes must be unique")
    systems = []
    for row in indexes:
        if row < 0 or row >= len(lattices):
            raise IndexError(row)
        start, stop = int(offsets[row]), int(offsets[row + 1])
        systems.append(
            factory(
                numbers=np.array(numbers[start:stop], copy=True),
                cell=np.array(lattices[row], copy=True),
                scaled_positions=np.array(frac[start:stop], copy=True),
                pbc=True,
            )
        )
    return systems


def _as_csr(value: Any) -> scipy_sparse.csr_matrix:
    if scipy_sparse.issparse(value):
        return value.tocsr().astype(np.float32)
    if hasattr(value, "to_scipy_sparse"):
        return value.to_scipy_sparse().tocsr().astype(np.float32)
    if hasattr(value, "todense") and not isinstance(value, np.ndarray):
        value = value.todense()
    return scipy_sparse.csr_matrix(np.asarray(value, dtype=np.float32))


def _csr_row_norms(matrix: Any) -> np.ndarray:
    """Return row L2 norms for SciPy sparse matrices and sparse arrays."""
    squared = matrix.multiply(matrix).sum(axis=1)
    return np.sqrt(np.asarray(squared).reshape(-1))


def compute_soap_batch(
    cache_dir: Path | str,
    row_indexes: Sequence[int],
    config: SoapConfig,
    *,
    soap_factory: Callable[..., Any] | None = None,
    atoms_factory: Callable[..., Any] | None = None,
    n_jobs: int = 1,
) -> scipy_sparse.csr_matrix:
    """Compute and validate one normalized SOAP batch."""
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    indexes = [int(value) for value in row_indexes]
    if not indexes:
        raise ValueError("row_indexes must not be empty")
    systems = build_ase_systems(
        cache_dir,
        indexes,
        atoms_factory=atoms_factory,
    )
    factory = soap_factory or _default_soap_factory
    soap = factory(**config.descriptor_kwargs())
    matrix = _as_csr(soap.create(systems, n_jobs=n_jobs))
    expected_columns = int(soap.get_number_of_features())
    if matrix.shape != (len(indexes), expected_columns):
        raise ValueError(
            f"SOAP shape mismatch: expected {(len(indexes), expected_columns)}, got {matrix.shape}"
        )
    if not np.isfinite(matrix.data).all():
        raise ValueError("SOAP contains non-finite values")
    norms = _csr_row_norms(matrix)
    if np.any(~np.isfinite(norms)):
        raise ValueError("SOAP norms contain non-finite values")
    if np.any(norms <= 0):
        raise ValueError("SOAP contains zero-norm rows")
    normalized = (scipy_sparse.diags(1.0 / norms) @ matrix).tocsr().astype(np.float32)
    normalized.sort_indices()
    check_norms = _csr_row_norms(normalized)
    if not np.allclose(check_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("SOAP normalization failed")
    return normalized


def _validate_reusable_shard(
    path: Path,
    receipt: SoapShardReceipt,
    expected_ids: Sequence[str],
) -> scipy_sparse.csr_matrix:
    if not path.is_file():
        raise SoapSnapshotConflict(f"formal shard is missing: {path.name}")
    if _sha256_file(path) != receipt.file_sha256:
        raise SoapSnapshotConflict(f"formal shard hash mismatch: {path.name}")
    if receipt.material_ids_sha256 != _ids_sha256(expected_ids):
        raise SoapSnapshotConflict(f"formal shard material ID hash mismatch: {path.name}")
    matrix = scipy_sparse.load_npz(path).tocsr().astype(np.float32)
    if matrix.shape != (receipt.rows, receipt.columns):
        raise SoapSnapshotConflict(f"formal shard shape mismatch: {path.name}")
    if matrix.nnz != receipt.nnz:
        raise SoapSnapshotConflict(f"formal shard nnz mismatch: {path.name}")
    if receipt.file_bytes != path.stat().st_size:
        raise SoapSnapshotConflict(f"formal shard byte count mismatch: {path.name}")
    if not np.isfinite(matrix.data).all():
        raise SoapSnapshotConflict(f"formal shard contains non-finite values: {path.name}")
    norms = _csr_row_norms(matrix)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise SoapSnapshotConflict(f"formal shard norm mismatch: {path.name}")
    return matrix


def run_soap_snapshot(
    cache_dir: Path | str,
    output_dir: Path | str,
    *,
    config: SoapConfig = SoapConfig(),
    batch_size: int = 128,
    n_jobs: int = 1,
    max_materials: int | None = None,
    soap_factory: Callable[..., Any] | None = None,
    atoms_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Materialize or strictly resume a deterministic SOAP snapshot."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    if max_materials is not None and max_materials <= 0:
        raise ValueError("max_materials must be positive")
    cache_root = Path(cache_dir)
    destination = Path(output_dir)
    ids = np.load(cache_root / "material_ids.npy", allow_pickle=False).astype(str)
    if max_materials is not None:
        ids = ids[:max_materials]
    material_ids = ids.tolist()
    if len(material_ids) != len(set(material_ids)):
        raise ValueError("cache material IDs must be unique")
    if not material_ids:
        raise ValueError("cache must contain at least one material")
    cache_ids_file_sha256 = _sha256_file(cache_root / "material_ids.npy")
    config_payload = _config_payload(config)
    checkpoint_path = destination / "checkpoint.json"
    shards_dir = destination / "shards"
    destination.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(exist_ok=True)

    receipts: list[SoapShardReceipt] = []
    feature_count: int | None = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_contract = {
            "schema_version": "soap_descriptors_v1",
            "cache_material_ids_file_sha256": cache_ids_file_sha256,
            "material_count": len(material_ids),
            "batch_size": batch_size,
            "config": config_payload,
        }
        for key, expected in expected_contract.items():
            if checkpoint.get(key) != expected:
                raise SoapSnapshotConflict(f"checkpoint {key} mismatch")
        receipts = [SoapShardReceipt(**item) for item in checkpoint.get("receipts", [])]
        feature_count = checkpoint.get("feature_count")
    elif any(destination.iterdir()) and list(shards_dir.iterdir()):
        raise SoapSnapshotConflict("SOAP shards exist without checkpoint")

    receipt_by_batch = {item.batch_index: item for item in receipts}
    computed_batches = 0
    reused_batches = 0
    matrices: list[scipy_sparse.csr_matrix] = []
    batch_count = (len(material_ids) + batch_size - 1) // batch_size
    for batch_index in range(batch_count):
        row_start = batch_index * batch_size
        row_stop = min(len(material_ids), row_start + batch_size)
        batch_ids = material_ids[row_start:row_stop]
        shard_path = shards_dir / f"batch-{batch_index:05d}.npz"
        existing = receipt_by_batch.get(batch_index)
        if existing is not None:
            if (existing.row_start, existing.row_stop) != (row_start, row_stop):
                raise SoapSnapshotConflict(f"receipt row interval mismatch: {batch_index}")
            matrix = _validate_reusable_shard(shard_path, existing, batch_ids)
            if feature_count is not None and matrix.shape[1] != feature_count:
                raise SoapSnapshotConflict(f"receipt feature count mismatch: {batch_index}")
            reused_batches += 1
        else:
            if shard_path.exists():
                raise SoapSnapshotConflict(f"formal shard exists without receipt: {shard_path.name}")
            matrix = compute_soap_batch(
                cache_root,
                range(row_start, row_stop),
                config,
                soap_factory=soap_factory,
                atoms_factory=atoms_factory,
                n_jobs=n_jobs,
            )
            if feature_count is None:
                feature_count = matrix.shape[1]
            elif matrix.shape[1] != feature_count:
                raise ValueError("SOAP feature count changed between batches")
            _save_sparse_atomic(shard_path, matrix)
            receipt = SoapShardReceipt(
                batch_index=batch_index,
                row_start=row_start,
                row_stop=row_stop,
                material_ids_sha256=_ids_sha256(batch_ids),
                rows=matrix.shape[0],
                columns=matrix.shape[1],
                nnz=matrix.nnz,
                file_bytes=shard_path.stat().st_size,
                file_sha256=_sha256_file(shard_path),
            )
            receipts.append(receipt)
            receipts.sort(key=lambda value: value.batch_index)
            receipt_by_batch[batch_index] = receipt
            computed_batches += 1
            checkpoint = {
                "schema_version": "soap_descriptors_v1",
                "cache_material_ids_file_sha256": cache_ids_file_sha256,
                "material_count": len(material_ids),
                "batch_size": batch_size,
                "config": config_payload,
                "feature_count": feature_count,
                "receipts": [asdict(item) for item in receipts],
            }
            _write_json_atomic(checkpoint_path, checkpoint)
        matrices.append(matrix)

    combined = scipy_sparse.vstack(matrices, format="csr", dtype=np.float32)
    combined.sort_indices()
    if combined.shape != (len(material_ids), int(feature_count or 0)):
        raise ValueError("combined SOAP matrix shape mismatch")
    aggregate_content_sha256 = _csr_content_sha256(combined)
    feature_path = destination / "soap_features.npz"
    ids_path = destination / "soap_material_ids.npy"
    config_path = destination / "soap_config.json"
    metadata_path = destination / "soap_metadata.json"
    if feature_path.exists():
        published = scipy_sparse.load_npz(feature_path).tocsr().astype(np.float32)
        if _csr_content_sha256(published) != aggregate_content_sha256:
            raise SoapSnapshotConflict("published SOAP aggregate content hash mismatch")
    else:
        _save_sparse_atomic(feature_path, combined)
    if ids_path.exists():
        if np.load(ids_path, allow_pickle=False).astype(str).tolist() != material_ids:
            raise SoapSnapshotConflict("published SOAP material ID order mismatch")
    else:
        _save_npy_atomic(ids_path, np.asarray(material_ids))
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config_payload:
            raise SoapSnapshotConflict("published SOAP config mismatch")
    else:
        _write_json_atomic(config_path, config_payload)

    metadata = {
        "schema_version": "soap_descriptors_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "complete": True,
        "material_count": len(material_ids),
        "feature_count": int(feature_count or 0),
        "batch_count": batch_count,
        "batch_size": batch_size,
        "n_jobs": n_jobs,
        "aggregate_content_sha256": aggregate_content_sha256,
        "output_file_sha256": {
            "soap_features.npz": _sha256_file(feature_path),
            "soap_material_ids.npy": _sha256_file(ids_path),
            "soap_config.json": _sha256_file(config_path),
        },
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "material_count", "feature_count", "batch_count", "batch_size", "aggregate_content_sha256"):
            if old.get(key) != metadata.get(key):
                raise SoapSnapshotConflict(f"published SOAP metadata {key} mismatch")
    else:
        _write_json_atomic(metadata_path, metadata)
    return {
        "status": "complete",
        "complete": True,
        "material_count": len(material_ids),
        "feature_count": int(feature_count or 0),
        "batch_count": batch_count,
        "computed_batches": computed_batches,
        "reused_batches": reused_batches,
        "soap_create_calls": computed_batches,
        "aggregate_content_sha256": aggregate_content_sha256,
        "output_dir": str(destination.resolve()),
    }
