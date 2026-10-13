import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice

from crystal_graph_cache import build_periodic_edges, materialize_graph_cache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_cache(root: Path) -> None:
    root.mkdir()
    arrays = {
        "material_ids.npy": np.asarray(["mp-1", "mp-2"]),
        "lattice_matrices.npy": np.stack([np.eye(3), np.eye(3)]),
        "fractional_coordinates.npy": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "atomic_numbers.npy": np.asarray([14, 8], dtype=np.int16),
        "site_offsets.npy": np.asarray([0, 1, 2], dtype=np.int64),
    }
    for name, value in arrays.items():
        np.save(root / name, value, allow_pickle=False)
    metadata = {
        "schema_version": "crystal_cache_v1",
        "status": "complete",
        "complete": True,
        "material_count": 2,
        "total_sites": 2,
        "output_file_sha256": {name: _sha256(root / name) for name in arrays},
    }
    (root / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_simple_cubic_periodic_neighbors_and_deterministic_cap():
    lattice = Lattice.cubic(1.0).matrix
    frac = np.asarray([[0.0, 0.0, 0.0]])
    full = build_periodic_edges(lattice, frac, cutoff=1.01, max_neighbors=6)
    assert full["edge_index"].shape == (2, 6)
    np.testing.assert_allclose(full["edge_distances"], 1.0)
    assert {tuple(value) for value in full["edge_image_vectors"].tolist()} == {
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
    }
    capped = build_periodic_edges(lattice, frac, cutoff=1.01, max_neighbors=4)
    np.testing.assert_array_equal(capped["edge_image_vectors"], full["edge_image_vectors"][:4])
    np.testing.assert_array_equal(capped["neighbor_counts"], np.asarray([4], dtype=np.int16))


def test_materialize_graph_cache_validates_and_reuses(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "graphs"
    _source_cache(source)
    first = materialize_graph_cache(source, output, cutoff=1.01, max_neighbors=6)
    second = materialize_graph_cache(source, output, cutoff=1.01, max_neighbors=6)
    assert first["complete"] is True and first["reused"] is False
    assert second["complete"] is True and second["reused"] is True
    metadata = json.loads((output / "graph_metadata.json").read_text(encoding="utf-8"))
    assert metadata["material_count"] == 2
    assert metadata["total_sites"] == 2
    assert metadata["total_edges"] == 12
    assert metadata["insufficient_neighbor_site_count"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "material_ids.npy", "atomic_numbers.npy", "site_offsets.npy", "edge_index.npy",
        "edge_distances.npy", "edge_image_vectors.npy", "graph_edge_offsets.npy",
        "neighbor_counts.npy", "graph_audit.json", "graph_metadata.json"
    }
    distances = np.load(output / "edge_distances.npy", allow_pickle=False)
    assert np.all((distances > 0) & (distances <= 1.01))


def test_existing_corrupt_output_fails_closed(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "graphs"
    _source_cache(source)
    materialize_graph_cache(source, output, cutoff=1.01, max_neighbors=6)
    (output / "edge_distances.npy").write_bytes(b"corrupt")
    try:
        materialize_graph_cache(source, output, cutoff=1.01, max_neighbors=6)
    except FileExistsError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("corrupt existing output was reused")
