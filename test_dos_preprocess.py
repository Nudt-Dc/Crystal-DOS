import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "dos_preprocess.py"


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError("dos_preprocess.py must implement the preprocessing pipeline")
    spec = importlib.util.spec_from_file_location("dos_preprocess", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_arrays(energy=None, density=None):
    energy = np.asarray(
        energy if energy is not None else [-10.0, -1.0, 8.0], dtype=float
    )
    density = np.asarray(
        density if density is not None else [0.0, 2.0, 0.0], dtype=float
    )
    return {
        "energies_ev": energy + 1.5,
        "efermi_ev": np.asarray(1.5),
        "energy_minus_efermi_ev": energy,
        "density_up": density,
        "density_down": np.zeros_like(density),
        "density_total": density,
        "spin_up_present": np.asarray(True),
        "spin_down_present": np.asarray(False),
    }


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sample(root, *, incomplete_second=False):
    source = root / "source"
    files_dir = source / "files"
    files_dir.mkdir(parents=True)
    material_ids = ["mp-a", "mp-b"]
    selected = pd.DataFrame(
        {
            "material_id": material_ids,
            "formula_pretty": ["Si", "Al"],
            "k_vrh_gpa": [100.0, 80.0],
            "g_vrh_gpa": [50.0, 30.0],
        }
    )
    selected.to_csv(source / "selected_materials.csv", index=False)

    manifest_rows = []
    for index, material_id in enumerate(material_ids):
        energy = [-9.0, -1.0, 8.0] if incomplete_second and index == 1 else None
        path = files_dir / f"{material_id}.npz"
        np.savez_compressed(path, **make_arrays(energy=energy))
        manifest_rows.append(
            {
                "material_id": material_id,
                "status": "success",
                "sha256": file_sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        source / "dos_sample_manifest.csv", index=False
    )
    return source


class DosPreprocessTests(unittest.TestCase):
    def test_preprocess_returns_fixed_grid_float32_unit_area(self):
        module = load_module()

        result = module.preprocess_dos_arrays(
            make_arrays(), module.DosPreprocessConfig()
        )

        self.assertEqual(result["feature"].shape, (1, 576))
        self.assertEqual(result["feature"].dtype, np.float32)
        self.assertEqual(result["energy_grid_ev"].dtype, np.float32)
        self.assertTrue(np.isfinite(result["feature"]).all())
        self.assertGreaterEqual(float(result["feature"].min()), 0.0)
        self.assertAlmostEqual(
            float(
                np.trapezoid(
                    result["feature"][0], result["energy_grid_ev"]
                )
            ),
            1.0,
            places=5,
        )

    def test_preprocess_rejects_incomplete_energy_coverage(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "does not cover"):
            module.preprocess_dos_arrays(
                make_arrays(energy=[-9.9, 0.0, 8.0]),
                module.DosPreprocessConfig(),
            )

    def test_preprocess_rejects_density_sum_mismatch(self):
        module = load_module()
        arrays = make_arrays()
        arrays["density_total"] = arrays["density_total"] + 1.0

        with self.assertRaisesRegex(ValueError, "spin sum"):
            module.preprocess_dos_arrays(arrays, module.DosPreprocessConfig())

    def test_preprocess_rejects_material_negative_density(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "negative"):
            module.preprocess_dos_arrays(
                make_arrays(density=[0.0, -1e-4, 0.0]),
                module.DosPreprocessConfig(),
            )

    def test_config_rejects_invalid_grid(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "point_count"):
            module.DosPreprocessConfig(point_count=1)

    def test_evaluation_masks_are_reproducible_exact_and_nested(self):
        module = load_module()
        rates = (0.0, 0.25, 0.5, 0.75, 1.0)

        first = module.build_evaluation_masks(50, rates, 5, 20260907)
        second = module.build_evaluation_masks(50, rates, 5, 20260907)

        expected_missing = [0, 13, 25, 38, 50]
        keys = [
            "missing_rate_000",
            "missing_rate_025",
            "missing_rate_050",
            "missing_rate_075",
            "missing_rate_100",
        ]
        for key, missing_count in zip(keys, expected_missing, strict=True):
            np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(first[key].shape, (5, 50))
            self.assertEqual(first[key].dtype, np.uint8)
            self.assertTrue(
                np.all((first[key] == 0).sum(axis=1) == missing_count)
            )

        for repeat in range(5):
            previous_missing = set()
            for key in keys:
                current_missing = set(np.flatnonzero(first[key][repeat] == 0))
                self.assertTrue(previous_missing.issubset(current_missing))
                previous_missing = current_missing

    def test_evaluation_masks_reject_invalid_arguments(self):
        module = load_module()

        invalid_calls = [
            (0, (0.0,), 1),
            (10, (-0.1,), 1),
            (10, (1.1,), 1),
            (10, (0.5,), 0),
        ]
        for material_count, rates, repeats in invalid_calls:
            with self.subTest(
                material_count=material_count, rates=rates, repeats=repeats
            ):
                with self.assertRaises(ValueError):
                    module.build_evaluation_masks(
                        material_count, rates, repeats, 20260907
                    )

    def test_materialize_writes_ordered_model_ready_outputs(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = write_sample(root)
            destination = root / "output"

            manifest, summary = module.materialize_preprocessed_dataset(
                source,
                destination,
                config=module.DosPreprocessConfig(),
                missing_rates=(0.0, 0.5, 1.0),
                repeats=2,
                base_seed=7,
            )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["success_count"], 2)
            self.assertEqual(summary["failed_count"], 0)
            np.testing.assert_array_equal(
                np.load(destination / "material_ids.npy"), ["mp-a", "mp-b"]
            )
            self.assertEqual(
                np.load(destination / "dos_features.npy").shape, (2, 1, 576)
            )
            self.assertTrue(
                (np.load(destination / "dos_observed_mask.npy") == 1).all()
            )
            self.assertTrue((destination / "preprocessing_config.json").is_file())
            self.assertTrue((destination / "quality_summary.json").is_file())
            self.assertEqual(
                manifest["source_sha256_matches"].tolist(), [True, True]
            )

    def test_materialize_records_partial_result_without_publishing_features(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = write_sample(root, incomplete_second=True)
            destination = root / "output"

            manifest, summary = module.materialize_preprocessed_dataset(
                source,
                destination,
                config=module.DosPreprocessConfig(),
                missing_rates=(0.0, 0.5, 1.0),
                repeats=2,
                base_seed=7,
            )

            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertFalse((destination / "dos_features.npy").exists())
            failed = manifest.loc[manifest["status"] == "failed"].iloc[0]
            self.assertEqual(failed["error_type"], "ValueError")
            self.assertTrue(failed["source_sha256_matches"])

    def test_strict_window_mode_excludes_only_incomplete_coverage(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = write_sample(root, incomplete_second=True)
            (source / "dos_sample_manifest.csv").replace(
                source / "full_dos_manifest.csv"
            )
            destination = root / "output"

            manifest, summary = module.materialize_preprocessed_dataset(
                source,
                destination,
                config=module.DosPreprocessConfig(),
                missing_rates=(0.0, 0.5, 1.0),
                repeats=2,
                base_seed=7,
                source_manifest_name="full_dos_manifest.csv",
                coverage_policy="exclude",
                spec_version="dos-full-preprocessing-v2-strict-window",
            )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["excluded_count"], 1)
            self.assertEqual(summary["failed_count"], 0)
            np.testing.assert_array_equal(
                np.load(destination / "material_ids.npy"), ["mp-a"]
            )
            self.assertEqual(
                np.load(destination / "dos_features.npy").shape, (1, 1, 576)
            )
            self.assertEqual(manifest["status"].tolist(), ["success", "excluded"])
            exclusions = pd.read_csv(destination / "coverage_exclusions.csv")
            self.assertEqual(exclusions["material_id"].tolist(), ["mp-b"])
            config = json.loads(
                (destination / "preprocessing_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["spec_version"],
                "dos-full-preprocessing-v2-strict-window",
            )


if __name__ == "__main__":
    unittest.main()
