# Input and output contracts

The release preserves the original loaders; all paths are supplied by CLI arguments. The training loader is `load_diagnostic_data` in `diagnostic_baselines.py`. Arrays must have the ID/row order recorded in the master table and matching SHA-256 metadata. Merely renaming unrelated arrays to these file names is insufficient.

## Training inputs

| CLI directory | Required contents |
| --- | --- |
| `--master-dir` | `master_table.parquet`, `master_metadata.json` |
| `--crystal-cache-dir` | `material_ids.npy`, `cache_metadata.json`; complete preprocessing cache additionally includes lattice/site arrays below |
| `--graph-cache-dir` | Complete graph cache below |
| `--dos-dir` | `material_ids.npy`, `dos_features.npy`, and `quality_summary.json` or `preprocessing_config.json` |
| `--split-dir` | `split_metadata.json`, `soap_loco_folds.parquet`; `soap_svd_256.npy` when SOAP embeddings are stored here |
| `--soap-dir` (AFLOW amended splits) | Original `split_metadata.json` and `soap_svd_256.npy`, bound by the amendment metadata |
| `--controls-dir` | `controls_metadata.json` and `loco/loco_NN/` with the fixed per-unit controls |

The minimum master columns consumed by the training loader are `material_id`, `crystal_row`, `dos_row`, `dos_observed`, `k_vrh_gpa`, `g_vrh_gpa`. IDs are unique strings; crystal rows index the crystal cache. Observed DOS rows are valid integer indices, and missing DOS rows are null. K and G are finite values in GPa. Low-label masking happens after loading: only selected training K/G values contribute to the property loss or target normalization.

`dos_features.npy` is a float array of shape `(observed_materials, 1, 576)`. The associated Unicode ID vector maps each spectrum to the master. DOS is aligned to the Fermi level, checked for the fixed -10 to +8 eV window, interpolated and area-normalized by the source-specific preprocessing module. The training runner then standardizes each bin with observed training DOS statistics.

`soap_loco_folds.parquet` includes `fold`, `material_id`, `split`, `stable_order_key`, `crystal_row`, and `dos_row`. For each fold every master ID appears exactly once, with `split` equal to `train`, `validation`, or `test`. `soap_svd_256.npy` is a floating matrix aligned to master rows. The core neural model does not consume SOAP; this artifact belongs to the fixed structure-based split contract.

Each control unit includes `train_material_ids.npy`, `low_label_masks.npz`, `paired_test_material_ids.npy`, `paired_test_dos_rows.npy`, `full_test_material_ids.npy`, `evaluation_missingness_masks.npz`, `shuffled_dos_source_rows.npy`, and `control_summary.json`. Low-label NPZ keys are `label_fraction_001`, `label_fraction_005`, `label_fraction_010`; each has five mask rows, and the paper uses row/repeat 0. The loader verifies the complete control contract, even though the paper's low-label validation runner never evaluates the held-out test partition.

## Structure and graph caches

The structure cache used by `run_crystal_graph_cache.py` contains `material_ids.npy`, `lattice_matrices.npy`, `fractional_coordinates.npy`, `atomic_numbers.npy`, `site_offsets.npy`, and `cache_metadata.json`. Pymatgen builds periodic edges with an 8 Angstrom cutoff and at most 12 neighbors per atom for the paper recipe.

The graph cache inventory is exactly: `material_ids.npy`, `atomic_numbers.npy`, `site_offsets.npy`, `edge_index.npy`, `edge_distances.npy`, `edge_image_vectors.npy`, `graph_edge_offsets.npy`, `neighbor_counts.npy`, `graph_audit.json`, and `graph_metadata.json`. Edges point from neighbor to center. Graph metadata records counts, schema and output hashes. The bundled official CGCNN atom features cover elements 1 through 100.

## Preprocessing entrypoints

```bash
python run_dos_preprocess.py --sample-dir data/extracted_mp_dos --output-dir data/dos --source-manifest dos_sample_manifest.csv --coverage-policy exclude
python run_crystal_graph_cache.py --source-cache-dir data/crystals --output-dir data/graphs --cutoff 8.0 --max-neighbors 12
python run_soap_descriptors.py --cache-dir data/crystals --output-dir data/soap --n-jobs 1
python run_experiment_controls.py --master-dir data/master --dos-dir data/dos --split-dir data/splits --output-dir data/controls
```

The MP DOS input is the project's validated extracted-spectrum format, with its manifest and per-spectrum arrays (`energies_ev`, `efermi_ev`, `energy_minus_efermi_ev`, `density_up`, `density_down`, `density_total`, spin-presence flags). AFLOW uses downloaded cohort records and the cohort audit's selected entries. These entrypoints do not provide a bulk-download service or redistribute the source snapshots. The master/cohort generators preserve dataset-specific contracts; consult `--help` and the corresponding module before preparing a different resource.

## Checkpoint and inference

The matched runner saves checkpoint schema `crystal_dos_low_label_loco_checkpoint_v1`, model `state_dict`, DOS/target normalizers, selected epoch, fold, fraction and variant. Crystal-only inference uses only `state_dict`, `normalizers.target_mean`, `normalizers.target_scale`, the graph cache, and CGCNN atom initialization. Predictions are inverse-transformed to GPa. No claim of experimental validation is implied by these computational predictions.
