# Source map

The 42 research source modules in this flat release are byte-for-byte copies of the original files with the same names in the project's `work/src/` directory. Their historical module names and docstrings are retained, including v1/v2 dependencies imported for helpers. The paper's matched training entrypoint is specifically `crystal_dos_low_label_loco_v1.py`, which constructs `PrivilegedCrystalDosModelV3`; older helper modules are not alternate recommended paper recipes.

| Purpose | Modules |
| --- | --- |
| Matched structure/paired/shuffled training | `crystal_dos_low_label_loco_v1.py`, `run_aflow_low_label_matrix_worker.py` |
| Model | `crystal_dos_privileged.py`, `crystal_dos_privileged_v3.py` |
| Training inputs and controls | `diagnostic_baselines.py`, `privileged_training_data.py`, `privileged_training_shuffle.py`, `experiment_controls.py` |
| Graph cache and official CGCNN adapter | `crystal_cgcnn_smoke.py`, `crystal_graph_cache.py`, `official_cgcnn_iid.py` |
| Crystal-only prediction kernel | `outer_target_blind_cgcnn_batch.py`, `outer_target_blind_cgcnn_predict.py` |
| MP preprocessing | `dos_preprocess.py`, `dos_labeled_subset.py`, `crystal_dos_master.py`, `crystal_cache.py`, corresponding `run_*.py` entrypoints |
| AFLOW preprocessing | `aflow_confirmatory_*.py`, corresponding `run_*.py` entrypoints |
| SOAP and partitions | `soap_descriptors.py`, `experiment_splits.py`, `aflow_confirmatory_split_amendment.py`, corresponding `run_*.py` entrypoints |
| Import dependencies retained intact | `privileged_iid_freeze.py`, `privileged_iid_train.py`, `privileged_iid_train_v2.py`, `privileged_loco_train_v2.py`, `crystal_dos_privileged_v2.py`, `audit_aflow_confirmatory_cohort.py`, `jarvis_mp_overlap_audit.py` |

Nine original synthetic-test files are copied from `work/tests/`. Only the source-file lookup in `test_dos_preprocess.py` changes to the flat layout. `test_release_entrypoints.py` is a release integration test for the included official model and crystal-only method. `predict_crystal_only.py` is a small new argument/file-I/O wrapper around the unchanged prediction implementation. Documentation and requirements files are added for this release.

No research algorithm, loss, model layer, split rule, seed derivation, normalization rule or checkpoint format was rewritten for this release.
