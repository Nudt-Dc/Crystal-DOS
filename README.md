# Crystal + DOS: core implementation

Core implementation of training-time electronic-structure knowledge transfer for crystal-only prediction of bulk and shear moduli, K and G (GPa).

The paper's matched low-label experiment uses `crystal_dos_low_label_loco_v1.py` and `PrivilegedCrystalDosModelV3`. All three conditions share the same model, structure batches, label masks and crystal-only validation procedure. The conditions are `lowlabel_structure`, `lowlabel_full_correct`, and `lowlabel_full_shuffled`. DOS is used only during training; inference accepts crystal graphs and the checkpoint's target normalizers.

This source release contains the original model/training/preprocessing modules and their local import dependencies, a small crystal-only prediction CLI, and synthetic unit tests. It does not contain raw materials datasets, pretrained checkpoints, the complete baseline suite, or the external-evaluation execution infrastructure. Code availability is distinct from full numerical reproduction of the paper, which also requires the original data snapshots and frozen partitions/control masks. This repository does not redistribute MP or AFLOW raw data.

## Environment and tests

Use Python 3.11. The core dependencies are the versions recorded in the training environment; CUDA-enabled PyTorch may be installed from its official wheel index when needed. Preprocessing has additional dependencies.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-preprocessing.txt
python -m pytest -q
```

The tests use synthetic arrays, synthetic structures, and temporary files. They do not download datasets or run scientific training. `ENVIRONMENT.md` records the scope of the version information. The published project source modules are unchanged from the original implementation; see `SOURCE_MAP.md` for the only layout adaptations and added interface.

## Model and matched conditions

The structure encoder uses the included official CGCNN model with three convolutions, 64 atom features and a 128-dimensional crystal representation. A two-layer 1D CNN encodes DOS to 64 dimensions. Crystal-derived DOS latents, alignment projection, binary observation mask, and two separate K/G heads are implemented in `crystal_dos_privileged_v3.py` and `crystal_dos_privileged.py`.

The paper's low-label runner uses masked smooth-L1 property loss, symmetric contrastive alignment (weight 0.02; temperature 0.1), and stop-gradient latent distillation (weight 0.1). Paired and shuffled conditions drop 50% of the available training DOS inputs per epoch while retaining teacher supervision. The structure condition uses zero observation/teacher masks and zero auxiliary weights. Shuffling is a deterministic derangement within observed training DOS rows; validation and test rows are untouched. Every training structure participates each epoch, but K/G supervision and target standardization use only the selected labeled training rows.

The protocol is 30 epochs, logical batch size 256, Adam learning rate 0.001, label fractions 1%, 5%, 10%, label-mask repeat 0, and seeds 20260910/20260911/20260912. The reported MP comparison uses `loco_01` through `loco_09`; AFLOW uses `loco_00` through `loco_08`. These are 243 runs per resource across the three conditions. The selected checkpoint minimizes the mean of the K/G validation MAEs standardized by their labeled-training scales. Validation is crystal-only.

## Training

Run from this directory. The paths below refer to your locally prepared dataset directories, described in `DATA_SCHEMA.md`. Data must have consistent identifiers, row order, metadata and hashes; the loader validates those contracts.

```bash
python crystal_dos_low_label_loco_v1.py train \
  --master-dir data/master \
  --crystal-cache-dir data/crystals \
  --graph-cache-dir data/graphs \
  --dos-dir data/dos \
  --split-dir data/splits \
  --controls-dir data/controls \
  --official-model-py cgcnn_model.py \
  --atom-init-json atom_init.json \
  --evaluation-unit loco_01 \
  --fraction-key label_fraction_010 \
  --label-repeat 0 \
  --variant lowlabel_full_correct \
  --epochs 30 --batch-size 256 --learning-rate 0.001 \
  --seed 20260910 --device cuda:0 \
  --output-dir results/mp/loco_01/10pct/seed20260910/paired
```

Repeat with `lowlabel_structure` and `lowlabel_full_shuffled` using identical data, fold, label fraction and seed, and distinct output directories. For AFLOW's amended split, also provide `--soap-dir data/soap_splits` pointing to the original SOAP embedding source. `run_aflow_low_label_matrix_worker.py` accepts explicit paths, folds, fractions and seeds for sequential runs. `--device cpu` is supported for small technical checks.

Each training run writes `checkpoint.pt`, `epoch_history.csv`, `validation_predictions.csv`, label/DOS assignment CSVs and a run summary. The `audit` subcommand accepts the same input paths plus `--result-dir` and `--output-json` and checks the stored checkpoint/predictions. It performs validation inference when invoked; it is not required for importing the package or running its synthetic tests.

## Crystal-only inference

Supply a checkpoint from the matched runner and a graph cache generated with the same graph settings (8 Angstrom cutoff, 12 neighbors). No DOS or K/G labels are required.

```bash
python predict_crystal_only.py \
  --checkpoint results/mp/loco_01/10pct/seed20260910/paired/checkpoint.pt \
  --graph-cache-dir data/new_graphs \
  --output predictions.csv --device cpu
```

The output columns are `material_id`, `k_prediction_gpa`, and `g_prediction_gpa`, in graph-cache ID order. This CLI loads the model/normalizers and calls the original `predict_target_blind_cgcnn` kernel and `predict_structure_only` method. It adds no training or prediction algorithm.

## Preparing inputs

`run_dos_preprocess.py` handles validated MP-style extracted total DOS; `run_aflow_confirmatory_preprocess.py` handles the AFLOW cohort format. Both use the fixed -10 to +8 eV, 576-point representation. The source modules preserve the original format checks and missingness handling.

`run_crystal_dos_master.py`, `run_crystal_cache.py`, and `run_crystal_graph_cache.py` build MP master/structure/graph caches. The corresponding AFLOW master/cache entrypoints are included. `run_soap_descriptors.py`, `run_experiment_splits.py`, and `run_experiment_controls.py` implement the descriptor, split and control generation rules; `aflow_confirmatory_split_amendment.py` contains the deterministic AFLOW split-coarsening rule. Use each entrypoint's `--help` and `DATA_SCHEMA.md`. Historical cohort counts in CLI defaults describe the original datasets and must be supplied explicitly for other data. Regenerating partitions from a new data snapshot is a new experiment, not exact reproduction of the original partitions.

## Attribution and licensing

The included CGCNN files retain their original MIT license in `LICENSE-CGCNN.txt`; see `THIRD_PARTY.md` for exact provenance. No repository-wide license has been assigned to the remaining project code in this release. Public source availability alone does not grant an additional license.
