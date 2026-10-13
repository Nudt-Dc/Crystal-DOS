"""Low-label Crystal + electronic-DOS privileged-learning utilities and runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from crystal_cgcnn_smoke import CrystalGraphCache, subset_sha256
from crystal_dos_privileged import OfficialCgcnnEncoder
from crystal_dos_privileged_v3 import PrivilegedCrystalDosModelV3
from diagnostic_baselines import DiagnosticData, load_diagnostic_data
from official_cgcnn_iid import batch_official_graphs, build_atom_feature_lookup
from privileged_iid_freeze import sha256_file
from privileged_iid_train import signature_sha256
from privileged_loco_train_v2 import _require_loco_unit
from privileged_training_data import TrainNormalizers, build_epoch_plan
from privileged_training_shuffle import build_training_dos_assignment


FRACTIONS = {
    "label_fraction_001": 0.01,
    "label_fraction_005": 0.05,
    "label_fraction_010": 0.10,
}
VARIANT_CONFIG = {
    "lowlabel_structure": (0.0, 0.0, "always_missing", "identity"),
    "lowlabel_full_correct": (0.02, 0.1, "deterministic_dropout", "identity"),
    "lowlabel_full_shuffled": (
        0.02,
        0.1,
        "deterministic_dropout",
        "within_train_observed_sattolo",
    ),
}
VARIANTS = tuple(VARIANT_CONFIG)
INVENTORY = {
    "checkpoint.pt",
    "epoch_history.csv",
    "validation_predictions.csv",
    "label_assignment.csv",
    "dos_assignment.csv",
    "quality_report.md",
    "run_summary.json",
}


def compute_low_label_normalizers(
    data: DiagnosticData, membership: np.ndarray
) -> TrainNormalizers:
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    labelled = train[np.asarray(membership[train], dtype=bool)]
    paired = np.asarray(data.paired_partitions["train"], dtype=np.int64)
    if len(labelled) < 2 or len(paired) < 2:
        raise ValueError("low-label normalization requires at least two rows")
    dos = np.asarray(data.dos_features[data.dos_rows[paired]], dtype=np.float64)
    targets = np.asarray(data.targets[labelled], dtype=np.float64)
    dos_mean, dos_scale = dos.mean(axis=0), dos.std(axis=0, ddof=1)
    target_mean, target_scale = targets.mean(axis=0), targets.std(axis=0, ddof=1)
    dos_scale[dos_scale < 1e-8] = 1.0
    if not all(np.isfinite(x).all() for x in (dos_mean, dos_scale, target_mean, target_scale)):
        raise ValueError("low-label normalizers must be finite")
    if np.any(dos_scale <= 0) or np.any(target_scale <= 0):
        raise ValueError("low-label scales must be positive")
    return TrainNormalizers(
        dos_mean.astype(np.float32),
        dos_scale.astype(np.float32),
        target_mean.astype(np.float32),
        target_scale.astype(np.float32),
    )


def low_label_losses(
    output,
    standardized_targets: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    alignment_weight: float,
    distillation_weight: float,
    teacher_mask: torch.Tensor,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    labelled = label_mask.reshape(-1).to(output.prediction.device).bool()
    observed = teacher_mask.reshape(-1).to(output.prediction.device).bool()
    if labelled.shape[0] != output.prediction.shape[0]:
        raise ValueError("label mask length mismatch")
    if observed.shape[0] != output.prediction.shape[0]:
        raise ValueError("teacher mask length mismatch")
    if labelled.any():
        by_target = F.smooth_l1_loss(
            output.prediction[labelled], standardized_targets[labelled], reduction="none"
        ).mean(dim=0)
    else:
        by_target = output.prediction.sum(dim=0) * 0.0
    property_loss = by_target.mean()
    if int(observed.sum()) >= 2:
        crystal = F.normalize(output.aligned_crystal_latent[observed], dim=1)
        dos = F.normalize(output.observed_dos_latent[observed], dim=1)
        logits = crystal @ dos.T / float(temperature)
        labels = torch.arange(len(logits), device=logits.device)
        alignment = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
    else:
        alignment = output.prediction.sum() * 0.0
    if observed.any():
        distillation = F.mse_loss(
            output.predicted_dos_latent[observed],
            output.observed_dos_latent[observed].detach(),
        )
    else:
        distillation = output.prediction.sum() * 0.0
    total = property_loss + alignment_weight * alignment + distillation_weight * distillation
    return {
        "total": total,
        "property": property_loss,
        "property_k": by_target[0],
        "property_g": by_target[1],
        "alignment": alignment,
        "distillation": distillation,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _control_root(controls_dir: Path | str, evaluation_unit: str) -> Path:
    return Path(controls_dir) / "loco" / _require_loco_unit(evaluation_unit)


def load_label_membership(
    data: DiagnosticData,
    controls_dir: Path | str,
    *,
    evaluation_unit: str,
    fraction_key: str,
    repeat: int,
) -> tuple[np.ndarray, dict[str, str]]:
    if fraction_key not in FRACTIONS or repeat != 0:
        raise ValueError("pilot requires a frozen fraction key and repeat 0")
    root = _control_root(controls_dir, evaluation_unit)
    ids_path = root / "train_material_ids.npy"
    masks_path = root / "low_label_masks.npz"
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    control_ids = np.load(ids_path, allow_pickle=False)
    if control_ids.ndim != 1 or not np.array_equal(control_ids, data.material_ids[train]):
        raise ValueError("control train IDs do not align with the training partition")
    with np.load(masks_path, allow_pickle=False) as archive:
        if fraction_key not in archive.files:
            raise ValueError("requested fraction is absent from controls")
        matrix = np.asarray(archive[fraction_key])
    if matrix.shape != (5, len(train)):
        raise ValueError("low-label control shape is invalid")
    relative = np.asarray(matrix[repeat], dtype=np.uint8)
    if not np.all((relative == 0) | (relative == 1)):
        raise ValueError("low-label control must be binary")
    expected = int(math.floor(FRACTIONS[fraction_key] * len(train) + 0.5))
    if int(relative.sum()) != expected or expected < 2:
        raise ValueError("low-label count differs from the frozen fraction")
    membership = np.zeros(len(data.material_ids), dtype=np.uint8)
    membership[train] = relative
    return membership, {
        "controls/train_material_ids.npy": sha256_file(ids_path),
        "controls/low_label_masks.npz": sha256_file(masks_path),
    }


def low_label_batch_arrays(
    data: DiagnosticData,
    positions: np.ndarray,
    normalizers: TrainNormalizers,
    membership: np.ndarray,
    assigned_dos_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(positions, dtype=np.int64)
    teacher = np.asarray(data.observed[selected], dtype=np.float32)
    labelled = np.asarray(membership[selected], dtype=np.float32)
    dos = np.zeros((len(selected), 1, data.dos_features.shape[2]), dtype=np.float32)
    available = teacher == 1
    if available.any():
        rows = np.asarray(assigned_dos_rows[selected[available]], dtype=np.int64)
        if np.any(rows < 0):
            raise ValueError("observed row has no assigned DOS")
        raw = np.asarray(data.dos_features[rows], dtype=np.float32)
        dos[available] = (raw - normalizers.dos_mean) / normalizers.dos_scale
    targets = np.zeros((len(selected), 2), dtype=np.float32)
    use = labelled == 1
    if use.any():
        targets[use] = (
            np.asarray(data.targets[selected[use]], dtype=np.float32)
            - normalizers.target_mean
        ) / normalizers.target_scale
    if not np.isfinite(dos).all() or not np.isfinite(targets).all():
        raise ValueError("low-label batch contains non-finite values")
    return dos, targets, teacher, labelled


def _batches(positions: np.ndarray, batch_size: int) -> list[np.ndarray]:
    result = [positions[i : i + batch_size] for i in range(0, len(positions), batch_size)]
    if len(result) > 1 and len(result[-1]) == 1:
        result[-2] = np.concatenate([result[-2], result[-1]])
        result.pop()
    return result


def evaluate_crystal_only(
    model: PrivilegedCrystalDosModelV3,
    data: DiagnosticData,
    cache: CrystalGraphCache,
    atom_lookup: np.ndarray,
    normalizers: TrainNormalizers,
    positions: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    blocks = []
    model.eval()
    with torch.no_grad():
        for selected in _batches(positions, batch_size):
            graph = batch_official_graphs(
                cache, selected, data.targets, atom_lookup, target_index=0, device=device
            )
            blocks.append(model.predict_structure_only(graph).cpu().numpy())
    prediction_std = np.concatenate(blocks)
    truth_std = (
        np.asarray(data.targets[positions], dtype=np.float32) - normalizers.target_mean
    ) / normalizers.target_scale
    target_scores = np.mean(np.abs(prediction_std - truth_std), axis=0)
    prediction = prediction_std * normalizers.target_scale + normalizers.target_mean
    truth = np.asarray(data.targets[positions], dtype=np.float64)
    residual = prediction - truth
    metrics = {
        "selection_score": float(target_scores.mean()),
        "target_selection_score": {"k": float(target_scores[0]), "g": float(target_scores[1])},
        "crystal_only": {
            "mae_gpa": np.mean(np.abs(residual), axis=0).astype(float).tolist(),
            "rmse_gpa": np.sqrt(np.mean(residual**2, axis=0)).astype(float).tolist(),
        },
    }
    return prediction.astype(np.float32), metrics


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "crystal_dos_privileged.py",
        "crystal_dos_privileged_v3.py",
        "privileged_training_data.py",
        "privileged_training_shuffle.py",
        "crystal_dos_low_label_loco_v1.py",
    )
    return {name: sha256_file(root / name) for name in names}


def build_signature(
    data: DiagnosticData,
    cache: CrystalGraphCache,
    controls_dir: Path | str,
    official_model_py: Path | str,
    atom_init_json: Path | str,
    *,
    evaluation_unit: str,
    fraction_key: str,
    label_repeat: int,
    variant: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    base_seed: int,
) -> tuple[dict, np.ndarray]:
    evaluation_unit = _require_loco_unit(evaluation_unit)
    if variant not in VARIANTS:
        raise ValueError("unknown low-label variant")
    membership, control_hashes = load_label_membership(
        data,
        controls_dir,
        evaluation_unit=evaluation_unit,
        fraction_key=fraction_key,
        repeat=label_repeat,
    )
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    validation = np.asarray(data.partitions["validation"], dtype=np.int64)
    labelled = train[np.asarray(membership[train], dtype=bool)]
    alignment, distillation, input_policy, assignment_policy = VARIANT_CONFIG[variant]
    signature = {
        "schema_version": "crystal_dos_low_label_loco_signature_v1",
        "evaluation_unit": evaluation_unit,
        "fraction_key": fraction_key,
        "label_repeat": label_repeat,
        "variant": variant,
        "source_sha256": {**dict(data.source_sha256), **control_hashes},
        "graph_metadata_sha256": sha256_file(cache.root / "graph_metadata.json"),
        "official_model_sha256": sha256_file(official_model_py),
        "atom_init_sha256": sha256_file(atom_init_json),
        "implementation_sha256": _implementation_hashes(),
        "partition_material_ids_sha256": {
            "train": subset_sha256(data.material_ids[train]),
            "labelled_train": subset_sha256(data.material_ids[labelled]),
            "validation": subset_sha256(data.material_ids[validation]),
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "base_seed": base_seed,
            "label_fraction": FRACTIONS[fraction_key],
            "property_loss": "label_masked_equal_target_smooth_l1",
            "target_normalization": "labelled_training_rows_only",
            "dos_normalization": "all_observed_training_rows_only",
            "alignment_weight": alignment,
            "distillation_weight": distillation,
            "modality_dropout": 1.0 if variant == "lowlabel_structure" else 0.5,
            "input_policy": input_policy,
            "dos_assignment_policy": assignment_policy,
            "selection": "crystal_only_validation_mean_standardized_k_g_mae",
            "student_inference_modalities": ["crystal"],
        },
    }
    return signature, membership


def build_dos_assignment(data, evaluation_unit: str, base_seed: int, variant: str):
    control = build_training_dos_assignment(
        data, evaluation_unit=evaluation_unit, base_seed=base_seed
    )
    assigned = np.asarray(data.dos_rows, dtype=np.int64).copy()
    if variant == "lowlabel_full_shuffled":
        assigned = control.full_dos_rows.copy()
    return control, assigned


def assignment_csv_text(data, membership, control, assigned_rows) -> tuple[str, str]:
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    labelled = train[np.asarray(membership[train], dtype=bool)]
    labels = io.StringIO(newline="")
    writer = csv.writer(labels, lineterminator="\n")
    writer.writerow(["material_id", "master_position"])
    for position in labelled:
        writer.writerow([data.material_ids[position], int(position)])
    dos = io.StringIO(newline="")
    writer = csv.writer(dos, lineterminator="\n")
    writer.writerow(["material_id", "master_position", "original_dos_row", "assigned_dos_row"])
    for position in control.paired_train_positions:
        writer.writerow(
            [data.material_ids[position], int(position), int(data.dos_rows[position]), int(assigned_rows[position])]
        )
    return labels.getvalue(), dos.getvalue()


def train_run(
    data: DiagnosticData,
    cache: CrystalGraphCache,
    controls_dir: Path | str,
    official_model_py: Path | str,
    atom_init_json: Path | str,
    output_dir: Path | str,
    *,
    evaluation_unit: str,
    fraction_key: str,
    label_repeat: int,
    variant: str,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    base_seed: int = 20260910,
    device: str = "cuda:0",
) -> dict:
    if epochs < 1 or batch_size < 2 or learning_rate <= 0:
        raise ValueError("invalid low-label training configuration")
    destination = Path(output_dir).resolve()
    signature, membership = build_signature(
        data,
        cache,
        controls_dir,
        official_model_py,
        atom_init_json,
        evaluation_unit=evaluation_unit,
        fraction_key=fraction_key,
        label_repeat=label_repeat,
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        base_seed=base_seed,
    )
    signature_hash = signature_sha256(signature)
    if destination.exists():
        summary = json.loads((destination / "run_summary.json").read_text(encoding="utf-8"))
        if {path.name for path in destination.iterdir()} != INVENTORY:
            raise FileExistsError("existing low-label inventory differs")
        if summary.get("run_signature_sha256") != signature_hash:
            raise FileExistsError("existing low-label signature differs")
        return {"status": "complete", "complete": True, "reused": True}

    normalizers = compute_low_label_normalizers(data, membership)
    control, assigned_rows = build_dos_assignment(data, evaluation_unit, base_seed, variant)
    atom_lookup, _ = build_atom_feature_lookup(atom_init_json)
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    validation = np.asarray(data.partitions["validation"], dtype=np.int64)
    labelled = train[np.asarray(membership[train], dtype=bool)]
    selected_device = torch.device(device)
    torch.set_num_threads(1)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = PrivilegedCrystalDosModelV3(OfficialCgcnnEncoder(official_model_py)).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    alignment_weight, distillation_weight, _, _ = VARIANT_CONFIG[variant]
    graph_targets = np.zeros_like(np.asarray(data.targets), dtype=np.float32)
    history = []
    best_state = None
    best_metrics = None
    best_score = math.inf
    best_epoch = -1

    for epoch in range(epochs):
        plan = build_epoch_plan(
            train,
            data.observed,
            variant="base" if variant == "lowlabel_structure" else "full",
            batch_size=batch_size,
            base_seed=base_seed,
            evaluation_unit=evaluation_unit,
            epoch=epoch,
        )
        input_mask = plan.input_mask.copy()
        teacher_mask = plan.teacher_mask.copy()
        if variant == "lowlabel_structure":
            input_mask[:] = 0
            teacher_mask[:] = 0
        totals = {name: 0.0 for name in ("total", "property", "alignment", "distillation")}
        batch_count = 0
        model.train()
        for positions in plan.batches:
            graph = batch_official_graphs(
                cache,
                positions,
                graph_targets,
                atom_lookup,
                target_index=0,
                device=selected_device,
            )
            dos_np, target_np, _, label_np = low_label_batch_arrays(
                data, positions, normalizers, membership, assigned_rows
            )
            output = model(
                graph,
                torch.as_tensor(dos_np, device=selected_device),
                torch.as_tensor(input_mask[positions], device=selected_device),
            )
            losses = low_label_losses(
                output,
                torch.as_tensor(target_np, device=selected_device),
                torch.as_tensor(label_np, device=selected_device),
                alignment_weight=alignment_weight,
                distillation_weight=distillation_weight,
                teacher_mask=torch.as_tensor(teacher_mask[positions], device=selected_device),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().item())
            batch_count += 1
        prediction, metrics = evaluate_crystal_only(
            model,
            data,
            cache,
            atom_lookup,
            normalizers,
            validation,
            batch_size=batch_size,
            device=selected_device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                **{f"train_{name}": totals[name] / batch_count for name in totals},
                "validation_selection_score": metrics["selection_score"],
                "validation_k_selection_score": metrics["target_selection_score"]["k"],
                "validation_g_selection_score": metrics["target_selection_score"]["g"],
            }
        )
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("no finite low-label validation checkpoint selected")
    model.load_state_dict(best_state)
    prediction, best_metrics = evaluate_crystal_only(
        model,
        data,
        cache,
        atom_lookup,
        normalizers,
        validation,
        batch_size=batch_size,
        device=selected_device,
    )
    label_text, dos_text = assignment_csv_text(data, membership, control, assigned_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        torch.save(
            {
                "schema_version": "crystal_dos_low_label_loco_checkpoint_v1",
                "state_dict": best_state,
                "normalizers": {
                    name: torch.as_tensor(getattr(normalizers, name))
                    for name in normalizers.__dataclass_fields__
                },
                "run_signature_sha256": signature_hash,
                "best_epoch": best_epoch,
                "evaluation_unit": evaluation_unit,
                "fraction_key": fraction_key,
                "variant": variant,
            },
            staging / "checkpoint.pt",
        )
        history_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            history_buffer, fieldnames=list(history[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)
        (staging / "epoch_history.csv").write_text(
            history_buffer.getvalue(), encoding="utf-8"
        )
        fields = [
            "material_id",
            "k_true_gpa",
            "g_true_gpa",
            "k_prediction_gpa",
            "g_prediction_gpa",
        ]
        prediction_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            prediction_buffer, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        truth = np.asarray(data.targets[validation], dtype=np.float64)
        for row, position in enumerate(validation):
            writer.writerow(
                dict(zip(fields, [data.material_ids[position], *truth[row], *prediction[row]]))
            )
        (staging / "validation_predictions.csv").write_text(
            prediction_buffer.getvalue(), encoding="utf-8"
        )
        (staging / "label_assignment.csv").write_text(label_text, encoding="utf-8")
        (staging / "dos_assignment.csv").write_text(dos_text, encoding="utf-8")
        (staging / "quality_report.md").write_text(
            f"# Low-label {evaluation_unit} {fraction_key} {variant}\n\n"
            f"Status: complete. Labelled training rows: {len(labelled)}. "
            f"Best epoch: {best_epoch}. Crystal-only validation score: {best_score:.8g}. "
            "No outer-test inference was performed.\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": "crystal_dos_low_label_loco_train_v1",
            "status": "complete",
            "complete": True,
            "evaluation_unit": evaluation_unit,
            "fraction_key": fraction_key,
            "label_repeat": label_repeat,
            "variant": variant,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "sample_counts": {
                "train": int(len(train)),
                "labelled_train": int(len(labelled)),
                "paired_train": int(len(control.paired_train_positions)),
                "validation": int(len(validation)),
                "test": 0,
            },
            "label_membership_sha256": _array_sha256(membership),
            "assigned_dos_rows_sha256": _array_sha256(assigned_rows),
            "run_signature": signature,
            "run_signature_sha256": signature_hash,
            "student_inference_modalities": ["crystal"],
            "unlabelled_targets_used": False,
            "test_metrics_read": False,
            "test_inference_performed": False,
            "output_file_sha256": {
                name: sha256_file(staging / name)
                for name in INVENTORY - {"run_summary.json"}
            },
        }
        (staging / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "complete": True,
        "reused": False,
        "output_dir": str(destination),
    }


def audit_run(
    data: DiagnosticData,
    cache: CrystalGraphCache,
    controls_dir: Path | str,
    official_model_py: Path | str,
    atom_init_json: Path | str,
    result_dir: Path | str,
    *,
    evaluation_unit: str,
    device: str = "cuda:0",
) -> dict:
    root = Path(result_dir).resolve()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(root / "checkpoint.pt", map_location=device, weights_only=True)
    signature = summary["run_signature"]
    config = signature["config"]
    expected, membership = build_signature(
        data,
        cache,
        controls_dir,
        official_model_py,
        atom_init_json,
        evaluation_unit=evaluation_unit,
        fraction_key=summary["fraction_key"],
        label_repeat=summary["label_repeat"],
        variant=summary["variant"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        base_seed=config["base_seed"],
    )
    normalizers = TrainNormalizers(
        **{name: value.cpu().numpy() for name, value in checkpoint["normalizers"].items()}
    )
    fresh_normalizers = compute_low_label_normalizers(data, membership)
    control, assigned_rows = build_dos_assignment(
        data, evaluation_unit, config["base_seed"], summary["variant"]
    )
    expected_label_text, expected_dos_text = assignment_csv_text(
        data, membership, control, assigned_rows
    )
    atom_lookup, _ = build_atom_feature_lookup(atom_init_json)
    model = PrivilegedCrystalDosModelV3(OfficialCgcnnEncoder(official_model_py)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    validation = np.asarray(data.partitions["validation"], dtype=np.int64)
    prediction, metrics = evaluate_crystal_only(
        model,
        data,
        cache,
        atom_lookup,
        normalizers,
        validation,
        batch_size=config["batch_size"],
        device=torch.device(device),
    )
    with (root / "validation_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    recorded = np.asarray(
        [[float(row["k_prediction_gpa"]), float(row["g_prediction_gpa"])] for row in rows]
    )
    checks = {
        "summary_complete": summary.get("complete") is True,
        "identity": summary.get("evaluation_unit") == evaluation_unit,
        "test_sealed": summary.get("sample_counts", {}).get("test") == 0
        and summary.get("test_metrics_read") is False
        and summary.get("test_inference_performed") is False,
        "crystal_only_inference": summary.get("student_inference_modalities") == ["crystal"],
        "unlabelled_targets_unused": summary.get("unlabelled_targets_used") is False,
        "exact_inventory": {path.name for path in root.iterdir()} == INVENTORY,
        "output_hashes": all(
            summary["output_file_sha256"].get(name) == sha256_file(root / name)
            for name in INVENTORY - {"run_summary.json"}
        ),
        "signature": signature == expected
        and summary.get("run_signature_sha256") == signature_sha256(expected),
        "checkpoint_binding": checkpoint.get("run_signature_sha256")
        == summary.get("run_signature_sha256"),
        "normalizers": all(
            np.array_equal(getattr(normalizers, name), getattr(fresh_normalizers, name))
            for name in normalizers.__dataclass_fields__
        ),
        "label_membership": summary.get("label_membership_sha256")
        == _array_sha256(membership)
        and (root / "label_assignment.csv").read_text(encoding="utf-8")
        == expected_label_text,
        "dos_assignment": summary.get("assigned_dos_rows_sha256")
        == _array_sha256(assigned_rows)
        and (root / "dos_assignment.csv").read_text(encoding="utf-8")
        == expected_dos_text,
        "prediction_ids": [row["material_id"] for row in rows]
        == data.material_ids[validation].tolist(),
        "recomputed_predictions": np.allclose(recorded, prediction, rtol=0, atol=1e-4),
        "recomputed_metrics": math.isclose(
            float(summary["validation_metrics"]["selection_score"]),
            float(metrics["selection_score"]),
            rel_tol=0,
            abs_tol=1e-6,
        ),
    }
    complete = all(checks.values())
    return {
        "schema_version": "crystal_dos_low_label_loco_audit_v1",
        "status": "complete" if complete else "failed",
        "complete": complete,
        "evaluation_unit": evaluation_unit,
        "fraction_key": summary.get("fraction_key"),
        "variant": summary.get("variant"),
        "checks": checks,
        "max_abs_prediction_difference_gpa": float(np.max(np.abs(recorded - prediction))),
        "result_summary_sha256": sha256_file(root / "run_summary.json"),
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    for name in (
        "master-dir",
        "crystal-cache-dir",
        "graph-cache-dir",
        "dos-dir",
        "split-dir",
        "controls-dir",
        "official-model-py",
        "atom-init-json",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--soap-dir")
    parser.add_argument("--evaluation-unit", required=True)
    parser.add_argument("--device", default="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", allow_abbrev=False)
    _add_common_arguments(train_parser)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--fraction-key", choices=tuple(FRACTIONS), required=True)
    train_parser.add_argument("--label-repeat", type=int, default=0)
    train_parser.add_argument("--variant", choices=VARIANTS, required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--seed", type=int, default=20260910)
    audit_parser = commands.add_parser("audit", allow_abbrev=False)
    _add_common_arguments(audit_parser)
    audit_parser.add_argument("--result-dir", required=True)
    audit_parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    data = load_diagnostic_data(
        args.master_dir,
        args.crystal_cache_dir,
        args.dos_dir,
        args.split_dir,
        args.controls_dir,
        evaluation_unit=args.evaluation_unit,
        soap_dir=args.soap_dir,
    )
    cache = CrystalGraphCache(args.graph_cache_dir, expected_material_ids=data.material_ids)
    if args.command == "train":
        result = train_run(
            data,
            cache,
            args.controls_dir,
            args.official_model_py,
            args.atom_init_json,
            args.output_dir,
            evaluation_unit=args.evaluation_unit,
            fraction_key=args.fraction_key,
            label_repeat=args.label_repeat,
            variant=args.variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_seed=args.seed,
            device=args.device,
        )
    else:
        result = audit_run(
            data,
            cache,
            args.controls_dir,
            args.official_model_py,
            args.atom_init_json,
            args.result_dir,
            evaluation_unit=args.evaluation_unit,
            device=args.device,
        )
        output = Path(args.output_json)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite low-label audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if not result.get("complete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
