"""Train/validation-only SOAP-LOCO v2 privileged-information experiment."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crystal_cgcnn_smoke import CrystalGraphCache, subset_sha256
from crystal_dos_privileged import OfficialCgcnnEncoder, PrivilegedCrystalDosModel
from crystal_dos_privileged_v2 import privileged_losses_v2
from diagnostic_baselines import DiagnosticData
from official_cgcnn_iid import batch_official_graphs, build_atom_feature_lookup
from privileged_iid_freeze import sha256_file
from privileged_iid_train import INVENTORY, _evaluate, signature_sha256
from privileged_iid_train_v2 import VARIANT_CONFIG_V2, _evaluate_v2
from privileged_training_data import (
    TrainNormalizers,
    build_epoch_plan,
    compute_train_normalizers,
    normalized_batch_arrays,
)


LOCO_VARIANTS_V2 = ("structure_only_matched", "distill", "full")


def _require_loco_unit(evaluation_unit: str) -> str:
    valid = {f"loco_{index:02d}" for index in range(10)}
    if evaluation_unit not in valid:
        raise ValueError("evaluation_unit must be loco_00 through loco_09")
    return evaluation_unit


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "crystal_dos_privileged.py",
        "crystal_dos_privileged_v2.py",
        "privileged_training_data.py",
        "privileged_iid_train.py",
        "privileged_iid_train_v2.py",
        "privileged_loco_train_v2.py",
    )
    return {name: sha256_file(root / name) for name in names}


def _signature_invariant(signature: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(signature))
    value.pop("variant", None)
    config = value["config"]
    config.pop("base_seed", None)
    for key in ("alignment_weight", "distillation_weight", "modality_dropout", "input_policy"):
        config.pop(key, None)
    return value


def build_signature_v2(
    data: DiagnosticData,
    cache: CrystalGraphCache,
    official_model_py: Path | str,
    atom_init_json: Path | str,
    *,
    evaluation_unit: str,
    variant: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    base_seed: int,
) -> dict[str, Any]:
    evaluation_unit = _require_loco_unit(evaluation_unit)
    if data.evaluation_unit != evaluation_unit:
        raise ValueError("loaded data evaluation_unit differs from requested LOCO fold")
    if variant not in LOCO_VARIANTS_V2:
        raise ValueError(f"unknown SOAP-LOCO v2 variant: {variant}")
    return {
        "schema_version": "privileged_loco_train_signature_v2",
        "evaluation_unit": evaluation_unit,
        "variant": variant,
        "source_sha256": dict(data.source_sha256),
        "graph_metadata_sha256": sha256_file(cache.root / "graph_metadata.json"),
        "official_model_sha256": sha256_file(official_model_py),
        "atom_init_sha256": sha256_file(atom_init_json),
        "implementation_sha256": _implementation_hashes(),
        "partition_material_ids_sha256": {
            name: subset_sha256(data.material_ids[data.partitions[name]])
            for name in ("train", "validation")
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "base_seed": base_seed,
            "property_loss": "smooth_l1",
            "alignment_objective": "symmetric_inbatch_infonce",
            "temperature": 0.1,
            "selection": "mean_standardized_mae_natural_and_all_missing",
            **VARIANT_CONFIG_V2[variant],
        },
    }


def train_validation_run_v2(
    data: DiagnosticData,
    cache: CrystalGraphCache,
    official_model_py: Path | str,
    atom_init_json: Path | str,
    output_dir: Path | str,
    *,
    evaluation_unit: str,
    variant: str,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    base_seed: int = 20260910,
    device: str = "cuda:0",
) -> dict[str, Any]:
    evaluation_unit = _require_loco_unit(evaluation_unit)
    if variant not in LOCO_VARIANTS_V2 or epochs < 1 or batch_size < 2 or learning_rate <= 0:
        raise ValueError("invalid SOAP-LOCO v2 training configuration")
    destination = Path(output_dir).resolve()
    signature = build_signature_v2(
        data,
        cache,
        official_model_py,
        atom_init_json,
        evaluation_unit=evaluation_unit,
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        base_seed=base_seed,
    )
    signature_hash = signature_sha256(signature)
    if destination.exists():
        summary = json.loads((destination / "run_summary.json").read_text(encoding="utf-8"))
        if {path.name for path in destination.iterdir()} != INVENTORY or summary.get("run_signature_sha256") != signature_hash:
            raise FileExistsError("existing SOAP-LOCO v2 output inventory or signature differs")
        return {"status": "complete", "complete": True, "reused": True, "output_dir": str(destination)}

    normalizers = compute_train_normalizers(data)
    atom_lookup, _ = build_atom_feature_lookup(atom_init_json)
    train = np.asarray(data.partitions["train"], dtype=np.int64)
    validation = np.asarray(data.partitions["validation"], dtype=np.int64)
    selected_device = torch.device(device)
    torch.set_num_threads(1)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = PrivilegedCrystalDosModel(OfficialCgcnnEncoder(official_model_py)).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    config = VARIANT_CONFIG_V2[variant]
    force_missing = config["input_policy"] == "always_missing"
    history: list[dict[str, Any]] = []
    best_state = None
    best_metrics = None
    best_score, best_epoch = math.inf, -1

    for epoch in range(epochs):
        plan_variant = "base" if force_missing else variant
        plan = build_epoch_plan(
            train,
            data.observed,
            variant=plan_variant,
            batch_size=batch_size,
            base_seed=base_seed,
            evaluation_unit=evaluation_unit,
            epoch=epoch,
        )
        input_mask = np.zeros_like(plan.input_mask) if force_missing else plan.input_mask
        model.train()
        sums = {name: 0.0 for name in ("total", "property", "alignment", "distillation")}
        count = 0
        for positions in plan.batches:
            graph = batch_official_graphs(cache, positions, data.targets, atom_lookup, target_index=0, device=selected_device)
            dos_np, target_np, teacher_np = normalized_batch_arrays(data, positions, normalizers)
            if force_missing:
                teacher_np = np.zeros_like(teacher_np)
            output = model(
                graph,
                torch.as_tensor(dos_np, device=selected_device),
                torch.as_tensor(input_mask[positions], device=selected_device),
            )
            losses = privileged_losses_v2(
                output,
                torch.as_tensor(target_np, device=selected_device),
                alignment_weight=float(config["alignment_weight"]),
                distillation_weight=float(config["distillation_weight"]),
                teacher_mask=torch.as_tensor(teacher_np, device=selected_device),
                temperature=0.1,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for name in sums:
                sums[name] += float(losses[name].detach().item()) * len(positions)
            count += len(positions)
        natural, missing, metrics = _evaluate_v2(
            model,
            data,
            cache,
            atom_lookup,
            normalizers,
            validation,
            batch_size=batch_size,
            device=selected_device,
            force_missing=force_missing,
        )
        history.append({
            "epoch": epoch + 1,
            **{f"train_{name}": sums[name] / count for name in sums},
            "validation_selection_score": metrics["selection_score"],
        })
        if metrics["selection_score"] < best_score:
            best_score, best_epoch, best_metrics = metrics["selection_score"], epoch + 1, metrics
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("no finite SOAP-LOCO v2 validation checkpoint selected")
    model.load_state_dict(best_state)
    natural, missing, best_metrics = _evaluate_v2(
        model,
        data,
        cache,
        atom_lookup,
        normalizers,
        validation,
        batch_size=batch_size,
        device=selected_device,
        force_missing=force_missing,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        torch.save({
            "schema_version": "privileged_loco_checkpoint_v2",
            "state_dict": best_state,
            "normalizers": {
                key: torch.as_tensor(getattr(normalizers, key))
                for key in normalizers.__dataclass_fields__
            },
            "run_signature_sha256": signature_hash,
            "best_epoch": best_epoch,
            "evaluation_unit": evaluation_unit,
            "variant": variant,
        }, staging / "checkpoint.pt")
        history_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(history_buffer, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
        (staging / "epoch_history.csv").write_text(history_buffer.getvalue(), encoding="utf-8")
        fields = [
            "material_id", "k_true_gpa", "g_true_gpa", "k_natural_gpa", "g_natural_gpa",
            "k_all_missing_gpa", "g_all_missing_gpa",
        ]
        prediction_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(prediction_buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        truth = np.asarray(data.targets[validation], dtype=np.float64)
        for row, position in enumerate(validation):
            writer.writerow(dict(zip(fields, [data.material_ids[position], *truth[row], *natural[row], *missing[row]])))
        (staging / "validation_predictions.csv").write_text(prediction_buffer.getvalue(), encoding="utf-8")
        (staging / "quality_report.md").write_text(
            f"# Crystal+DOS SOAP-LOCO v2 {evaluation_unit} {variant} train-validation run\n\n"
            f"Status: complete. Best epoch: {best_epoch}; validation selection score: {best_score:.8g}. "
            "No outer-test inference was performed.\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": "privileged_loco_train_v2",
            "status": "complete",
            "complete": True,
            "evaluation_unit": evaluation_unit,
            "variant": variant,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "sample_counts": {"train": len(train), "validation": len(validation), "test": 0},
            "run_signature": signature,
            "run_signature_sha256": signature_hash,
            "test_inference_performed": False,
            "output_file_sha256": {
                name: sha256_file(staging / name) for name in INVENTORY - {"run_summary.json"}
            },
        }
        (staging / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "complete", "complete": True, "reused": False, "output_dir": str(destination)}


def audit_validation_run_v2(
    data: DiagnosticData,
    cache: CrystalGraphCache,
    official_model_py: Path | str,
    atom_init_json: Path | str,
    result_dir: Path | str,
    *,
    evaluation_unit: str,
    device: str = "cuda:0",
) -> dict[str, Any]:
    evaluation_unit = _require_loco_unit(evaluation_unit)
    root = Path(result_dir).resolve()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(root / "checkpoint.pt", map_location=device, weights_only=True)
    signature = summary["run_signature"]
    variant, config = signature["variant"], signature["config"]
    expected = build_signature_v2(
        data,
        cache,
        official_model_py,
        atom_init_json,
        evaluation_unit=evaluation_unit,
        variant=variant,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        base_seed=config["base_seed"],
    )
    normalizers = TrainNormalizers(**{
        key: value.cpu().numpy() for key, value in checkpoint["normalizers"].items()
    })
    atom_lookup, _ = build_atom_feature_lookup(atom_init_json)
    model = PrivilegedCrystalDosModel(OfficialCgcnnEncoder(official_model_py)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    validation = np.asarray(data.partitions["validation"], dtype=np.int64)
    natural, missing, metrics = _evaluate_v2(
        model,
        data,
        cache,
        atom_lookup,
        normalizers,
        validation,
        batch_size=config["batch_size"],
        device=torch.device(device),
        force_missing=config["input_policy"] == "always_missing",
    )
    with (root / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    recorded = np.asarray([
        [float(row[name]) for name in (
            "k_natural_gpa", "g_natural_gpa", "k_all_missing_gpa", "g_all_missing_gpa"
        )]
        for row in rows
    ])
    fresh = np.concatenate([natural, missing], axis=1)
    checks = {
        "summary_complete": summary.get("complete") is True and summary.get("test_inference_performed") is False,
        "evaluation_unit": summary.get("evaluation_unit") == evaluation_unit,
        "test_count_zero": summary.get("sample_counts", {}).get("test") == 0,
        "exact_inventory": {path.name for path in root.iterdir()} == INVENTORY,
        "output_hashes": all(
            summary["output_file_sha256"].get(name) == sha256_file(root / name)
            for name in INVENTORY - {"run_summary.json"}
        ),
        "signature": signature == expected and summary.get("run_signature_sha256") == signature_sha256(expected),
        "checkpoint_binding": (
            checkpoint.get("run_signature_sha256") == summary.get("run_signature_sha256")
            and checkpoint.get("evaluation_unit") == evaluation_unit
        ),
        "prediction_ids": [row["material_id"] for row in rows] == data.material_ids[validation].tolist(),
        "recomputed_predictions": np.allclose(recorded, fresh, rtol=0, atol=1e-4),
        "recomputed_selection": math.isclose(
            metrics["selection_score"], summary["validation_metrics"]["selection_score"], rel_tol=0, abs_tol=1e-6
        ),
    }
    complete = all(checks.values())
    return {
        "schema_version": "privileged_loco_train_audit_v2",
        "status": "complete" if complete else "failed",
        "complete": complete,
        "evaluation_unit": evaluation_unit,
        "variant": variant,
        "checks": checks,
        "max_abs_prediction_difference_gpa": float(np.max(np.abs(recorded - fresh))),
        "result_summary_sha256": sha256_file(root / "run_summary.json"),
    }
