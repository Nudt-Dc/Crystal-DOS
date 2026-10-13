"""Validate and freeze formal IID train/validation checkpoints before test inference."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from privileged_training_data import VARIANT_CONFIG


SEEDS = (20260910, 20260911, 20260912)
VARIANTS = ("base", "align", "distill", "align_distill", "full")
INVENTORY = {"freeze_manifest.json", "validation_aggregate.csv", "quality_report.md"}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _invariant_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(signature))
    value.pop("variant", None)
    config = value["config"]
    config.pop("base_seed", None)
    for key in ("alignment_weight", "distillation_weight", "modality_dropout"):
        config.pop(key, None)
    return value


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    return statistics.mean(numbers), statistics.stdev(numbers)


def build_freeze_payload(result_root: Path | str, audit_root: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = Path(result_root).resolve()
    audits = Path(audit_root).resolve()
    rows: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    common: dict[str, Any] | None = None
    for seed in SEEDS:
        for variant in VARIANTS:
            run = results / f"seed{seed}" / variant
            audit_path = audits / f"seed{seed}" / f"{variant}_audit.json"
            summary_path = run / "run_summary.json"
            checkpoint_path = run / "checkpoint.pt"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if summary.get("complete") is not True or summary.get("test_inference_performed") is not False:
                raise ValueError(f"unfrozen or test-exposed training summary: seed={seed} variant={variant}")
            if summary.get("variant") != variant or int(summary["run_signature"]["config"]["base_seed"]) != seed:
                raise ValueError(f"training identity mismatch: seed={seed} variant={variant}")
            expected_weights = {key: float(value) for key, value in VARIANT_CONFIG[variant].items()}
            actual_weights = {key: float(summary["run_signature"]["config"][key]) for key in expected_weights}
            if actual_weights != expected_weights:
                raise ValueError(f"frozen variant configuration mismatch: {variant}")
            summary_hash = sha256_file(summary_path)
            if audit.get("complete") is not True or audit.get("result_summary_sha256") != summary_hash:
                raise ValueError(f"training audit binding failed: seed={seed} variant={variant}")
            invariant = _invariant_signature(summary["run_signature"])
            if common is None:
                common = invariant
            elif _canonical(invariant) != _canonical(common):
                raise ValueError(f"seed/variant invariant drift: seed={seed} variant={variant}")
            metric = summary["validation_metrics"]
            rows.append({
                "seed": seed, "variant": variant, "best_epoch": int(summary["best_epoch"]),
                "selection_score": float(metric["selection_score"]),
                "natural_k_mae_gpa": float(metric["natural"]["mae_gpa"][0]),
                "natural_g_mae_gpa": float(metric["natural"]["mae_gpa"][1]),
                "all_missing_k_mae_gpa": float(metric["all_missing"]["mae_gpa"][0]),
                "all_missing_g_mae_gpa": float(metric["all_missing"]["mae_gpa"][1]),
            })
            bindings[f"seed{seed}/{variant}"] = {
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "run_summary_sha256": summary_hash,
                "audit_sha256": sha256_file(audit_path),
                "run_signature_sha256": summary["run_signature_sha256"],
                "best_epoch": int(summary["best_epoch"]),
            }
    aggregates: dict[str, Any] = {}
    metric_names = [key for key in rows[0] if key not in {"seed", "variant"}]
    for variant in VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant]
        aggregates[variant] = {
            key: dict(zip(("mean", "sample_sd"), _mean_sd([row[key] for row in chosen])))
            for key in metric_names
        }
    payload = {
        "schema_version": "privileged_iid_validation_freeze_v1",
        "status": "complete", "complete": True,
        "seeds": list(SEEDS), "variants": list(VARIANTS), "run_count": len(rows),
        "selection_policy": "mean_standardized_mae_natural_and_all_missing",
        "test_inference_authorized_after_freeze": True,
        "test_metrics_read": False,
        "common_training_invariant": common,
        "checkpoint_bindings": bindings,
        "validation_aggregate": aggregates,
    }
    return payload, rows


def materialize_freeze(result_root: Path | str, audit_root: Path | str, output_dir: Path | str) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite freeze output: {destination}")
    payload, rows = build_freeze_payload(result_root, audit_root)
    staging = destination.with_name(f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        (staging / "validation_aggregate.csv").write_text(buffer.getvalue(), encoding="utf-8")
        report = [
            "# Crystal+DOS privileged IID validation freeze v1", "",
            "All 15 train/validation-only checkpoints passed independent reload audits and are frozen before test inference.", "",
            "| Variant | Selection score mean ± sample SD | Best epoch mean ± sample SD |", "|---|---:|---:|",
        ]
        for variant in VARIANTS:
            item = payload["validation_aggregate"][variant]
            report.append(
                f"| {variant} | {item['selection_score']['mean']:.6f} ± {item['selection_score']['sample_sd']:.6f} | "
                f"{item['best_epoch']['mean']:.2f} ± {item['best_epoch']['sample_sd']:.2f} |"
            )
        report += ["", "No test metrics or test predictions were read while creating this freeze.", ""]
        (staging / "quality_report.md").write_text("\n".join(report), encoding="utf-8")
        payload["output_file_sha256"] = {
            name: sha256_file(staging / name) for name in ("validation_aggregate.csv", "quality_report.md")
        }
        (staging / "freeze_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if {path.name for path in staging.iterdir()} != INVENTORY:
            raise RuntimeError("freeze inventory mismatch")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "complete", "complete": True, "run_count": len(rows), "output_dir": str(destination)}
