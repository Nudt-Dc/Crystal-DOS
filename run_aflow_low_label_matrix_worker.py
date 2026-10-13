"""Sequential worker for one frozen AFLOW low-label variant on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crystal_cgcnn_smoke import CrystalGraphCache
from crystal_dos_low_label_loco_v1 import FRACTIONS, VARIANTS, audit_run, train_run
from diagnostic_baselines import load_diagnostic_data


def _items(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("comma-separated list must not be empty")
    return result


def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in (
        "master-dir", "crystal-cache-dir", "graph-cache-dir", "dos-dir", "split-dir",
        "soap-dir", "controls-dir", "official-model-py", "atom-init-json",
        "output-root", "audit-root",
    ):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--folds", required=True, type=_items)
    p.add_argument("--fractions", required=True, type=_items)
    p.add_argument("--seeds", required=True, type=_items)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--device", required=True)
    a = p.parse_args()
    if any(value not in FRACTIONS for value in a.fractions):
        raise ValueError("unknown low-label fraction")
    seeds = [int(value) for value in a.seeds]
    output_root, audit_root = Path(a.output_root), Path(a.audit_root)
    cache = None
    completed = 0
    for fold in a.folds:
        data = load_diagnostic_data(
            a.master_dir, a.crystal_cache_dir, a.dos_dir, a.split_dir, a.controls_dir,
            evaluation_unit=fold, soap_dir=a.soap_dir,
        )
        if cache is None:
            cache = CrystalGraphCache(a.graph_cache_dir, expected_material_ids=data.material_ids)
        for fraction in a.fractions:
            for seed in seeds:
                result_dir = output_root / fold / fraction / f"seed{seed}" / a.variant
                audit_json = audit_root / fold / fraction / f"seed{seed}" / f"{a.variant}_audit.json"
                train_run(
                    data, cache, a.controls_dir, a.official_model_py, a.atom_init_json,
                    result_dir, evaluation_unit=fold, fraction_key=fraction, label_repeat=0,
                    variant=a.variant, epochs=a.epochs, batch_size=a.batch_size,
                    learning_rate=a.learning_rate, base_seed=seed, device=a.device,
                )
                audit_result = audit_run(
                    data, cache, a.controls_dir, a.official_model_py, a.atom_init_json,
                    result_dir, evaluation_unit=fold, device=a.device,
                )
                if audit_json.exists():
                    raise FileExistsError(f"refusing to overwrite low-label audit: {audit_json}")
                audit_json.parent.mkdir(parents=True, exist_ok=True)
                audit_json.write_text(
                    json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                if not audit_result["complete"]:
                    raise RuntimeError(f"low-label audit failed: {fold} {fraction} {seed} {a.variant}")
                completed += 1
                print(json.dumps({
                    "status": "progress", "completed": completed, "fold": fold,
                    "fraction": fraction, "seed": seed, "variant": a.variant,
                }, sort_keys=True), flush=True)
    print(json.dumps({"status": "complete", "complete": True, "completed": completed, "variant": a.variant}, sort_keys=True))


if __name__ == "__main__":
    main()

