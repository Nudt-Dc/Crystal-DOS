"""Offline CLI for deterministic total-DOS preprocessing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dos_preprocess import DosPreprocessConfig, materialize_preprocessed_dataset


MISSING_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess a validated total-DOS sample."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy-min-ev", type=float, default=-10.0)
    parser.add_argument("--energy-max-ev", type=float, default=8.0)
    parser.add_argument("--point-count", type=int, default=576)
    parser.add_argument("--negative-tolerance", type=float, default=1e-8)
    parser.add_argument("--min-area", type=float, default=1e-12)
    parser.add_argument("--base-seed", type=int, default=20260907)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--source-manifest", default="dos_sample_manifest.csv"
    )
    parser.add_argument(
        "--coverage-policy", choices=("fail", "exclude"), default="fail"
    )
    parser.add_argument("--spec-version", default="dos-preprocessing-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DosPreprocessConfig(
        energy_min_ev=args.energy_min_ev,
        energy_max_ev=args.energy_max_ev,
        point_count=args.point_count,
        negative_tolerance=args.negative_tolerance,
        min_area=args.min_area,
    )
    _manifest, summary = materialize_preprocessed_dataset(
        args.sample_dir,
        args.output_dir,
        config=config,
        missing_rates=MISSING_RATES,
        repeats=args.repeats,
        base_seed=args.base_seed,
        source_manifest_name=args.source_manifest,
        coverage_policy=args.coverage_policy,
        spec_version=args.spec_version,
    )
    receipt = {
        "status": summary["status"],
        "output_dir": str(args.output_dir.resolve()),
        "source_count": summary.get("source_count", summary["success_count"]),
        "success_count": summary["success_count"],
        "excluded_count": summary.get("excluded_count", 0),
        "failed_count": summary["failed_count"],
        "feature_shape": summary.get("feature_shape"),
        "missing_rates": list(MISSING_RATES),
        "base_seed": args.base_seed,
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
