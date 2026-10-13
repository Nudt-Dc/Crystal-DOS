"""CLI for structure-only SVD, canonical KMeans, IID and LOCO splits."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from experiment_splits import materialize_experiment_splits


def build_parser():
    parser = argparse.ArgumentParser(description="Publish deterministic experiment splits.")
    parser.add_argument("--master-table", type=Path, required=True)
    parser.add_argument("--soap-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iid-train", type=int, default=8546)
    parser.add_argument("--iid-validation", type=int, default=1068)
    parser.add_argument("--iid-test", type=int, default=1068)
    parser.add_argument("--minimum-total", type=int, default=300)
    parser.add_argument("--minimum-observed", type=int, default=150)
    parser.add_argument("--diagnostic-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, materialize_fn: Callable = materialize_experiment_splits) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("iid_train", "iid_validation", "iid_test", "minimum_total"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_observed < 0:
        parser.error("--minimum-observed must be non-negative")
    result = materialize_fn(
        args.master_table, args.soap_dir, args.output_dir,
        iid_sizes=(args.iid_train, args.iid_validation, args.iid_test),
        minimum_total=args.minimum_total,
        minimum_observed=args.minimum_observed,
        diagnostic_smoke=args.diagnostic_smoke,
    )
    receipt = {key: result.get(key) for key in ("status", "complete", "row_count", "cluster_gate_failure_count", "output_dir")}
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
