"""CLI for publishing the strict Crystal + DOS master population."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from crystal_dos_master import materialize_master_table


def build_parser():
    parser = argparse.ArgumentParser(description="Build the strict Crystal + DOS master table.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--dos-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=10682)
    parser.add_argument("--expected-observed", type=int, default=9459)
    parser.add_argument("--expected-natural-missing", type=int, default=1057)
    parser.add_argument("--expected-window-insufficient", type=int, default=166)
    parser.add_argument("--max-materials", type=int)
    return parser


def main(argv: Sequence[str] | None = None, *, materialize_fn: Callable = materialize_master_table) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("expected_count", "expected_observed", "expected_natural_missing", "expected_window_insufficient"):
        if getattr(args, name) < 0 or (name == "expected_count" and getattr(args, name) == 0):
            parser.error(f"--{name.replace('_', '-')} must be {'positive' if name == 'expected_count' else 'non-negative'}")
    if args.max_materials is not None and args.max_materials <= 0:
        parser.error("--max-materials must be positive")
    result = materialize_fn(
        args.inventory, args.structure_manifest, args.dos_ids, args.output_dir,
        expected_count=args.expected_count,
        expected_dos_counts={
            "observed": args.expected_observed,
            "natural_missing": args.expected_natural_missing,
            "window_insufficient": args.expected_window_insufficient,
        },
        max_materials=args.max_materials,
    )
    receipt = {key: result.get(key) for key in ("status", "complete", "row_count", "full_input_count", "output_dir")}
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
