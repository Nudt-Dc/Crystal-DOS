"""CLI for publishing the framework-independent crystal cache."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from crystal_cache import pack_structure_cache
from crystal_dos_master import validate_master_table


def build_parser():
    parser = argparse.ArgumentParser(description="Pack validated structures into arrays.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--master-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=10682)
    parser.add_argument("--max-materials", type=int)
    return parser


def main(argv: Sequence[str] | None = None, *, pack_fn: Callable = pack_structure_cache, validate_fn: Callable = validate_master_table) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    if args.max_materials is not None and args.max_materials <= 0:
        parser.error("--max-materials must be positive")
    master = pd.read_parquet(args.master_table)
    audit = validate_fn(
        master,
        expected_count=args.expected_count,
        require_dense_dos_rows=args.max_materials is None,
    )
    if not audit.get("complete"):
        raise ValueError("full master contract failed before cache selection")
    selected = master.iloc[:args.max_materials].copy() if args.max_materials is not None else master
    result = pack_fn(args.snapshot_dir, selected, args.output_dir)
    receipt = {key: result.get(key) for key in ("status", "complete", "material_count", "total_sites", "output_dir")}
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
