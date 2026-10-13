"""CLI for resumable compressed periodic SOAP generation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from soap_descriptors import run_soap_snapshot


def build_parser():
    parser = argparse.ArgumentParser(description="Generate or resume SOAP descriptors.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-materials", type=int)
    return parser


def main(argv: Sequence[str] | None = None, *, run_fn: Callable = run_soap_snapshot) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("batch_size", "n_jobs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_materials is not None and args.max_materials <= 0:
        parser.error("--max-materials must be positive")
    result = run_fn(
        args.cache_dir, args.output_dir,
        batch_size=args.batch_size, n_jobs=args.n_jobs, max_materials=args.max_materials,
    )
    keys = ("status", "complete", "material_count", "feature_count", "batch_count", "computed_batches", "reused_batches", "soap_create_calls", "output_dir")
    receipt = {key: result.get(key) for key in keys}
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
