"""Command-line entry point for publishing reviewed experiment controls."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from experiment_controls import materialize_experiment_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish deterministic experiment control artifacts.",
        allow_abbrev=False,
    )
    parser.add_argument("--master-dir", type=Path, required=True)
    parser.add_argument("--dos-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-smoke", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    materialize_fn: Callable = materialize_experiment_controls,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved_output = args.output_dir.resolve(strict=False)
    resolved_sources = tuple(
        path.resolve(strict=False)
        for path in (args.master_dir, args.dos_dir, args.split_dir)
    )
    if resolved_output in resolved_sources:
        parser.error("--output-dir must differ from each source directory")

    result = materialize_fn(
        args.master_dir,
        args.dos_dir,
        args.split_dir,
        args.output_dir,
        diagnostic_smoke=args.diagnostic_smoke,
    )
    receipt = {
        key: result.get(key)
        for key in (
            "status",
            "complete",
            "evaluation_unit_count",
            "output_dir",
            "reused",
            "source_sha256",
        )
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if bool(result.get("complete")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
