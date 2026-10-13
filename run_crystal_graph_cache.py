"""Build the deterministic Crystal+DOS periodic graph cache."""

from __future__ import annotations

import argparse
import json
import sys

from crystal_graph_cache import materialize_graph_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_graph_cache(
            args.source_cache_dir,
            args.output_dir,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "complete": False, "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
