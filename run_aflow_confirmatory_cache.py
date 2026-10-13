"""CLI for the independent AFLOW crystal cache."""

from __future__ import annotations

import argparse
import json

from aflow_confirmatory_cache import materialize_cache


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_cache(
        args.master_dir, args.cohort_dir, args.output_dir
    ), sort_keys=True))


if __name__ == "__main__":
    main()

