"""CLI for the AFLOW confirmatory master table."""

from __future__ import annotations

import argparse
import json

from aflow_confirmatory_master import materialize_master


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--overlap-dir", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_master(
        args.overlap_dir, args.preprocessed_dir, args.output_dir
    ), sort_keys=True))


if __name__ == "__main__":
    main()

