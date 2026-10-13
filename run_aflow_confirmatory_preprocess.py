"""CLI for AFLOW confirmatory total-DOS preprocessing."""

from __future__ import annotations

import argparse
import json

from aflow_confirmatory_preprocess import materialize_preprocessed


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--records", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--cohort-audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=5564)
    parser.add_argument("--energy-min-ev", type=float, default=-10.0)
    parser.add_argument("--energy-max-ev", type=float, default=8.0)
    parser.add_argument("--point-count", type=int, default=576)
    args = parser.parse_args()
    result = materialize_preprocessed(
        args.records,
        args.cohort_dir,
        args.cohort_audit_dir,
        args.output_dir,
        expected_count=args.expected_count,
        energy_min_ev=args.energy_min_ev,
        energy_max_ev=args.energy_max_ev,
        point_count=args.point_count,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

