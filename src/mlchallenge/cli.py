"""Command-line entry points for auditable project operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlchallenge.audit import audit_data, write_audit
from mlchallenge.contracts import load_test_data, load_training_data
from mlchallenge.folds import build_fold_manifest
from mlchallenge.submission import validate_submission_frames


def _audit(args: argparse.Namespace) -> int:
    report = audit_data(args.data_root)
    write_audit(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _make_folds(args: argparse.Namespace) -> int:
    data = load_training_data(args.data_root)
    manifest = build_fold_manifest(
        data.source1,
        data.source2,
        data.source3,
        data.ground_truth,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, sep="\t", index=False)
    print(f"Wrote {len(manifest)} records to {output}")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    test = load_test_data(args.data_root)
    matching = pd.read_csv(
        args.matching, sep="\t", dtype="string", keep_default_na=False, na_filter=False
    )
    candidates = pd.read_csv(
        args.candidates, sep="\t", dtype="string", keep_default_na=False, na_filter=False
    )
    issues = validate_submission_frames(matching, candidates, test)
    if issues:
        for number, issue in enumerate(issues, start=1):
            print(f"{number}. {issue}")
        return 1
    print("PASS: repository preflight checks succeeded")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlchallenge",
        description="Auditable tools for the Amazon Business Entity Resolution Challenge",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="validate and fingerprint challenge data")
    audit_parser.add_argument("--data-root", default="data")
    audit_parser.add_argument("--output", default="reports/data_audit.json")
    audit_parser.set_defaults(func=_audit)

    folds_parser = subparsers.add_parser(
        "make-folds", help="build a deterministic entity-group fold manifest"
    )
    folds_parser.add_argument("--data-root", default="data")
    folds_parser.add_argument("--output", default="artifacts/fold_manifest.tsv")
    folds_parser.add_argument("--n-splits", type=int, required=True)
    folds_parser.add_argument("--seed", type=int, required=True)
    folds_parser.set_defaults(func=_make_folds)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate the two submission TSV files locally"
    )
    preflight_parser.add_argument("--data-root", default="data")
    preflight_parser.add_argument("--matching", default="output/matching_results.tsv")
    preflight_parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    preflight_parser.set_defaults(func=_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
