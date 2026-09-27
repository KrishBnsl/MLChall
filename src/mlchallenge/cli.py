"""Command-line entry points for auditable project operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlchallenge.contracts import load_test_data, load_training_data
from mlchallenge.diagnostics import (
    analyze_errors,
    evaluate_source_thresholds,
    tune_source_thresholds,
)
from mlchallenge.folds import build_fold_manifest
from mlchallenge.large_data import DuckDBRuntime, audit_dataset_out_of_core, write_json_report
from mlchallenge.large_matching import (
    MatcherConfig,
    SamplingConfig,
    build_training_sample,
    evaluate_at_threshold,
    prepare_test_split,
    score_candidates,
    threshold_curve,
    train_matchers,
    write_submission_out_of_core,
)
from mlchallenge.scalable_candidates import (
    BlockingConfig,
    generate_candidates_out_of_core,
    generate_partitioned_candidates,
)
from mlchallenge.sparse_name_candidates import SparseNameConfig, generate_sparse_name_candidates
from mlchallenge.splits import (
    ExperimentSplitConfig,
    SplitConfig,
    create_development_splits,
    create_experiment_splits,
)
from mlchallenge.submission import validate_submission_frames


def _runtime(args: argparse.Namespace) -> DuckDBRuntime:
    return DuckDBRuntime(
        memory_limit=args.memory_limit,
        threads=args.threads,
        max_temp_directory_size=args.max_temp_size,
    )


def _audit(args: argparse.Namespace) -> int:
    report = audit_dataset_out_of_core(
        args.data_root,
        temp_directory=args.temp_directory,
        runtime=_runtime(args),
    )
    write_json_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["contract_status"] == "PASS" else 1


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


def _make_splits(args: argparse.Namespace) -> int:
    metadata = create_development_splits(
        args.data_root,
        args.output_root,
        config=SplitConfig(
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            local_test_fraction=args.local_test_fraction,
            seed=args.seed,
        ),
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
        audit_report=args.audit_report,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _make_experiment_splits(args: argparse.Namespace) -> int:
    metadata = create_experiment_splits(
        args.parent_split_directory,
        args.output_root,
        config=ExperimentSplitConfig(
            fit_fraction=args.fit_fraction,
            tuning_fraction=args.tuning_fraction,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        ),
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _build_candidates(args: argparse.Namespace) -> int:
    generator = (
        generate_partitioned_candidates if args.partitioned else generate_candidates_out_of_core
    )
    keyword_arguments = {
        "config": BlockingConfig(
            fuzzy_top_k_per_source=args.top_k_per_source,
            max_key_document_frequency=args.max_key_df,
            max_query_keys_per_source=args.max_query_keys,
            minimum_token_length=args.minimum_token_length,
            query_batch_size=args.query_batch_size,
        ),
        "runtime": _runtime(args),
        "temp_directory": args.temp_directory,
    }
    if args.partitioned:
        keyword_arguments["resume"] = not args.no_resume
    report = generator(
        args.split_directory,
        args.output,
        args.report,
        **keyword_arguments,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_sparse_name_candidates(args: argparse.Namespace) -> int:
    report = generate_sparse_name_candidates(
        args.split_directory,
        args.output,
        args.report,
        config=SparseNameConfig(
            top_k_per_source=args.top_k_per_source,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            n_features=args.n_features,
            query_batch_size=args.query_batch_size,
            minimum_similarity=args.minimum_similarity,
            multiplication_threads=args.multiplication_threads,
            address_weight=args.address_weight,
        ),
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
        union_candidate_path=args.union_candidates,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _prepare_test(args: argparse.Namespace) -> int:
    report = prepare_test_split(
        args.data_root,
        args.output_directory,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_training_sample(args: argparse.Namespace) -> int:
    report = build_training_sample(
        args.candidates,
        args.split_directory,
        args.output,
        args.report,
        sampling=SamplingConfig(
            hard_negatives_per_source=args.hard_negatives_per_source,
            random_negatives_per_source=args.random_negatives_per_source,
            model_hard_negatives_per_source=args.model_hard_negatives_per_source,
            cv_folds=args.cv_folds,
            calibration_modulus=args.calibration_modulus,
            seed=args.seed,
        ),
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
        mining_scores_path=args.mining_scores,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _train_matcher(args: argparse.Namespace) -> int:
    report = train_matchers(
        args.sample,
        args.model_directory,
        args.report,
        config=MatcherConfig(
            model_family=args.model_family,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            threads=args.threads,
            random_state=args.seed,
        ),
        cv_folds=args.cv_folds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _score_candidates(args: argparse.Namespace) -> int:
    report = score_candidates(
        args.candidates,
        args.split_directory,
        args.output,
        args.report,
        model_path=args.model,
        oof_model_directory=args.oof_model_directory,
        cv_folds=args.cv_folds,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
        vectors_per_chunk=args.vectors_per_chunk,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _tune_threshold(args: argparse.Namespace) -> int:
    report = threshold_curve(
        args.scores,
        args.split_directory,
        args.curve,
        args.report,
        steps=args.steps,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _evaluate_matcher(args: argparse.Namespace) -> int:
    report = evaluate_at_threshold(
        args.scores,
        args.split_directory,
        args.report,
        threshold=args.threshold,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _tune_source_thresholds(args: argparse.Namespace) -> int:
    report = tune_source_thresholds(
        args.scores,
        args.split_directory,
        args.report,
        global_threshold=args.global_threshold,
        search_radius=args.search_radius,
        step=args.step,
        rounds=args.rounds,
        minimum_gain=args.minimum_gain,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _evaluate_source_thresholds(args: argparse.Namespace) -> int:
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["thresholds"]
    report = evaluate_source_thresholds(
        args.scores,
        args.split_directory,
        args.report,
        thresholds=thresholds,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _analyze_errors(args: argparse.Namespace) -> int:
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["thresholds"]
    else:
        thresholds = {"source2": args.threshold, "source3": args.threshold}
    report = analyze_errors(
        args.scores,
        args.candidates,
        args.split_directory,
        args.output_directory,
        args.report,
        thresholds=thresholds,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _write_submission(args: argparse.Namespace) -> int:
    thresholds = None
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["thresholds"]
    report = write_submission_out_of_core(
        args.scores,
        args.candidates,
        args.test_split_directory,
        args.output_directory,
        args.report,
        threshold=args.threshold,
        thresholds_by_source=thresholds,
        runtime=_runtime(args),
        temp_directory=args.temp_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
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

    audit_parser = subparsers.add_parser(
        "audit", help="out-of-core validation and fingerprinting of challenge data"
    )
    audit_parser.add_argument("--data-root", default="data")
    audit_parser.add_argument("--output", default="reports/data_audit.json")
    audit_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    audit_parser.add_argument("--memory-limit", default="3GB")
    audit_parser.add_argument("--max-temp-size", default="8GB")
    audit_parser.add_argument("--threads", type=int, default=4)
    audit_parser.set_defaults(func=_audit)

    folds_parser = subparsers.add_parser(
        "make-folds", help="build a deterministic entity-group fold manifest"
    )
    folds_parser.add_argument("--data-root", default="data")
    folds_parser.add_argument("--output", default="artifacts/fold_manifest.tsv")
    folds_parser.add_argument("--n-splits", type=int, required=True)
    folds_parser.add_argument("--seed", type=int, required=True)
    folds_parser.set_defaults(func=_make_folds)

    splits_parser = subparsers.add_parser(
        "make-splits",
        help="create deterministic entity-disjoint train/validation/local-test Parquet splits",
    )
    splits_parser.add_argument("--data-root", default="student_resource/dataset")
    splits_parser.add_argument("--output-root", default="artifacts/splits")
    splits_parser.add_argument("--audit-report", default="reports/data_audit.json")
    splits_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    splits_parser.add_argument("--memory-limit", default="3GB")
    splits_parser.add_argument("--max-temp-size", default="8GB")
    splits_parser.add_argument("--threads", type=int, default=4)
    splits_parser.add_argument("--train-fraction", type=float, default=0.90)
    splits_parser.add_argument("--validation-fraction", type=float, default=0.05)
    splits_parser.add_argument("--local-test-fraction", type=float, default=0.05)
    splits_parser.add_argument("--seed", type=int, default=20260925)
    splits_parser.add_argument("--force", action="store_true")
    splits_parser.set_defaults(func=_make_splits)

    experiment_splits_parser = subparsers.add_parser(
        "make-experiment-splits",
        help="reserve fresh fit/tuning/holdout entities from the unused training partition",
    )
    experiment_splits_parser.add_argument(
        "--parent-split-directory", default="artifacts/splits/train"
    )
    experiment_splits_parser.add_argument(
        "--output-root", default="artifacts/experiment_splits_v2"
    )
    experiment_splits_parser.add_argument("--fit-fraction", type=float, default=8.0 / 9.0)
    experiment_splits_parser.add_argument("--tuning-fraction", type=float, default=1.0 / 18.0)
    experiment_splits_parser.add_argument("--holdout-fraction", type=float, default=1.0 / 18.0)
    experiment_splits_parser.add_argument("--seed", type=int, default=20260927)
    experiment_splits_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    experiment_splits_parser.add_argument("--memory-limit", default="3GB")
    experiment_splits_parser.add_argument("--max-temp-size", default="8GB")
    experiment_splits_parser.add_argument("--threads", type=int, default=4)
    experiment_splits_parser.add_argument("--force", action="store_true")
    experiment_splits_parser.set_defaults(func=_make_experiment_splits)

    candidate_parser = subparsers.add_parser(
        "build-candidates",
        help="generate and evaluate out-of-core candidates for one labeled split",
    )
    candidate_parser.add_argument("--split-directory", required=True)
    candidate_parser.add_argument("--output", required=True)
    candidate_parser.add_argument("--report", required=True)
    candidate_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    candidate_parser.add_argument("--memory-limit", default="3GB")
    candidate_parser.add_argument("--max-temp-size", default="8GB")
    candidate_parser.add_argument("--threads", type=int, default=4)
    candidate_parser.add_argument("--top-k-per-source", type=int, default=50)
    candidate_parser.add_argument("--max-key-df", type=int, default=200)
    candidate_parser.add_argument("--max-query-keys", type=int, default=6)
    candidate_parser.add_argument("--minimum-token-length", type=int, default=3)
    candidate_parser.add_argument("--query-batch-size", type=int, default=25_000)
    candidate_parser.add_argument(
        "--partitioned",
        action="store_true",
        help="bound memory by processing one country/source/query batch at a time",
    )
    candidate_parser.add_argument("--no-resume", action="store_true")
    candidate_parser.set_defaults(func=_build_candidates)

    sparse_parser = subparsers.add_parser(
        "build-sparse-name-candidates",
        help="build hashed character TF-IDF top-k candidates and optionally union another blocker",
    )
    sparse_parser.add_argument("--split-directory", required=True)
    sparse_parser.add_argument("--output", required=True)
    sparse_parser.add_argument("--report", required=True)
    sparse_parser.add_argument("--union-candidates")
    sparse_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    sparse_parser.add_argument("--memory-limit", default="3GB")
    sparse_parser.add_argument("--max-temp-size", default="8GB")
    sparse_parser.add_argument("--threads", type=int, default=2)
    sparse_parser.add_argument("--top-k-per-source", type=int, default=100)
    sparse_parser.add_argument("--ngram-min", type=int, default=2)
    sparse_parser.add_argument("--ngram-max", type=int, default=5)
    sparse_parser.add_argument("--n-features", type=int, default=1_048_576)
    sparse_parser.add_argument("--query-batch-size", type=int, default=5_000)
    sparse_parser.add_argument("--minimum-similarity", type=float, default=0.10)
    sparse_parser.add_argument("--multiplication-threads", type=int, default=2)
    sparse_parser.add_argument("--address-weight", type=float, default=0.35)
    sparse_parser.set_defaults(func=_build_sparse_name_candidates)

    prepare_test_parser = subparsers.add_parser(
        "prepare-test", help="convert official test TSV files to pipeline Parquet inputs"
    )
    prepare_test_parser.add_argument("--data-root", default="data")
    prepare_test_parser.add_argument("--output-directory", default="artifacts/test_split")
    prepare_test_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    prepare_test_parser.add_argument("--memory-limit", default="3GB")
    prepare_test_parser.add_argument("--max-temp-size", default="8GB")
    prepare_test_parser.add_argument("--threads", type=int, default=4)
    prepare_test_parser.set_defaults(func=_prepare_test)

    sample_parser = subparsers.add_parser(
        "build-training-sample",
        help="build vectorized features with all positives and weighted hard negatives",
    )
    sample_parser.add_argument("--candidates", required=True)
    sample_parser.add_argument("--split-directory", required=True)
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--report", required=True)
    sample_parser.add_argument("--hard-negatives-per-source", type=int, default=5)
    sample_parser.add_argument("--random-negatives-per-source", type=int, default=1)
    sample_parser.add_argument("--model-hard-negatives-per-source", type=int, default=0)
    sample_parser.add_argument("--mining-scores")
    sample_parser.add_argument("--cv-folds", type=int, default=3)
    sample_parser.add_argument("--calibration-modulus", type=int, default=10)
    sample_parser.add_argument("--seed", type=int, default=20260925)
    sample_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    sample_parser.add_argument("--memory-limit", default="3GB")
    sample_parser.add_argument("--max-temp-size", default="8GB")
    sample_parser.add_argument("--threads", type=int, default=4)
    sample_parser.set_defaults(func=_build_training_sample)

    train_parser = subparsers.add_parser(
        "train-matcher", help="fit calibrated group-disjoint OOF and final pair matchers"
    )
    train_parser.add_argument("--sample", required=True)
    train_parser.add_argument("--model-directory", required=True)
    train_parser.add_argument("--report", required=True)
    train_parser.add_argument("--cv-folds", type=int, default=3)
    train_parser.add_argument(
        "--model-family",
        choices=("hist_gradient_boosting", "lightgbm", "xgboost", "catboost"),
        default="hist_gradient_boosting",
    )
    train_parser.add_argument("--learning-rate", type=float, default=0.08)
    train_parser.add_argument("--max-iter", type=int, default=140)
    train_parser.add_argument("--max-leaf-nodes", type=int, default=31)
    train_parser.add_argument("--min-samples-leaf", type=int, default=40)
    train_parser.add_argument("--l2-regularization", type=float, default=1.0)
    train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=20260925)
    train_parser.set_defaults(func=_train_matcher)

    score_parser = subparsers.add_parser(
        "score-candidates", help="stream candidate features through a final or OOF matcher"
    )
    score_parser.add_argument("--candidates", required=True)
    score_parser.add_argument("--split-directory", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--report", required=True)
    score_model = score_parser.add_mutually_exclusive_group(required=True)
    score_model.add_argument("--model")
    score_model.add_argument("--oof-model-directory")
    score_parser.add_argument("--cv-folds", type=int, default=3)
    score_parser.add_argument("--vectors-per-chunk", type=int, default=128)
    score_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    score_parser.add_argument("--memory-limit", default="3GB")
    score_parser.add_argument("--max-temp-size", default="8GB")
    score_parser.add_argument("--threads", type=int, default=4)
    score_parser.set_defaults(func=_score_candidates)

    threshold_parser = subparsers.add_parser(
        "tune-threshold", help="select the macro F0.5 threshold from OOF scores"
    )
    threshold_parser.add_argument("--scores", required=True)
    threshold_parser.add_argument("--split-directory", required=True)
    threshold_parser.add_argument("--curve", required=True)
    threshold_parser.add_argument("--report", required=True)
    threshold_parser.add_argument("--steps", type=int, default=201)
    threshold_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    threshold_parser.add_argument("--memory-limit", default="3GB")
    threshold_parser.add_argument("--max-temp-size", default="8GB")
    threshold_parser.add_argument("--threads", type=int, default=4)
    threshold_parser.set_defaults(func=_tune_threshold)

    evaluation_parser = subparsers.add_parser(
        "evaluate-matcher", help="evaluate a frozen threshold using exact macro F0.5"
    )
    evaluation_parser.add_argument("--scores", required=True)
    evaluation_parser.add_argument("--split-directory", required=True)
    evaluation_parser.add_argument("--report", required=True)
    evaluation_parser.add_argument("--threshold", required=True, type=float)
    evaluation_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    evaluation_parser.add_argument("--memory-limit", default="3GB")
    evaluation_parser.add_argument("--max-temp-size", default="8GB")
    evaluation_parser.add_argument("--threads", type=int, default=4)
    evaluation_parser.set_defaults(func=_evaluate_matcher)

    segmented_parser = subparsers.add_parser(
        "tune-source-thresholds",
        help="tune two guarded source-specific thresholds from OOF predictions",
    )
    segmented_parser.add_argument("--scores", required=True)
    segmented_parser.add_argument("--split-directory", required=True)
    segmented_parser.add_argument("--report", required=True)
    segmented_parser.add_argument("--global-threshold", required=True, type=float)
    segmented_parser.add_argument("--search-radius", type=float, default=0.08)
    segmented_parser.add_argument("--step", type=float, default=0.01)
    segmented_parser.add_argument("--rounds", type=int, default=2)
    segmented_parser.add_argument("--minimum-gain", type=float, default=0.0001)
    segmented_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    segmented_parser.add_argument("--memory-limit", default="3GB")
    segmented_parser.add_argument("--max-temp-size", default="8GB")
    segmented_parser.add_argument("--threads", type=int, default=4)
    segmented_parser.set_defaults(func=_tune_source_thresholds)

    segmented_eval_parser = subparsers.add_parser(
        "evaluate-source-thresholds",
        help="evaluate frozen source-specific thresholds on a separate entity holdout",
    )
    segmented_eval_parser.add_argument("--scores", required=True)
    segmented_eval_parser.add_argument("--split-directory", required=True)
    segmented_eval_parser.add_argument("--thresholds", required=True)
    segmented_eval_parser.add_argument("--report", required=True)
    segmented_eval_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    segmented_eval_parser.add_argument("--memory-limit", default="3GB")
    segmented_eval_parser.add_argument("--max-temp-size", default="8GB")
    segmented_eval_parser.add_argument("--threads", type=int, default=4)
    segmented_eval_parser.set_defaults(func=_evaluate_source_thresholds)

    errors_parser = subparsers.add_parser(
        "analyze-errors",
        help="write pair-level blocker/matcher errors and macro-F0.5 slices",
    )
    errors_parser.add_argument("--scores", required=True)
    errors_parser.add_argument("--candidates", required=True)
    errors_parser.add_argument("--split-directory", required=True)
    errors_parser.add_argument("--output-directory", required=True)
    errors_parser.add_argument("--report", required=True)
    error_threshold = errors_parser.add_mutually_exclusive_group(required=True)
    error_threshold.add_argument("--threshold", type=float)
    error_threshold.add_argument("--thresholds")
    errors_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    errors_parser.add_argument("--memory-limit", default="3GB")
    errors_parser.add_argument("--max-temp-size", default="8GB")
    errors_parser.add_argument("--threads", type=int, default=4)
    errors_parser.set_defaults(func=_analyze_errors)

    submission_parser = subparsers.add_parser(
        "write-submission", help="write and verify both required test submission TSV files"
    )
    submission_parser.add_argument("--scores", required=True)
    submission_parser.add_argument("--candidates", required=True)
    submission_parser.add_argument("--test-split-directory", required=True)
    submission_parser.add_argument("--output-directory", default="output")
    submission_parser.add_argument("--report", required=True)
    submission_threshold = submission_parser.add_mutually_exclusive_group(required=True)
    submission_threshold.add_argument("--threshold", type=float)
    submission_threshold.add_argument("--thresholds")
    submission_parser.add_argument("--temp-directory", default="artifacts/duckdb_tmp")
    submission_parser.add_argument("--memory-limit", default="3GB")
    submission_parser.add_argument("--max-temp-size", default="8GB")
    submission_parser.add_argument("--threads", type=int, default=4)
    submission_parser.set_defaults(func=_write_submission)

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
