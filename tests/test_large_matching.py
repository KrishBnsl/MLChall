from __future__ import annotations

import duckdb
import pandas as pd

from mlchallenge.diagnostics import (
    analyze_errors,
    evaluate_source_thresholds,
    tune_source_thresholds,
)
from mlchallenge.large_data import DuckDBRuntime
from mlchallenge.large_matching import (
    MatcherConfig,
    SamplingConfig,
    build_mining_pool,
    build_training_sample,
    evaluate_at_threshold,
    score_candidates,
    threshold_curve,
    train_matchers,
)


def _write_parquet(frame: pd.DataFrame, path) -> None:
    connection = duckdb.connect()
    connection.register("frame", frame)
    try:
        connection.execute(f"COPY frame TO '{path}' (FORMAT PARQUET)")
    finally:
        connection.close()


def _matching_fixture(root) -> tuple[object, object]:
    split = root / "split"
    split.mkdir()
    source1_rows = []
    source2_rows = []
    source3_rows = []
    truth_rows = []
    candidate_rows = []
    for index in range(30):
        source1_id = f"S1-{index:03d}"
        true_id = f"S2-{index:03d}"
        decoy_id = f"S2-D{index:03d}"
        source3_id = f"S3-{index:03d}"
        source1_rows.append((source1_id, f"Example Company {index}", f"{index} Main Road", "US"))
        is_singleton = index % 10 == 0
        source2_rows.extend(
            [
                (
                    true_id,
                    f"No Match Target {index}" if is_singleton else f"Example Co {index}",
                    f"{index + 900} Remote Lane" if is_singleton else f"{index} Main Rd",
                    "US",
                ),
                (decoy_id, f"Example Company {index + 100}", f"{index + 100} Main Rd", "US"),
            ]
        )
        source3_rows.append(
            (source3_id, f"Unrelated Enterprise {index}", f"{index + 500} Other Ave", "US")
        )
        truth_rows.append((source1_id, "" if is_singleton else true_id))
        candidate_rows.extend(
            [
                (
                    source1_id,
                    true_id,
                    "source2",
                    0,
                    0,
                    0.5 if is_singleton else 4.0,
                    1 if is_singleton else 2,
                    0.20 if is_singleton else 0.95,
                ),
                (source1_id, decoy_id, "source2", 0, 0, 2.0, 1, 0.55),
                (source1_id, source3_id, "source3", 0, 0, 1.0, 1, 0.35),
            ]
        )
    record_columns = ("entity_id", "business_name", "business_address", "country")
    _write_parquet(pd.DataFrame(source1_rows, columns=record_columns), split / "source1.parquet")
    _write_parquet(pd.DataFrame(source2_rows, columns=record_columns), split / "source2.parquet")
    _write_parquet(pd.DataFrame(source3_rows, columns=record_columns), split / "source3.parquet")
    _write_parquet(
        pd.DataFrame(truth_rows, columns=("source1_entity_id", "matched_entity_ids")),
        split / "ground_truth.parquet",
    )
    candidates = root / "candidates.parquet"
    _write_parquet(
        pd.DataFrame(
            candidate_rows,
            columns=(
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "exact_name",
                "exact_address",
                "fuzzy_score",
                "shared_blocking_keys",
                "char_name_score",
            ),
        ),
        candidates,
    )
    return split, candidates


def test_large_matching_pipeline_end_to_end(tmp_path) -> None:
    split, candidates = _matching_fixture(tmp_path)
    runtime = DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB")
    mining_pool = tmp_path / "mining_pool.parquet"
    mining_pool_report = build_mining_pool(
        candidates,
        mining_pool,
        tmp_path / "mining_pool.json",
        candidates_per_source=1,
        runtime=runtime,
        temp_directory=tmp_path / "mining_pool_tmp",
    )
    assert mining_pool_report["rows"] == 60
    assert mining_pool_report["candidate_count_per_entity_source"]["max"] == 1

    sample = tmp_path / "training_sample.parquet"
    sample_report = build_training_sample(
        candidates,
        split,
        sample,
        tmp_path / "sample_report.json",
        sampling=SamplingConfig(
            hard_negatives_per_source=1,
            random_negatives_per_source=0,
            cv_folds=3,
            calibration_modulus=3,
            seed=17,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "sample_tmp",
    )
    assert sample_report["positive_rows"] == 27
    assert sample_report["negative_rows"] > 0

    models = tmp_path / "models"
    train_report = train_matchers(
        sample,
        models,
        tmp_path / "train_report.json",
        config=MatcherConfig(max_iter=30, min_samples_leaf=2, random_state=17),
        cv_folds=3,
    )
    assert set(train_report["artifacts"]) == {"oof_fold_0", "oof_fold_1", "oof_fold_2", "final"}

    scores = tmp_path / "scores.parquet"
    score_report = score_candidates(
        candidates,
        split,
        scores,
        tmp_path / "score_report.json",
        model_path=None,
        oof_model_directory=models,
        cv_folds=3,
        runtime=runtime,
        temp_directory=tmp_path / "score_tmp",
        vectors_per_chunk=1,
    )
    assert score_report["rows"] == 90

    threshold_report = threshold_curve(
        scores,
        split,
        tmp_path / "threshold_curve.csv",
        tmp_path / "threshold_report.json",
        steps=51,
        runtime=runtime,
        temp_directory=tmp_path / "threshold_tmp",
    )
    evaluation = evaluate_at_threshold(
        scores,
        split,
        tmp_path / "evaluation.json",
        threshold=threshold_report["threshold"],
        runtime=runtime,
        temp_directory=tmp_path / "evaluation_tmp",
    )
    assert evaluation["macro_f0_5"] >= 0.9
    assert evaluation["singleton_accuracy"] == 1.0

    segmented = tune_source_thresholds(
        scores,
        split,
        tmp_path / "segmented_thresholds.json",
        global_threshold=threshold_report["threshold"],
        search_radius=0.04,
        step=0.02,
        rounds=1,
        minimum_gain=0.0,
        runtime=runtime,
        temp_directory=tmp_path / "segmented_tmp",
    )
    segmented_evaluation = evaluate_source_thresholds(
        scores,
        split,
        tmp_path / "segmented_evaluation.json",
        thresholds=segmented["thresholds"],
        runtime=runtime,
        temp_directory=tmp_path / "segmented_eval_tmp",
    )
    assert segmented_evaluation["macro_f0_5"] >= 0.9

    diagnostics = analyze_errors(
        scores,
        candidates,
        split,
        tmp_path / "error_analysis",
        tmp_path / "error_analysis.json",
        thresholds=segmented["thresholds"],
        runtime=runtime,
        temp_directory=tmp_path / "error_tmp",
    )
    assert diagnostics["artifacts"]["pair_errors"]["bytes"] > 0
    cardinality_slice = pd.read_csv(diagnostics["artifacts"]["slices"]["truth_cardinality"]["path"])
    singleton_score = cardinality_slice.loc[
        cardinality_slice["segment"].astype(str) == "0", "macro_f0_5"
    ].iloc[0]
    assert singleton_score == 1.0

    mined_sample = tmp_path / "mined_sample.parquet"
    mined_report = build_training_sample(
        candidates,
        split,
        mined_sample,
        tmp_path / "mined_sample.json",
        sampling=SamplingConfig(
            hard_negatives_per_source=1,
            random_negatives_per_source=0,
            model_hard_negatives_per_source=1,
            cv_folds=3,
            calibration_modulus=3,
            seed=17,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "mined_sample_tmp",
        mining_scores_path=scores,
    )
    assert mined_report["hard_example_mining"] is not None
    assert mined_report["positive_rows"] == 27
