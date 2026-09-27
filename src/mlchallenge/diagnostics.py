"""Out-of-core error analysis and conservative segmented-threshold selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from mlchallenge.audit import sha256_file
from mlchallenge.large_data import DuckDBRuntime, connect_out_of_core


def _sql_path(path: str | Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def _threshold_predicate(thresholds: dict[str, float], alias: str = "scores") -> str:
    source2 = float(thresholds["source2"])
    source3 = float(thresholds["source3"])
    return (
        f"(({alias}.candidate_source = 'source2' AND {alias}.score >= {source2}) OR "
        f"({alias}.candidate_source = 'source3' AND {alias}.score >= {source3}))"
    )


def _evaluate_thresholds_sql(
    connection: duckdb.DuckDBPyConnection,
    scores: Path,
    split: Path,
    thresholds: dict[str, float],
    *,
    scores_relation: str | None = None,
) -> dict[str, Any]:
    predicate = _threshold_predicate(thresholds)
    score_input = scores_relation or f"read_parquet({_sql_path(scores)})"
    row = connection.execute(
        f"""
        WITH truth_pairs AS (
            SELECT source1_entity_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM read_parquet({_sql_path(split / 'ground_truth.parquet')})
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        ), truth_counts AS (
            SELECT source1_entity_id, count(*) AS true_count
            FROM truth_pairs GROUP BY source1_entity_id
        ), selected AS (
            SELECT scores.source1_entity_id, count(*) AS predicted_count,
                   count(truth.target_entity_id) AS true_positive
            FROM {score_input} scores
            LEFT JOIN truth_pairs truth
              ON scores.source1_entity_id = truth.source1_entity_id
             AND scores.candidate_entity_id = truth.target_entity_id
            WHERE {predicate}
            GROUP BY scores.source1_entity_id
        ), per_entity AS (
            SELECT source1.entity_id,
                   coalesce(truth_counts.true_count, 0) AS true_count,
                   coalesce(selected.predicted_count, 0) AS predicted_count,
                   coalesce(selected.true_positive, 0) AS true_positive
            FROM read_parquet({_sql_path(split / 'source1.parquet')}) source1
            LEFT JOIN truth_counts ON source1.entity_id = truth_counts.source1_entity_id
            LEFT JOIN selected ON source1.entity_id = selected.source1_entity_id
        ), scored AS (
            SELECT *, CASE
                WHEN true_count = 0 AND predicted_count = 0 THEN 1.0
                WHEN true_positive = 0 THEN 0.0
                ELSE 1.25 * true_positive
                     / (1.25 * true_positive + 0.25 * (true_count - true_positive)
                        + (predicted_count - true_positive))
            END AS entity_f0_5
            FROM per_entity
        )
        SELECT count(*), avg(entity_f0_5), sum(true_positive),
               sum(predicted_count - true_positive), sum(true_count - true_positive),
               count_if(true_count = 0 AND predicted_count = 0)::DOUBLE
                   / count_if(true_count = 0),
               count_if(predicted_count = 0)
        FROM scored
        """
    ).fetchone()
    tp, fp, fn = int(row[2]), int(row[3]), int(row[4])
    return {
        "entities": int(row[0]),
        "macro_f0_5": float(row[1]),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "micro_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "micro_recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "singleton_accuracy": float(row[5]),
        "empty_predictions": int(row[6]),
    }


def evaluate_source_thresholds(
    scores_path: str | Path,
    split_directory: str | Path,
    report_path: str | Path,
    *,
    thresholds: dict[str, float],
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Evaluate two frozen source thresholds using exact macro entity-level F0.5."""

    if set(thresholds) != {"source2", "source3"}:
        raise ValueError("thresholds must contain exactly source2 and source3")
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("source thresholds must be in [0, 1]")
    scores = Path(scores_path).resolve()
    split = Path(split_directory).resolve()
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        report = _evaluate_thresholds_sql(connection, scores, split, thresholds)
    finally:
        connection.close()
    report.update({"thresholds": thresholds, "scores_sha256": sha256_file(scores)})
    output = Path(report_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _optimize_one_source(
    connection: duckdb.DuckDBPyConnection,
    scores: Path,
    split: Path,
    thresholds: dict[str, float],
    source: str,
    grid: list[float],
    scores_relation: str,
) -> tuple[float, float]:
    rows: list[tuple[float, float]] = []
    for candidate in grid:
        trial = dict(thresholds)
        trial[source] = candidate
        metrics = _evaluate_thresholds_sql(
            connection, scores, split, trial, scores_relation=scores_relation
        )
        rows.append((candidate, float(metrics["macro_f0_5"])))
    return max(rows, key=lambda item: (item[1], item[0]))


def tune_source_thresholds(
    scores_path: str | Path,
    split_directory: str | Path,
    report_path: str | Path,
    *,
    global_threshold: float,
    search_radius: float = 0.08,
    step: float = 0.01,
    rounds: int = 2,
    minimum_gain: float = 0.0001,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Coordinate-search two source thresholds and reject negligible complexity."""

    if not 0 <= global_threshold <= 1:
        raise ValueError("global threshold must be in [0, 1]")
    if search_radius <= 0 or step <= 0 or rounds < 1:
        raise ValueError("search radius, step, and rounds must be positive")
    scores = Path(scores_path).resolve()
    split = Path(split_directory).resolve()
    lower = max(0.0, global_threshold - search_radius)
    upper = min(1.0, global_threshold + search_radius)
    count = int(round((upper - lower) / step))
    grid = sorted(
        {round(lower + index * step, 10) for index in range(count + 1)}
        | {global_threshold}
    )
    current = {"source2": global_threshold, "source3": global_threshold}
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    trace: list[dict[str, Any]] = []
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE threshold_scores AS
            SELECT * FROM read_parquet({_sql_path(scores)}) WHERE score >= {lower}
            """
        )
        baseline = _evaluate_thresholds_sql(
            connection, scores, split, current, scores_relation="threshold_scores"
        )
        for round_index in range(rounds):
            for source in ("source2", "source3"):
                threshold, score = _optimize_one_source(
                    connection,
                    scores,
                    split,
                    current,
                    source,
                    grid,
                    "threshold_scores",
                )
                current[source] = threshold
                trace.append(
                    {
                        "round": round_index + 1,
                        "source": source,
                        "selected_threshold": threshold,
                        "macro_f0_5": score,
                    }
                )
        selected = _evaluate_thresholds_sql(
            connection, scores, split, current, scores_relation="threshold_scores"
        )
    finally:
        connection.close()
    gain = float(selected["macro_f0_5"] - baseline["macro_f0_5"])
    accepted = gain >= minimum_gain
    if not accepted:
        current = {"source2": global_threshold, "source3": global_threshold}
        selected = baseline
    report = {
        "global_threshold": global_threshold,
        "thresholds": current,
        "accepted": accepted,
        "minimum_gain": minimum_gain,
        "observed_gain_before_guardrail": gain,
        "baseline": baseline,
        "selected": selected,
        "search": {
            "radius": search_radius,
            "step": step,
            "rounds": rounds,
            "grid": grid,
            "trace": trace,
        },
        "scores_sha256": sha256_file(scores),
    }
    output = Path(report_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _optional_candidate_expression(columns: set[str], name: str, default: str) -> str:
    return f"coalesce(candidates.{name}, {default})" if name in columns else default


def analyze_errors(
    scores_path: str | Path,
    candidates_path: str | Path,
    split_directory: str | Path,
    output_directory: str | Path,
    report_path: str | Path,
    *,
    thresholds: dict[str, float],
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Write pair-level errors and country/cardinality/candidate-count F0.5 slices."""

    scores = Path(scores_path).resolve()
    candidates = Path(candidates_path).resolve()
    split = Path(split_directory).resolve()
    output = Path(output_directory).resolve()
    report_output = Path(report_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    predicate = _threshold_predicate(thresholds)
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        candidate_columns = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({_sql_path(candidates)})"
            ).fetchall()
        }
        exact_name = _optional_candidate_expression(candidate_columns, "exact_name", "0")
        exact_address = _optional_candidate_expression(candidate_columns, "exact_address", "0")
        retrieval = _optional_candidate_expression(candidate_columns, "char_name_score", "0.0")
        errors_path = output / "pair_errors.parquet"
        connection.execute(
            f"""
            COPY (
                WITH truth_pairs AS (
                    SELECT source1_entity_id,
                           unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
                    FROM read_parquet({_sql_path(split / 'ground_truth.parquet')})
                    WHERE coalesce(trim(matched_entity_ids), '') <> ''
                ), false_negatives AS (
                    SELECT truth.source1_entity_id,
                           truth.target_entity_id AS candidate_entity_id,
                           CASE WHEN starts_with(truth.target_entity_id, 'S2-')
                                THEN 'source2' ELSE 'source3' END AS candidate_source,
                           CASE WHEN candidates.candidate_entity_id IS NULL
                                THEN 'blocking_false_negative'
                                ELSE 'matcher_false_negative' END AS error_type,
                           scores.score,
                           {exact_name}::FLOAT AS exact_name,
                           {exact_address}::FLOAT AS exact_address,
                           {retrieval}::FLOAT AS retrieval_score
                    FROM truth_pairs truth
                    LEFT JOIN read_parquet({_sql_path(candidates)}) candidates
                      ON truth.source1_entity_id = candidates.source1_entity_id
                     AND truth.target_entity_id = candidates.candidate_entity_id
                    LEFT JOIN read_parquet({_sql_path(scores)}) scores
                      ON truth.source1_entity_id = scores.source1_entity_id
                     AND truth.target_entity_id = scores.candidate_entity_id
                    WHERE candidates.candidate_entity_id IS NULL
                       OR NOT coalesce({_threshold_predicate(thresholds, 'scores')}, false)
                ), false_positives AS (
                    SELECT scores.source1_entity_id, scores.candidate_entity_id,
                           scores.candidate_source, 'false_positive' AS error_type,
                           scores.score,
                           {exact_name}::FLOAT AS exact_name,
                           {exact_address}::FLOAT AS exact_address,
                           {retrieval}::FLOAT AS retrieval_score
                    FROM read_parquet({_sql_path(scores)}) scores
                    JOIN read_parquet({_sql_path(candidates)}) candidates
                      ON scores.source1_entity_id = candidates.source1_entity_id
                     AND scores.candidate_entity_id = candidates.candidate_entity_id
                    LEFT JOIN truth_pairs truth
                      ON scores.source1_entity_id = truth.source1_entity_id
                     AND scores.candidate_entity_id = truth.target_entity_id
                    WHERE {predicate} AND truth.target_entity_id IS NULL
                )
                SELECT errors.*, source1.country
                FROM (
                    SELECT * FROM false_negatives
                    UNION ALL
                    SELECT * FROM false_positives
                ) errors
                JOIN read_parquet({_sql_path(split / 'source1.parquet')}) source1
                  ON errors.source1_entity_id = source1.entity_id
                ORDER BY error_type, score DESC NULLS LAST, source1_entity_id,
                         candidate_entity_id
            ) TO {_sql_path(errors_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )

        per_entity_path = output / "per_entity_metrics.parquet"
        connection.execute(
            f"""
            COPY (
                WITH truth_pairs AS (
                    SELECT source1_entity_id,
                           unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
                    FROM read_parquet({_sql_path(split / 'ground_truth.parquet')})
                    WHERE coalesce(trim(matched_entity_ids), '') <> ''
                ), truth_counts AS (
                    SELECT source1_entity_id, count(*) AS true_count
                    FROM truth_pairs GROUP BY source1_entity_id
                ), candidate_counts AS (
                    SELECT source1_entity_id, count(*) AS candidate_count
                    FROM read_parquet({_sql_path(candidates)}) GROUP BY source1_entity_id
                ), selected AS (
                    SELECT scores.source1_entity_id, count(*) AS predicted_count,
                           count(truth.target_entity_id) AS true_positive
                    FROM read_parquet({_sql_path(scores)}) scores
                    LEFT JOIN truth_pairs truth
                      ON scores.source1_entity_id = truth.source1_entity_id
                     AND scores.candidate_entity_id = truth.target_entity_id
                    WHERE {predicate}
                    GROUP BY scores.source1_entity_id
                ), entity_base AS (
                    SELECT source1.entity_id AS source1_entity_id, source1.country,
                           coalesce(truth_counts.true_count, 0) AS true_count,
                           coalesce(candidate_counts.candidate_count, 0) AS candidate_count,
                           coalesce(selected.predicted_count, 0) AS predicted_count,
                           coalesce(selected.true_positive, 0) AS true_positive
                    FROM read_parquet({_sql_path(split / 'source1.parquet')}) source1
                    LEFT JOIN truth_counts
                      ON source1.entity_id = truth_counts.source1_entity_id
                    LEFT JOIN candidate_counts
                      ON source1.entity_id = candidate_counts.source1_entity_id
                    LEFT JOIN selected ON source1.entity_id = selected.source1_entity_id
                )
                SELECT *,
                       CASE
                           WHEN true_count = 0 AND predicted_count = 0 THEN 1.0
                           WHEN true_positive = 0 THEN 0.0
                           ELSE 1.25 * true_positive
                                / (1.25 * true_positive
                                   + 0.25 * (true_count - true_positive)
                                   + (predicted_count - true_positive))
                       END AS entity_f0_5
                FROM entity_base
            ) TO {_sql_path(per_entity_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        slices: dict[str, pd.DataFrame] = {}
        slices["country"] = connection.execute(
            """
            SELECT country AS segment, count(*) AS entities, avg(entity_f0_5) AS macro_f0_5,
                   sum(true_positive) AS true_positive,
                   sum(predicted_count - true_positive) AS false_positive,
                   sum(true_count - true_positive) AS false_negative
            FROM read_parquet(?) GROUP BY country ORDER BY country
            """,
            [str(per_entity_path)],
        ).fetchdf()
        slices["truth_cardinality"] = connection.execute(
            """
            SELECT CASE WHEN true_count >= 5 THEN '5+' ELSE true_count::VARCHAR END AS segment,
                   count(*) AS entities, avg(entity_f0_5) AS macro_f0_5,
                   sum(true_positive) AS true_positive,
                   sum(predicted_count - true_positive) AS false_positive,
                   sum(true_count - true_positive) AS false_negative
            FROM read_parquet(?) GROUP BY 1 ORDER BY min(true_count)
            """,
            [str(per_entity_path)],
        ).fetchdf()
        slices["candidate_count"] = connection.execute(
            """
            SELECT CASE
                       WHEN candidate_count = 0 THEN '0'
                       WHEN candidate_count <= 50 THEN '1-50'
                       WHEN candidate_count <= 150 THEN '51-150'
                       WHEN candidate_count <= 250 THEN '151-250'
                       ELSE '251+'
                   END AS segment,
                   count(*) AS entities, avg(entity_f0_5) AS macro_f0_5,
                   sum(true_positive) AS true_positive,
                   sum(predicted_count - true_positive) AS false_positive,
                   sum(true_count - true_positive) AS false_negative
            FROM read_parquet(?) GROUP BY 1 ORDER BY min(candidate_count)
            """,
            [str(per_entity_path)],
        ).fetchdf()
        slice_artifacts: dict[str, dict[str, Any]] = {}
        for name, frame in slices.items():
            path = output / f"slice_{name}.csv"
            frame.to_csv(path, index=False)
            slice_artifacts[name] = {
                "path": str(path),
                "rows": len(frame),
                "sha256": sha256_file(path),
            }
        error_counts = connection.execute(
            """
            SELECT error_type, candidate_source, country, count(*) AS rows
            FROM read_parquet(?)
            GROUP BY error_type, candidate_source, country
            ORDER BY error_type, candidate_source, country
            """,
            [str(errors_path)],
        ).fetchdf()
        error_counts_path = output / "error_counts.csv"
        error_counts.to_csv(error_counts_path, index=False)
        error_total_rows = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(errors_path)]
            ).fetchone()[0]
        )
    finally:
        connection.close()

    report = {
        "thresholds": thresholds,
        "error_rows": error_total_rows,
        "artifacts": {
            "pair_errors": {
                "path": str(errors_path),
                "bytes": errors_path.stat().st_size,
                "sha256": sha256_file(errors_path),
            },
            "per_entity_metrics": {
                "path": str(per_entity_path),
                "bytes": per_entity_path.stat().st_size,
                "sha256": sha256_file(per_entity_path),
            },
            "error_counts": {
                "path": str(error_counts_path),
                "rows": len(error_counts),
                "sha256": sha256_file(error_counts_path),
            },
            "slices": slice_artifacts,
        },
        "scores_sha256": sha256_file(scores),
        "candidates_sha256": sha256_file(candidates),
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
