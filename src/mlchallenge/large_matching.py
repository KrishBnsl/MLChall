"""Bounded-memory feature, training, scoring, and submission pipeline."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from mlchallenge.audit import sha256_file
from mlchallenge.large_data import DuckDBRuntime, connect_out_of_core

LARGE_FEATURE_COLUMNS = (
    "block_exact_name",
    "block_exact_address",
    "retrieval_score",
    "log_fuzzy_score",
    "shared_blocking_keys",
    "candidate_rank_source_log",
    "candidate_rank_reciprocal",
    "candidate_count_log",
    "source_candidate_count_log",
    "retrieval_gap_from_best",
    "retrieval_ratio_to_best",
    "blocker_agreement",
    "block_exact_both",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_containment",
    "name_shared_token_count_log",
    "name_edit_similarity",
    "name_exact",
    "name_sorted_exact",
    "name_first_token_exact",
    "name_last_token_exact",
    "address_jaro_winkler",
    "address_token_jaccard",
    "address_token_containment",
    "address_shared_token_count_log",
    "address_edit_similarity",
    "address_exact",
    "address_digit_jaccard",
    "address_number_overlap_count_log",
    "address_number_subset",
    "same_first_address_number",
    "conflicting_first_address_number",
    "country_exact",
    "name_missing_either",
    "address_missing_either",
    "candidate_is_source3",
    "query_name_frequency_log",
    "candidate_name_frequency_log",
    "candidate_address_frequency_log",
    "name_length_ratio",
    "address_length_ratio",
)


@dataclass(frozen=True)
class SamplingConfig:
    hard_negatives_per_source: int = 5
    random_negatives_per_source: int = 1
    model_hard_negatives_per_source: int = 0
    cv_folds: int = 3
    calibration_modulus: int = 10
    seed: int = 20260925

    def __post_init__(self) -> None:
        if self.hard_negatives_per_source < 1:
            raise ValueError("hard_negatives_per_source must be positive")
        if self.random_negatives_per_source < 0:
            raise ValueError("random_negatives_per_source cannot be negative")
        if self.model_hard_negatives_per_source < 0:
            raise ValueError("model_hard_negatives_per_source cannot be negative")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least two")
        if self.calibration_modulus < 3:
            raise ValueError("calibration_modulus must be at least three")


@dataclass(frozen=True)
class MatcherConfig:
    model_family: str = "hist_gradient_boosting"
    learning_rate: float = 0.08
    max_iter: int = 140
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    threads: int = 4
    random_state: int = 20260925

    def __post_init__(self) -> None:
        supported = {"hist_gradient_boosting", "lightgbm", "xgboost", "catboost"}
        if self.model_family not in supported:
            raise ValueError(f"unsupported model family: {self.model_family}")
        if self.threads < 1:
            raise ValueError("threads must be positive")


@dataclass
class CalibratedMatcher:
    """A nonlinear pair classifier followed by weighted sigmoid calibration."""

    estimator: Any
    calibrator: LogisticRegression | None
    feature_columns: tuple[str, ...]
    config: dict[str, Any]

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=False)
        raw = np.clip(self.estimator.predict_proba(matrix)[:, 1], 1e-7, 1 - 1e-7)
        if self.calibrator is None:
            return raw
        logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]


def _sql_path(path: str | Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def _normalized(column: str) -> str:
    return (
        "trim(regexp_replace(replace(lower(coalesce("
        f"{column}, '')), '&', ' and '), "
        "'[^\\p{L}\\p{N}\\p{M}]+', ' ', 'g'))"
    )


def _candidate_columns(connection: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM read_parquet({_sql_path(path)})").fetchall()
    return {str(row[0]) for row in rows}


def _optional_candidate_column(columns: set[str], name: str, default: str) -> str:
    return f"coalesce({name}, {default})" if name in columns else default


def _feature_query(
    connection: duckdb.DuckDBPyConnection,
    candidates_path: Path,
    split_directory: Path,
    *,
    sampling: SamplingConfig | None = None,
    mining_scores_path: Path | None = None,
) -> str:
    """Return one SQL query for either sampled labeled features or all inference features."""

    source1 = split_directory / "source1.parquet"
    source2 = split_directory / "source2.parquet"
    source3 = split_directory / "source3.parquet"
    for path in (candidates_path, source1, source2, source3):
        if not path.is_file():
            raise FileNotFoundError(f"required feature input is missing: {path}")
    truth = split_directory / "ground_truth.parquet"
    if sampling is not None and not truth.is_file():
        raise FileNotFoundError(f"labeled feature sampling requires {truth}")
    if mining_scores_path is not None:
        if sampling is None:
            raise ValueError("hard-example mining scores require labeled sampling")
        if not mining_scores_path.is_file():
            raise FileNotFoundError(
                f"hard-example mining score file is missing: {mining_scores_path}"
            )

    columns = _candidate_columns(connection, candidates_path)
    exact_name = _optional_candidate_column(columns, "exact_name", "0")
    exact_address = _optional_candidate_column(columns, "exact_address", "0")
    fuzzy_score = _optional_candidate_column(columns, "fuzzy_score", "0.0")
    shared_keys = _optional_candidate_column(columns, "shared_blocking_keys", "0")
    retrieval_score = _optional_candidate_column(columns, "char_name_score", "0.0")
    seed = sampling.seed if sampling is not None else 20260925
    cv_folds = sampling.cv_folds if sampling is not None else 3
    calibration_modulus = sampling.calibration_modulus if sampling is not None else 10

    if sampling is None:
        truth_cte = ""
        label_expression = "0 AS label,"
        sample_ctes = "selected AS (SELECT * FROM ranked),"
        sample_columns = ""
    else:
        truth_cte = f"""
        truth_pairs AS (
            SELECT source1_entity_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM read_parquet({_sql_path(truth)})
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        ),
        """
        label_expression = "CASE WHEN truth.target_entity_id IS NULL THEN 0 ELSE 1 END AS label,"
        random_limit = sampling.random_negatives_per_source
        model_hard_limit = sampling.model_hard_negatives_per_source
        sample_ctes = f"""
        sample_ranked AS (
            SELECT *,
                row_number() OVER (
                    PARTITION BY source1_entity_id, candidate_source, label
                    ORDER BY block_strength DESC, candidate_entity_id
                ) AS label_strength_rank,
                row_number() OVER (
                    PARTITION BY source1_entity_id, candidate_source, label
                    ORDER BY hash(candidate_entity_id || ':{seed}:negative')
                ) AS label_random_rank,
                row_number() OVER (
                    PARTITION BY source1_entity_id, candidate_source, label
                    ORDER BY model_score DESC, block_strength DESC, candidate_entity_id
                ) AS label_model_rank,
                count_if(label = 0) OVER (
                    PARTITION BY source1_entity_id, candidate_source
                ) AS total_negative_count
            FROM ranked
        ),
        selected_unweighted AS (
            SELECT *
            FROM sample_ranked
            WHERE label = 1
               OR label_strength_rank <= {sampling.hard_negatives_per_source}
               OR (label_model_rank <= {model_hard_limit} AND label = 0)
               OR (label_random_rank <= {random_limit} AND label = 0)
        ),
        selected AS (
            SELECT *,
                count_if(label = 0) OVER (
                    PARTITION BY source1_entity_id, candidate_source
                ) AS sampled_negative_count
            FROM selected_unweighted
        ),
        """
        sample_columns = """
            label::UTINYINT AS label,
            CASE WHEN label = 1 THEN 1.0
                 ELSE total_negative_count::DOUBLE / greatest(sampled_negative_count, 1)
            END::FLOAT AS sample_weight,
        """

    record_name = _normalized("business_name")
    record_address = _normalized("business_address")
    truth_join = ""
    if sampling is not None:
        truth_join = (
            "LEFT JOIN truth_pairs truth "
            "ON candidates.source1_entity_id = truth.source1_entity_id "
            "AND candidates.candidate_entity_id = truth.target_entity_id"
        )

    model_score_expression = "0.0"
    model_score_join = ""
    if mining_scores_path is not None:
        model_score_expression = "coalesce(mining.score, 0.0)"
        model_score_join = (
            f"LEFT JOIN read_parquet({_sql_path(mining_scores_path)}) mining "
            "ON candidates.source1_entity_id = mining.source1_entity_id "
            "AND candidates.candidate_entity_id = mining.candidate_entity_id"
        )

    return f"""
        WITH
        {truth_cte}
        raw_candidates AS (
            SELECT
                candidates.source1_entity_id,
                candidates.candidate_entity_id,
                candidates.candidate_source,
                ({exact_name})::DOUBLE AS block_exact_name,
                ({exact_address})::DOUBLE AS block_exact_address,
                ({fuzzy_score})::DOUBLE AS fuzzy_score,
                ({shared_keys})::DOUBLE AS shared_blocking_keys,
                ({retrieval_score})::DOUBLE AS retrieval_score,
                ({model_score_expression})::DOUBLE AS model_score
            FROM read_parquet({_sql_path(candidates_path)}) candidates
            {model_score_join}
        ),
        labeled AS (
            SELECT candidates.*,
                {label_expression}
                4.0 * candidates.block_exact_name
                + 4.0 * candidates.block_exact_address
                + 2.0 * candidates.retrieval_score
                + 0.10 * ln(1.0 + greatest(candidates.fuzzy_score, 0.0))
                + 0.01 * candidates.shared_blocking_keys AS block_strength
            FROM raw_candidates candidates
            {truth_join}
        ),
        ranked AS (
            SELECT *,
                row_number() OVER (
                    PARTITION BY source1_entity_id, candidate_source
                    ORDER BY block_strength DESC, candidate_entity_id
                ) AS candidate_rank_source,
                count(*) OVER (PARTITION BY source1_entity_id) AS candidate_count,
                count(*) OVER (PARTITION BY source1_entity_id, candidate_source)
                    AS source_candidate_count,
                max(retrieval_score) OVER (PARTITION BY source1_entity_id)
                    AS best_retrieval_score
            FROM labeled
        ),
        {sample_ctes}
        query_base AS (
            SELECT entity_id, lower(trim(coalesce(country, ''))) AS left_country,
                   {record_name} AS left_name,
                   {record_address} AS left_address
            FROM read_parquet({_sql_path(source1)})
        ),
        query_enriched AS (
            SELECT *, count(*) OVER (PARTITION BY left_country, left_name)
                AS query_name_frequency
            FROM query_base
        ),
        target_base AS (
            SELECT entity_id, 'source2' AS record_source,
                   lower(trim(coalesce(country, ''))) AS right_country,
                   {record_name} AS right_name,
                   {record_address} AS right_address
            FROM read_parquet({_sql_path(source2)})
            UNION ALL
            SELECT entity_id, 'source3' AS record_source,
                   lower(trim(coalesce(country, ''))) AS right_country,
                   {record_name} AS right_name,
                   {record_address} AS right_address
            FROM read_parquet({_sql_path(source3)})
        ),
        target_enriched AS (
            SELECT *,
                count(*) OVER (PARTITION BY record_source, right_country, right_name)
                    AS candidate_name_frequency,
                count(*) OVER (PARTITION BY record_source, right_country, right_address)
                    AS candidate_address_frequency
            FROM target_base
        ),
        joined AS (
            SELECT selected.*,
                query.left_name,
                query.left_address,
                query.left_country,
                query.query_name_frequency,
                target.right_name,
                target.right_address,
                target.right_country,
                target.candidate_name_frequency,
                target.candidate_address_frequency
            FROM selected
            JOIN query_enriched query
              ON selected.source1_entity_id = query.entity_id
            JOIN target_enriched target
              ON selected.candidate_entity_id = target.entity_id
             AND selected.candidate_source = target.record_source
        ),
        tokenized AS (
            SELECT *,
                list_distinct(list_filter(string_split(left_name, ' '), item -> item <> ''))
                    AS left_name_tokens,
                list_distinct(list_filter(string_split(right_name, ' '), item -> item <> ''))
                    AS right_name_tokens,
                list_distinct(list_filter(string_split(left_address, ' '), item -> item <> ''))
                    AS left_address_tokens,
                list_distinct(list_filter(string_split(right_address, ' '), item -> item <> ''))
                    AS right_address_tokens,
                list_distinct(regexp_extract_all(left_address, '\\d+')) AS left_digits,
                list_distinct(regexp_extract_all(right_address, '\\d+')) AS right_digits
            FROM joined
        ),
        feature_base AS (
            SELECT *,
                list_unique(list_intersect(left_name_tokens, right_name_tokens))
                    AS shared_name_token_count,
                list_unique(list_intersect(left_address_tokens, right_address_tokens))
                    AS shared_address_token_count,
                list_unique(list_intersect(left_digits, right_digits))
                    AS shared_address_number_count
            FROM tokenized
        )
        SELECT
            source1_entity_id,
            candidate_entity_id,
            candidate_source,
            {sample_columns}
            (hash(source1_entity_id || ':{seed}:fold') % {cv_folds})::UTINYINT AS cv_fold,
            (hash(source1_entity_id || ':{seed}:calibration')
                % {calibration_modulus})::UTINYINT AS calibration_bucket,
            block_exact_name::FLOAT AS block_exact_name,
            block_exact_address::FLOAT AS block_exact_address,
            retrieval_score::FLOAT AS retrieval_score,
            ln(1.0 + greatest(fuzzy_score, 0.0))::FLOAT AS log_fuzzy_score,
            shared_blocking_keys::FLOAT AS shared_blocking_keys,
            ln(1.0 + candidate_rank_source)::FLOAT AS candidate_rank_source_log,
            (1.0 / candidate_rank_source)::FLOAT AS candidate_rank_reciprocal,
            ln(1.0 + candidate_count)::FLOAT AS candidate_count_log,
            ln(1.0 + source_candidate_count)::FLOAT AS source_candidate_count_log,
            (best_retrieval_score - retrieval_score)::FLOAT AS retrieval_gap_from_best,
            CASE WHEN best_retrieval_score <= 0 THEN 0.0
                 ELSE retrieval_score / best_retrieval_score END::FLOAT
                 AS retrieval_ratio_to_best,
            ((block_exact_name > 0 OR block_exact_address > 0
                OR shared_blocking_keys > 0) AND retrieval_score > 0)::INT::FLOAT
                AS blocker_agreement,
            (block_exact_name > 0 AND block_exact_address > 0)::INT::FLOAT
                AS block_exact_both,
            CASE WHEN left_name = '' OR right_name = '' THEN 0.0
                 ELSE jaro_winkler_similarity(left_name, right_name) END::FLOAT
                 AS name_jaro_winkler,
            CASE
                WHEN list_unique(list_concat(left_name_tokens, right_name_tokens)) = 0 THEN 0.0
                ELSE list_unique(list_intersect(left_name_tokens, right_name_tokens))::DOUBLE
                     / list_unique(list_concat(left_name_tokens, right_name_tokens))
            END::FLOAT AS name_token_jaccard,
            CASE WHEN least(len(left_name_tokens), len(right_name_tokens)) = 0 THEN 0.0
                 ELSE shared_name_token_count::DOUBLE
                      / least(len(left_name_tokens), len(right_name_tokens))
            END::FLOAT AS name_token_containment,
            ln(1.0 + shared_name_token_count)::FLOAT AS name_shared_token_count_log,
            CASE WHEN greatest(length(left_name), length(right_name)) = 0 THEN 0.0
                 ELSE 1.0 - levenshtein(left_name, right_name)::DOUBLE
                      / greatest(length(left_name), length(right_name))
            END::FLOAT AS name_edit_similarity,
            (left_name <> '' AND replace(left_name, ' ', '') = replace(right_name, ' ', ''))::INT
                ::FLOAT AS name_exact,
            (len(left_name_tokens) > 0 AND len(right_name_tokens) > 0
                AND list_sort(left_name_tokens) = list_sort(right_name_tokens))::INT::FLOAT
                AS name_sorted_exact,
            (len(left_name_tokens) > 0 AND len(right_name_tokens) > 0
                AND left_name_tokens[1] = right_name_tokens[1])::INT::FLOAT
                AS name_first_token_exact,
            (len(left_name_tokens) > 0 AND len(right_name_tokens) > 0
                AND left_name_tokens[-1] = right_name_tokens[-1])::INT::FLOAT
                AS name_last_token_exact,
            CASE WHEN left_address = '' OR right_address = '' THEN 0.0
                 ELSE jaro_winkler_similarity(left_address, right_address) END::FLOAT
                 AS address_jaro_winkler,
            CASE
                WHEN list_unique(
                    list_concat(left_address_tokens, right_address_tokens)
                ) = 0 THEN 0.0
                ELSE list_unique(list_intersect(left_address_tokens, right_address_tokens))::DOUBLE
                     / list_unique(list_concat(left_address_tokens, right_address_tokens))
            END::FLOAT AS address_token_jaccard,
            CASE WHEN least(len(left_address_tokens), len(right_address_tokens)) = 0 THEN 0.0
                 ELSE shared_address_token_count::DOUBLE
                      / least(len(left_address_tokens), len(right_address_tokens))
            END::FLOAT AS address_token_containment,
            ln(1.0 + shared_address_token_count)::FLOAT
                AS address_shared_token_count_log,
            CASE WHEN greatest(length(left_address), length(right_address)) = 0 THEN 0.0
                 ELSE 1.0 - levenshtein(left_address, right_address)::DOUBLE
                      / greatest(length(left_address), length(right_address))
            END::FLOAT AS address_edit_similarity,
            (left_address <> ''
                AND replace(left_address, ' ', '') = replace(right_address, ' ', ''))
                ::INT::FLOAT AS address_exact,
            CASE
                WHEN list_unique(list_concat(left_digits, right_digits)) = 0 THEN 0.0
                ELSE list_unique(list_intersect(left_digits, right_digits))::DOUBLE
                     / list_unique(list_concat(left_digits, right_digits))
            END::FLOAT AS address_digit_jaccard,
            ln(1.0 + shared_address_number_count)::FLOAT
                AS address_number_overlap_count_log,
            (len(left_digits) > 0 AND len(right_digits) > 0
                AND shared_address_number_count = least(len(left_digits), len(right_digits)))
                ::INT::FLOAT AS address_number_subset,
            (len(left_digits) > 0 AND len(right_digits) > 0
                AND left_digits[1] = right_digits[1])::INT::FLOAT AS same_first_address_number,
            (len(left_digits) > 0 AND len(right_digits) > 0
                AND left_digits[1] <> right_digits[1])::INT::FLOAT
                AS conflicting_first_address_number,
            (left_country <> '' AND left_country = right_country)::INT::FLOAT AS country_exact,
            (left_name = '' OR right_name = '')::INT::FLOAT AS name_missing_either,
            (left_address = '' OR right_address = '')::INT::FLOAT AS address_missing_either,
            starts_with(candidate_entity_id, 'S3-')::INT::FLOAT AS candidate_is_source3,
            ln(1.0 + query_name_frequency)::FLOAT AS query_name_frequency_log,
            ln(1.0 + candidate_name_frequency)::FLOAT AS candidate_name_frequency_log,
            ln(1.0 + candidate_address_frequency)::FLOAT AS candidate_address_frequency_log,
            CASE WHEN greatest(length(left_name), length(right_name)) = 0 THEN 0.0
                 ELSE least(length(left_name), length(right_name))::DOUBLE
                      / greatest(length(left_name), length(right_name))
            END::FLOAT AS name_length_ratio,
            CASE WHEN greatest(length(left_address), length(right_address)) = 0 THEN 0.0
                 ELSE least(length(left_address), length(right_address))::DOUBLE
                      / greatest(length(left_address), length(right_address))
            END::FLOAT AS address_length_ratio
        FROM feature_base
    """


def prepare_test_split(
    data_root: str | Path,
    output_directory: str | Path,
    *,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Convert the official test TSVs into the same Parquet layout as development splits."""

    root = Path(data_root).resolve() / "test"
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "source1": root / "test_source1.tsv",
        "source2": root / "test_source2.tsv",
        "source3": root / "test_source3.tsv",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required test file is missing: {path}")
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    written: dict[str, dict[str, Any]] = {}
    try:
        for source, path in source_paths.items():
            destination = output / f"{source}.parquet"
            connection.execute(
                f"""
                COPY (
                    SELECT * FROM read_csv({_sql_path(path)}, delim='\\t', header=true,
                                           all_varchar=true)
                ) TO {_sql_path(destination)}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )
            rows = connection.execute(
                f"SELECT count(*) FROM read_parquet({_sql_path(destination)})"
            ).fetchone()[0]
            written[source] = {
                "path": str(destination),
                "rows": int(rows),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
    finally:
        connection.close()
    report = {"runtime": asdict(runtime), "artifacts": written}
    report_path = output / "test_split_metadata.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_training_sample(
    candidates_path: str | Path,
    split_directory: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    sampling: SamplingConfig,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
    mining_scores_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a weighted sample containing all positives and deterministic hard negatives."""

    candidates = Path(candidates_path).resolve()
    split = Path(split_directory).resolve()
    output = Path(output_path).resolve()
    report_output = Path(report_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    mining_scores = Path(mining_scores_path).resolve() if mining_scores_path is not None else None
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        query = _feature_query(
            connection,
            candidates,
            split,
            sampling=sampling,
            mining_scores_path=mining_scores,
        )
        connection.execute(
            f"COPY ({query}) TO {_sql_path(output)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        row = connection.execute(
            f"""
            SELECT count(*), count_if(label = 1), count_if(label = 0),
                   count(DISTINCT source1_entity_id), min(sample_weight), max(sample_weight)
            FROM read_parquet({_sql_path(output)})
            """
        ).fetchone()
    finally:
        connection.close()
    report = {
        "sampling": asdict(sampling),
        "runtime": asdict(runtime),
        "rows": int(row[0]),
        "positive_rows": int(row[1]),
        "negative_rows": int(row[2]),
        "entities": int(row[3]),
        "sample_weight_min": float(row[4]),
        "sample_weight_max": float(row[5]),
        "hard_example_mining": (
            {"scores_path": str(mining_scores), "scores_sha256": sha256_file(mining_scores)}
            if mining_scores is not None
            else None
        ),
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _make_tree_estimator(config: MatcherConfig) -> Any:
    if config.model_family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            early_stopping=False,
            random_state=config.random_state,
        )
    if config.model_family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:  # pragma: no cover - optional experiment dependency
            raise RuntimeError("lightgbm is not installed; install the benchmark extra") from error
        return LGBMClassifier(
            objective="binary",
            learning_rate=config.learning_rate,
            n_estimators=config.max_iter,
            num_leaves=config.max_leaf_nodes,
            min_child_samples=config.min_samples_leaf,
            reg_lambda=config.l2_regularization,
            n_jobs=config.threads,
            random_state=config.random_state,
            verbosity=-1,
        )
    if config.model_family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:  # pragma: no cover - optional experiment dependency
            raise RuntimeError("xgboost is not installed; install the benchmark extra") from error
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            grow_policy="lossguide",
            learning_rate=config.learning_rate,
            n_estimators=config.max_iter,
            max_leaves=config.max_leaf_nodes,
            min_child_weight=max(1.0, config.min_samples_leaf / 10.0),
            reg_lambda=config.l2_regularization,
            n_jobs=config.threads,
            random_state=config.random_state,
        )
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:  # pragma: no cover - optional experiment dependency
        raise RuntimeError("catboost is not installed; install the benchmark extra") from error
    return CatBoostClassifier(
        loss_function="Logloss",
        learning_rate=config.learning_rate,
        iterations=config.max_iter,
        depth=max(4, min(10, round(math.log2(config.max_leaf_nodes)))),
        l2_leaf_reg=config.l2_regularization,
        thread_count=config.threads,
        random_seed=config.random_state,
        verbose=False,
        allow_writing_files=False,
    )


def _fit_one_matcher(
    frame: pd.DataFrame,
    base_mask: np.ndarray,
    calibration_mask: np.ndarray,
    config: MatcherConfig,
) -> CalibratedMatcher:
    base = frame.loc[base_mask]
    calibration = frame.loc[calibration_mask]
    if base["label"].nunique() != 2:
        raise ValueError("base training partition must contain positive and negative examples")
    estimator = _make_tree_estimator(config)
    estimator.fit(
        base.loc[:, LARGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False),
        base["label"].to_numpy(dtype=np.int8, copy=False),
        sample_weight=base["sample_weight"].to_numpy(dtype=np.float64, copy=False),
    )
    calibrator: LogisticRegression | None = None
    if len(calibration) and calibration["label"].nunique() == 2:
        raw = np.clip(
            estimator.predict_proba(
                calibration.loc[:, LARGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
            )[:, 1],
            1e-7,
            1 - 1e-7,
        )
        logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        calibrator = LogisticRegression(C=10.0, max_iter=500, random_state=config.random_state)
        calibrator.fit(
            logits,
            calibration["label"].to_numpy(dtype=np.int8, copy=False),
            sample_weight=calibration["sample_weight"].to_numpy(dtype=np.float64, copy=False),
        )
    return CalibratedMatcher(
        estimator=estimator,
        calibrator=calibrator,
        feature_columns=LARGE_FEATURE_COLUMNS,
        config=asdict(config),
    )


def train_matchers(
    sample_path: str | Path,
    model_directory: str | Path,
    report_path: str | Path,
    *,
    config: MatcherConfig,
    cv_folds: int,
) -> dict[str, Any]:
    """Fit group-disjoint OOF models and one final calibrated model."""

    sample = Path(sample_path).resolve()
    model_dir = Path(model_directory).resolve()
    report_output = Path(report_path).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        select_columns = [
            "label",
            "sample_weight",
            "cv_fold",
            "calibration_bucket",
            *LARGE_FEATURE_COLUMNS,
        ]
        casts = [
            "label::UTINYINT AS label",
            "sample_weight::FLOAT AS sample_weight",
            "cv_fold::UTINYINT AS cv_fold",
            "calibration_bucket::UTINYINT AS calibration_bucket",
            *(f"{column}::FLOAT AS {column}" for column in LARGE_FEATURE_COLUMNS),
        ]
        del select_columns
        frame = connection.execute(
            f"SELECT {', '.join(casts)} FROM read_parquet({_sql_path(sample)})"
        ).fetchdf()
    finally:
        connection.close()
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("training sample must contain positive and negative examples")

    artifacts: dict[str, dict[str, Any]] = {}
    for fold in range(cv_folds):
        eligible = frame["cv_fold"].to_numpy() != fold
        calibration = eligible & (frame["calibration_bucket"].to_numpy() == 0)
        base = eligible & ~calibration
        matcher = _fit_one_matcher(frame, base, calibration, config)
        path = model_dir / f"oof_fold_{fold}.joblib"
        joblib.dump(matcher, path, compress=3)
        artifacts[f"oof_fold_{fold}"] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "base_rows": int(base.sum()),
            "calibration_rows": int(calibration.sum()),
        }

    final_calibration = frame["calibration_bucket"].to_numpy() == 0
    final_base = ~final_calibration
    final_matcher = _fit_one_matcher(frame, final_base, final_calibration, config)
    final_path = model_dir / "final_matcher.joblib"
    joblib.dump(final_matcher, final_path, compress=3)
    artifacts["final"] = {
        "path": str(final_path),
        "bytes": final_path.stat().st_size,
        "sha256": sha256_file(final_path),
        "base_rows": int(final_base.sum()),
        "calibration_rows": int(final_calibration.sum()),
    }
    report = {
        "config": asdict(config),
        "cv_folds": cv_folds,
        "sample_rows": len(frame),
        "positive_rows": int(frame["label"].sum()),
        "artifacts": artifacts,
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _write_score_chunk(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    output: Path,
) -> None:
    connection.register("score_chunk", frame)
    try:
        connection.execute(
            f"COPY score_chunk TO {_sql_path(output)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.unregister("score_chunk")


def score_candidates(
    candidates_path: str | Path,
    split_directory: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    model_path: str | Path | None,
    oof_model_directory: str | Path | None,
    cv_folds: int,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
    vectors_per_chunk: int = 128,
) -> dict[str, Any]:
    """Stream vectorized features through either a final model or OOF model set."""

    if (model_path is None) == (oof_model_directory is None):
        raise ValueError("provide exactly one of model_path or oof_model_directory")
    candidates = Path(candidates_path).resolve()
    split = Path(split_directory).resolve()
    output = Path(output_path).resolve()
    report_output = Path(report_path).resolve()
    chunk_directory = output.parent / f".{output.stem}_score_chunks"
    if output.exists() or chunk_directory.exists():
        raise FileExistsError(f"score output already exists: {output} or {chunk_directory}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    chunk_directory.mkdir(parents=True)

    final_matcher: CalibratedMatcher | None = None
    oof_matchers: dict[int, CalibratedMatcher] = {}
    if model_path is not None:
        final_matcher = joblib.load(Path(model_path))
    else:
        model_dir = Path(oof_model_directory).resolve()  # type: ignore[arg-type]
        oof_matchers = {
            fold: joblib.load(model_dir / f"oof_fold_{fold}.joblib") for fold in range(cv_folds)
        }

    reader = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    writer = duckdb.connect()
    chunks: list[Path] = []
    rows = 0
    score_min = math.inf
    score_max = -math.inf
    try:
        query = _feature_query(reader, candidates, split, sampling=None)
        reader.execute(query)
        chunk_number = 0
        while True:
            frame = reader.fetch_df_chunk(vectors_per_chunk=vectors_per_chunk)
            if frame.empty:
                break
            scores = np.empty(len(frame), dtype=np.float32)
            if final_matcher is not None:
                scores[:] = final_matcher.predict_scores(frame).astype(np.float32)
            else:
                folds = frame["cv_fold"].to_numpy(dtype=np.int8, copy=False)
                for fold, matcher in oof_matchers.items():
                    mask = folds == fold
                    if mask.any():
                        scores[mask] = matcher.predict_scores(frame.loc[mask]).astype(np.float32)
            score_frame = frame.loc[
                :, ["source1_entity_id", "candidate_entity_id", "candidate_source"]
            ].copy()
            score_frame["score"] = scores
            chunk_path = chunk_directory / f"scores_{chunk_number:05d}.parquet"
            _write_score_chunk(writer, score_frame, chunk_path)
            chunks.append(chunk_path)
            rows += len(score_frame)
            score_min = min(score_min, float(scores.min()))
            score_max = max(score_max, float(scores.max()))
            chunk_number += 1
        if not chunks:
            raise RuntimeError("candidate scoring produced no rows")
        writer.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet({_sql_path(chunk_directory / "*.parquet")})
                ORDER BY source1_entity_id, candidate_entity_id
            ) TO {_sql_path(output)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    finally:
        reader.close()
        writer.close()
        if chunk_directory.exists():
            shutil.rmtree(chunk_directory)
    report = {
        "rows": rows,
        "score_min": score_min,
        "score_max": score_max,
        "runtime": asdict(runtime),
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "scoring_mode": "final" if final_matcher is not None else "out_of_fold",
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def threshold_curve(
    scores_path: str | Path,
    split_directory: str | Path,
    curve_path: str | Path,
    report_path: str | Path,
    *,
    steps: int = 201,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Select a global threshold by exact organizer macro F0.5 on an OOF score file."""

    if steps < 3:
        raise ValueError("threshold steps must be at least three")
    scores = Path(scores_path).resolve()
    split = Path(split_directory).resolve()
    curve_output = Path(curve_path).resolve()
    report_output = Path(report_path).resolve()
    curve_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    source1 = split / "source1.parquet"
    truth = split / "ground_truth.parquet"
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        curve = connection.execute(
            f"""
            WITH truth_pairs AS (
                SELECT source1_entity_id,
                       unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
                FROM read_parquet({_sql_path(truth)})
                WHERE coalesce(trim(matched_entity_ids), '') <> ''
            ),
            truth_counts AS (
                SELECT source1_entity_id, count(*) AS true_count
                FROM truth_pairs GROUP BY source1_entity_id
            ),
            score_bins AS (
                SELECT scores.source1_entity_id,
                       least({steps - 1}, greatest(0,
                           floor(scores.score * {steps - 1})::INTEGER)) AS bin,
                       count(*) AS predicted_delta,
                       count(truth.target_entity_id) AS tp_delta
                FROM read_parquet({_sql_path(scores)}) scores
                LEFT JOIN truth_pairs truth
                  ON scores.source1_entity_id = truth.source1_entity_id
                 AND scores.candidate_entity_id = truth.target_entity_id
                GROUP BY scores.source1_entity_id, bin
            ),
            grid AS (
                SELECT range AS threshold_index,
                       range::DOUBLE / {steps - 1} AS threshold
                FROM range({steps})
            ),
            entity_grid AS (
                SELECT source1.entity_id AS source1_entity_id,
                       grid.threshold_index, grid.threshold,
                       coalesce(truth_counts.true_count, 0) AS true_count,
                       coalesce(score_bins.predicted_delta, 0) AS predicted_delta,
                       coalesce(score_bins.tp_delta, 0) AS tp_delta
                FROM read_parquet({_sql_path(source1)}) source1
                CROSS JOIN grid
                LEFT JOIN truth_counts
                  ON source1.entity_id = truth_counts.source1_entity_id
                LEFT JOIN score_bins
                  ON source1.entity_id = score_bins.source1_entity_id
                 AND grid.threshold_index = score_bins.bin
            ),
            cumulative AS (
                SELECT *,
                    sum(predicted_delta) OVER (
                        PARTITION BY source1_entity_id ORDER BY threshold_index DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS predicted_count,
                    sum(tp_delta) OVER (
                        PARTITION BY source1_entity_id ORDER BY threshold_index DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS true_positive
                FROM entity_grid
            ),
            scored AS (
                SELECT *,
                    CASE
                        WHEN true_count = 0 AND predicted_count = 0 THEN 1.0
                        WHEN true_positive = 0 THEN 0.0
                        ELSE 1.25 * true_positive
                             / (1.25 * true_positive
                                + 0.25 * (true_count - true_positive)
                                + (predicted_count - true_positive))
                    END AS entity_f0_5
                FROM cumulative
            )
            SELECT threshold_index, threshold, avg(entity_f0_5) AS macro_f0_5,
                   sum(true_positive) AS true_positive,
                   sum(predicted_count - true_positive) AS false_positive,
                   sum(true_count - true_positive) AS false_negative,
                   count_if(true_count = 0 AND predicted_count = 0)::DOUBLE
                     / count_if(true_count = 0) AS singleton_accuracy
            FROM scored
            GROUP BY threshold_index, threshold
            ORDER BY threshold_index
            """
        ).fetchdf()
    finally:
        connection.close()
    curve.to_csv(curve_output, index=False)
    best = curve.sort_values(
        ["macro_f0_5", "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    report = {
        "threshold": float(best["threshold"]),
        "macro_f0_5": float(best["macro_f0_5"]),
        "true_positive": int(best["true_positive"]),
        "false_positive": int(best["false_positive"]),
        "false_negative": int(best["false_negative"]),
        "singleton_accuracy": float(best["singleton_accuracy"]),
        "steps": steps,
        "curve_path": str(curve_output),
        "scores_sha256": sha256_file(scores),
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_at_threshold(
    scores_path: str | Path,
    split_directory: str | Path,
    report_path: str | Path,
    *,
    threshold: float,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Evaluate one frozen threshold using the exact macro entity-level F0.5 definition."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    scores = Path(scores_path).resolve()
    split = Path(split_directory).resolve()
    source1 = split / "source1.parquet"
    truth = split / "ground_truth.parquet"
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        row = connection.execute(
            f"""
            WITH truth_pairs AS (
                SELECT source1_entity_id,
                       unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
                FROM read_parquet({_sql_path(truth)})
                WHERE coalesce(trim(matched_entity_ids), '') <> ''
            ),
            truth_counts AS (
                SELECT source1_entity_id, count(*) AS true_count
                FROM truth_pairs GROUP BY source1_entity_id
            ),
            selected AS (
                SELECT scores.source1_entity_id,
                       count(*) AS predicted_count,
                       count(truth.target_entity_id) AS true_positive
                FROM read_parquet({_sql_path(scores)}) scores
                LEFT JOIN truth_pairs truth
                  ON scores.source1_entity_id = truth.source1_entity_id
                 AND scores.candidate_entity_id = truth.target_entity_id
                WHERE scores.score >= ?
                GROUP BY scores.source1_entity_id
            ),
            per_entity AS (
                SELECT source1.entity_id,
                       coalesce(truth_counts.true_count, 0) AS true_count,
                       coalesce(selected.predicted_count, 0) AS predicted_count,
                       coalesce(selected.true_positive, 0) AS true_positive
                FROM read_parquet({_sql_path(source1)}) source1
                LEFT JOIN truth_counts ON source1.entity_id = truth_counts.source1_entity_id
                LEFT JOIN selected ON source1.entity_id = selected.source1_entity_id
            ),
            scored AS (
                SELECT *, CASE
                    WHEN true_count = 0 AND predicted_count = 0 THEN 1.0
                    WHEN true_positive = 0 THEN 0.0
                    ELSE 1.25 * true_positive
                         / (1.25 * true_positive
                            + 0.25 * (true_count - true_positive)
                            + (predicted_count - true_positive))
                END AS entity_f0_5
                FROM per_entity
            )
            SELECT count(*) AS entities, avg(entity_f0_5) AS macro_f0_5,
                   sum(true_positive) AS true_positive,
                   sum(predicted_count - true_positive) AS false_positive,
                   sum(true_count - true_positive) AS false_negative,
                   count_if(true_count = 0 AND predicted_count = 0)::DOUBLE
                       / count_if(true_count = 0) AS singleton_accuracy,
                   count_if(predicted_count = 0) AS empty_predictions
            FROM scored
            """,
            [threshold],
        ).fetchone()
    finally:
        connection.close()
    tp, fp, fn = int(row[2]), int(row[3]), int(row[4])
    report = {
        "threshold": threshold,
        "entities": int(row[0]),
        "macro_f0_5": float(row[1]),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "micro_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "micro_recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "singleton_accuracy": float(row[5]),
        "empty_predictions": int(row[6]),
        "scores_sha256": sha256_file(scores),
    }
    output = Path(report_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_submission_out_of_core(
    scores_path: str | Path,
    candidates_path: str | Path,
    test_split_directory: str | Path,
    output_directory: str | Path,
    report_path: str | Path,
    *,
    threshold: float | None,
    thresholds_by_source: dict[str, float] | None = None,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Write both required TSVs and verify coverage, IDs, uniqueness, and subset rules."""

    if (threshold is None) == (thresholds_by_source is None):
        raise ValueError("provide exactly one global threshold or source-threshold mapping")
    if thresholds_by_source is not None:
        if set(thresholds_by_source) != {"source2", "source3"}:
            raise ValueError("source thresholds must contain exactly source2 and source3")
        score_predicate = (
            "((scores.candidate_source = 'source2' AND scores.score >= "
            f"{float(thresholds_by_source['source2'])}) OR "
            "(scores.candidate_source = 'source3' AND scores.score >= "
            f"{float(thresholds_by_source['source3'])}))"
        )
    else:
        score_predicate = f"scores.score >= {float(threshold)}"
    scores = Path(scores_path).resolve()
    candidates = Path(candidates_path).resolve()
    split = Path(test_split_directory).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matching_path = output / "matching_results.tsv"
    candidate_path = output / "candidate_pairs.tsv"
    source1 = split / "source1.parquet"
    source2 = split / "source2.parquet"
    source3 = split / "source3.parquet"
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        connection.execute(
            f"""
            COPY (
                SELECT source1.entity_id AS source1_entity_id,
                       coalesce(string_agg(scores.candidate_entity_id, ','
                                           ORDER BY scores.candidate_entity_id), '')
                           AS matched_entity_ids
                FROM read_parquet({_sql_path(source1)}) source1
                LEFT JOIN read_parquet({_sql_path(scores)}) scores
                  ON source1.entity_id = scores.source1_entity_id
                 AND {score_predicate}
                GROUP BY source1.entity_id
                ORDER BY source1.entity_id
            ) TO {_sql_path(matching_path)}
            (FORMAT CSV, DELIMITER '\\t', HEADER true, NULL '')
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT source1.entity_id AS source1_entity_id,
                       coalesce(string_agg(candidates.candidate_entity_id, ','
                                           ORDER BY candidates.candidate_entity_id), '')
                           AS candidate_entity_ids
                FROM read_parquet({_sql_path(source1)}) source1
                LEFT JOIN read_parquet({_sql_path(candidates)}) candidates
                  ON source1.entity_id = candidates.source1_entity_id
                GROUP BY source1.entity_id
                ORDER BY source1.entity_id
            ) TO {_sql_path(candidate_path)}
            (FORMAT CSV, DELIMITER '\\t', HEADER true, NULL '')
            """
        )
        checks = connection.execute(
            f"""
            WITH allowed_targets AS (
                SELECT entity_id FROM read_parquet({_sql_path(source2)})
                UNION ALL
                SELECT entity_id FROM read_parquet({_sql_path(source3)})
            ), invalid_candidates AS (
                SELECT count(*) AS failures
                FROM read_parquet({_sql_path(candidates)}) candidates
                ANTI JOIN allowed_targets targets
                  ON candidates.candidate_entity_id = targets.entity_id
            ), duplicate_candidates AS (
                SELECT count(*) AS failures FROM (
                    SELECT source1_entity_id, candidate_entity_id
                    FROM read_parquet({_sql_path(candidates)})
                    GROUP BY source1_entity_id, candidate_entity_id HAVING count(*) > 1
                ) duplicates
            ), outside_candidates AS (
                SELECT count(*) AS failures
                FROM read_parquet({_sql_path(scores)}) scores
                ANTI JOIN read_parquet({_sql_path(candidates)}) candidates
                  ON scores.source1_entity_id = candidates.source1_entity_id
                 AND scores.candidate_entity_id = candidates.candidate_entity_id
                WHERE {score_predicate}
            )
            SELECT
                (SELECT count(*) FROM read_parquet({_sql_path(source1)})) AS expected_rows,
                (SELECT count(*) FROM read_csv({_sql_path(matching_path)}, delim='\\t',
                                               header=true, all_varchar=true)) AS matching_rows,
                (SELECT count(*) FROM read_csv({_sql_path(candidate_path)}, delim='\\t',
                                               header=true, all_varchar=true)) AS candidate_rows,
                (SELECT failures FROM invalid_candidates),
                (SELECT failures FROM duplicate_candidates),
                (SELECT failures FROM outside_candidates)
            """
        ).fetchone()
    finally:
        connection.close()
    failures = {
        "matching_row_count_mismatch": abs(int(checks[0]) - int(checks[1])),
        "candidate_row_count_mismatch": abs(int(checks[0]) - int(checks[2])),
        "invalid_candidate_ids": int(checks[3]),
        "duplicate_candidate_pairs": int(checks[4]),
        "matches_outside_candidates": int(checks[5]),
    }
    report = {
        "status": "PASS" if not any(failures.values()) else "FAIL",
        "threshold": threshold,
        "thresholds_by_source": thresholds_by_source,
        "verification": failures,
        "matching_results": {
            "path": str(matching_path),
            "bytes": matching_path.stat().st_size,
            "sha256": sha256_file(matching_path),
        },
        "candidate_pairs": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256_file(candidate_path),
        },
    }
    report_output = Path(report_path).resolve()
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(f"submission verification failed: {failures}")
    return report
