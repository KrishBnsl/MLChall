"""Out-of-core multi-key candidate generation for million-row entity sources."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from mlchallenge.audit import sha256_file
from mlchallenge.large_data import DuckDBRuntime, connect_out_of_core


@dataclass(frozen=True)
class BlockingConfig:
    fuzzy_top_k_per_source: int = 50
    max_key_document_frequency: int = 200
    max_query_keys_per_source: int = 6
    minimum_token_length: int = 3
    query_batch_size: int = 25_000

    def __post_init__(self) -> None:
        if self.fuzzy_top_k_per_source < 1:
            raise ValueError("fuzzy_top_k_per_source must be positive")
        if self.max_key_document_frequency < 1:
            raise ValueError("max_key_document_frequency must be positive")
        if self.max_query_keys_per_source < 1:
            raise ValueError("max_query_keys_per_source must be positive")
        if self.minimum_token_length < 1:
            raise ValueError("minimum_token_length must be positive")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _normalization_sql(column: str) -> str:
    return (
        "trim(regexp_replace(replace(lower(coalesce("
        f"{column}, '')), '&', ' and '), "
        "'[^\\p{L}\\p{N}\\p{M}]+', ' ', 'g'))"
    )


def _register_split_views(
    connection: duckdb.DuckDBPyConnection,
    split_directory: str | Path,
) -> dict[str, Path]:
    root = Path(split_directory).resolve()
    paths = {
        "split_source1": root / "source1.parquet",
        "split_source2": root / "source2.parquet",
        "split_source3": root / "source3.parquet",
    }
    ground_truth = root / "ground_truth.parquet"
    if ground_truth.is_file():
        paths["split_ground_truth"] = ground_truth
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required split artifact is missing: {path}")
        connection.from_parquet(str(path)).create_view(name, replace=True)
    return paths


def _materialize_normalized_records(connection: duckdb.DuckDBPyConnection) -> None:
    name1 = _normalization_sql("business_name")
    address1 = _normalization_sql("business_address")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE query_records AS
        SELECT
            entity_id AS source1_entity_id,
            country,
            {name1} AS normalized_name,
            {address1} AS normalized_address
        FROM split_source1
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target_records AS
        SELECT
            entity_id AS candidate_entity_id,
            'source2' AS candidate_source,
            country,
            {name1} AS normalized_name,
            {address1} AS normalized_address
        FROM split_source2
        UNION ALL
        SELECT
            entity_id AS candidate_entity_id,
            'source3' AS candidate_source,
            country,
            {name1} AS normalized_name,
            {address1} AS normalized_address
        FROM split_source3
        """
    )


def _materialize_exact_candidates(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE exact_candidates AS
        SELECT
            source1_entity_id,
            candidate_entity_id,
            candidate_source,
            max(exact_name) AS exact_name,
            max(exact_address) AS exact_address
        FROM (
            SELECT
                query.source1_entity_id,
                target.candidate_entity_id,
                target.candidate_source,
                1 AS exact_name,
                0 AS exact_address
            FROM query_records query
            JOIN target_records target
              ON query.country = target.country
             AND query.normalized_name = target.normalized_name
            WHERE query.normalized_name <> ''
            UNION ALL
            SELECT
                query.source1_entity_id,
                target.candidate_entity_id,
                target.candidate_source,
                0 AS exact_name,
                1 AS exact_address
            FROM query_records query
            JOIN target_records target
              ON query.country = target.country
             AND query.normalized_address = target.normalized_address
            WHERE query.normalized_address <> ''
        ) exact_union
        GROUP BY source1_entity_id, candidate_entity_id, candidate_source
        """
    )


def _materialize_keys(
    connection: duckdb.DuckDBPyConnection,
    config: BlockingConfig,
) -> None:
    minimum_length = config.minimum_token_length
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target_keys AS
        WITH name_tokens AS (
            SELECT
                candidate_entity_id,
                candidate_source,
                country,
                token
            FROM target_records,
                 unnest(string_split(normalized_name, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), address_tokens AS (
            SELECT
                candidate_entity_id,
                candidate_source,
                country,
                token
            FROM target_records,
                 unnest(string_split(normalized_address, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), all_keys AS (
            SELECT candidate_entity_id, candidate_source, country,
                   'name_exact' AS key_type, token AS key, 1.0 AS key_weight
            FROM name_tokens
            UNION ALL
            SELECT candidate_entity_id, candidate_source, country,
                   'name_prefix' AS key_type, left(token, 4) AS key, 0.65 AS key_weight
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT candidate_entity_id, candidate_source, country,
                   'name_suffix' AS key_type, right(token, 4) AS key, 0.65 AS key_weight
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT candidate_entity_id, candidate_source, country,
                   CASE WHEN regexp_matches(token, '^\\d+$')
                        THEN 'address_number' ELSE 'address_exact' END AS key_type,
                   token AS key,
                   CASE WHEN regexp_matches(token, '^\\d+$') THEN 0.90 ELSE 0.45 END
                       AS key_weight
            FROM address_tokens
        )
        SELECT
            candidate_entity_id,
            candidate_source,
            country,
            key_type,
            key,
            max(key_weight) AS key_weight
        FROM all_keys
        GROUP BY candidate_entity_id, candidate_source, country, key_type, key
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE target_key_stats AS
        SELECT
            candidate_source,
            country,
            key_type,
            key,
            count(*) AS document_frequency
        FROM target_keys
        GROUP BY candidate_source, country, key_type, key
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE target_pool_sizes AS
        SELECT candidate_source, country, count(*) AS target_count
        FROM target_records
        GROUP BY candidate_source, country
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE query_keys AS
        WITH name_tokens AS (
            SELECT source1_entity_id, country, token
            FROM query_records,
                 unnest(string_split(normalized_name, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), address_tokens AS (
            SELECT source1_entity_id, country, token
            FROM query_records,
                 unnest(string_split(normalized_address, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), all_keys AS (
            SELECT source1_entity_id, country,
                   'name_exact' AS key_type, token AS key, 1.0 AS key_weight
            FROM name_tokens
            UNION ALL
            SELECT source1_entity_id, country,
                   'name_prefix' AS key_type, left(token, 4) AS key, 0.65 AS key_weight
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT source1_entity_id, country,
                   'name_suffix' AS key_type, right(token, 4) AS key, 0.65 AS key_weight
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT source1_entity_id, country,
                   CASE WHEN regexp_matches(token, '^\\d+$')
                        THEN 'address_number' ELSE 'address_exact' END AS key_type,
                   token AS key,
                   CASE WHEN regexp_matches(token, '^\\d+$') THEN 0.90 ELSE 0.45 END
                       AS key_weight
            FROM address_tokens
        )
        SELECT source1_entity_id, country, key_type, key, max(key_weight) AS key_weight
        FROM all_keys
        GROUP BY source1_entity_id, country, key_type, key
        """
    )


def _materialize_fuzzy_candidates(
    connection: duckdb.DuckDBPyConnection,
    config: BlockingConfig,
) -> None:
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE selected_query_keys AS
        SELECT
            query.source1_entity_id,
            stats.candidate_source,
            query.country,
            query.key_type,
            query.key,
            query.key_weight,
            stats.document_frequency,
            ln((pools.target_count + 1.0) / (stats.document_frequency + 1.0)) AS idf
        FROM query_keys query
        JOIN target_key_stats stats
          ON query.country = stats.country
         AND query.key_type = stats.key_type
         AND query.key = stats.key
        JOIN target_pool_sizes pools
          ON stats.candidate_source = pools.candidate_source
         AND stats.country = pools.country
        WHERE stats.document_frequency <= {config.max_key_document_frequency}
        QUALIFY row_number() OVER (
            PARTITION BY query.source1_entity_id, stats.candidate_source
            ORDER BY stats.document_frequency, query.key_weight DESC,
                     query.key_type, query.key
        ) <= {config.max_query_keys_per_source}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE fuzzy_candidates AS
        WITH scored AS (
            SELECT
                query.source1_entity_id,
                target.candidate_entity_id,
                target.candidate_source,
                sum(query.key_weight * query.idf) AS fuzzy_score,
                count(*) AS shared_blocking_keys
            FROM selected_query_keys query
            JOIN target_keys target
              ON query.candidate_source = target.candidate_source
             AND query.country = target.country
             AND query.key_type = target.key_type
             AND query.key = target.key
            GROUP BY query.source1_entity_id,
                     target.candidate_entity_id,
                     target.candidate_source
        )
        SELECT *
        FROM scored
        QUALIFY row_number() OVER (
            PARTITION BY source1_entity_id, candidate_source
            ORDER BY fuzzy_score DESC, shared_blocking_keys DESC, candidate_entity_id
        ) <= {config.fuzzy_top_k_per_source}
        """
    )


def _materialize_final_candidates(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE final_candidates AS
        SELECT
            source1_entity_id,
            candidate_entity_id,
            candidate_source,
            max(exact_name) AS exact_name,
            max(exact_address) AS exact_address,
            max(fuzzy_score) AS fuzzy_score,
            max(shared_blocking_keys) AS shared_blocking_keys
        FROM (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                exact_name,
                exact_address,
                0.0 AS fuzzy_score,
                0 AS shared_blocking_keys
            FROM exact_candidates
            UNION ALL
            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                0 AS exact_name,
                0 AS exact_address,
                fuzzy_score,
                shared_blocking_keys
            FROM fuzzy_candidates
        ) candidate_union
        GROUP BY source1_entity_id, candidate_entity_id, candidate_source
        """
    )


def _candidate_report(
    connection: duckdb.DuckDBPyConnection,
    *,
    has_ground_truth: bool = True,
) -> dict[str, Any]:
    counts = connection.execute(
        """
        WITH per_query AS (
            SELECT source1_entity_id, count(*) AS candidate_count
            FROM final_candidates
            GROUP BY source1_entity_id
        )
        SELECT
            (SELECT count(*) FROM query_records) AS query_entities,
            (SELECT count(*) FROM final_candidates) AS candidate_pairs,
            count(*) AS entities_with_candidates,
            count_if(candidate_count = 0) AS entities_with_zero_candidates,
            min(candidate_count),
            approx_quantile(candidate_count, [0.5, 0.9, 0.95, 0.99]),
            max(candidate_count),
            avg(candidate_count)
        FROM per_query
        """
    ).fetchone()
    # per_query omits zero-candidate entities, so compute that count from total queries.
    zero_candidates = int(counts[0] - counts[2])
    report: dict[str, Any] = {
        "query_entities": int(counts[0]),
        "candidate_pairs": int(counts[1]),
        "entities_with_candidates": int(counts[2]),
        "entities_with_zero_candidates": zero_candidates,
        "candidate_count_nonzero": {
            "min": int(counts[4]),
            "median": int(counts[5][0]),
            "p90": int(counts[5][1]),
            "p95": int(counts[5][2]),
            "p99": int(counts[5][3]),
            "max": int(counts[6]),
            "mean": float(counts[7]),
        },
        "labeled_evaluation": has_ground_truth,
    }
    if not has_ground_truth:
        return report

    recall = connection.execute(
        """
        WITH truth AS (
            SELECT
                source1_entity_id,
                unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM split_ground_truth
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        ), found AS (
            SELECT
                truth.source1_entity_id,
                truth.target_entity_id,
                candidates.candidate_source,
                candidates.candidate_entity_id IS NOT NULL AS retrieved
            FROM truth
            LEFT JOIN final_candidates candidates
              ON truth.source1_entity_id = candidates.source1_entity_id
             AND truth.target_entity_id = candidates.candidate_entity_id
        ), entity_recall AS (
            SELECT
                source1_entity_id,
                count(*) AS true_matches,
                count_if(retrieved) AS retrieved_matches
            FROM found
            GROUP BY source1_entity_id
        )
        SELECT
            (SELECT count(*) FROM truth) AS positive_pairs,
            (SELECT count_if(retrieved) FROM found) AS positive_pairs_retrieved,
            (SELECT count(*) FROM entity_recall) AS matched_entities,
            (SELECT count_if(true_matches = retrieved_matches) FROM entity_recall)
                AS entities_with_all_matches_retrieved
        """
    ).fetchone()
    source_recall = connection.execute(
        """
        WITH truth AS (
            SELECT
                source1_entity_id,
                unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM split_ground_truth
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        )
        SELECT
            CASE WHEN starts_with(truth.target_entity_id, 'S2-') THEN 'source2' ELSE 'source3' END,
            count(*) AS positives,
            count_if(candidates.candidate_entity_id IS NOT NULL) AS retrieved
        FROM truth
        LEFT JOIN final_candidates candidates
          ON truth.source1_entity_id = candidates.source1_entity_id
         AND truth.target_entity_id = candidates.candidate_entity_id
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    report.update(
        {
            "positive_pairs": int(recall[0]),
            "positive_pairs_retrieved": int(recall[1]),
            "positive_pair_recall": float(recall[1] / recall[0]),
            "matched_entities": int(recall[2]),
            "entities_with_all_matches_retrieved": int(recall[3]),
            "complete_entity_recall": float(recall[3] / recall[2]),
            "source_recall": {
                source: {
                    "positive_pairs": int(positives),
                    "retrieved": int(retrieved),
                    "recall": float(retrieved / positives),
                }
                for source, positives, retrieved in source_recall
            },
        }
    )
    return report


def generate_candidates_out_of_core(
    split_directory: str | Path,
    output_parquet: str | Path,
    report_path: str | Path,
    *,
    config: BlockingConfig,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
) -> dict[str, Any]:
    """Generate candidates without labels, then evaluate blocking against held-out truth."""

    output = Path(output_parquet).resolve()
    report_output = Path(report_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        inputs = _register_split_views(connection, split_directory)
        _materialize_normalized_records(connection)
        _materialize_exact_candidates(connection)
        _materialize_keys(connection, config)
        _materialize_fuzzy_candidates(connection, config)
        _materialize_final_candidates(connection)
        report = _candidate_report(
            connection,
            has_ground_truth="split_ground_truth" in inputs,
        )
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM final_candidates
                ORDER BY source1_entity_id, candidate_source,
                         exact_name DESC, exact_address DESC,
                         fuzzy_score DESC, candidate_entity_id
            ) TO {_sql_path(output)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    finally:
        connection.close()
    report.update(
        {
            "blocking_version": 1,
            "engine": {"name": "duckdb", "version": duckdb.__version__},
            "runtime": asdict(runtime),
            "config": asdict(config),
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in inputs.items()
            },
            "candidate_artifact": {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
        }
    )
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _partitioned_candidate_report(
    connection: duckdb.DuckDBPyConnection,
    candidate_path: Path,
    split_directory: Path,
) -> dict[str, Any]:
    source1 = split_directory / "source1.parquet"
    truth = split_directory / "ground_truth.parquet"
    row = connection.execute(
        """
        WITH per_query AS (
            SELECT source1_entity_id, count(*) AS candidate_count
            FROM read_parquet(?)
            GROUP BY source1_entity_id
        )
        SELECT
            (SELECT count(*) FROM read_parquet(?)) AS query_entities,
            (SELECT count(*) FROM read_parquet(?)) AS candidate_pairs,
            count(*) AS entities_with_candidates,
            min(candidate_count),
            approx_quantile(candidate_count, [0.5, 0.9, 0.95, 0.99]),
            max(candidate_count), avg(candidate_count)
        FROM per_query
        """,
        [str(candidate_path), str(source1), str(candidate_path)],
    ).fetchone()
    report: dict[str, Any] = {
        "query_entities": int(row[0]),
        "candidate_pairs": int(row[1]),
        "entities_with_candidates": int(row[2]),
        "entities_with_zero_candidates": int(row[0] - row[2]),
        "candidate_count_nonzero": {
            "min": int(row[3]),
            "median": int(row[4][0]),
            "p90": int(row[4][1]),
            "p95": int(row[4][2]),
            "p99": int(row[4][3]),
            "max": int(row[5]),
            "mean": float(row[6]),
        },
        "labeled_evaluation": truth.is_file(),
    }
    duplicate_count = connection.execute(
        """
        SELECT count(*) FROM (
            SELECT source1_entity_id, candidate_entity_id
            FROM read_parquet(?)
            GROUP BY source1_entity_id, candidate_entity_id
            HAVING count(*) > 1
        ) duplicates
        """,
        [str(candidate_path)],
    ).fetchone()[0]
    report["duplicate_pairs"] = int(duplicate_count)
    if duplicate_count:
        raise RuntimeError(f"partitioned blocker wrote {duplicate_count} duplicate candidate pairs")
    if not truth.is_file():
        return report

    recall = connection.execute(
        """
        WITH truth AS (
            SELECT source1_entity_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM read_parquet(?)
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        ), found AS (
            SELECT truth.source1_entity_id, truth.target_entity_id,
                   candidates.candidate_entity_id IS NOT NULL AS retrieved
            FROM truth
            LEFT JOIN read_parquet(?) candidates
              ON truth.source1_entity_id = candidates.source1_entity_id
             AND truth.target_entity_id = candidates.candidate_entity_id
        ), per_entity AS (
            SELECT source1_entity_id, count(*) AS positives, count_if(retrieved) AS retrieved
            FROM found GROUP BY source1_entity_id
        )
        SELECT
            (SELECT count(*) FROM found),
            (SELECT count_if(retrieved) FROM found),
            count(*), count_if(positives = retrieved)
        FROM per_entity
        """,
        [str(truth), str(candidate_path)],
    ).fetchone()
    source_rows = connection.execute(
        """
        WITH truth AS (
            SELECT source1_entity_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM read_parquet(?)
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        )
        SELECT
            CASE WHEN starts_with(truth.target_entity_id, 'S2-')
                 THEN 'source2' ELSE 'source3' END AS candidate_source,
            count(*) AS positives,
            count_if(candidates.candidate_entity_id IS NOT NULL) AS retrieved
        FROM truth
        LEFT JOIN read_parquet(?) candidates
          ON truth.source1_entity_id = candidates.source1_entity_id
         AND truth.target_entity_id = candidates.candidate_entity_id
        GROUP BY 1 ORDER BY 1
        """,
        [str(truth), str(candidate_path)],
    ).fetchall()
    report.update(
        {
            "positive_pairs": int(recall[0]),
            "positive_pairs_retrieved": int(recall[1]),
            "positive_pair_recall": float(recall[1] / recall[0]),
            "matched_entities": int(recall[2]),
            "entities_with_all_matches_retrieved": int(recall[3]),
            "complete_entity_recall": float(recall[3] / recall[2]),
            "source_recall": {
                source: {
                    "positive_pairs": int(positives),
                    "retrieved": int(retrieved),
                    "recall": float(retrieved / positives),
                }
                for source, positives, retrieved in source_rows
            },
        }
    )
    return report


def _materialize_partition_target(
    connection: duckdb.DuckDBPyConnection,
    target_path: Path,
    country: str,
    config: BlockingConfig,
) -> int:
    name = _normalization_sql("business_name")
    address = _normalization_sql("business_address")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE partition_target_records AS
        SELECT entity_id AS candidate_entity_id,
               {name} AS normalized_name,
               {address} AS normalized_address
        FROM read_parquet({_sql_path(target_path)})
        WHERE country = ?
        """,
        [country],
    )
    target_count = int(
        connection.execute("SELECT count(*) FROM partition_target_records").fetchone()[0]
    )
    minimum_length = config.minimum_token_length
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE partition_target_keys AS
        WITH name_tokens AS (
            SELECT candidate_entity_id, token
            FROM partition_target_records,
                 unnest(string_split(normalized_name, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), address_tokens AS (
            SELECT candidate_entity_id, token
            FROM partition_target_records,
                 unnest(string_split(normalized_address, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), all_keys AS (
            SELECT candidate_entity_id, 'name_exact' AS key_type,
                   token AS key, 1.0 AS key_weight FROM name_tokens
            UNION ALL
            SELECT candidate_entity_id, 'name_prefix', left(token, 4), 0.65
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT candidate_entity_id, 'name_suffix', right(token, 4), 0.65
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT candidate_entity_id,
                   CASE WHEN regexp_matches(token, '^\\d+$')
                        THEN 'address_number' ELSE 'address_exact' END,
                   token,
                   CASE WHEN regexp_matches(token, '^\\d+$') THEN 0.90 ELSE 0.45 END
            FROM address_tokens
        )
        SELECT candidate_entity_id, key_type, key, max(key_weight) AS key_weight
        FROM all_keys
        GROUP BY candidate_entity_id, key_type, key
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE partition_target_key_stats AS
        SELECT key_type, key, count(*) AS document_frequency
        FROM partition_target_keys
        GROUP BY key_type, key
        """
    )
    return target_count


def _partition_candidate_sql(
    *,
    source: str,
    start_row: int,
    stop_row: int,
    target_count: int,
    config: BlockingConfig,
) -> str:
    minimum_length = config.minimum_token_length
    return f"""
        WITH batch_queries AS (
            SELECT * FROM partition_queries
            WHERE batch_row > {start_row} AND batch_row <= {stop_row}
        ), exact_candidates AS (
            SELECT source1_entity_id, candidate_entity_id,
                   max(exact_name) AS exact_name,
                   max(exact_address) AS exact_address
            FROM (
                SELECT query.source1_entity_id, target.candidate_entity_id,
                       1 AS exact_name, 0 AS exact_address
                FROM batch_queries query
                JOIN partition_target_records target
                  ON query.normalized_name = target.normalized_name
                WHERE query.normalized_name <> ''
                UNION ALL
                SELECT query.source1_entity_id, target.candidate_entity_id,
                       0 AS exact_name, 1 AS exact_address
                FROM batch_queries query
                JOIN partition_target_records target
                  ON query.normalized_address = target.normalized_address
                WHERE query.normalized_address <> ''
            ) exact_union
            GROUP BY source1_entity_id, candidate_entity_id
        ), name_tokens AS (
            SELECT source1_entity_id, token
            FROM batch_queries,
                 unnest(string_split(normalized_name, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), address_tokens AS (
            SELECT source1_entity_id, token
            FROM batch_queries,
                 unnest(string_split(normalized_address, ' ')) AS tokens(token)
            WHERE length(token) >= {minimum_length}
        ), all_query_keys AS (
            SELECT source1_entity_id, 'name_exact' AS key_type,
                   token AS key, 1.0 AS key_weight FROM name_tokens
            UNION ALL
            SELECT source1_entity_id, 'name_prefix', left(token, 4), 0.65
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT source1_entity_id, 'name_suffix', right(token, 4), 0.65
            FROM name_tokens WHERE length(token) >= 5
            UNION ALL
            SELECT source1_entity_id,
                   CASE WHEN regexp_matches(token, '^\\d+$')
                        THEN 'address_number' ELSE 'address_exact' END,
                   token,
                   CASE WHEN regexp_matches(token, '^\\d+$') THEN 0.90 ELSE 0.45 END
            FROM address_tokens
        ), query_keys AS (
            SELECT source1_entity_id, key_type, key, max(key_weight) AS key_weight
            FROM all_query_keys
            GROUP BY source1_entity_id, key_type, key
        ), selected_query_keys AS (
            SELECT query.source1_entity_id, query.key_type, query.key, query.key_weight,
                   stats.document_frequency,
                   ln(({target_count} + 1.0) / (stats.document_frequency + 1.0)) AS idf
            FROM query_keys query
            JOIN partition_target_key_stats stats
              ON query.key_type = stats.key_type AND query.key = stats.key
            WHERE stats.document_frequency <= {config.max_key_document_frequency}
            QUALIFY row_number() OVER (
                PARTITION BY query.source1_entity_id
                ORDER BY stats.document_frequency, query.key_weight DESC,
                         query.key_type, query.key
            ) <= {config.max_query_keys_per_source}
        ), fuzzy_candidates AS (
            SELECT query.source1_entity_id, target.candidate_entity_id,
                   sum(query.key_weight * query.idf) AS fuzzy_score,
                   count(*) AS shared_blocking_keys
            FROM selected_query_keys query
            JOIN partition_target_keys target
              ON query.key_type = target.key_type AND query.key = target.key
            GROUP BY query.source1_entity_id, target.candidate_entity_id
            QUALIFY row_number() OVER (
                PARTITION BY query.source1_entity_id
                ORDER BY fuzzy_score DESC, shared_blocking_keys DESC,
                         target.candidate_entity_id
            ) <= {config.fuzzy_top_k_per_source}
        )
        SELECT source1_entity_id, candidate_entity_id,
               '{source}' AS candidate_source,
               max(exact_name)::UTINYINT AS exact_name,
               max(exact_address)::UTINYINT AS exact_address,
               max(fuzzy_score)::FLOAT AS fuzzy_score,
               max(shared_blocking_keys)::USMALLINT AS shared_blocking_keys
        FROM (
            SELECT source1_entity_id, candidate_entity_id,
                   exact_name, exact_address, 0.0 AS fuzzy_score,
                   0 AS shared_blocking_keys
            FROM exact_candidates
            UNION ALL
            SELECT source1_entity_id, candidate_entity_id,
                   0, 0, fuzzy_score, shared_blocking_keys
            FROM fuzzy_candidates
        ) candidate_union
        GROUP BY source1_entity_id, candidate_entity_id
    """


def generate_partitioned_candidates(
    split_directory: str | Path,
    output_parquet: str | Path,
    report_path: str | Path,
    *,
    config: BlockingConfig,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Generate rule candidates by country, source, and bounded query batches.

    Only one country/source target index and one query batch are joined at a time. Completed
    chunks are resumable, which keeps the eventual full test run within bounded memory and disk.
    """

    split = Path(split_directory).resolve()
    output = Path(output_parquet).resolve()
    report_output = Path(report_path).resolve()
    source_paths = {name: split / f"{name}.parquet" for name in ("source1", "source2", "source3")}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required split artifact is missing: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    chunk_directory = output.parent / f".{output.stem}_partition_chunks"
    manifest_path = chunk_directory / "manifest.json"
    signature_payload = {
        "config": asdict(config),
        "inputs": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if chunk_directory.exists() and not resume:
        shutil.rmtree(chunk_directory)
    chunk_directory.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(
                f"existing partition chunks have a different configuration: {chunk_directory}"
            )
    else:
        manifest_path.write_text(
            json.dumps({"signature": signature, **signature_payload}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    chunks: list[Path] = []
    partition_rows: list[dict[str, Any]] = []
    try:
        countries = connection.execute(
            "SELECT DISTINCT country FROM read_parquet(?) ORDER BY country",
            [str(source_paths["source1"])],
        ).fetchall()
        name = _normalization_sql("business_name")
        address = _normalization_sql("business_address")
        for country_index, (country_value,) in enumerate(countries):
            country = str(country_value)
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE partition_queries AS
                SELECT row_number() OVER (ORDER BY entity_id) AS batch_row,
                       entity_id AS source1_entity_id,
                       {name} AS normalized_name,
                       {address} AS normalized_address
                FROM read_parquet({_sql_path(source_paths['source1'])})
                WHERE country = ?
                """,
                [country],
            )
            query_count = int(
                connection.execute("SELECT count(*) FROM partition_queries").fetchone()[0]
            )
            for source in ("source2", "source3"):
                target_count = _materialize_partition_target(
                    connection, source_paths[source], country, config
                )
                written_rows = 0
                for batch_number, start in enumerate(
                    range(0, query_count, config.query_batch_size)
                ):
                    stop = min(start + config.query_batch_size, query_count)
                    chunk = chunk_directory / (
                        f"country_{country_index:03d}_{source}_{batch_number:05d}.parquet"
                    )
                    if not chunk.is_file():
                        query = _partition_candidate_sql(
                            source=source,
                            start_row=start,
                            stop_row=stop,
                            target_count=target_count,
                            config=config,
                        )
                        connection.execute(
                            f"COPY ({query}) TO {_sql_path(chunk)} "
                            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
                        )
                    chunk_rows = int(
                        connection.execute(
                            "SELECT count(*) FROM read_parquet(?)", [str(chunk)]
                        ).fetchone()[0]
                    )
                    written_rows += chunk_rows
                    chunks.append(chunk)
                partition_rows.append(
                    {
                        "country": country,
                        "source": source,
                        "query_rows": query_count,
                        "target_rows": target_count,
                        "candidate_rows": written_rows,
                    }
                )
        if not chunks:
            raise RuntimeError("partitioned blocking produced no candidate chunks")
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet({_sql_path(chunk_directory / '*.parquet')})
            ) TO {_sql_path(output)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        report = _partitioned_candidate_report(connection, output, split)
    finally:
        connection.close()

    report.update(
        {
            "blocking_version": 3,
            "method": "country_source_query_batch_rule_blocking",
            "engine": {"name": "duckdb", "version": duckdb.__version__},
            "runtime": asdict(runtime),
            "config": asdict(config),
            "resume_signature": signature,
            "partitions": partition_rows,
            "candidate_artifact": {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
        }
    )
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(chunk_directory)
    return report
