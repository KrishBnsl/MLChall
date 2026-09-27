"""Memory-bounded character TF-IDF retrieval with fused sparse top-k selection."""

from __future__ import annotations

import json
import shutil
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from mlchallenge.audit import sha256_file
from mlchallenge.large_data import DuckDBRuntime, connect_out_of_core
from mlchallenge.normalization import normalize_text


@dataclass(frozen=True)
class SparseNameConfig:
    top_k_per_source: int = 100
    ngram_min: int = 2
    ngram_max: int = 5
    n_features: int = 1_048_576
    query_batch_size: int = 5_000
    minimum_similarity: float = 0.10
    multiplication_threads: int = 2
    address_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.top_k_per_source < 1:
            raise ValueError("top_k_per_source must be positive")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("invalid n-gram range")
        if self.n_features < 2:
            raise ValueError("n_features must be at least two")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        if not 0 <= self.minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be in [0, 1]")
        if self.multiplication_threads < 1:
            raise ValueError("multiplication_threads must be positive")
        if not 0 <= self.address_weight < 1:
            raise ValueError("address_weight must be in [0, 1)")


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _retrieval_name(value: object) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    decomposed = unicodedata.normalize("NFKD", normalized)
    accent_folded = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    accent_folded = normalize_text(accent_folded)
    return normalized if accent_folded == normalized else f"{normalized} {accent_folded}"


def _load_country_group(
    connection: duckdb.DuckDBPyConnection,
    split_directory: Path,
    *,
    source: str,
    country: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = connection.execute(
        """
        SELECT entity_id, business_name, business_address
        FROM read_parquet(?)
        WHERE country = ?
        ORDER BY entity_id
        """,
        [str(split_directory / "source1.parquet"), country],
    ).fetchdf()
    target = connection.execute(
        """
        SELECT entity_id, business_name, business_address
        FROM read_parquet(?)
        WHERE country = ?
        ORDER BY entity_id
        """,
        [str(split_directory / f"{source}.parquet"), country],
    ).fetchdf()
    return query, target


def _write_frame_parquet(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.register("candidate_batch", frame)
    try:
        connection.execute(
            f"""
            COPY candidate_batch TO {_sql_path(output)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.unregister("candidate_batch")


def _generate_group_chunks(
    connection: duckdb.DuckDBPyConnection,
    split_directory: Path,
    chunk_directory: Path,
    *,
    source: str,
    country: str,
    config: SparseNameConfig,
) -> list[Path]:
    query, target = _load_country_group(
        connection,
        split_directory,
        source=source,
        country=country,
    )
    if query.empty or target.empty:
        return []
    query_names = query["business_name"].map(_retrieval_name)
    target_names = target["business_name"].map(_retrieval_name)
    name_vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(config.ngram_min, config.ngram_max),
        n_features=config.n_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
        dtype=np.float32,
    )
    target_name_counts = name_vectorizer.transform(target_names)
    name_transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
    target_name_matrix = name_transformer.fit_transform(target_name_counts).astype(np.float32)
    target_matrix = target_name_matrix
    address_vectorizer = None
    address_transformer = None
    if config.address_weight > 0:
        address_vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            n_features=config.n_features,
            alternate_sign=False,
            norm=None,
            lowercase=False,
            dtype=np.float32,
        )
        target_addresses = target["business_address"].map(_retrieval_name)
        target_address_counts = address_vectorizer.transform(target_addresses)
        address_transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
        target_address_matrix = address_transformer.fit_transform(target_address_counts).astype(
            np.float32
        )
        target_matrix = normalize(
            sparse.hstack(
                [
                    target_name_matrix * np.sqrt(1.0 - config.address_weight),
                    target_address_matrix * np.sqrt(config.address_weight),
                ],
                format="csr",
            ),
            norm="l2",
            copy=False,
        )
    target_transpose = target_matrix.transpose().tocsr()
    target_ids = target["entity_id"].astype(str).to_numpy()
    query_ids = query["entity_id"].astype(str).to_numpy()
    written: list[Path] = []
    for batch_number, start in enumerate(range(0, len(query), config.query_batch_size)):
        stop = min(start + config.query_batch_size, len(query))
        query_name_counts = name_vectorizer.transform(query_names.iloc[start:stop])
        query_name_matrix = name_transformer.transform(query_name_counts).astype(np.float32)
        query_matrix = query_name_matrix
        if address_vectorizer is not None and address_transformer is not None:
            query_addresses = query["business_address"].iloc[start:stop].map(_retrieval_name)
            query_address_counts = address_vectorizer.transform(query_addresses)
            query_address_matrix = address_transformer.transform(query_address_counts).astype(
                np.float32
            )
            query_matrix = normalize(
                sparse.hstack(
                    [
                        query_name_matrix * np.sqrt(1.0 - config.address_weight),
                        query_address_matrix * np.sqrt(config.address_weight),
                    ],
                    format="csr",
                ),
                norm="l2",
                copy=False,
            )
        similarities = sp_matmul_topn(
            query_matrix,
            target_transpose,
            top_n=min(config.top_k_per_source, len(target)),
            threshold=config.minimum_similarity,
            sort=True,
            n_threads=config.multiplication_threads,
        ).tocsr()
        row_indices = np.repeat(
            np.arange(similarities.shape[0], dtype=np.int64),
            np.diff(similarities.indptr),
        )
        if not len(row_indices):
            continue
        frame = pd.DataFrame(
            {
                "source1_entity_id": query_ids[start:stop][row_indices],
                "candidate_entity_id": target_ids[similarities.indices],
                "candidate_source": source,
                "char_name_score": similarities.data.astype(np.float32),
            }
        )
        output = chunk_directory / f"{source}_{country}_{batch_number:05d}.parquet"
        _write_frame_parquet(connection, frame, output)
        written.append(output)
    return written


def _candidate_metrics(
    connection: duckdb.DuckDBPyConnection,
    candidate_path: Path,
    split_directory: Path,
) -> dict[str, Any]:
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
            max(candidate_count),
            avg(candidate_count)
        FROM per_query
        """,
        [
            str(candidate_path),
            str(split_directory / "source1.parquet"),
            str(candidate_path),
        ],
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
        "labeled_evaluation": (split_directory / "ground_truth.parquet").is_file(),
    }
    if not report["labeled_evaluation"]:
        return report

    recall = connection.execute(
        """
        WITH truth AS (
            SELECT
                source1_entity_id,
                unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
            FROM read_parquet(?)
            WHERE coalesce(trim(matched_entity_ids), '') <> ''
        ), found AS (
            SELECT
                truth.source1_entity_id,
                truth.target_entity_id,
                candidates.candidate_entity_id IS NOT NULL AS retrieved
            FROM truth
            LEFT JOIN read_parquet(?) candidates
              ON truth.source1_entity_id = candidates.source1_entity_id
             AND truth.target_entity_id = candidates.candidate_entity_id
        ), per_entity AS (
            SELECT source1_entity_id, count(*) AS positives, count_if(retrieved) AS retrieved
            FROM found
            GROUP BY source1_entity_id
        )
        SELECT
            (SELECT count(*) FROM found),
            (SELECT count_if(retrieved) FROM found),
            count(*),
            count_if(positives = retrieved)
        FROM per_entity
        """,
        [str(split_directory / "ground_truth.parquet"), str(candidate_path)],
    ).fetchone()
    report.update(
        {
            "positive_pairs": int(recall[0]),
            "positive_pairs_retrieved": int(recall[1]),
            "positive_pair_recall": float(recall[1] / recall[0]),
            "matched_entities": int(recall[2]),
            "entities_with_all_matches_retrieved": int(recall[3]),
            "complete_entity_recall": float(recall[3] / recall[2]),
        }
    )
    return report


def generate_sparse_name_candidates(
    split_directory: str | Path,
    output_parquet: str | Path,
    report_path: str | Path,
    *,
    config: SparseNameConfig,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
    union_candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate name candidates and optionally union them with an earlier blocker."""

    split_root = Path(split_directory).resolve()
    output = Path(output_parquet).resolve()
    report_output = Path(report_path).resolve()
    chunk_directory = output.parent / f".{output.stem}_chunks"
    if chunk_directory.exists():
        shutil.rmtree(chunk_directory)
    chunk_directory.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    chunks: list[Path] = []
    try:
        countries = connection.execute(
            "SELECT DISTINCT country FROM read_parquet(?) ORDER BY country",
            [str(split_root / "source1.parquet")],
        ).fetchall()
        for (country,) in countries:
            for source in ("source2", "source3"):
                chunks.extend(
                    _generate_group_chunks(
                        connection,
                        split_root,
                        chunk_directory,
                        source=source,
                        country=str(country),
                        config=config,
                    )
                )
        if not chunks:
            raise RuntimeError("sparse retrieval produced no candidate chunks")
        chunk_glob = _sql_path(chunk_directory / "*.parquet")
        if union_candidate_path is None:
            union_sql = f"""
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source,
                    0 AS exact_name,
                    0 AS exact_address,
                    0.0 AS fuzzy_score,
                    0 AS shared_blocking_keys,
                    max(char_name_score) AS char_name_score
                FROM read_parquet({chunk_glob})
                GROUP BY source1_entity_id, candidate_entity_id, candidate_source
            """
        else:
            earlier = Path(union_candidate_path).resolve()
            if not earlier.is_file():
                raise FileNotFoundError(f"union candidate artifact is missing: {earlier}")
            union_sql = f"""
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source,
                    max(exact_name) AS exact_name,
                    max(exact_address) AS exact_address,
                    max(fuzzy_score) AS fuzzy_score,
                    max(shared_blocking_keys) AS shared_blocking_keys,
                    max(char_name_score) AS char_name_score
                FROM (
                    SELECT
                        source1_entity_id, candidate_entity_id, candidate_source,
                        exact_name, exact_address, fuzzy_score, shared_blocking_keys,
                        0.0 AS char_name_score
                    FROM read_parquet({_sql_path(earlier)})
                    UNION ALL
                    SELECT
                        source1_entity_id, candidate_entity_id, candidate_source,
                        0 AS exact_name, 0 AS exact_address, 0.0 AS fuzzy_score,
                        0 AS shared_blocking_keys, char_name_score
                    FROM read_parquet({chunk_glob})
                ) candidates
                GROUP BY source1_entity_id, candidate_entity_id, candidate_source
            """
        connection.execute(
            f"""
            COPY ({union_sql}) TO {_sql_path(output)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        report = _candidate_metrics(connection, output, split_root)
    finally:
        connection.close()
        if chunk_directory.exists():
            shutil.rmtree(chunk_directory)
    report.update(
        {
            "blocking_version": 2,
            "method": "hashed_char_tfidf_name_topn",
            "config": asdict(config),
            "runtime": asdict(runtime),
            "engine": {
                "name": "sparse-dot-topn",
                "version": __import__("sparse_dot_topn").__version__,
            },
            "union_candidate_path": (
                str(Path(union_candidate_path).resolve()) if union_candidate_path else None
            ),
            "candidate_artifact": {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
        }
    )
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
