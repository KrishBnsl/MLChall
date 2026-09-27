"""Deterministic, entity-disjoint development splits for the full training data."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from mlchallenge.audit import sha256_file
from mlchallenge.large_data import (
    DuckDBRuntime,
    connect_out_of_core,
    register_dataset_views,
)

SPLIT_NAMES = ("train", "validation", "local_test")
EXPERIMENT_SPLIT_NAMES = ("fit", "tuning", "holdout")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.90
    validation_fraction: float = 0.05
    local_test_fraction: float = 0.05
    seed: int = 20260925

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.local_test_fraction)
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("all split fractions must be strictly between zero and one")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to one")


@dataclass(frozen=True)
class ExperimentSplitConfig:
    """Nested split carved only from the previously unused training partition."""

    fit_fraction: float = 8.0 / 9.0
    tuning_fraction: float = 1.0 / 18.0
    holdout_fraction: float = 1.0 / 18.0
    seed: int = 20260927

    def __post_init__(self) -> None:
        fractions = (self.fit_fraction, self.tuning_fraction, self.holdout_fraction)
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("all experiment split fractions must be strictly between zero and one")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValueError("experiment split fractions must sum to one")


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _copy_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY ({query}) TO {_sql_path(output)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def _build_assignment_tables(
    connection: duckdb.DuckDBPyConnection,
    config: SplitConfig,
) -> None:
    validation_fraction = config.validation_fraction
    local_test_fraction = config.local_test_fraction
    seed = config.seed
    connection.execute(
        """
        CREATE OR REPLACE TABLE positive_pairs AS
        SELECT
            source1_entity_id,
            unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
        FROM train_ground_truth
        WHERE coalesce(trim(matched_entity_ids), '') <> ''
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE source1_match_counts AS
        SELECT
            source1_entity_id,
            CASE
                WHEN coalesce(trim(matched_entity_ids), '') = '' THEN 0
                ELSE 1 + length(matched_entity_ids)
                    - length(replace(matched_entity_ids, ',', ''))
            END AS match_count
        FROM train_ground_truth
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE source1_assignments AS
        WITH base AS (
            SELECT
                source1.entity_id,
                source1.country,
                counts.match_count,
                count(*) OVER (
                    PARTITION BY source1.country, counts.match_count
                ) AS stratum_size,
                row_number() OVER (
                    PARTITION BY source1.country, counts.match_count
                    ORDER BY md5_number_lower('{seed}:' || source1.entity_id)
                ) AS stratum_rank
            FROM train_source1 source1
            JOIN source1_match_counts counts
              ON source1.entity_id = counts.source1_entity_id
        ), allocations AS (
            SELECT
                *,
                CASE
                    WHEN stratum_size >= 3
                    THEN greatest(1, floor(stratum_size * {validation_fraction}))
                    ELSE 0
                END AS validation_size,
                CASE
                    WHEN stratum_size >= 3
                    THEN greatest(1, floor(stratum_size * {local_test_fraction}))
                    ELSE 0
                END AS local_test_size
            FROM base
        )
        SELECT
            entity_id,
            country,
            match_count,
            CASE
                WHEN stratum_rank <= validation_size THEN 'validation'
                WHEN stratum_rank <= validation_size + local_test_size THEN 'local_test'
                ELSE 'train'
            END AS split,
            '{seed}' AS split_seed
        FROM allocations
        """
    )
    validation_threshold = round(config.validation_fraction * 10_000)
    local_test_threshold = round((config.validation_fraction + config.local_test_fraction) * 10_000)
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE target_assignments AS
        WITH targets AS (
            SELECT entity_id, 'source2' AS source, country
            FROM train_source2
            UNION ALL
            SELECT entity_id, 'source3' AS source, country
            FROM train_source3
        ), owned AS (
            SELECT
                targets.entity_id,
                targets.source,
                targets.country,
                pairs.source1_entity_id AS owner_source1_entity_id,
                assignments.split AS owner_split
            FROM targets
            LEFT JOIN positive_pairs pairs
              ON targets.entity_id = pairs.target_entity_id
            LEFT JOIN source1_assignments assignments
              ON pairs.source1_entity_id = assignments.entity_id
        )
        SELECT
            entity_id,
            source,
            country,
            owner_source1_entity_id,
            coalesce(
                owner_split,
                CASE
                    WHEN md5_number_lower('{seed}:unmatched:' || entity_id) % 10000
                         < {validation_threshold}
                    THEN 'validation'
                    WHEN md5_number_lower('{seed}:unmatched:' || entity_id) % 10000
                         < {local_test_threshold}
                    THEN 'local_test'
                    ELSE 'train'
                END
            ) AS split,
            '{seed}' AS split_seed
        FROM owned
        """
    )


def _verify_assignments(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    checks = {
        "source1_assignment_count_mismatch": connection.execute(
            """
            SELECT abs(
                (SELECT count(*) FROM train_source1)
                - (SELECT count(*) FROM source1_assignments)
            )
            """
        ).fetchone()[0],
        "target_assignment_count_mismatch": connection.execute(
            """
            SELECT abs(
                (SELECT count(*) FROM train_source2) + (SELECT count(*) FROM train_source3)
                - (SELECT count(*) FROM target_assignments)
            )
            """
        ).fetchone()[0],
        "duplicate_source1_assignments": connection.execute(
            """
            SELECT count(*)
            FROM (
                SELECT entity_id
                FROM source1_assignments
                GROUP BY entity_id
                HAVING count(*) > 1
            ) duplicates
            """
        ).fetchone()[0],
        "duplicate_target_assignments": connection.execute(
            """
            SELECT count(*)
            FROM (
                SELECT entity_id
                FROM target_assignments
                GROUP BY entity_id
                HAVING count(*) > 1
            ) duplicates
            """
        ).fetchone()[0],
        "positive_pair_split_mismatches": connection.execute(
            """
            SELECT count(*)
            FROM positive_pairs pairs
            JOIN source1_assignments source1
              ON pairs.source1_entity_id = source1.entity_id
            JOIN target_assignments target
              ON pairs.target_entity_id = target.entity_id
            WHERE source1.split <> target.split
            """
        ).fetchone()[0],
        "ground_truth_source1_without_assignment": connection.execute(
            """
            SELECT count(*)
            FROM train_ground_truth truth
            ANTI JOIN source1_assignments assignments
              ON truth.source1_entity_id = assignments.entity_id
            """
        ).fetchone()[0],
        "invalid_source1_split_values": connection.execute(
            """
            SELECT count(*)
            FROM source1_assignments
            WHERE split NOT IN ('train', 'validation', 'local_test')
            """
        ).fetchone()[0],
        "invalid_target_split_values": connection.execute(
            """
            SELECT count(*)
            FROM target_assignments
            WHERE split NOT IN ('train', 'validation', 'local_test')
            """
        ).fetchone()[0],
    }
    failures = {key: int(value) for key, value in checks.items() if value}
    if failures:
        raise RuntimeError(f"split assignment verification failed: {failures}")
    return {key: int(value) for key, value in checks.items()}


def _write_split_parquets(
    connection: duckdb.DuckDBPyConnection,
    output_root: Path,
    *,
    split_mapping: dict[str, str] | None = None,
) -> list[Path]:
    mapping = split_mapping or {name: name for name in SPLIT_NAMES}
    written: list[Path] = []
    manifest_dir = output_root / "manifests"
    source1_manifest = manifest_dir / "source1_assignments.parquet"
    target_manifest = manifest_dir / "target_assignments.parquet"
    _copy_to_parquet(
        connection,
        "SELECT * FROM source1_assignments ORDER BY entity_id",
        source1_manifest,
    )
    _copy_to_parquet(
        connection,
        "SELECT * FROM target_assignments ORDER BY source, entity_id",
        target_manifest,
    )
    written.extend((source1_manifest, target_manifest))

    for assignment_split, output_name in mapping.items():
        split_dir = output_root / output_name
        source1_path = split_dir / "source1.parquet"
        source2_path = split_dir / "source2.parquet"
        source3_path = split_dir / "source3.parquet"
        truth_path = split_dir / "ground_truth.parquet"
        _copy_to_parquet(
            connection,
            f"""
            SELECT source1.*
            FROM train_source1 source1
            JOIN source1_assignments assignments
              ON source1.entity_id = assignments.entity_id
            WHERE assignments.split = '{assignment_split}'
            """,
            source1_path,
        )
        _copy_to_parquet(
            connection,
            f"""
            SELECT source2.*
            FROM train_source2 source2
            JOIN target_assignments assignments
              ON source2.entity_id = assignments.entity_id
            WHERE assignments.split = '{assignment_split}'
            """,
            source2_path,
        )
        _copy_to_parquet(
            connection,
            f"""
            SELECT source3.*
            FROM train_source3 source3
            JOIN target_assignments assignments
              ON source3.entity_id = assignments.entity_id
            WHERE assignments.split = '{assignment_split}'
            """,
            source3_path,
        )
        _copy_to_parquet(
            connection,
            f"""
            SELECT
                truth.source1_entity_id,
                coalesce(truth.matched_entity_ids, '') AS matched_entity_ids
            FROM train_ground_truth truth
            JOIN source1_assignments assignments
              ON truth.source1_entity_id = assignments.entity_id
            WHERE assignments.split = '{assignment_split}'
            """,
            truth_path,
        )
        written.extend((source1_path, source2_path, source3_path, truth_path))
    return written


def _rename_summary_splits(
    summary: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    return {
        section: {mapping.get(name, name): values for name, values in entries.items()}
        for section, entries in summary.items()
    }


def _split_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    source1_counts = connection.execute(
        """
        SELECT split, country, match_count, count(*)
        FROM source1_assignments
        GROUP BY split, country, match_count
        ORDER BY split, country, match_count
        """
    ).fetchall()
    target_counts = connection.execute(
        """
        SELECT split, source, country,
               count(*) AS records,
               count_if(owner_source1_entity_id IS NOT NULL) AS positive_owned_records,
               count_if(owner_source1_entity_id IS NULL) AS unmatched_records
        FROM target_assignments
        GROUP BY split, source, country
        ORDER BY split, source, country
        """
    ).fetchall()
    source1: dict[str, dict[str, Any]] = {}
    for split, country, match_count, count in source1_counts:
        entry = source1.setdefault(split, {"records": 0, "countries": {}, "match_counts": {}})
        entry["records"] += int(count)
        entry["countries"][country] = entry["countries"].get(country, 0) + int(count)
        entry["match_counts"][str(match_count)] = entry["match_counts"].get(
            str(match_count), 0
        ) + int(count)
    targets: dict[str, dict[str, Any]] = {}
    for split, source, country, records, positive_owned, unmatched in target_counts:
        split_entry = targets.setdefault(split, {})
        source_entry = split_entry.setdefault(
            source,
            {"records": 0, "positive_owned_records": 0, "unmatched_records": 0, "countries": {}},
        )
        source_entry["records"] += int(records)
        source_entry["positive_owned_records"] += int(positive_owned)
        source_entry["unmatched_records"] += int(unmatched)
        source_entry["countries"][country] = int(records)
    return {"source1": source1, "targets": targets}


def create_development_splits(
    data_root: str | Path,
    output_root: str | Path,
    *,
    config: SplitConfig,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
    audit_report: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create compressed physical splits plus complete assignment manifests."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"split output already exists and is not empty: {output}; use force explicitly"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    database_path = output / "split_manifest.duckdb"
    connection = connect_out_of_core(
        database=database_path,
        temp_directory=temp_directory,
        runtime=runtime,
    )
    try:
        register_dataset_views(connection, data_root, include_test=False)
        _build_assignment_tables(connection, config)
        checks = _verify_assignments(connection)
        written = _write_split_parquets(connection, output)
        summary = _split_summary(connection)
    finally:
        connection.close()

    audit_reference = None
    if audit_report is not None:
        audit_path = Path(audit_report).resolve()
        if audit_path.is_file():
            audit_reference = {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
            }
    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    metadata = {
        "split_version": 1,
        "engine": {"name": "duckdb", "version": duckdb.__version__},
        "runtime": asdict(runtime),
        "config": asdict(config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "audit_report": audit_reference,
        "verification": checks,
        "summary": summary,
        "artifacts": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in written
        },
    }
    metadata_path = output / "split_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def create_experiment_splits(
    parent_split_directory: str | Path,
    output_root: str | Path,
    *,
    config: ExperimentSplitConfig,
    runtime: DuckDBRuntime,
    temp_directory: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Create a fresh fit/tuning/holdout split from an unused parent partition.

    This deliberately accepts Parquet split artifacts rather than the original training TSVs.
    It lets later experiments reserve a new holdout without moving any entity previously used by
    the first validation/local-test workflow into that holdout.
    """

    parent = Path(parent_split_directory).resolve()
    parent_paths = {
        "source1": parent / "source1.parquet",
        "source2": parent / "source2.parquet",
        "source3": parent / "source3.parquet",
        "ground_truth": parent / "ground_truth.parquet",
    }
    for path in parent_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required parent split artifact is missing: {path}")

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"experiment split output already exists and is not empty: {output}; "
                "use force explicitly"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    database_path = output / "experiment_split_manifest.duckdb"
    connection = connect_out_of_core(
        database=database_path,
        temp_directory=temp_directory,
        runtime=runtime,
    )
    mapping = {"train": "fit", "validation": "tuning", "local_test": "holdout"}
    try:
        connection.from_parquet(str(parent_paths["source1"])).create_view(
            "train_source1", replace=True
        )
        connection.from_parquet(str(parent_paths["source2"])).create_view(
            "train_source2", replace=True
        )
        connection.from_parquet(str(parent_paths["source3"])).create_view(
            "train_source3", replace=True
        )
        connection.from_parquet(str(parent_paths["ground_truth"])).create_view(
            "train_ground_truth", replace=True
        )
        _build_assignment_tables(
            connection,
            SplitConfig(
                train_fraction=config.fit_fraction,
                validation_fraction=config.tuning_fraction,
                local_test_fraction=config.holdout_fraction,
                seed=config.seed,
            ),
        )
        checks = _verify_assignments(connection)
        written = _write_split_parquets(connection, output, split_mapping=mapping)
        summary = _rename_summary_splits(_split_summary(connection), mapping)
    finally:
        connection.close()

    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    metadata = {
        "split_version": 2,
        "purpose": "fresh_model_selection_holdout",
        "engine": {"name": "duckdb", "version": duckdb.__version__},
        "runtime": asdict(runtime),
        "config": asdict(config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "parent_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in parent_paths.items()
        },
        "verification": checks,
        "summary": summary,
        "artifacts": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in written
        },
    }
    metadata_path = output / "experiment_split_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
