"""Out-of-core data access and auditing for the full challenge bundle."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from mlchallenge.audit import sha256_file
from mlchallenge.contracts import GROUND_TRUTH_COLUMNS, RECORD_COLUMNS, ContractError

DATASET_FILES = {
    "train_source1": Path("train/train_source1.tsv"),
    "train_source2": Path("train/train_source2.tsv"),
    "train_source3": Path("train/train_source3.tsv"),
    "train_ground_truth": Path("train/train_ground_truth.tsv"),
    "test_source1": Path("test/test_source1.tsv"),
    "test_source2": Path("test/test_source2.tsv"),
    "test_source3": Path("test/test_source3.tsv"),
}

SOURCE_PREFIXES = {
    "train_source1": "S1-",
    "train_source2": "S2-",
    "train_source3": "S3-",
    "test_source1": "S1-",
    "test_source2": "S2-",
    "test_source3": "S3-",
}


@dataclass(frozen=True)
class DuckDBRuntime:
    memory_limit: str = "3GB"
    threads: int = 4
    max_temp_directory_size: str = "8GB"


def validate_dataset_headers(data_root: str | Path) -> dict[str, Path]:
    """Validate exact headers without loading any data rows."""

    root = Path(data_root).resolve()
    paths = {name: root / relative for name, relative in DATASET_FILES.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required challenge file is missing: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            header = stream.readline().rstrip("\r\n").split("\t")
        expected = GROUND_TRUTH_COLUMNS if name == "train_ground_truth" else RECORD_COLUMNS
        if tuple(header) != expected:
            raise ContractError(
                f"{path} columns are {tuple(header)}; expected exactly {expected}. "
                "Confirm that the file is tab-separated and unmodified."
            )
    return paths


def connect_out_of_core(
    *,
    database: str | Path = ":memory:",
    temp_directory: str | Path,
    runtime: DuckDBRuntime,
) -> duckdb.DuckDBPyConnection:
    temp_path = Path(temp_directory).resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit='{runtime.memory_limit}'")
    connection.execute(f"SET threads={runtime.threads}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET max_temp_directory_size='{runtime.max_temp_directory_size}'")
    escaped_temp = str(temp_path).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped_temp}'")
    return connection


def register_dataset_views(
    connection: duckdb.DuckDBPyConnection,
    data_root: str | Path,
    *,
    include_test: bool = True,
) -> dict[str, Path]:
    paths = validate_dataset_headers(data_root)
    selected = paths if include_test else {k: v for k, v in paths.items() if k.startswith("train_")}
    for name, path in selected.items():
        relation = connection.read_csv(
            str(path),
            delimiter="\t",
            header=True,
            all_varchar=True,
        )
        relation.create_view(name, replace=True)
    return selected


def _record_summary(
    connection: duckdb.DuckDBPyConnection,
    view: str,
    prefix: str,
) -> dict[str, Any]:
    row = connection.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count_if(coalesce(trim(entity_id), '') = '') AS empty_entity_id,
            count_if(NOT starts_with(entity_id, ?)) AS wrong_prefix,
            count_if(coalesce(trim(business_name), '') = '') AS empty_business_name,
            count_if(coalesce(trim(business_address), '') = '') AS empty_business_address,
            count_if(coalesce(trim(country), '') = '') AS empty_country,
            approx_quantile(length(coalesce(business_name, '')), [0.0, 0.5, 0.95, 0.99, 1.0])
                AS name_length,
            approx_quantile(length(coalesce(business_address, '')), [0.0, 0.5, 0.95, 0.99, 1.0])
                AS address_length
        FROM {view}
        """,
        [prefix],
    ).fetchone()
    duplicate_groups, duplicate_extra_rows = connection.execute(
        f"""
        SELECT count(*), coalesce(sum(record_count - 1), 0)
        FROM (
            SELECT entity_id, count(*) AS record_count
            FROM {view}
            GROUP BY entity_id
            HAVING count(*) > 1
        ) duplicates
        """
    ).fetchone()
    countries = connection.execute(
        f"""
        SELECT coalesce(country, ''), count(*)
        FROM {view}
        GROUP BY country
        ORDER BY count(*) DESC, country
        """
    ).fetchall()
    return {
        "rows": int(row[0]),
        "empty_entity_id": int(row[1]),
        "wrong_prefix": int(row[2]),
        "empty_business_name": int(row[3]),
        "empty_business_address": int(row[4]),
        "empty_country": int(row[5]),
        "name_length": {
            key: int(value)
            for key, value in zip(("min", "median", "p95", "p99", "max"), row[6], strict=True)
        },
        "address_length": {
            key: int(value)
            for key, value in zip(("min", "median", "p95", "p99", "max"), row[7], strict=True)
        },
        "duplicate_id_groups": int(duplicate_groups),
        "duplicate_extra_rows": int(duplicate_extra_rows),
        "countries": {str(country): int(count) for country, count in countries},
    }


def _ground_truth_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE positive_pairs AS
        SELECT
            source1_entity_id,
            unnest(string_split(matched_entity_ids, ',')) AS target_entity_id
        FROM train_ground_truth
        WHERE coalesce(trim(matched_entity_ids), '') <> ''
        """
    )
    cardinality = connection.execute(
        """
        WITH cardinalities AS (
            SELECT
                source1_entity_id,
                CASE
                    WHEN coalesce(trim(matched_entity_ids), '') = '' THEN 0
                    ELSE 1 + length(matched_entity_ids)
                        - length(replace(matched_entity_ids, ',', ''))
                END AS match_count
            FROM train_ground_truth
        )
        SELECT
            count(*) AS source1_entities,
            count_if(match_count = 0) AS singletons,
            sum(match_count) AS positive_pairs,
            min(match_count),
            approx_quantile(match_count, [0.5, 0.9, 0.95, 0.99]),
            max(match_count),
            avg(match_count)
        FROM cardinalities
        """
    ).fetchone()
    distribution = connection.execute(
        """
        SELECT match_count, count(*)
        FROM (
            SELECT
                CASE
                    WHEN coalesce(trim(matched_entity_ids), '') = '' THEN 0
                    ELSE 1 + length(matched_entity_ids)
                        - length(replace(matched_entity_ids, ',', ''))
                END AS match_count
            FROM train_ground_truth
        ) cardinalities
        GROUP BY match_count
        ORDER BY match_count
        """
    ).fetchall()
    pair_stats = connection.execute(
        """
        SELECT
            count(*) AS pair_count,
            count_if(starts_with(target_entity_id, 'S2-')) AS source2_pairs,
            count_if(starts_with(target_entity_id, 'S3-')) AS source3_pairs,
            count_if(NOT starts_with(target_entity_id, 'S2-')
                     AND NOT starts_with(target_entity_id, 'S3-')) AS invalid_prefix_pairs,
            count(DISTINCT target_entity_id) AS distinct_targets
        FROM positive_pairs
        """
    ).fetchone()
    duplicate_source1 = connection.execute(
        """
        SELECT count(*)
        FROM (
            SELECT source1_entity_id
            FROM train_ground_truth
            GROUP BY source1_entity_id
            HAVING count(*) > 1
        ) duplicates
        """
    ).fetchone()[0]
    missing_gt_s1 = connection.execute(
        """
        SELECT count(*)
        FROM train_source1 source1
        ANTI JOIN train_ground_truth truth
          ON source1.entity_id = truth.source1_entity_id
        """
    ).fetchone()[0]
    unknown_gt_s1 = connection.execute(
        """
        SELECT count(*)
        FROM train_ground_truth truth
        ANTI JOIN train_source1 source1
          ON source1.entity_id = truth.source1_entity_id
        """
    ).fetchone()[0]
    missing_source2_targets = connection.execute(
        """
        SELECT count(*)
        FROM positive_pairs pairs
        ANTI JOIN train_source2 source2
          ON pairs.target_entity_id = source2.entity_id
        WHERE starts_with(pairs.target_entity_id, 'S2-')
        """
    ).fetchone()[0]
    missing_source3_targets = connection.execute(
        """
        SELECT count(*)
        FROM positive_pairs pairs
        ANTI JOIN train_source3 source3
          ON pairs.target_entity_id = source3.entity_id
        WHERE starts_with(pairs.target_entity_id, 'S3-')
        """
    ).fetchone()[0]
    return {
        "source1_entities": int(cardinality[0]),
        "singleton_entities": int(cardinality[1]),
        "singleton_fraction": float(cardinality[1] / cardinality[0]),
        "positive_pairs": int(cardinality[2]),
        "match_count": {
            "min": int(cardinality[3]),
            "median": int(cardinality[4][0]),
            "p90": int(cardinality[4][1]),
            "p95": int(cardinality[4][2]),
            "p99": int(cardinality[4][3]),
            "max": int(cardinality[5]),
            "mean": float(cardinality[6]),
        },
        "match_count_distribution": {str(key): int(value) for key, value in distribution},
        "source2_positive_pairs": int(pair_stats[1]),
        "source3_positive_pairs": int(pair_stats[2]),
        "invalid_target_prefix_pairs": int(pair_stats[3]),
        "distinct_positive_targets": int(pair_stats[4]),
        "duplicate_target_ownership_rows": int(pair_stats[0] - pair_stats[4]),
        "duplicate_source1_ground_truth_rows": int(duplicate_source1),
        "source1_without_ground_truth": int(missing_gt_s1),
        "unknown_ground_truth_source1": int(unknown_gt_s1),
        "missing_source2_positive_targets": int(missing_source2_targets),
        "missing_source3_positive_targets": int(missing_source3_targets),
    }


def audit_dataset_out_of_core(
    data_root: str | Path,
    *,
    temp_directory: str | Path,
    runtime: DuckDBRuntime,
) -> dict[str, Any]:
    """Fully audit the multi-gigabyte dataset with bounded memory and disk spill."""

    connection = connect_out_of_core(temp_directory=temp_directory, runtime=runtime)
    try:
        paths = register_dataset_views(connection, data_root)
        records = {
            name: _record_summary(connection, name, SOURCE_PREFIXES[name])
            for name in SOURCE_PREFIXES
        }
        truth = _ground_truth_summary(connection)
        files = {
            str(path.relative_to(Path(data_root).resolve())): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(paths.values())
        }
        failures: list[str] = []
        for name, summary in records.items():
            if summary["empty_entity_id"]:
                failures.append(f"{name} contains empty entity IDs")
            if summary["wrong_prefix"]:
                failures.append(f"{name} contains IDs with invalid source prefixes")
            if summary["duplicate_id_groups"]:
                failures.append(f"{name} contains duplicate entity IDs")
        for key in (
            "invalid_target_prefix_pairs",
            "duplicate_target_ownership_rows",
            "duplicate_source1_ground_truth_rows",
            "source1_without_ground_truth",
            "unknown_ground_truth_source1",
            "missing_source2_positive_targets",
            "missing_source3_positive_targets",
        ):
            if truth[key]:
                failures.append(f"ground-truth invariant failed: {key}={truth[key]}")
        return {
            "audit_version": 2,
            "engine": {"name": "duckdb", "version": duckdb.__version__},
            "runtime": asdict(runtime),
            "files": files,
            "records": records,
            "ground_truth": truth,
            "contract_failures": failures,
            "contract_status": "PASS" if not failures else "FAIL",
        }
    finally:
        connection.close()


def write_json_report(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
