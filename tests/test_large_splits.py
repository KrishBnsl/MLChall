from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import mlchallenge.sparse_name_candidates as sparse_module
from mlchallenge.large_data import DuckDBRuntime
from mlchallenge.scalable_candidates import (
    BlockingConfig,
    generate_candidates_out_of_core,
    generate_partitioned_candidates,
)
from mlchallenge.sparse_name_candidates import (
    SparseNameConfig,
    generate_sparse_name_candidates,
    merge_candidate_artifacts,
)
from mlchallenge.splits import (
    ExperimentSplitConfig,
    SplitConfig,
    create_development_splits,
    create_experiment_splits,
    create_scale_matched_evaluation_splits,
)


def _write_split_fixture(root) -> None:
    train = root / "train"
    test = root / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    source1_rows = []
    source2_rows = []
    source3_rows = []
    truth_rows = []
    for index in range(12):
        source1_rows.append((f"S1-{index}", f"Business {index}", f"{index} Main Rd", "US"))
        source2_rows.append((f"S2-{index}", f"Business {index}", f"{index} Main Road", "US"))
        source3_rows.append((f"S3-{index}", f"Other {index}", f"{index} Other Road", "US"))
        truth_rows.append((f"S1-{index}", f"S2-{index}"))
    source2_rows.append(("S2-unmatched", "Unmatched Two", "Unknown", "US"))
    source3_rows.append(("S3-unmatched", "Unmatched Three", "Unknown", "US"))
    columns = ("entity_id", "business_name", "business_address", "country")
    source1 = pd.DataFrame(source1_rows, columns=columns)
    source2 = pd.DataFrame(source2_rows, columns=columns)
    source3 = pd.DataFrame(source3_rows, columns=columns)
    truth = pd.DataFrame(
        truth_rows,
        columns=("source1_entity_id", "matched_entity_ids"),
    )
    source1.to_csv(train / "train_source1.tsv", sep="\t", index=False)
    source2.to_csv(train / "train_source2.tsv", sep="\t", index=False)
    source3.to_csv(train / "train_source3.tsv", sep="\t", index=False)
    truth.to_csv(train / "train_ground_truth.tsv", sep="\t", index=False)
    source1.to_csv(test / "test_source1.tsv", sep="\t", index=False)
    source2.to_csv(test / "test_source2.tsv", sep="\t", index=False)
    source3.to_csv(test / "test_source3.tsv", sep="\t", index=False)


def test_out_of_core_split_creation_is_complete_and_leakage_free(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    _write_split_fixture(data_root)
    output_root = tmp_path / "splits"
    metadata = create_development_splits(
        data_root,
        output_root,
        config=SplitConfig(
            train_fraction=0.50,
            validation_fraction=0.25,
            local_test_fraction=0.25,
            seed=17,
        ),
        runtime=DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB"),
        temp_directory=tmp_path / "duckdb_tmp",
    )
    assert all(value == 0 for value in metadata["verification"].values())
    counts = metadata["summary"]["source1"]
    assert set(counts) == {"train", "validation", "local_test"}
    assert sum(item["records"] for item in counts.values()) == 12
    assert all(item["records"] > 0 for item in counts.values())
    for split in ("train", "validation", "local_test"):
        assert (output_root / split / "source1.parquet").is_file()
        assert (output_root / split / "ground_truth.parquet").is_file()

    connection = duckdb.connect()
    mismatches = connection.execute(
        """
        SELECT count(*)
        FROM read_parquet(?) source1
        JOIN read_parquet(?) target
          ON source1.entity_id = target.owner_source1_entity_id
        WHERE source1.split <> target.split
        """,
        [
            str(output_root / "manifests" / "source1_assignments.parquet"),
            str(output_root / "manifests" / "target_assignments.parquet"),
        ],
    ).fetchone()[0]
    assert mismatches == 0

    candidate_path = tmp_path / "validation_candidates.parquet"
    report_path = tmp_path / "validation_candidates.json"
    report = generate_candidates_out_of_core(
        output_root / "validation",
        candidate_path,
        report_path,
        config=BlockingConfig(
            fuzzy_top_k_per_source=10,
            max_key_document_frequency=20,
            max_query_keys_per_source=4,
            minimum_token_length=2,
        ),
        runtime=DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB"),
        temp_directory=tmp_path / "candidate_tmp",
    )
    assert report["positive_pair_recall"] == 1.0
    assert report["entities_with_zero_candidates"] == 0
    assert candidate_path.is_file()
    assert report_path.is_file()

    sparse_output = tmp_path / "validation_sparse_union.parquet"
    sparse_report_path = tmp_path / "validation_sparse_union.json"
    sparse_report = generate_sparse_name_candidates(
        output_root / "validation",
        sparse_output,
        sparse_report_path,
        config=SparseNameConfig(
            top_k_per_source=5,
            ngram_min=2,
            ngram_max=4,
            n_features=4096,
            query_batch_size=2,
            minimum_similarity=0.0,
            multiplication_threads=1,
        ),
        runtime=DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB"),
        temp_directory=tmp_path / "sparse_tmp",
        union_candidate_path=candidate_path,
    )
    assert sparse_report["positive_pair_recall"] == 1.0
    assert sparse_output.is_file()
    assert sparse_report_path.is_file()

    merged_output = tmp_path / "validation_merged.parquet"
    merged_report = merge_candidate_artifacts(
        [candidate_path, sparse_output],
        merged_output,
        tmp_path / "validation_merged.json",
        split_directory=output_root / "validation",
        runtime=DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB"),
        temp_directory=tmp_path / "merge_tmp",
    )
    assert merged_report["positive_pair_recall"] == 1.0
    assert merged_report["duplicate_pairs"] == 0
    assert merged_output.is_file()


def test_fresh_nested_holdout_and_partitioned_blocker(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    _write_split_fixture(data_root)
    runtime = DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB")
    original = tmp_path / "original_splits"
    create_development_splits(
        data_root,
        original,
        config=SplitConfig(
            train_fraction=0.50,
            validation_fraction=0.25,
            local_test_fraction=0.25,
            seed=17,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "original_tmp",
    )
    nested = tmp_path / "experiment_splits"
    metadata = create_experiment_splits(
        original / "train",
        nested,
        config=ExperimentSplitConfig(
            fit_fraction=0.50,
            tuning_fraction=0.25,
            holdout_fraction=0.25,
            seed=19,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "nested_tmp",
    )
    assert set(metadata["summary"]["source1"]) == {"fit", "tuning", "holdout"}
    assert all(value == 0 for value in metadata["verification"].values())

    candidates = tmp_path / "partitioned_candidates.parquet"
    report = generate_partitioned_candidates(
        nested / "holdout",
        candidates,
        tmp_path / "partitioned_report.json",
        config=BlockingConfig(
            fuzzy_top_k_per_source=5,
            max_key_document_frequency=20,
            max_query_keys_per_source=4,
            minimum_token_length=2,
            query_batch_size=1,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "partitioned_tmp",
    )
    assert report["positive_pair_recall"] == 1.0
    assert report["duplicate_pairs"] == 0
    assert candidates.is_file()


def test_scale_matched_views_keep_queries_held_out_and_add_all_targets(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    _write_split_fixture(data_root)
    runtime = DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB")
    original = tmp_path / "original_splits"
    create_development_splits(
        data_root,
        original,
        config=SplitConfig(
            train_fraction=0.50,
            validation_fraction=0.25,
            local_test_fraction=0.25,
            seed=17,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "original_tmp",
    )
    nested = tmp_path / "experiment_splits"
    create_experiment_splits(
        original / "train",
        nested,
        config=ExperimentSplitConfig(
            fit_fraction=0.50,
            tuning_fraction=0.25,
            holdout_fraction=0.25,
            seed=19,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "nested_tmp",
    )

    scaled = tmp_path / "scale_matched"
    metadata = create_scale_matched_evaluation_splits(
        nested,
        scaled,
        runtime=runtime,
        temp_directory=tmp_path / "scaled_tmp",
    )
    assert all(value == 0 for value in metadata["verification"].values())

    connection = duckdb.connect()
    expected_source2 = sum(
        connection.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(nested / split / "source2.parquet")],
        ).fetchone()[0]
        for split in ("fit", "tuning", "holdout")
    )
    actual_source2 = connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(scaled / "holdout" / "source2.parquet")],
    ).fetchone()[0]
    assert actual_source2 == expected_source2

    expected_queries = connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(nested / "holdout" / "source1.parquet")],
    ).fetchone()[0]
    actual_queries = connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(scaled / "holdout" / "source1.parquet")],
    ).fetchone()[0]
    assert actual_queries == expected_queries
    assert metadata["summary"]["evaluation_splits"]["holdout"]["distractor_target_records"] > 0


def test_sparse_candidate_build_resumes_completed_batches(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "dataset"
    _write_split_fixture(data_root)
    runtime = DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB")
    splits = tmp_path / "splits"
    create_development_splits(
        data_root,
        splits,
        config=SplitConfig(
            train_fraction=0.50,
            validation_fraction=0.25,
            local_test_fraction=0.25,
            seed=17,
        ),
        runtime=runtime,
        temp_directory=tmp_path / "split_tmp",
    )
    output = tmp_path / "resumable_candidates.parquet"
    config = SparseNameConfig(
        top_k_per_source=3,
        ngram_min=2,
        ngram_max=3,
        n_features=1024,
        query_batch_size=1,
        minimum_similarity=0.0,
        multiplication_threads=1,
    )
    original_matmul = sparse_module.sp_matmul_topn
    calls = 0

    def fail_after_first_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(sparse_module, "sp_matmul_topn", fail_after_first_batch)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        generate_sparse_name_candidates(
            splits / "validation",
            output,
            tmp_path / "interrupted_report.json",
            config=config,
            runtime=runtime,
            temp_directory=tmp_path / "candidate_tmp",
        )
    chunk_directory = output.parent / f".{output.stem}_chunks"
    assert len(list(chunk_directory.glob("*.parquet"))) == 1

    monkeypatch.setattr(sparse_module, "sp_matmul_topn", original_matmul)
    report = generate_sparse_name_candidates(
        splits / "validation",
        output,
        tmp_path / "resumed_report.json",
        config=config,
        runtime=runtime,
        temp_directory=tmp_path / "candidate_tmp",
    )
    assert report["positive_pair_recall"] == 1.0
    assert output.is_file()
    assert not chunk_directory.exists()
