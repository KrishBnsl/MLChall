from __future__ import annotations

import json

import pandas as pd
import pytest

from mlchallenge.cli import main
from mlchallenge.contracts import ContractError, load_test_data, load_training_data
from mlchallenge.large_data import DuckDBRuntime, audit_dataset_out_of_core
from mlchallenge.submission import write_submission_files


def test_loaders_and_audit_report(dataset_root) -> None:
    training = load_training_data(dataset_root)
    test = load_test_data(dataset_root)
    assert len(training.source1) == 4
    assert len(test.source3) == 4

    report = audit_dataset_out_of_core(
        dataset_root,
        temp_directory=dataset_root / "tmp",
        runtime=DuckDBRuntime(memory_limit="256MB", threads=1, max_temp_directory_size="1GB"),
    )
    assert report["ground_truth"]["singleton_entities"] == 1
    assert report["ground_truth"]["positive_pairs"] == 6
    assert len(report["files"]) == 7
    assert all(len(item["sha256"]) == 64 for item in report["files"].values())
    assert report["contract_status"] == "PASS"


def test_cli_audit_make_folds_and_preflight(dataset_root, tmp_path, capsys) -> None:
    audit_path = tmp_path / "reports" / "audit.json"
    assert (
        main(
            [
                "audit",
                "--data-root",
                str(dataset_root),
                "--output",
                str(audit_path),
                "--temp-directory",
                str(tmp_path / "duckdb_tmp"),
                "--memory-limit",
                "256MB",
                "--max-temp-size",
                "1GB",
                "--threads",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(audit_path.read_text())["audit_version"] == 2

    manifest_path = tmp_path / "artifacts" / "folds.tsv"
    assert (
        main(
            [
                "make-folds",
                "--data-root",
                str(dataset_root),
                "--output",
                str(manifest_path),
                "--n-splits",
                "2",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    manifest = pd.read_csv(manifest_path, sep="\t")
    assert len(manifest) == 13

    test = load_test_data(dataset_root)
    predictions = {entity_id: set() for entity_id in test.source1["entity_id"].astype(str)}
    candidates = {entity_id: set() for entity_id in test.source1["entity_id"].astype(str)}
    matching_path, candidate_path = write_submission_files(
        test, predictions, candidates, tmp_path / "output"
    )
    assert (
        main(
            [
                "preflight",
                "--data-root",
                str(dataset_root),
                "--matching",
                str(matching_path),
                "--candidates",
                str(candidate_path),
            ]
        )
        == 0
    )
    assert "PASS" in capsys.readouterr().out


def test_loader_rejects_wrong_source_prefix(dataset_root) -> None:
    path = dataset_root / "test" / "test_source2.tsv"
    frame = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
    frame.loc[0, "entity_id"] = "S1-wrong"
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ContractError, match="wrong prefix"):
        load_test_data(dataset_root)


def test_loader_rejects_non_tab_or_changed_schema(dataset_root) -> None:
    path = dataset_root / "train" / "train_source1.tsv"
    path.write_text("entity_id,business_name,business_address,country\nS1-1,A,B,US\n")
    with pytest.raises(ContractError, match="tab-separated"):
        load_training_data(dataset_root)
