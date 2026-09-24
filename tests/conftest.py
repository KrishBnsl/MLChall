from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_frames() -> dict[str, pd.DataFrame]:
    source1 = pd.DataFrame(
        [
            ("S1-001", "Acme Pvt. Ltd.", "12 MG Road Bengaluru 560001", "India"),
            ("S1-002", "Blue River Bakery", "4 Lake Street Austin TX", "US"),
            ("S1-003", "Maison Lumiere", "8 Rue Victor Hugo Paris", "France"),
            ("S1-004", "No Match Shop", "Unknown", "US"),
        ],
        columns=("entity_id", "business_name", "business_address", "country"),
        dtype="string",
    )
    source2 = pd.DataFrame(
        [
            ("S2-101", "ACME Private Limited", "12 M.G. Rd, Bangalore 560001", "India"),
            ("S2-102", "Blue River Bakes", "4 Lake St, Austin", "US"),
            ("S2-103", "Unrelated Store", "9 Hill Road", "US"),
            ("S2-104", "Maison Lumiere", "8 rue Victor Hugo", "France"),
            ("S2-105", "Independent Target", "No known address", "US"),
        ],
        columns=("entity_id", "business_name", "business_address", "country"),
        dtype="string",
    )
    source3 = pd.DataFrame(
        [
            ("S3-201", "Acme Pvt Ltd", "12 MG Road Bengaluru", "India"),
            ("S3-202", "Blue River Bakery", "Lake Street Austin Texas", "US"),
            ("S3-203", "Different Company", "100 Other Ave", "US"),
            ("S3-204", "Lumiere Maison", "Paris 8 Victor Hugo", "France"),
        ],
        columns=("entity_id", "business_name", "business_address", "country"),
        dtype="string",
    )
    truth = pd.DataFrame(
        [
            ("S1-001", "S2-101,S3-201"),
            ("S1-002", "S2-102,S3-202"),
            ("S1-003", "S2-104,S3-204"),
            ("S1-004", ""),
        ],
        columns=("source1_entity_id", "matched_entity_ids"),
        dtype="string",
    )
    return {"source1": source1, "source2": source2, "source3": source3, "truth": truth}


@pytest.fixture
def dataset_root(tmp_path, synthetic_frames) -> object:
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    synthetic_frames["source1"].to_csv(train / "train_source1.tsv", sep="\t", index=False)
    synthetic_frames["source2"].to_csv(train / "train_source2.tsv", sep="\t", index=False)
    synthetic_frames["source3"].to_csv(train / "train_source3.tsv", sep="\t", index=False)
    synthetic_frames["truth"].to_csv(train / "train_ground_truth.tsv", sep="\t", index=False)
    synthetic_frames["source1"].to_csv(test / "test_source1.tsv", sep="\t", index=False)
    synthetic_frames["source2"].to_csv(test / "test_source2.tsv", sep="\t", index=False)
    synthetic_frames["source3"].to_csv(test / "test_source3.tsv", sep="\t", index=False)
    return tmp_path
