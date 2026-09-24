from __future__ import annotations

from mlchallenge.contracts import ChallengeTestData
from mlchallenge.submission import validate_submission_frames, write_submission_files


def test_submission_round_trip_and_subset_rule(tmp_path, synthetic_frames) -> None:
    frames = synthetic_frames
    test_data = ChallengeTestData(frames["source1"], frames["source2"], frames["source3"])
    predictions = {
        "S1-001": {"S2-101"},
        "S1-002": {"S3-202"},
        "S1-003": set(),
        "S1-004": set(),
    }
    candidates = {
        "S1-001": {"S2-101", "S3-201"},
        "S1-002": {"S2-102", "S3-202"},
        "S1-003": {"S2-104"},
        "S1-004": set(),
    }
    matching_path, candidate_path = write_submission_files(
        test_data, predictions, candidates, tmp_path
    )
    assert matching_path.is_file()
    assert candidate_path.is_file()

    matching = __import__("pandas").read_csv(
        matching_path, sep="\t", dtype="string", keep_default_na=False
    )
    candidate = __import__("pandas").read_csv(
        candidate_path, sep="\t", dtype="string", keep_default_na=False
    )
    assert validate_submission_frames(matching, candidate, test_data) == []

    matching.loc[0, "matched_entity_ids"] = "S2-103"
    issues = validate_submission_frames(matching, candidate, test_data)
    assert any("absent from candidates" in issue for issue in issues)
