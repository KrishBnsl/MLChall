from mlchallenge.normalization import candidate_text, compact_text, digit_set, normalize_text


def test_normalization_is_unicode_preserving_and_deterministic() -> None:
    assert normalize_text("  ACME & Sons, Pvt. Ltd.  ") == "acme and sons pvt ltd"
    assert normalize_text("Cafe\u0301 de Paris") == "café de paris"
    assert compact_text("M.G. Road") == "mgroad"
    assert digit_set("12 MG Road, PIN 560001") == frozenset({"12", "560001"})


def test_candidate_text_keeps_field_boundaries() -> None:
    assert candidate_text("Acme", "12 Road") == "name acme address 12 road"
