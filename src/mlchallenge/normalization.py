"""Deterministic, conservative text views used by retrieval and features."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^\w]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def normalize_text(value: object) -> str:
    """Return a Unicode-preserving canonical text form.

    NFKC and case folding remove representational differences without transliterating scripts,
    deleting legal suffixes, consulting an external service, or assuming a particular country.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def compact_text(value: object) -> str:
    """Return a whitespace-free normalized view for exact-ish comparisons."""

    return normalize_text(value).replace(" ", "")


def token_set(value: object) -> frozenset[str]:
    """Return unique normalized tokens."""

    normalized = normalize_text(value)
    return frozenset(normalized.split()) if normalized else frozenset()


def digit_set(value: object) -> frozenset[str]:
    """Return digit runs, retaining useful address-number evidence."""

    return frozenset(_DIGITS.findall(unicodedata.normalize("NFKC", str(value or ""))))


def candidate_text(name: object, address: object) -> str:
    """Build a field-aware retrieval string without learning from labels."""

    normalized_name = normalize_text(name)
    normalized_address = normalize_text(address)
    # Sentinels preserve which field supplied each character sequence.
    return f"name {normalized_name} address {normalized_address}".strip()
