# Decision record

## ADR-001: Raw data is immutable and untracked

- Status: accepted
- Decision: raw challenge files live under ignored `data/` paths and are fingerprinted before use.
- Reason: protects provenance, prevents accidental publication, and makes runs auditable.

## ADR-002: Validate by entity groups, not pair rows

- Status: accepted
- Decision: all records known to describe one business remain in one fold.
- Reason: pair-row splitting leaks business identity and inflates validation performance.

## ADR-003: Begin with a transparent CPU baseline

- Status: accepted for measurement, not selected as final
- Decision: exact character TF-IDF retrieval plus explicit string-similarity features and logistic
  regression form the first end-to-end baseline.
- Reason: it is inspectable, deterministic, license-safe, and establishes where errors arise before
  introducing approximate retrieval, boosting, or neural encoders.

## ADR-004: Do not hard-code countries

- Status: accepted
- Decision: country is retained as open text and used only through general comparison features unless
  validation supports another rule.
- Reason: France occurs only in test, and the statement explicitly defines an open label set.

## ADR-005: No forced match

- Status: accepted
- Decision: every candidate is independently thresholded and a Source 1 entity may receive an empty
  prediction.
- Reason: singletons are part of the metric and false merges are heavily penalized.

## Pending decisions requiring real data

- exact versus approximate candidate retrieval;
- candidate budget and blocking ensembles;
- validation fold count and rare-slice treatment;
- normalization ablations, including transliteration and legal-suffix handling;
- model family, calibration, and threshold policy;
- whether any compliant pretrained encoder is useful enough to justify its cost.

