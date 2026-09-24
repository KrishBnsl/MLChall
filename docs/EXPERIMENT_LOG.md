# Experiment log

Every run receives an immutable ID. Entries include the Git commit, data fingerprints, configuration,
command, environment, seed, runtime, peak memory, blocking diagnostics, validation results, output
hashes, observations, and decision. Failed experiments are retained because they prevent repeated
mistakes.

## E000 - Repository foundation

- Date: 2026-09-25
- Data: unavailable
- Purpose: establish contracts, validation policy, transparent baseline components, and tests.
- Result: no model was trained and no metric was calculated.
- Decision: wait for the official dataset and student resource bundle; audit before selecting any
  dataset-dependent parameters.

## Entry template

```text
Experiment ID:
Date/time and timezone:
Git commit:
Data audit/report hash:
Configuration and hash:
Command:
Environment/lock hash:
Seed(s):
Hardware:
Runtime/peak memory:
Blocking recall and candidate statistics:
Outer-fold macro F0.5 (mean, spread, confidence interval):
Slice results:
Output artifact hashes:
Observations:
Decision and rationale:
```

