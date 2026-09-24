# Amazon Business Entity Resolution Challenge

This repository is the reviewable working area for the Amazon ML Challenge described in
`docs/specification/amazon_ml_challenge_problem_statement.pdf`. The goal is an accurate,
reproducible entity-resolution system that maps every Source 1 business to zero or more records
from Sources 2 and 3, while respecting the competition's precision-heavy macro F0.5 metric and
fair-play rules.

## Current status

Phase 0 foundations are implemented. The challenge dataset and official student resource bundle
have not been provided yet, so no data profile, validation split, trained model, threshold, runtime
estimate, or performance claim exists. Those omissions are deliberate.

Implemented now:

- strict TSV and identifier contracts;
- deterministic text normalization with no internet lookup;
- exact character TF-IDF candidate generation suitable as a transparent baseline;
- pairwise similarity features and a conservative baseline classifier;
- entity-level macro F0.5 scoring, including correct singleton behavior;
- group-disjoint fold-manifest construction to prevent cross-source entity leakage;
- data fingerprinting and audit reports;
- output preflight validation;
- synthetic unit tests for correctness properties;
- decision, validation, experiment, and risk documentation.

Not yet claimed or selected:

- dataset-specific blocking limits or approximate-nearest-neighbor technology;
- final feature set, model family, calibration method, or threshold;
- final cross-validation design parameters;
- leaderboard score or expected final-validation score.

## Repository layout

```text
MLChallenge/
|-- configs/                  # Reviewable, versioned experiment configuration
|-- data/                     # Local challenge data; ignored by Git
|-- docs/                     # Protocols, decisions, risk register, source statement
|-- src/mlchallenge/          # Reusable pipeline code
|-- tests/                    # Synthetic correctness tests only
|-- artifacts/                # Local fitted models/manifests; ignored by Git
|-- reports/                  # Local audits/evaluations; ignored by Git
`-- output/                   # Required submission TSV files; ignored by Git
```

## Required data gate

Place the official files without renaming them:

```text
data/
|-- train/
|   |-- train_source1.tsv
|   |-- train_source2.tsv
|   |-- train_source3.tsv
|   `-- train_ground_truth.tsv
`-- test/
    |-- test_source1.tsv
    |-- test_source2.tsv
    `-- test_source3.tsv
```

Also retain the organizer-provided `utils/validate_submission.py` and
`Documentation_template.md` when the full student resource bundle is available. The official
validator remains the final authority; the repository's preflight checks are additional guards.

## Reproducible setup

The lock file is generated with `uv` and should be used when available:

```bash
uv sync --extra dev
uv run pytest
```

Without `uv`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## First commands after data arrival

```bash
uv run mlchallenge audit --data-root data --output reports/data_audit.json
uv run mlchallenge make-folds \
  --data-root data \
  --output artifacts/fold_manifest.tsv \
  --n-splits 5 \
  --seed 20260925
```

The audit must be reviewed before running experiments. In particular, candidate counts, memory
strategy, fold counts, and model search space will be chosen from measured dataset properties.

## Validation principles

- Split by real-world entity groups, never by candidate-pair rows.
- Fit candidate indices, vectorizers, feature transformations, calibration, and thresholds only
  inside their permitted training partition.
- Measure blocking recall separately because the matcher cannot recover missed candidates.
- Keep threshold and hyperparameter selection inside inner validation; report only untouched outer
  fold performance as the model-selection estimate.
- Preserve empty predictions. Never force one match per Source 1 entity.
- Do not use leaderboard feedback as a training label or repeatedly tune to it.
- Fingerprint raw data and version every experiment configuration.

The complete protocol is in `docs/VALIDATION_PROTOCOL.md`.

## Fair-play boundary

Only the supplied challenge data may provide business identity evidence. The pipeline must not use
business registries, geocoders, search engines, entity-resolution services, or any other external
lookup. If a pretrained model is later evaluated, its license and provenance must be documented and
must comply with the stated MIT/Apache 2.0 and parameter-count restrictions.

