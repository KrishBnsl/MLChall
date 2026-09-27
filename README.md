# Amazon Business Entity Resolution Challenge

This repository is the reviewable working area for the Amazon ML Challenge described in
`docs/specification/amazon_ml_challenge_problem_statement.pdf`. The goal is an accurate,
reproducible entity-resolution system that maps every Source 1 business to zero or more records
from Sources 2 and 3, while respecting the competition's precision-heavy macro F0.5 metric and
fair-play rules.

## Current status

The official student resource bundle has been audited and the large-data pipeline is implemented.
The best completed entity-group OOF tuning estimate is **0.973524 macro F0.5** with LightGBM and
two rounds of hard-example mining. This is not an official test score. The official test labels are
not available, and 0.99 has not been achieved or claimed.

Implemented and verified:

- strict TSV and identifier contracts;
- deterministic text normalization with no internet lookup;
- deterministic entity-disjoint fit/tuning/holdout partitions;
- country/source/query-batch rule blocking with resumable artifacts;
- resumable hashed character TF-IDF name/address retrieval and blocker union;
- compact structured, frequency, rank, token, edit, and address-number features;
- deterministic hard-negative sampling and iterative model-hard-example mining;
- HistGradientBoosting, LightGBM, XGBoost, and CatBoost comparison support;
- global and guarded source-specific threshold evaluation;
- entity-level macro F0.5 scoring, including correct singleton behavior;
- scale-matched evaluation views that add the full target pool only as distractors;
- data fingerprinting and audit reports;
- output preflight validation;
- synthetic regression tests for correctness, leakage, resume, and diagnostic properties;
- decision, validation, experiment, and risk documentation.

Selected production design:

- top-100-per-source rule plus sparse TF-IDF candidate union;
- calibrated LightGBM with 350 estimators and 63 leaves;
- one global threshold (source-specific thresholds failed the minimum-gain guard);
- no neural network: the structured tree model is simpler, faster, and better supported by the
  validation evidence.

See `docs/MODEL_SELECTION_V3.md` for the model-family, blocking, mining, and complexity gates.

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

The lock file is generated with `uv` and includes an optional benchmark extra:

```bash
uv sync --extra dev --extra benchmark
uv run pytest
```

Without `uv`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Large-data workflow

```bash
uv run mlchallenge audit --data-root data --output reports/data_audit.json
uv run mlchallenge make-splits --data-root student_resource/dataset
uv run mlchallenge make-experiment-splits \
  --parent-split-directory artifacts/splits/train \
  --output-root artifacts/experiment_splits_v2
uv run mlchallenge make-scale-evaluation-splits \
  --experiment-split-root artifacts/experiment_splits_v2 \
  --output-root artifacts/scale_evaluation_splits_v3
```

Run `mlchallenge --help` or a subcommand's `--help` for the bounded-memory candidate, sampling,
training, scoring, threshold, error-analysis, submission, and preflight commands. Large artifacts
are intentionally ignored by Git and every report fingerprints its inputs and outputs.

## Validation principles

- Split by real-world entity groups, never by candidate-pair rows.
- Fit candidate indices, vectorizers, feature transformations, calibration, and thresholds only
  inside their permitted training partition.
- Measure blocking recall separately because the matcher cannot recover missed candidates.
- Select model structure with entity-group OOF predictions, tune the final fit-only model on the
  scale-matched tuning queries, and evaluate the frozen pipeline once on holdout.
- Preserve empty predictions. Never force one match per Source 1 entity.
- Do not use leaderboard feedback as a training label or repeatedly tune to it.
- Fingerprint raw data and version every experiment configuration.

The complete protocol is in `docs/VALIDATION_PROTOCOL.md`.

## Fair-play boundary

Only the supplied challenge data may provide business identity evidence. The pipeline must not use
business registries, geocoders, search engines, entity-resolution services, or any other external
lookup. If a pretrained model is later evaluated, its license and provenance must be documented and
must comply with the stated MIT/Apache 2.0 and parameter-count restrictions.
