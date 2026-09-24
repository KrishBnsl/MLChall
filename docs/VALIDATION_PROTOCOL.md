# Leakage-resistant validation protocol

This protocol is intentionally stricter than a random candidate-pair split. Pair-level splitting can
place the same business text in training and validation under different source records, producing an
optimistic score that will not generalize.

## 1. Audit before modeling

Record and review:

- SHA-256 and size of every raw file;
- row counts, identifier uniqueness, missingness, and text-length distributions;
- country labels and counts without hard-coding their values;
- ground-truth match cardinality and singleton rate;
- whether any Source 2/3 ID maps to multiple Source 1 IDs;
- exact and near-duplicate rates within and across sources;
- estimated all-pairs size and measured candidate-generation memory/runtime.

No search space is finalized until this audit is reviewed.

## 2. Entity-group fold manifest

The Source 1 record and all of its labeled Source 2/3 matches form one indivisible entity group. A
group receives one fold and none of its IDs may appear in another fold. Unmatched Source 2/3 records
are assigned deterministically and exclusively to folds so they can act as realistic decoys without
cross-fold row reuse.

The versioned split logic writes a manifest containing `entity_id`, `source`, `fold`, and
`entity_group`. The manifest and raw-data hashes are saved with every experiment.

Before training, automated checks must prove:

- no entity ID occurs twice;
- matched records share the Source 1 entity's fold;
- train and validation ID sets are disjoint for every outer fold;
- every raw record is accounted for exactly once.

## 3. Nested model selection

Use outer group-disjoint folds for the performance estimate and inner group-disjoint folds for all
choices, including:

- blocking parameters and candidate count;
- normalization variants and features;
- model hyperparameters;
- probability calibration;
- decision threshold and any per-country/per-source rule.

For each outer fold, refit every learned component from scratch on the outer-training partition.
The outer-validation partition is touched once for that configuration. Aggregate per-entity scores
across untouched outer folds and report the mean with fold dispersion and bootstrap confidence
intervals when sample size permits.

Leaderboard submissions are not evidence for model selection. They are sparse operational checks
after a choice is frozen.

## 4. Candidate generation evaluation

Evaluate blocking before the classifier:

- positive-pair recall overall and by source, country, text-missingness, and match cardinality;
- fraction of Source 1 entities with every true match retrieved;
- candidates per query (median, p95, maximum);
- reduction ratio relative to all valid cross-source pairs;
- wall time and peak memory.

Never inject missed validation positives into candidates. Doing so would hide the real recall ceiling.
Training-positive augmentation, if ever studied, must be an explicit ablation and cannot change the
validation candidate set.

## 5. Matching evaluation

Primary metric: macro entity-level F0.5 exactly as specified. Include singletons: truth-empty and
prediction-empty scores 1.0; a false match on a singleton scores 0.0.

Also record precision, recall, pairwise confusion counts, singleton accuracy, calibration metrics,
and slice results. Secondary metrics diagnose behavior but never replace the organizer metric.

Threshold selection occurs only from inner out-of-fold predictions. The system never forces a
positive prediction because zero-match entities are valid and rewarded.

## 6. Final fit and test inference

After the approach is frozen from nested validation:

1. Train once on all supplied training data.
2. Generate test candidates without looking up external business information.
3. Apply the frozen model and decision policy.
4. Write both required TSV files.
5. Run repository preflight checks and then the organizer's validator.
6. Archive configuration, Git commit, environment lock, data hashes, runtime, and output hashes.

The provided test Source 2/3 text is necessarily used to construct its label-free retrieval index
and retrieval vocabulary; validation mirrors that operation within each held-out candidate pool.
Test data is never used to choose blocking parameters or to fit classifier weights, calibration,
decision thresholds, or supervised feature transformations. Any additional transductive use must be
competition-legal, predeclared, and tested as a separate ablation.
