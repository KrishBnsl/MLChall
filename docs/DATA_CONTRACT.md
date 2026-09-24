# Data contract

This contract is derived from the supplied problem statement. The audit code fails loudly on schema
or identifier violations instead of silently repairing unknown data.

## Record files

Every `*_source1.tsv`, `*_source2.tsv`, and `*_source3.tsv` file must be tab-separated and contain:

| Column | Contract |
|---|---|
| `entity_id` | Non-empty, unique in its file, and prefixed with `S1-`, `S2-`, or `S3-` according to the source file |
| `business_name` | String; empty is allowed because noise/missingness is expected |
| `business_address` | String; empty is allowed because noise/missingness is expected |
| `country` | Open-set string label; the code never enumerates only US and India |

The loader preserves text as strings and does not interpret strings such as `NA` as missing values.
Unexpected columns are rejected by default so a changed organizer schema cannot silently alter a run.

## Ground truth

`train_ground_truth.tsv` must contain exactly:

- `source1_entity_id`
- `matched_entity_ids`

There must be exactly one ground-truth row for every training Source 1 ID. Match lists may be empty,
must contain unique comma-separated IDs, and may refer only to IDs present in training Sources 2 or
3. A Source 2 or Source 3 record assigned to multiple Source 1 entities is treated as an ambiguous
entity-group violation until the actual data is reviewed.

## Submission files

`matching_results.tsv` and `candidate_pairs.tsv` must be tab-separated, have exactly one row per test
Source 1 ID, contain no duplicate IDs per list, and reference only test Source 2 or Source 3 IDs.
Every predicted final match must be present in that Source 1 entity's submitted candidate list.

The official organizer validator is the final format authority.

## Raw-data immutability

Raw files are never modified. Each audit records file size and SHA-256. Derived folds, candidates,
features, models, predictions, and reports live outside `data/`.

