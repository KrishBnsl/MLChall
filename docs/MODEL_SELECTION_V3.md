# Model selection v3

This document records the leakage-safe model selection decisions for the expanded entity-resolution
pipeline. Numbers in the tuning tables are entity-group out-of-fold (OOF) estimates. They are not
official test scores and must not be presented as such.

## Objective and validation boundary

The optimization target is entity-level macro F0.5. F0.5 gives precision four times the weight of
recall:

```text
F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)
```

The original unused training partition was deterministically divided by real-world identity into:

| Partition | Source 1 entities | Permitted use |
| --- | ---: | --- |
| fit | 1,765,497 | feature/model fitting |
| tuning | 110,332 | model selection and threshold tuning |
| holdout | 110,332 | one final frozen evaluation only |

Target records owned by a Source 1 identity remain in the same partition. For a scale-sensitive
final check, tuning and holdout queries are also evaluated against the full target pool: target
records from other identity partitions are included only as nonmatching distractors. Labels and
queries remain held out.

## Candidate retrieval decisions

The accepted retriever is the union of:

- country/source/query-batch partitioned exact and token rules; and
- accent-folded hashed character TF-IDF over name plus structured address, top 100 per source.

The rule key document-frequency cap is 2,000 for the full target universe. On scale-matched tuning,
this recovered 83.5763% of positive pairs versus 71.2713% at the earlier cap of 200, while the
top-50 rule output kept volume bounded (11.68M versus 8.65M pairs). The higher cap preserves roughly
the same relative frequency cutoff used on the smaller tuning-only target pool.

On the original tuning target pool this produced 28,604,504 pairs, 97.7791% positive-pair recall,
and 93.8351% complete-entity recall. Candidate recall is reported separately because the matcher
cannot recover a positive pair that blocking omits.

Rejected retrieval variants:

| Variant | Candidate pairs | Pair recall | Decision |
| --- | ---: | ---: | --- |
| accepted top 100 union | 28,604,504 | 0.977791 | keep |
| name-only union | 30,944,789 | approximately unchanged | reject: one extra positive only |
| top 200 union | 47,635,347 | 0.980695 | reject: 66% more pairs and lower F0.5 |

The top-200 matcher score was 0.971603 versus 0.973524 for top 100. The small oracle-recall gain did
not survive end-to-end precision-heavy evaluation.

## Feature set

The matcher uses compact, explainable pair features rather than raw-text memorization:

- blocker agreement, exact-name/address indicators, retrieval score/rank/gap/ratio;
- normalized name and address Jaro-Winkler, edit, token Jaccard, and containment similarities;
- sorted-name, first/last-name-token, address-number overlap/subset/conflict features;
- source-relative candidate counts and query/candidate name/address frequency features; and
- missingness, country agreement, source indicator, and length ratios.

These features are deterministic and use only organizer-provided records. No external lookup or
pretrained identity model is used.

## Model-family comparison

The same Round-1 model-hard-negative sample was used for the tree-family comparison:

| Model | OOF macro F0.5 | Selected threshold | Decision |
| --- | ---: | ---: | --- |
| LightGBM | 0.972957 | 0.160 | selected |
| XGBoost | 0.970332 | 0.180 | reject |
| HistGradientBoosting (bounded 80-tree baseline) | 0.964906 | 0.125 | reject |
| CatBoost | 0.963913 | 0.175 | reject |

LightGBM was both the best scorer and the fastest practical implementation. A bounded
HistGradientBoosting run is retained as the dependency-light baseline. A neural network is not
justified by the evidence: the inputs are structured similarity features, the tree model is easier
to audit, and adding a neural text model would increase overfitting and deployment risk without a
validated gain.

## Hard-example mining

Training began with all retrieved positives, blocker-hard negatives, and deterministic random
negatives. The OOF matcher then scored candidates, and the highest-scoring false pairs were added
for a second round. A third sample increased model-hard negatives from 5 to 10 per source/query.

| Round | OOF macro F0.5 | Threshold | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| no model-hard negatives | 0.949044 | 0.515 | 346,895 | 8,686 | 34,981 |
| 5 model-hard/source | 0.972957 | 0.160 | 359,569 | 2,689 | 22,307 |
| 10 model-hard/source | **0.973524** | **0.210** | 360,038 | 2,593 | 21,838 |

The Round-2 gain was only 0.000567, so further mining was stopped. This is an explicit complexity
gate against repeatedly fitting the tuning partition.

## Threshold and segmentation decision

Source-specific thresholds were evaluated with a minimum required gain of 0.0001 macro F0.5. The
largest observed gain was below that guard, so the production design retains one global threshold.
The threshold must be re-estimated using predictions from the final fit-only model on the
scale-matched tuning queries, then frozen before the holdout is scored once.

## Status of the 0.99 target

The best completed tuning OOF estimate is 0.973524. A score of 0.99 has not been achieved and is not
guaranteed. The remaining honest path is to train the selected simple LightGBM on the much larger
fit partition, tune once on scale-matched tuning, and report the untouched scale-matched holdout.
The pipeline must not claim 0.99 by reusing the holdout, forcing matches, or tuning to an official
leaderboard.
