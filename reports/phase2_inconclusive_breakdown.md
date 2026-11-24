# Breakdown of the INCONCLUSIVE FPT spans

Development only, from stored ranks (no model run). Heldout untouched. Rules are in the docstring of `scripts/diagnostics/fpt_inconclusive_breakdown.py`. Descriptive: no threshold is tested here.

## Match with the review profiles (bug-mining, span by span)

`phase2_inconclusive_review.md` sampled from 171 spans that were INCONCLUSIVE before the NEAR_MISS category existed (85 all pieces, 81 some, 5 none in the final-stack top ten at +/-1 frame). Recomputing `classify` on the stored ranks gives:

| Review profile | Now NEAR_MISS | Now INCONCLUSIVE |
|---|---:|---:|
| All pieces | 85 | 0 |
| Some pieces | 0 | 81 |
| No pieces | 0 | 5 |

The profile is the same quantity as the NEAR_MISS test, so the 85 "all" spans are the near misses and the 86 remaining INCONCLUSIVE spans are exactly the "some" and "none" ones. Stored categories that differ from the recomputed ones: bug-mining 85, other development 0 (the old INCONCLUSIVE spans that are now NEAR_MISS).

## What the INCONCLUSIVE spans look like

| | Bug-mining | Other development | Pooled |
|---|---:|---:|---:|
| INCONCLUSIVE spans | 86 | 79 | 165 |
| Final-stack profile "some" | 81 | 76 | 157 |
| Final-stack profile "none" | 5 | 3 | 8 |
| Exactly one piece outside the top ten | 46 | 44 | 90 |
| Two pieces outside | 23 | 20 | 43 |
| Three or more outside | 17 | 15 | 32 |
| Every piece within the top 20 | 18 | 17 | 35 |
| Every piece within the top 50 | 34 | 33 | 67 |
| Worst missing piece below rank 100 | 36 | 32 | 68 |
| Span contains a tokenizer unknown | 0 | 0 | 0 |
| All pieces in the top ten if the predictor had the reference prefix | 38 | 34 | 72 |
| ... and the first missing piece is the word's first piece | 1 | 1 | 2 |
| ... and the first missing piece comes later in the word | 37 | 33 | 70 |

## Where this leaves the post-encoder share (pooled, 365 spans)

Alternative readings of the same table, not a decision. They differ by what is allowed to count as "after the encoder". Each of the last two rows adds a different set of INCONCLUSIVE spans to the first row; the two sets overlap, so the rows are not cumulative.

| Reading | Spans | Share |
|---|---:|---:|
| NEAR_MISS + PREDICTOR_OVERRIDE_BUG (the `classify` labels) | 187 | 51.2% |
| First row plus INCONCLUSIVE spans with every piece within the top 20 | 222 | 60.8% |
| First row plus INCONCLUSIVE spans reachable with the reference prefix (oracle context) | 259 | 71.0% |

The last reading uses an oracle context, so it measures availability and not repairability. The top-20 reading changes the 10 in the NEAR_MISS rule, which was fixed before the first run. Neither is a preregistered test.
