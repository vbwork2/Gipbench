# Cascade test for the INCONCLUSIVE FPT spans

Development only, from stored ranks (no model run). Heldout untouched. The rules and the prediction are in the docstring of `scripts/diagnostics/fpt_inconclusive_cascade.py` and were fixed before the run.

Span types. START: the word's first piece is already outside the top ten. LOCAL: the first missing piece follows pieces that all have rank 1. CHAIN: an earlier piece of the word had rank 2 to 10, so it lost its frame to a competitor before the first missing piece.

| Span type | Bug-mining reachable | Bug-mining not reachable | Other development reachable | Other development not reachable | Pooled reachable | Pooled not reachable |
|---|---:|---:|---:|---:|---:|---:|
| START | 1 | 18 | 1 | 13 | 2 | 31 |
| LOCAL | 1 | 6 | 2 | 13 | 3 | 19 |
| CHAIN | 36 | 24 | 31 | 19 | 67 | 43 |

Reachable means every piece is in the top ten under the greedy prefix or under the reference prefix (oracle context).

Pooled: CHAIN is 67 of 72 reachable spans (93.1%) and 43 of 93 other spans (46.2%).

## Root of the CHAIN spans (the first earlier piece that lost its frame)

| Leader at the root piece | Bug-mining | Other development | Pooled |
|---|---:|---:|---:|
| Blank | 24 | 24 | 48 |
| Another piece | 36 | 26 | 62 |

## Reading

Registered prediction (CHAIN above half of the reachable spans and above its share among the others): supported.

A best rank of 1 is the best inside the probed window and does not prove the piece was emitted, so LOCAL may contain spans whose emission was wrong. The reference prefix is an oracle context. This test describes where the first loss happens. It does not test any repair.
