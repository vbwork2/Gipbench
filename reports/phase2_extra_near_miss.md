# Phase 2 diagnosis on the extra FPT pool (diagnostic part)

Diagnostic part of the extra pool only (`extra_diagnostic.jsonl`, sha256 checked against `phase1_extra_split_lock.json`). The new heldout was not opened. Same logit lens, emitted-greedy-prefix context and categories as Phase 2; rules in the docstrings of `fpt_pool_near_miss.py` and `fpt_extra_report.py`. The development columns reproduce `phase2_pool_near_miss.md`.

| | Other development (179 rec.) | Development pooled (279 rec.) | Extra diagnostic (1070 rec.) |
|---|---:|---:|---:|
| Spans | 186 | 365 | 1487 |
| Near miss | 98 (52.7%) | 183 (50.1%) | 763 (51.3%) |
| Encoder bug | 3 (1.6%) | 5 (1.4%) | 41 (2.8%) |
| Predictor override | 3 (1.6%) | 4 (1.1%) | 18 (1.2%) |
| True gap | 3 (1.6%) | 8 (2.2%) | 29 (2.0%) |
| Inconclusive | 79 (42.5%) | 165 (45.2%) | 636 (42.8%) |
| Near-miss spans where blank leads a non-top piece | 59 | 117 | 504 |
| Blank-led near-miss pieces (Phase 4 event unit) | 100 | 204 | 857 |
| Recordings with such a piece | 58 | 101 | 449 |
| Distinct reference words among near-miss spans | 92 | 161 | 599 |

## Cascade types of the INCONCLUSIVE spans (replicate of `phase2_inconclusive_cascade.md`)

| Span type | Other development reachable | Other development not reachable | Development pooled reachable | Development pooled not reachable | Extra diagnostic reachable | Extra diagnostic not reachable |
|---|---:|---:|---:|---:|---:|---:|
| START | 1 | 13 | 2 | 31 | 8 | 108 |
| LOCAL | 2 | 13 | 3 | 19 | 7 | 68 |
| CHAIN | 31 | 19 | 67 | 43 | 289 | 156 |

| CHAIN root leader | Other development | Development pooled | Extra diagnostic |
|---|---:|---:|---:|
| Blank | 24 | 48 | 204 |
| Another piece | 26 | 62 | 241 |

- Development pooled: CHAIN is 67 of 72 reachable spans (93.1%) and 43 of 93 other spans (46.2%). Registered cascade prediction: holds.
- Extra diagnostic: CHAIN is 289 of 304 reachable spans (95.1%) and 156 of 332 other spans (47.0%). Registered cascade prediction: holds.

## Bug or gap on the extra pool (same reading as `TODO_rea.md`)

- Post-encoder, near miss: 763 (51.3%).
- Post-encoder, chain: 445 (29.9%).
- Encoder bug and predictor override: 59 (4.0%).
- Gap-like (true gap, START, LOCAL): 220 (14.8%).

Descriptive. The extra pool includes 24 recordings (26 spans) whose text also occurs in the development pool; they were forced to the diagnostic side by the split. Blank-led pieces are not yet replayed through the Joiner (item 1.4), so they are candidate events, not replicated ones.
