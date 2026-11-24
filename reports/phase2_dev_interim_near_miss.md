# Phase 2 diagnosis on the interim CS dev set (cross-dataset check)

`data/dev/dev_interim_cs.jsonl`: natural speech from YouTube talks (VietSuperSpeech), 323 rows, 61 sources, labels listened to by a person. Rules and predictions were registered in the docstring of `scripts/diagnostics/cs_dev_near_miss.py` before the diagnosis ran. Locked benches not used.

Baseline greedy INT8: WER 4.91%, CS-WER 43.38%. 135 rows with a CS substitution or deletion, 174 spans.

CS words by the FPT rule (`cs_terms` + `en_like`): 521. Words inside `<en>` tags: 550. Tagged words the rule misses: 31; rule words without a tag: 2.

| Category | Spans | FPT development pool | FPT extra diagnostic |
|---|---:|---:|---:|
| Near miss | 124 (71.3%) | 50.1% | 51.3% |
| Encoder bug | 7 (4.0%) | 1.4% | 2.8% |
| Predictor override | 5 (2.9%) | 1.1% | 1.2% |
| True gap | 0 (0.0%) | 2.2% | 2.0% |
| Inconclusive | 38 (21.8%) | 45.2% | 42.8% |
| of which chain | 36 (20.7%) | 30.1% | 29.9% |
| Gap-like (true gap, START, LOCAL) | 2 (1.1%) | 17.3% | 14.8% |

Blank leads a non-top piece in 68 near-miss spans; 94 blank-led pieces in 58 rows; 93 distinct near-miss words.

| Span type | Reachable | Not reachable |
|---|---:|---:|
| START | 0 | 2 |
| LOCAL | 0 | 0 |
| CHAIN | 30 | 6 |

## Registered predictions

- H1 near miss at least 30%: holds (71.3%).
- H2 encoder bug below 25%: holds (4.0%).
- H3 cascade prediction: holds (CHAIN is 30 of 30 reachable spans (100.0%) and 6 of 8 other spans (75.0%)).

## Selection bias (read before comparing the gap share)

The rows come from the `vss_cs` pool, whose labels started as a Zipformer-30M pseudo-label. Every row kept has every English span confirmed by at least one of two ASR hypotheses (`agreement.confirmed`, 471 of 471 spans), and a person then reviewed it. English words that ASR systems do not hear were filtered out before this set existed. The low gap-like share is therefore partly built into the set and does not compare with FPT. The bug-side predictions (H1 to H3) are still informative: on words that an ASR can hear, Gipformer's CS errors are mostly near misses and cascades.

Descriptive beyond these three. One span is about 0.6% of the total.
