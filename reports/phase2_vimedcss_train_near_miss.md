# Phase 2 diagnosis on the ViMedCSS train pool (cross-dataset check)

`data/pools/vimedcss_train_v4/manifest.jsonl`: medical Vietnamese-English speech from YouTube, 3,868 rows, 7.5 h, original human labels. Rules and predictions were registered in the docstring of `scripts/diagnostics/cs_dev_near_miss.py` before the diagnosis ran. Locked benches not used.

Baseline greedy INT8: WER 18.37%, CS-WER 94.89%. 3099 rows with a CS substitution or deletion, 5236 spans. The diagnosis covers a seeded sample of 1200 of these rows (2113 spans).

CS words by the FPT rule (`cs_terms` + `en_like`): 6768. Words inside `<en>` tags: 5921. Tagged words the rule misses: 611; rule words without a tag: 1458.

| Category | Spans | FPT development pool | FPT extra diagnostic |
|---|---:|---:|---:|
| Near miss | 899 (42.5%) | 50.1% | 51.3% |
| Encoder bug | 41 (1.9%) | 1.4% | 2.8% |
| Predictor override | 28 (1.3%) | 1.1% | 1.2% |
| True gap | 81 (3.8%) | 2.2% | 2.0% |
| Inconclusive | 1064 (50.4%) | 45.2% | 42.8% |
| of which chain | 667 (31.6%) | 30.1% | 29.9% |
| Gap-like (true gap, START, LOCAL) | 478 (22.6%) | 17.3% | 14.8% |

Blank leads a non-top piece in 638 near-miss spans; 1260 blank-led pieces in 446 rows; 433 distinct near-miss words.

| Span type | Reachable | Not reachable |
|---|---:|---:|
| START | 22 | 269 |
| LOCAL | 8 | 98 |
| CHAIN | 336 | 331 |

## Registered predictions

- H1 near miss at least 30%: holds (42.5%).
- H2 encoder bug below 25%: holds (1.9%).
- H3 cascade prediction: holds (CHAIN is 336 of 366 reachable spans (91.8%) and 331 of 698 other spans (47.4%)).

## Labels

Original human labels of the ViMedCSS dataset, no ASR agreement filter, so the gap-like share is not biased toward words an ASR can hear. Medical terms (drug, disease and anatomy names) are frequent. This is the train split, a training pool of direction B; the ViMedCSS test split (bench E0) was not read.

Descriptive beyond these three. One span is about 0.05% of the total.
