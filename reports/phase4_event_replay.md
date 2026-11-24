# Replicated near-miss events per pool (Joiner replay)

Diagnostic data only: the sealed development pool and the diagnostic part of the extra pool (`extra_diagnostic.jsonl`, checked against `phase1_extra_split_lock.json`). Neither heldout was opened. Rules are those of `scripts/08_joiner_decomposition.py`, restated in the docstring of `scripts/diagnostics/fpt_event_replay.py`: an event is a near-miss piece below rank 1 with blank leading at its Phase 2 frame. It replicates when the greedy replay (INT8 encoder and decoder outputs, fp32 output layer) has blank as argmax at that frame and the piece in the top ten.

| Group | Events | Replicated | Share | Recordings with a replicated event | Distinct words |
|---|---:|---:|---:|---:|---:|
| Bug mining (100 rec.) | 104 | 66 | 63.5% | 34 | 40 |
| Other development (179 rec.) | 100 | 78 | 78.0% | 49 | 47 |
| Development pooled | 204 | 144 | 70.6% | 83 | 85 |
| Extra diagnostic (1,070 rec.) | 857 | 654 | 76.3% | 389 | 360 |

Harness check: replayed on the Phase 4 encoder cache, bug mining gives 65 of 104, the Phase 4 figure. The table uses freshly computed encoder outputs for every group. On those, bug mining gives 66. The INT8 encoder is sensitive to tiny feature differences: the NumPy filterbank matches kaldi-native-fbank to about 6e-5, and that alone moves encoder outputs by about 0.2. So one event in 104 flipping is the size of the numerical noise here, not a change of rule.

## What this changes

Phase 4 had 65 replicated events. The extra diagnostic part alone has 654, in 389 recordings and 360 distinct words, so a trigger (TODO_rea item 2.2) could be fit with folds by recording, by word and by normalized text without running out of events. The development pool adds 144 more. Bug mining replicates less often (63.5%) than the unselected recordings (76–78%). That fits its selection for repeated confusions, and the 62.5% estimate used so far was low.

Descriptive. Nothing was selected or tuned on these counts. The 24 extra-pool recordings that share a text with the development pool are on the diagnostic side, as the split rule says.
