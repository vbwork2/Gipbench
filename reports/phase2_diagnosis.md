# Phase 2 diagnosis

PyTorch and ONNX INT8 produced identical greedy transcripts on the ten calibration recordings. The FPT bug-mining split contained 100 recordings and 179 code-switching substitutions or deletions. The heldout remained sealed throughout diagnosis.

| Category | Emitted greedy prefix | Reference prefix |
|---|---:|---:|
| Near miss | 85 (47.49%) | 117 (65.36%) |
| Encoder bug | 2 (1.12%) | 11 (6.15%) |
| Predictor override | 1 (0.56%) | 0 (0.00%) |
| Operational true gap | 5 (2.79%) | 5 (2.79%) |
| Inconclusive | 86 (48.04%) | 46 (25.70%) |

A near miss has every target piece in the final-stack top ten within ±1 frame of its aligned position, but the span was still misrecognized. The answer was available to the Joiner, so the loss happened in score calibration or the search, after the encoder. The first version of this report had no near-miss category and counted these spans as inconclusive. The counts above come from re-applying the current rules to the same stored ranks with `scripts/diagnostics/fpt_phase2_reclassify.py`. No audio was re-run.

With the emitted greedy prefix, near misses cover 60 of the 100 recordings. The worst piece in each span sits at rank 1 in 7 spans, at ranks 2–3 in 39 and at ranks 4–10 in 39. Blank leads a non-top piece in 58 of the 85 spans. Across all near-miss pieces below rank one, blank is the leading competitor in 104 cases and the next most frequent competitor appears 8 times.

Near misses exceed the 30% score-calibration threshold in both contexts. Neither context reached the thresholds for an encoder sweep (25%), Joiner calibration from predictor override (30%), or fine-tuning (more than 70% operational gaps). The emitted-prefix analysis uses the decoder state available before each frame. Both analyses locate target pieces with forced reference alignment.

The operational gap label records BPE targets that stayed outside the top ten in the measured windows. Zero diagnosed spans contained an unknown SentencePiece ID. The frame-level evidence is in the local `phase2_bug_vs_gap.json`.
