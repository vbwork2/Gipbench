# Phase 2 diagnosis

PyTorch and ONNX INT8 produced identical greedy transcripts on the ten calibration recordings. The FPT bug-mining split contained 100 recordings and 179 code-switching substitutions or deletions. The heldout remained sealed throughout diagnosis.

| Category | Emitted greedy prefix | Reference prefix |
|---|---:|---:|
| Encoder bug | 2 (1.12%) | 11 (6.15%) |
| Predictor override | 1 (0.56%) | 0 (0.00%) |
| Operational true gap | 5 (2.79%) | 5 (2.79%) |
| Inconclusive | 171 (95.53%) | 163 (91.06%) |

Neither context reached the thresholds for an encoder sweep (25%), Joiner calibration (30%), or fine-tuning (more than 70% operational gaps). The emitted-prefix analysis uses the decoder state available before each frame. Both analyses locate target pieces with forced reference alignment.

The operational gap label records BPE targets that stayed outside the top ten in the measured windows. Zero diagnosed spans contained an unknown SentencePiece ID. The frame-level evidence is in the local `phase2_bug_vs_gap.json`.
