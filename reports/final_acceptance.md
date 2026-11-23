# Final acceptance

Primary candidate hotwords compared with beam. Overall result **FAIL**.

Heldout has 1,000 utterances, including 200 with CS reference tokens. The split unit is recording.
This is a decoding-only evaluation on identical model weights.

| System | Heldout WER | Heldout N-WER | Heldout CS-WER | Heldout insertion rate |
|---|---:|---:|---:|---:|
| beam | 9.43% | 7.52% | 81.64% | 1.07% |
| hotwords | 9.46% | 7.57% | 80.98% | 1.11% |
| greedy | 9.70% | 7.76% | 83.28% | 1.06% |

| Gate | Status | Evidence |
|---|---|---|
| G1 | FAIL | CS-WER gain +0.66 pp, p=0.2997, 95% CI [-1.79, +3.41] pp |
| G2 | PASS | 0/400 newly degraded clean utterances |
| G3 | PASS | Insertion rate delta +0.042 pp |
| G4 | PASS | N-WER delta +0.052 pp |
| G5 | N/A | Decoding-only comparison with identical weights |
