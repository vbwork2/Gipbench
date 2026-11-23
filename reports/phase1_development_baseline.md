# Gipformer 1.5 ONNX baseline

Dataset doof-ferb/fpt_fosd at revision e6ff950456eeaef7a257e79bf7f4597bb39c264c.

This run covers development splits. Heldout is reserved for Phase 7.
CS-WER includes attributed insertions and can exceed 100% on an error-selected split.

| Split | Decoder | Utterances | WER | CER | N-WER | CS-WER | Ins | Del | Sub |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bug_mining | greedy | 100 | 21.73% | 12.91% | 5.67% | 121.29% | 4.95% | 2.48% | 14.31% |
| bug_mining | modified beam (4) | 100 | 21.60% | 13.04% | 5.99% | 118.32% | 4.47% | 2.61% | 14.51% |
| clean_control | greedy | 400 | 0.00% | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00% |
| clean_control | modified beam (4) | 400 | 0.32% | 0.20% | 0.32% | n/a | 0.11% | 0.00% | 0.20% |
