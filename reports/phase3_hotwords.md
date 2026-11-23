# Phase 3 hotword sweep

Unbiased beam CS-WER: 119.31%.
Terms used: 181. Omitted unknown BPE terms: 0.
Clean-control utterances already changed by unbiased beam: 10.
The unbiased beam and hotword runs use the same batches over both development splits.

| Score | CS-WER | CS-WER gain | Newly changed clean | False hotword triggers | Pareto frontier |
|---:|---:|---:|---:|---:|:---|
| 1.0 | 103.96% | 15.35% | 0.25% | 0.00% | no |
| 1.5 | 98.51% | 20.79% | 0.25% | 0.25% | no |
| 2.0 | 97.52% | 21.78% | 0.25% | 0.25% | yes |
| 2.5 | 94.06% | 25.25% | 0.50% | 0.25% | no |
| 3.0 | 93.56% | 25.74% | 0.50% | 0.25% | yes |

Selected score: 2.0.
