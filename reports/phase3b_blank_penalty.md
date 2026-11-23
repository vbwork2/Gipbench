# Phase 3b blank-penalty sweep

Development only: bug mining 100, clean control 400 and a seeded random sample of 500 other development recordings (40 with CS words). The heldout stayed sealed. INT8 ONNX, same batches for every penalty.

Selection rule, fixed before the sweep: clean-control damage below 0.5%, random-sample WER no higher than at penalty 0, then the lowest bug-mining CS-WER, with ties going to the smaller penalty.

Phase 2 spans re-classified from stored ranks: {'OTHER': 94, 'NEAR_MISS': 85}. The fix rate counts spans that were wrong at penalty 0 with the same decoder and became correct. If the penalty acts through the near-miss mechanism, the near-miss fix rate should clearly exceed the rate for other spans.

## greedy

| Penalty | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | Random del | Clean damage | Near-miss fixed | Other fixed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 123.27% | 22.42% | 6.15% | 7.73% | 7.32% | 55.77% | 0.88% | 1.29% | 0.00% | 0/85 | 0/94 |
| 0.5 | 124.26% | 22.42% | 5.99% | 7.98% | 7.53% | 59.62% | 1.08% | 1.23% | 1.25% | 1/85 | 0/94 |
| 1 | 126.24% | 22.63% | 5.91% | 8.13% | 7.65% | 63.46% | 1.23% | 1.16% | 2.00% | 1/85 | 0/94 |
| 2 | 127.72% | 22.90% | 5.99% | 8.81% | 8.28% | 69.23% | 1.76% | 1.11% | 5.25% | 4/85 | 0/94 |
| 3 | 131.68% | 23.93% | 6.55% | 9.84% | 9.32% | 69.23% | 2.83% | 1.06% | 12.50% | 4/85 | 0/94 |
| 4 | 140.59% | 25.65% | 7.11% | 12.24% | 11.61% | 84.62% | 5.17% | 0.96% | 20.00% | 3/85 | 0/94 |
| 6 | 184.65% | 38.93% | 15.42% | 25.93% | 24.85% | 150.00% | 18.46% | 0.33% | 57.75% | 3/85 | 0/94 |

Random-sample CS-WER gain against penalty 0 (paired bootstrap, 1,000 rounds):

- 0.5: -3.85 pp, p=0.8991, 95% CI [-10.35, +0.00] pp
- 1: -7.69 pp, p=0.9411, 95% CI [-17.24, +0.00] pp
- 2: -13.46 pp, p=0.9870, 95% CI [-24.49, -3.63] pp
- 3: -13.46 pp, p=0.9770, 95% CI [-26.42, -1.96] pp
- 4: -28.85 pp, p=0.9950, 95% CI [-50.00, -10.53] pp
- 6: -94.23 pp, p=1.0000, 95% CI [-135.73, -54.34] pp

No penalty met the selection rule.

## beam

| Penalty | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | Random del | Clean damage | Near-miss fixed | Other fixed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 119.80% | 21.60% | 5.75% | 7.87% | 7.43% | 57.69% | 0.99% | 1.31% | 0.00% | 0/83 | 0/94 |
| 0.5 | 121.29% | 21.87% | 5.83% | 7.96% | 7.52% | 59.62% | 1.11% | 1.21% | 0.75% | 1/83 | 0/94 |
| 1 | 120.30% | 21.87% | 5.99% | 8.20% | 7.73% | 61.54% | 1.31% | 1.21% | 2.00% | 3/83 | 0/94 |
| 2 | 124.75% | 22.76% | 6.31% | 9.11% | 8.59% | 69.23% | 2.19% | 1.16% | 6.75% | 5/83 | 0/94 |
| 3 | 133.17% | 24.35% | 6.79% | 10.37% | 9.80% | 75.00% | 3.51% | 0.98% | 15.75% | 4/83 | 0/94 |
| 4 | 138.61% | 26.34% | 8.23% | 13.79% | 13.15% | 88.46% | 6.52% | 0.83% | 28.75% | 6/83 | 0/94 |
| 6 | 186.14% | 41.33% | 17.97% | 30.07% | 28.98% | 155.77% | 22.47% | 0.48% | 66.00% | 4/83 | 0/94 |

Random-sample CS-WER gain against penalty 0 (paired bootstrap, 1,000 rounds):

- 0.5: -1.92 pp, p=0.8392, 95% CI [-6.38, +0.00] pp
- 1: -3.85 pp, p=0.9500, 95% CI [-8.48, +0.00] pp
- 2: -11.54 pp, p=0.9540, 95% CI [-26.20, +0.00] pp
- 3: -17.31 pp, p=0.9770, 95% CI [-34.00, -2.38] pp
- 4: -30.77 pp, p=0.9940, 95% CI [-54.06, -11.11] pp
- 6: -98.08 pp, p=1.0000, 95% CI [-140.48, -59.25] pp

No penalty met the selection rule.

## Conclusion

This section was written after the sweep. The tables above are the script output.

Blank penalty is rejected for both decoders. At penalty 0.5, it already damaged 1.25% (greedy) and 0.75% (beam) of the clean controls and raised random-sample WER. Larger penalties added insertions on ordinary speech much faster than they fixed CS words. No penalty lowered bug-mining CS-WER below its penalty-0 value with either decoder. The best penalty recovered at most 6 of 83 near-miss spans.

The effect is specific even though it is small. Only near-miss spans were ever fixed. Across every run, 0 of 94 other spans changed from wrong to right. The near-miss category marks errors that can be reached after the encoder, but a uniform blank offset is too blunt to reach them. Before the full sweep, the stored Phase 2 ranks predicted this. Where blank leads a near-miss piece, the median log-probability margin is 5.7 nats. Only 6 of 58 blank-led spans have every margin at or below 2, and a penalty applies to every frame of every utterance.

A follow-up therefore needs to act only where it is needed. One example is a penalty or Joiner adjustment gated on frames where a non-blank piece is already within the top ten. A global decoder constant does not do this.
