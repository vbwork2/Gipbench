# Phase 3c conditional blank penalty

Development only: bug mining 100, clean control 400 and a seeded random sample of 500 other development recordings (40 with CS words). The heldout stayed sealed. INT8 ONNX greedy search, one symbol per frame, batch 1, decoded by `gipformer_mech.onnx_greedy` on encoder outputs cached from soxr-resampled 16 kHz audio.

Pre-registered rules are in the docstring of `scripts/07_eval_conditional_blank.py`.

## Decoder parity

The first parity criterion was 98% identical transcripts. Both penalties failed it, at 95.50% and 95.30%. Sherpa-onnx agrees with itself on only 91.20% of these rows when batch size and resampler change, so the criterion was revised before any sweep result existed: at least 91.2% identical and WER within 0.2 pp of sherpa on every set. See the script docstring.

| Blank penalty | Identical to sherpa-onnx greedy | Bug WER own / sherpa | Clean WER own / sherpa | Random WER own / sherpa |
|---:|---:|---:|---:|---:|
| 0 | 95.50% | 22.15% / 22.01% | 0.11% / 0.11% | 7.60% / 7.60% |
| 2 | 95.30% | 23.38% / 23.45% | 0.75% / 0.70% | 8.76% / 8.69% |

The continuation condition holds on 9.42% of frames at penalty 0. The random control fires at that rate.

Phase 2 spans re-classified from stored ranks: {'OTHER': 94, 'NEAR_MISS': 85}. Fix counts are relative to this decoder at penalty 0.

## uniform

| Setting | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | Random del | Clean damage | Near-miss fixed | Other fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120.79% | 22.15% | 6.23% | 7.60% | 7.18% | 55.77% | 0.84% | 1.24% | 0.00% | 0/85 | 0/94 |
| 1 (100.00% frames) | 126.24% | 23.04% | 6.39% | 8.13% | 7.65% | 63.46% | 1.28% | 1.21% | 1.75% | 1/85 | 0/94 |
| 2 (100.00% frames) | 128.22% | 23.38% | 6.47% | 8.76% | 8.25% | 67.31% | 1.77% | 1.11% | 5.75% | 3/85 | 0/94 |
| 3 (100.00% frames) | 133.66% | 24.42% | 6.79% | 9.94% | 9.40% | 71.15% | 2.85% | 1.08% | 11.25% | 5/85 | 0/94 |
| 4 (100.00% frames) | 143.56% | 26.41% | 7.51% | 12.49% | 11.88% | 82.69% | 5.32% | 0.99% | 21.00% | 2/85 | 0/94 |
| 6 (100.00% frames) | 172.28% | 37.83% | 16.13% | 24.90% | 23.87% | 144.23% | 17.52% | 0.63% | 54.00% | 2/85 | 0/94 |
| 8 (100.00% frames) | 283.66% | 80.19% | 47.36% | 70.26% | 68.78% | 240.38% | 62.59% | 0.13% | 88.50% | 1/85 | 0/94 |

Random-sample CS-WER gain against penalty 0 (paired bootstrap, 1,000 rounds):

- 1: -7.69 pp, p=0.9451, 95% CI [-17.07, +0.00] pp
- 2: -11.54 pp, p=0.9820, 95% CI [-22.23, -2.94] pp
- 3: -15.38 pp, p=0.9940, 95% CI [-26.99, -4.99] pp
- 4: -26.92 pp, p=0.9950, 95% CI [-47.46, -8.33] pp
- 6: -88.46 pp, p=1.0000, 95% CI [-130.37, -49.01] pp
- 8: -184.62 pp, p=1.0000, 95% CI [-248.09, -124.59] pp

No penalty met the selection rule.

## continuation

| Setting | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | Random del | Clean damage | Near-miss fixed | Other fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120.79% | 22.15% | 6.23% | 7.60% | 7.18% | 55.77% | 0.84% | 1.24% | 0.00% | 0/85 | 0/94 |
| 1 (9.31% frames) | 120.79% | 22.15% | 6.23% | 7.67% | 7.23% | 57.69% | 0.84% | 1.24% | 0.25% | 1/85 | 0/94 |
| 2 (9.21% frames) | 118.32% | 21.94% | 6.39% | 7.80% | 7.33% | 61.54% | 0.84% | 1.23% | 0.75% | 3/85 | 0/94 |
| 3 (9.14% frames) | 119.80% | 22.21% | 6.47% | 7.92% | 7.43% | 63.46% | 0.86% | 1.28% | 1.25% | 4/85 | 0/94 |
| 4 (9.05% frames) | 122.77% | 22.76% | 6.63% | 8.11% | 7.62% | 65.38% | 0.86% | 1.24% | 3.00% | 2/85 | 0/94 |
| 6 (8.79% frames) | 126.24% | 23.80% | 7.27% | 9.11% | 8.55% | 73.08% | 0.89% | 1.32% | 9.25% | 1/85 | 0/94 |
| 8 (8.33% frames) | 128.22% | 26.13% | 9.66% | 10.85% | 10.22% | 82.69% | 0.94% | 1.49% | 21.50% | 0/85 | 0/94 |

Random-sample CS-WER gain against penalty 0 (paired bootstrap, 1,000 rounds):

- 1: -1.92 pp, p=0.8841, 95% CI [-5.56, +0.00] pp
- 2: -5.77 pp, p=0.9800, 95% CI [-11.25, +0.00] pp
- 3: -7.69 pp, p=0.9900, 95% CI [-14.29, -1.79] pp
- 4: -9.62 pp, p=0.9880, 95% CI [-18.33, -1.89] pp
- 6: -17.31 pp, p=1.0000, 95% CI [-28.79, -6.52] pp
- 8: -26.92 pp, p=1.0000, 95% CI [-41.07, -13.04] pp

No penalty met the selection rule.

## word_start

| Setting | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | Random del | Clean damage | Near-miss fixed | Other fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120.79% | 22.15% | 6.23% | 7.60% | 7.18% | 55.77% | 0.84% | 1.24% | 0.00% | 0/85 | 0/94 |
| 1 (90.37% frames) | 126.24% | 22.97% | 6.31% | 8.10% | 7.63% | 61.54% | 1.28% | 1.21% | 1.50% | 0/85 | 0/94 |
| 2 (90.05% frames) | 130.20% | 23.59% | 6.39% | 8.61% | 8.13% | 63.46% | 1.79% | 1.13% | 5.00% | 0/85 | 0/94 |
| 3 (89.66% frames) | 131.68% | 24.00% | 6.63% | 9.47% | 8.99% | 65.38% | 2.67% | 1.09% | 9.25% | 2/85 | 0/94 |
| 4 (89.21% frames) | 136.63% | 25.24% | 7.27% | 11.72% | 11.19% | 73.08% | 5.00% | 1.01% | 18.00% | 2/85 | 0/94 |
| 6 (88.14% frames) | 150.50% | 33.29% | 14.38% | 22.47% | 21.71% | 109.62% | 15.93% | 0.68% | 48.00% | 4/85 | 0/94 |
| 8 (86.16% frames) | 222.77% | 65.82% | 40.50% | 63.40% | 62.52% | 165.38% | 57.31% | 0.18% | 84.75% | 4/85 | 0/94 |

Random-sample CS-WER gain against penalty 0 (paired bootstrap, 1,000 rounds):

- 1: -5.77 pp, p=0.8881, 95% CI [-16.22, +0.00] pp
- 2: -7.69 pp, p=0.9251, 95% CI [-18.75, +0.00] pp
- 3: -9.62 pp, p=0.9680, 95% CI [-20.01, -1.85] pp
- 4: -17.31 pp, p=0.9610, 95% CI [-37.10, -2.94] pp
- 6: -53.85 pp, p=0.9930, 95% CI [-92.51, -21.42] pp
- 8: -109.62 pp, p=0.9970, 95% CI [-176.92, -55.00] pp

No penalty met the selection rule.

## Rate-matched random control

Mean over 5 seeds, with the range in brackets.

| Penalty | Near-miss fixed | Other fixed | Bug CS-WER | Random WER | Clean damage |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4 [0, 1] | 0 [0, 0] | 121.39% [120.79%, 122.28%] | 7.66% [7.63%, 7.70%] | 0.15% [0.00%, 0.50%] |
| 2 | 0.4 [0, 1] | 0 [0, 0] | 121.78% [120.79%, 122.77%] | 7.71% [7.58%, 7.77%] | 0.65% [0.00%, 1.00%] |
| 3 | 0.6 [0, 1] | 0 [0, 0] | 122.38% [121.29%, 123.27%] | 7.85% [7.80%, 7.90%] | 1.35% [1.00%, 1.75%] |
| 4 | 0.8 [0, 2] | 0 [0, 0] | 122.67% [121.78%, 124.26%] | 8.01% [7.92%, 8.08%] | 3.45% [3.00%, 4.00%] |
| 6 | 2 [1, 3] | 0 [0, 0] | 122.87% [120.79%, 126.24%] | 9.02% [8.73%, 9.29%] | 10.25% [9.00%, 11.25%] |
| 8 | 2 [0, 4] | 0 [0, 0] | 130.40% [125.25%, 135.15%] | 12.34% [12.02%, 12.77%] | 28.40% [25.50%, 30.25%] |

Continuation near-miss fixes above every random seed at the same penalty: 1: no, 2: yes, 3: yes, 4: no, 6: no, 8: no.

## Conclusion

This section was written after the sweep. The sections above are the script output.

The continuation gate is rejected. No penalty met the selection rule. At penalty 2, bug-mining CS-WER fell from 120.79% to 118.32%, but clean damage was 0.75%, above the 0.5% limit. Random-sample WER also rose from 7.60% to 7.80%, and random-sample CS-WER rose from 55.77% to 61.54%. The random sample is the only set here that was not used to design the gate.

The gate removed most of the harm a uniform penalty does. At penalty 2, random-sample insertions stayed at 0.84% (1.77% with the uniform penalty), and random-sample WER rose by 0.20 pp instead of 1.16 pp. The remaining damage is mostly substitutions. A forced continuation turns a correct word into a longer wrong one, as in clean controls where TỦ became TỦA and HUÝT became HUUT. The word_start control behaved like the uniform penalty, and its insertions rose to 1.79% at penalty 2.

The near-miss effect is real but very small. At penalties 2 and 3, the continuation gate repaired more near-miss spans than every rate-matched random seed: 3 and 4, against a seed maximum of 1. No setting repaired any of the 94 other spans. Even so, at most 4 of 85 near misses were recovered, and larger penalties recovered fewer. This agrees with the stored margins (median 5.7 nats between blank and the target piece). Blank does not narrowly beat these pieces on one frame. It dominates them, so a per-frame offset small enough to keep ordinary words intact cannot reach them.

Decoder-side blank calibration for these errors is exhausted at the greedy level. The near misses show that the right pieces are available to the Joiner, but recovering them would need a change in how the Joiner scores blank against continuation pieces on those frames. That points to Joiner or encoder-side analysis, or training data. It does not point to a decoding constant.
