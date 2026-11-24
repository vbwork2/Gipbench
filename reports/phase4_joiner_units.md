# Phase 4 Joiner unit analysis

Development only. The heldout stayed sealed. The rules are in the docstring of `scripts/08_joiner_decomposition.py` and were fixed before the analysis ran.

The decomposition is exact for the fp32 output layer applied to the INT8 encoder and decoder outputs. On 3082 frames of 30 random-sample recordings this layer picks the same argmax as the deployed INT8 Joiner on 99.4% of frames. The largest logit difference per frame has a median of 0.75 nats.

Phase 2 events (near-miss pieces below rank one with blank leading): 104. Replicated under this analysis's own greedy replay: 65.
Blank-over-piece margin at replicated events, quantiles 10/25/50/75/90%: 2.2 / 3.5 / 6.4 / 9.1 / 13.0 nats.
Ordinary frames: 32542 blank frames from clean controls (selection); 51671 frames from the random sample (side effects).

## Concentration

Share of the summed positive excess contribution (event minus ordinary frames) carried by the top-k units of 512, using all events (descriptive, in-sample):

| k | 1 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| share | 4.5% | 18.3% | 30.1% | 43.9% | 64.1% | 81.1% | 96.0% |

Effective number of units (participation ratio of the excess): 127 of 512. A uniform spread would give 512 and a single unit would give 1.

## Held-out test with K = 10 units

| Held-out fold | Events | Median share carried | Random 10-unit sets: median, 95th pct | p | Target top-1 after zeroing | Random-sample frames with changed argmax |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 172.2% | -0.1%, 30.1% | 0.0010 | 37.5% | 11.1% (3677 blank, 2047 piece frames) |
| 1 | 33 | 173.8% | -1.8%, 34.6% | 0.0010 | 27.3% | 11.1% (3631 blank, 2105 piece frames) |

Pooled held-out median share: 173.8%.

| Rule | Result |
|---|---|
| P1 pooled median share at least 50% | pass |
| P2 changed-argmax rate below 1%, both folds | fail |
| P3 random-set p below 0.01, both folds | pass |

Units chosen on all events, in-sample, for reference only (median share 172.7%):

| Unit | Mean contribution at events | Mean at ordinary blank frames | Blank output weight |
|---:|---:|---:|---:|
| 149 | +2.063 | -0.001 | +0.898 |
| 17 | +2.165 | +0.352 | -2.110 |
| 251 | +1.169 | -0.531 | +0.240 |
| 96 | +0.887 | -0.587 | -5.098 |
| 483 | +1.762 | +0.428 | -1.906 |
| 14 | +0.494 | -0.762 | +2.954 |
| 134 | +0.532 | -0.629 | -2.976 |
| 300 | +2.084 | +0.947 | +2.894 |
| 434 | +0.880 | -0.123 | -0.148 |
| 282 | +1.259 | +0.417 | -2.481 |

**At least one rule failed.** The unit-level direction is closed.

## Conclusion

This section was written after the analysis. The sections above are the script output.

Static edits to these Joiner units are closed by rule P2. The result also contradicts the diffuse picture that was expected going in.

**The margin is concentrated, and the concentration is real.** Ten units chosen on one half of the recordings carry a median 173% of the blank-over-piece margin on the other half. The other units net push the margin the other way, which is why the share exceeds 100%. Random sets of ten units carry a median of about 0% (95th percentile 30–35%), so the selection does not come from chance (p = 0.001, the smallest value 1,000 draws allow). The two halves picked nine of the same ten units (14, 17, 96, 134, 149, 251, 300, 434, 483), so the choice is stable. The positive-excess curve in the concentration table is gross where the held-out share is net: the top ten units hold 30% of the summed positive excess and the effective unit count is 127. Both statements are true. The margin is a net of many opposing contributions, and a small set of units dominates the net.

**These units are central to ordinary decoding, not a switch for near misses.** Zeroing them changes the argmax on 8.2% of blank frames and 30% of non-blank frames in the random sample (51,671 frames, against 65 events). In exchange the target piece becomes the top choice at only 27–38% of held-out events (9 of 33 and 12 of 32). Several selected units have large blank output weights (unit 96: −5.1, unit 14: +3.0, unit 134: −3.0, unit 300: +2.9), and zeroing a unit removes its term from every logit at once, so the edit rewrites many ordinary decisions.

This matches the earlier result in the Qwen write-up that a hook worked while a direct weight edit broke ordinary sentences. The hook worked there because the frames to act on were identifiable (language-decision tokens). The blank experiments here found no structural trigger either: the word-continuation gate in `reports/phase3c_conditional_blank.md` repaired at most 4 of 85 near misses and still damaged clean controls.

**Not tested.** An ablation applied only at frames flagged by a learned trigger was not tried. With 65 replicated events from 100 recordings, a trigger would be fit on far too little data, and any result on the same events would be in-sample. A usable test needs more labeled near-miss frames than the development set contains.

**Consequence.** Within the Joiner, decoder-side and unit-level repairs are now both rejected on development data: a global or gated blank penalty (phases 3b and 3c) and static unit ablation (this phase). The near misses show that the right pieces are available to the Joiner. The fix has to change how the Joiner scores blank against them across the whole model, which is what training does. The planned X2 experiment (train only the decoder and Joiner) targets this directly, and the near-miss repair count can be reported alongside WER for each checkpoint.
