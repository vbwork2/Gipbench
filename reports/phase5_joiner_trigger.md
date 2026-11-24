# Phase 5: learned Joiner trigger for near misses (TODO_rea 2.2)

Diagnostic part of the extra pool only. Rules, grid, controls and pass rules are in the docstring of `scripts/09_joiner_trigger.py` and were committed before any probe was trained. Neither heldout was read.

Primary spans (near misses with a replicated blank-led event, wrong at the penalty-0 baseline): 430, of which 218 hold a rank-2 event (the ceiling of a blank penalty). Clean rows usable for damage: 254.

## Held-out fold 0

Train rows 479, test rows 472, positives 323, negatives 24645.

Training-fold grid:

| Penalty | Rate | Primary fixed | CS fixed | CS broken | Clean damage | Random WER | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.1% | 12/219 | 14 | 1 | 0.77% | 10.25% | False |
| 4 | 0.5% | 12/219 | 14 | 6 | 3.85% | 10.25% | False |
| 8 | 0.1% | 12/219 | 14 | 16 | 3.08% | 10.25% | False |
| 8 | 0.5% | 13/219 | 15 | 34 | 13.08% | 10.43% | False |
| baseline | | | | | | 10.25% | |

Fold fails: No eligible setting on the training fold.

## Held-out fold 1

Train rows 472, test rows 479, positives 331, negatives 25226.

Training-fold grid:

| Penalty | Rate | Primary fixed | CS fixed | CS broken | Clean damage | Random WER | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.1% | 11/211 | 15 | 8 | 0.81% | 8.88% | False |
| 4 | 0.5% | 11/211 | 14 | 11 | 1.61% | 8.88% | False |
| 8 | 0.1% | 9/211 | 10 | 20 | 1.61% | 8.88% | False |
| 8 | 0.5% | 8/211 | 10 | 41 | 14.52% | 8.88% | False |
| baseline | | | | | | 8.88% | |

Fold fails: No eligible setting on the training fold.

## Result

Registered pass (P1, P2 and P3 on both held-out folds): not passed.

## Post-hoc limitation check (not registered; changes no outcome)

`scripts/diagnostics/fpt_trigger_confound.py` refits the probe of each fold exactly as above and scores the baseline-trajectory blank frames at the 0.1% tau. Firing rate on the held-out fold (fold 0 / fold 1):

| Blank frames | Frames | Fires |
|---|---:|---:|
| Rows without events (the negatives' population) | 25 226 / 24 645 | 1.64% / 2.52% |
| Event rows, more than 10 frames from any listed event | 16 834 / 18 084 | 4.10% / 4.50% |
| Event rows, within 10 frames of an event (event frames excluded) | 3 773 / 3 784 | 23.9% / 27.5% |
| Replicated event frames | 331 / 323 | 41.4% / 44.0% |

On the training fold the negatives fire at 0.10% by construction and the event frames at 91% and 100%. Two readings: the 0.1% calibration does not transfer (held-out negatives fire 16 to 25 times as often), and the probe marks the neighbourhood of an English word rather than the event frame, so the penalty also lands on the frames around the target. That matches the CS words newly broken in the grid. A row-level component exists (far frames of event rows fire about twice as often as rows without events) but is smaller.
