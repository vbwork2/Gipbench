# Phase 6: static edit of Joiner output rows in the low-energy subspace (TODO_rea 2.3)

Diagnostic part of the extra pool only. Rules, grid, controls, pass rules and how a result is read are in the docstring of `scripts/10_joiner_null_edit.py`, committed before any edit was fitted. Neither heldout was read.

Replicated events 654, primary spans 430, clean rows usable 254.

## Held-out fold 0

Train rows 479 (323 events, 49314 ordinary frames), test rows 472 (331 events).

Static stage (baseline-trajectory frames; admissible needs train hit >= 10% and train change < 1%):

| r | alpha | Pieces edited | Test events covered | Train hit | Train change | Test hit | Test change | Admissible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 1 | 80 | 88.22% | 43.65% | 0.66% | 9.67% | 0.73% | True |
| 0.25 | 2 | 80 | 88.22% | 75.85% | 2.41% | 17.52% | 2.78% | False |
| 0.5 | 1 | 80 | 88.22% | 47.06% | 0.55% | 11.78% | 0.68% | True |
| 0.5 | 2 | 80 | 88.22% | 81.73% | 1.67% | 22.36% | 1.91% | False |

Training-fold decodes:

| r | alpha | Primary fixed | CS fixed | CS broken | Clean damage | Random WER | Event-row VI errors | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 1 | 22/219 | 25 | 32 | 14.62% | 10.86% | 148 | False |
| 0.5 | 1 | 30/219 | 31 | 11 | 3.85% | 10.37% | 144 | False |
| baseline | | | | | | 10.25% | 136 | |

Fold fails: No eligible setting on the training fold.

## Held-out fold 1

Train rows 472 (331 events, 48736 ordinary frames), test rows 479 (323 events).

Static stage (baseline-trajectory frames; admissible needs train hit >= 10% and train change < 1%):

| r | alpha | Pieces edited | Test events covered | Train hit | Train change | Test hit | Test change | Admissible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 1 | 77 | 90.09% | 41.39% | 0.64% | 8.05% | 0.80% | True |
| 0.25 | 2 | 77 | 90.09% | 72.81% | 2.35% | 18.58% | 3.55% | False |
| 0.5 | 1 | 77 | 90.09% | 43.81% | 0.60% | 11.15% | 0.71% | True |
| 0.5 | 2 | 77 | 90.09% | 77.64% | 1.66% | 20.74% | 2.31% | False |

Training-fold decodes:

| r | alpha | Primary fixed | CS fixed | CS broken | Clean damage | Random WER | Event-row VI errors | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 1 | 34/211 | 34 | 21 | 9.68% | 9.09% | 150 | False |
| 0.5 | 1 | 32/211 | 33 | 23 | 12.10% | 9.19% | 149 | False |
| baseline | | | | | | 8.88% | 146 | |

Fold fails: No eligible setting on the training fold.

## Result

Registered pass (P1, P2 and P3 on both held-out folds): not passed.

A failure closes static edits of `joiner.output_linear` for direction A; it does not close X2.
