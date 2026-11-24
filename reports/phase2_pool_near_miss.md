# Phase 2 near-miss count over the whole development pool

Development only. The heldout stayed sealed: the pool hash was checked against the lock and no heldout row was diagnosed. The rules are in the docstring of `scripts/diagnostics/fpt_pool_near_miss.py` and were fixed before the run.

Phase 2 diagnosed the 100 bug-mining recordings, which were chosen for their most repeated confusions. This run diagnoses the other 179 development recordings that have a CS substitution or deletion (186 spans), with the same logit lens, the same emitted-greedy-prefix context and the same categories. The bug-mining group is re-classified from its stored ranks, which reproduces the published counts (85 near miss, 2 encoder bug, 1 predictor override, 5 true gap, 86 inconclusive; 104 blank-led pieces).

| Category | Bug-mining (100 rec.) | Other development (179 rec.) | Pooled (279 rec.) |
|---|---:|---:|---:|
| Spans | 179 | 186 | 365 |
| Near miss | 85 (47.5%) | 98 (52.7%) | 183 (50.1%) |
| Encoder bug | 2 (1.1%) | 3 (1.6%) | 5 (1.4%) |
| Predictor override | 1 (0.6%) | 3 (1.6%) | 4 (1.1%) |
| True gap | 5 (2.8%) | 3 (1.6%) | 8 (2.2%) |
| Inconclusive | 86 (48.0%) | 79 (42.5%) | 165 (45.2%) |
| Near-miss spans where blank leads a non-top piece | 58 | 59 | 117 |
| Blank-led near-miss pieces (the unit of `phase4_joiner_units.md`) | 104 | 100 | 204 |
| Recordings with such a piece | 43 | 58 | 101 |
| Distinct reference words among near-miss spans | 74 | 92 | 161 |

The near-miss share does not come from how bug-mining was selected. The 179 recordings that were never ranked by repeated confusion give 52.7%, above the 47.5% of the selected set. Both exceed the 30% score-calibration threshold, and neither reaches the 25% encoder-sweep threshold (1.1% and 1.6%). Phase 2 stands on a population twice as large.

## What this changes for a trigger

Phase 4 had 104 blank-led pieces, of which 65 replicated under its own greedy replay (62.5%). The same rate on 204 pieces gives about 128 events, an estimate and not a measurement. Held out by recording, that is about 64 events per fold, close to the 32 and 33 per fold that Phase 4 had. A trigger fit on this set is still small, and it needs the Joiner and the replay to be run on the new recordings before any number is quoted. The set is diverse (161 distinct words over 183 spans), so a split by word as well as by recording is possible.

The development pool is not the whole dataset. The FPT snapshot has 25,917 rows and the pool holds 4,550, so about 20,000 rows have never been scored. Scoring them would raise the count several times, at the cost of a decode pass and a fresh exclusion check against the heldout hashes.

## Not done

- The Joiner replay that gives the 65 of 104 figure was not run on the new recordings.
- Bug-mining was not re-run. Its categories come from stored ranks, so a difference between this run's environment and the original one would show up only in the new group. The close agreement of the two groups is the only check.
- The inconclusive 45% still lacks a listening verdict (see `phase2_inconclusive_review.md`).
