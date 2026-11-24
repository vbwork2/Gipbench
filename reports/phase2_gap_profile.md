# Profile of the gap-like FPT error spans

Development only, from stored ranks and development reference texts. Heldout untouched. **Exploratory**: the features were chosen after reading 14 gap-like spans, nothing is registered and no threshold is applied. See the docstring of `scripts/diagnostics/fpt_gap_profile.py`.

| | near_miss | chain_reachable | chain_unreachable | start_local | true_gap |
|---|---:|---:|---:|---:|---:|
| Spans | 183 | 67 | 43 | 55 | 8 |
| Distinct reference words | 161 | 62 | 42 | 55 | 8 |
| Median pieces per word | 4 | 4 | 5 | 4 | 2.0 |
| Words of five or more pieces | 74 (40%) | 25 (37%) | 26 (60%) | 20 (36%) | 0 (0%) |
| Median rank of the weakest piece | 4 | 51 | 272 | 71 | 651.5 |
| Weakest piece below rank 100 | 0 (0%) | 20 (30%) | 26 (60%) | 22 (40%) | 6 (75%) |
| Weakest piece below rank 100 acoustically (no predictor context) | 69 (38%) | 39 (58%) | 32 (74%) | 38 (69%) | 7 (88%) |
| Weakest piece has at most two letters | 180 (98%) | 65 (97%) | 43 (100%) | 55 (100%) | 5 (62%) |
| Blank leads at the weakest piece | 90 (49%) | 56 (84%) | 33 (77%) | 39 (71%) | 4 (50%) |
| Word occurs in one development row only | 127 (69%) | 51 (76%) | 31 (72%) | 50 (91%) | 6 (75%) |
| Word occurs in five or more development rows | 10 (5%) | 3 (4%) | 0 (0%) | 0 (0%) | 0 (0%) |

Weakest pieces most often seen in the spans that stay missing even with the reference prefix (start_local, true_gap, chain_unreachable): `S` (7), `E` (5), `D` (4), `AR` (4), `R` (4), `Y` (4), `LA` (3), `J` (3), `T` (3), `AL` (3).

Reading limits: the weakest piece of a near miss is still in the top ten, so its rank row compares across groups but not as a pass mark. The occurrence count is a rarity proxy inside the development pool. The training frequency of the pieces and words is not available here, so whether these pieces are rare in the model's training data is not answered. A tokenizer check on the training text is the next step if the profile points that way.
