# Follow-up review of inconclusive FPT errors

Development-only audit on doof-ferb/fpt_fosd at revision e6ff950456eeaef7a257e79bf7f4597bb39c264c. Seed 42 selected 24 distinct recordings from 171 inconclusive spans.

| Final-stack top-10 pieces at ±1 frame | Inconclusive population | Reviewed recordings |
|---|---:|---:|
| All pieces | 85 | 10 |
| Some pieces | 81 | 9 |
| No pieces | 5 | 5 |

The replay reproduced all published final-stack ranks in 24/24 recordings. At ±1, 10/10 cases with every piece individually in the top ten also had top-ten support on distinct increasing frames.

| Window | Greedy-prefix ordered top 10 | Reference-prefix ordered top 10 | Acoustic ordered top 10 |
|---|---:|---:|---:|
| ±1 | 10 | 12 | 3 |
| ±2 | 10 | 12 | 3 |
| ±4 | 10 | 12 | 3 |

At ±4 frames, 2/24 recordings had ordered rank-one support with the greedy prefix. 2/24 had ordered top-ten support exclusively with the reference prefix.

Ordered top-ten support requires a distinct increasing frame for each target piece within its window. It measures the availability of the pieces, while complete greedy recognition also depends on their scores and the intervening emissions.

## Independent ASR cross-check

- Qwen3-ASR-0.6B (5eb144179a02acc5e5ba31e748d22b0cf3e303b0). The aligned hypothesis matched the normalized reference word in 7/24 recordings.
- faster-whisper large-v3 (edaa852ec7e145841d8ffdb056a99866b5f0a478). The aligned hypothesis matched the normalized reference word in 10/24 recordings.
- Both matched in 4, one matched in 9, and neither matched in 11. Of the last group, 4 had the same aligned substitution or deletion.
- Qwen language sensitivity on fpt_022456 (PASSWORD). Automatic language missed the target, and the Vietnamese setting matched it. This is a targeted follow-up outside the aggregate Qwen count.

The orthographic rule selected 24/24 target labels. ASR agreement triages label review. A human listening verdict is still pending.

## Ranked competitors on corroborated targets

Both independent ASRs matched these targets, and all pieces had ordered top-ten support with the greedy prefix. Each competitor leads at the best target frame within ±1.

| Recording | Target word | Pieces below rank one and their leading competitor |
|---|---|---|
| fpt_022372 | CHAPLIN | `▁CHA` (rank 2) vs `▁JA` |
| fpt_022073 | CROWELL | `L` (rank 2) vs `<blk>` |
| fpt_024310 | ANDY | `Y` (rank 2) vs `I` |
| fpt_010986 | SMITH | `H` (rank 2) vs `<blk>` |

## Sample details

| Recording | Target | Lexical rule | Greedy-prefix ordered top 10 at ±1 / ±4 | Qwen | Whisper |
|---|---|---|---|---|---|
| fpt_022372 | CHAPLIN | orthographic | yes / yes | match | match |
| fpt_011793 | SANDWICH | orthographic | no / no | substitution (HÒA) | match |
| fpt_022254 | DEAR | orthographic | no / no | substitution (ĐI) | substitution (CHÀO) |
| fpt_022073 | CROWELL | orthographic | yes / yes | match | match |
| fpt_019560 | SAMPRAS | orthographic | no / no | substitution (SAM) | substitution (SÁM) |
| fpt_022426 | JULIET | orthographic | no / no | substitution (LỊCH) | match |
| fpt_017165 | KATHARINE | orthographic | yes / yes | match | substitution (CATHERINE) |
| fpt_000299 | DAVE | orthographic | no / no | substitution (ĐÂY) | substitution (ĐÂY) |
| fpt_022241 | AIR | orthographic | no / no | deletion | substitution (E) |
| fpt_016307 | PICTURE | orthographic | yes / yes | substitution (PICTUREER) | match |
| fpt_003847 | JAYA | orthographic | no / no | match | substitution (ZAYA) |
| fpt_022095 | COUP | orthographic | no / no | substitution (KHẮP) | substitution (KHẮP) |
| fpt_019698 | TECHCOMBANK | orthographic | yes / yes | substitution (BÁNH) | substitution (MẠNH) |
| fpt_018699 | GIGGS | orthographic | no / no | substitution (GINK) | substitution (GINK) |
| fpt_020019 | PIXEL | orthographic | no / no | match | substitution (SÁU) |
| fpt_022456 | PASSWORD | orthographic | yes / yes | substitution (ตวอต) | match |
| fpt_022734 | INDEX | orthographic | no / no | substitution (INDUST) | substitution (INDUS) |
| fpt_024310 | ANDY | orthographic | yes / yes | match | match |
| fpt_022643 | BARCA | orthographic | no / no | substitution (PAKISTAN) | substitution (SÁT) |
| fpt_010986 | SMITH | orthographic | yes / yes | match | match |
| fpt_016579 | ENEMIES | orthographic | no / no | substitution (ANIMIST) | match |
| fpt_022602 | KATHERINE | orthographic | yes / yes | substitution (CATHERINE) | substitution (CATHERINE) |
| fpt_022097 | BAGHDAD | orthographic | no / no | substitution (BUDA) | substitution (ĐÀ) |
| fpt_015213 | LAYER | orthographic | yes / yes | substitution (WALEE) | match |

Full references, baseline hypotheses, transcriptions and WAVs are in the local review directory. The recording ID matches the WAV basename.

The sample deliberately overrepresents spans with no final-stack top-ten pieces. Its counts describe this review sample. The original Phase 2 population remains the input to the encoder, Joiner and fine-tuning decisions.
