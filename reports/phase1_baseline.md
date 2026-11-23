# Gipformer 1.5 ONNX baseline

Dataset doof-ferb/fpt_fosd at revision e6ff950456eeaef7a257e79bf7f4597bb39c264c.

This run covers the provisional scored pool. No heldout results or speaker-disjoint claims apply.

| Split | Decoder | Utterances | WER | CER | N-WER | CS-WER | Ins | Del | Sub |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| provisional_pool | greedy | 5550 | 8.68% | 5.86% | 7.76% | 78.82% | 0.83% | 1.90% | 5.94% |
| provisional_pool | modified beam (4) | 5550 | 8.67% | 5.86% | 7.75% | 78.82% | 0.90% | 1.87% | 5.90% |

Lexical CS candidates occur in 587 utterances.
Greedy returned zero word errors on 3,070 utterances.
Top detected terms by utterance: BUSH (11), TOKYO (9), NEW (8), TAXI (8), VTV (6), TV (6), SMITH (6), YORK (5), TOUR (5), FAX (5)

CS attribution is lexical and candidate labels may need reference review.
