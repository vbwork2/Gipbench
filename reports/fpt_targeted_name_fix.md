# Gipformer FPT reviewed-name regression

Development check **PASS**. Same INT8 decoder and batch size of 1 in both runs. The candidate supplies 4 reviewed name phrases at hotword score 3.

| Development recording | Target phrase | Baseline span | Candidate span |
|---|---|---|---|
| fpt_022372 | CHAPLIN | JAPLIN | CHAPLIN |
| fpt_022073 | CROWELL | CROWELL | CROWELL |
| fpt_024310 | ANDY MURRAY | ANDIMURAI | ANDY MURRAY |
| fpt_010986 | SMITH | SMIT | SMITH |

| Split | Decoder | WER | N-WER | CS-WER | Insertion rate |
|---|---|---:|---:|---:|---:|
| Bug mining | baseline | 21.39% | 5.99% | 116.83% | 4.33% |
| Bug mining | candidate | 20.98% | 5.91% | 114.36% | 4.33% |
| Clean control | baseline | 0.27% | 0.27% | N/A | 0.11% |
| Clean control | candidate | 0.27% | 0.27% | N/A | 0.11% |

- Exact target phrases 1/4 to 4/4.
- Newly incorrect clean recordings 0/400.
- Clean recordings with a newly introduced erroneous target phrase 0/400.
- Clean insertion-rate change +0.000 percentage points.

The name list was selected from the reviewed development cases. This is a targeted regression check for an opt-in decoding preset. The previous heldout was outside the run.

## Additional occurrences in the development pool

Three other development recordings contained these exact reference names. They are outside the 100 bug-mining and 400 clean-control recordings above.

| Recording | Reference phrase | Baseline span | Candidate span |
|---|---|---|---|
| fpt_004676 | SMITH | BẢO | (deleted) |
| fpt_016443 | ANDY MURRAY | ANDY MURAIS | ANDY MURAIS |
| fpt_021149 | SMITH | SUMIS | SMITH |

Exact matches improved from 0/3 to 1/3 on these additional recordings.
