# Gipformer FPT reviewed-name development probe

In-sample probe, criteria **met**. Same INT8 decoder and batch size of 1 in both runs. The candidate supplies 4 reviewed name phrases at hotword score 3.

The four phrases were taken from the four recordings in the first table, so the exact-phrase gain is in-sample. It shows that phrase biasing can reach those spans. It does not show that the preset generalizes. This report replaces an earlier version titled as a fix.

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

The bug-mining rows contain the four target recordings, so the bug-mining change is also in-sample. The clean-control rows are the useful part: they check for collateral damage.

- Exact target phrases 1/4 to 4/4 (in-sample; CROWELL was already correct).
- Newly incorrect clean recordings 0/400.
- Clean recordings with a newly introduced erroneous target phrase 0/400.
- Clean insertion-rate change +0.000 percentage points.

## Other development recordings with the same names

Three other development recordings contained these exact reference names. They are outside the 100 bug-mining and 400 clean-control recordings above, and they were not used to choose the names.

| Recording | Reference phrase | Baseline span | Candidate span |
|---|---|---|---|
| fpt_004676 | SMITH | BẢO | (deleted) |
| fpt_016443 | ANDY MURRAY | ANDY MURAIS | ANDY MURAIS |
| fpt_021149 | SMITH | SUMIS | SMITH |

Exact matches improved from 0/3 to 1/3 on these recordings. This is the closest thing to out-of-sample evidence in this report. In one recording the biased decoder dropped the span instead of producing the name, and in another the biased phrase was not recovered at all. Three recordings are too few to estimate a rate.

## Status

- The heldout was outside this run, and it was already used once for the hotword acceptance run. A new sealed heldout is needed before any claim about this preset.
- The preset lives in `configs/diagnostics/fpt/`, not in `configs/bench/`. One of the names (SMITH) also occurs in benchmark text. The bench guard only compares a hotword `catalog` path with the benchmark's own catalog and does not inspect plain word lists, so a preset that declares `provenance: external` would pass it. It must not be run on benchmarks.
