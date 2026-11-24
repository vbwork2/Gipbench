# Gipformer 1.5 mechanistic analysis

The analysis uses the Gipformer 1.5 Zipformer RNN-T checkpoint and the two FPT FOSD parquet shards at the revision pinned in `configs/pipeline.yaml`. The checkpoints and dataset stay outside Git. The audit opens the base checkpoint on CPU and verifies the SentencePiece IDs against the accompanying token table.

## Audit and decoding baseline

Run from the worktree root. Use a Python environment with the `mechanistic` project extra. Put the checkpoint, config, tokenizer, token table and ONNX triplet in the model directory and put the two parquet shards under `artifacts/fpt_fosd`.

```sh
python scripts/00_audit_model.py --model-dir artifacts/base_model
python scripts/01_build_manifests.py --model-dir artifacts/base_model
python scripts/02_eval_baseline.py --model-dir artifacts/base_model
```

The audit generates `configs/model_map.json` and `reports/phase0_audit.json`. Before sealing the splits, the baseline report measures the provisional pool. The automatic label audit checks for malformed or empty references, duplicate audio and invalid recordings. A reviewed exclusions JSONL with `utt_id` and `reason` can be supplied during sealing to remove transcript errors that require listening.

## Development splits and decoder parity

The FPT download has filenames and transcripts, with no verified speaker IDs. For this experiment, the split builder stratifies by lexical CS labels and checks overlap by utterance ID, source row and exact audio checksum. Each recording belongs to only one split. A person may appear in multiple splits, so the evaluation measures generalization to new recordings under that design. The builder binds the final manifests to their checksums and removes the heldout rows from the development pool. With a verified speaker map, `--speaker-map` can instead seal disjoint speaker groups.

```sh
python scripts/01_build_manifests.py --model-dir artifacts/base_model --split-by-recording
python scripts/02_eval_baseline.py --model-dir artifacts/base_model --report reports/phase1_development_baseline.md
python scripts/03_diagnose_bug_vs_gap.py --icefall /path/to/icefall --calibration-dir /path/to/calibration_wavs --parity-quantize int8
python scripts/04_eval_hotwords.py
```

Use ten or more calibration WAVs outside the FPT pool. On Windows, use the Ubuntu WSL environment with Icefall and k2 for the checkpoint audit and PyTorch diagnostics if the Windows PyTorch libraries cannot load. The Phase 2 parity check defaults to FP32 ONNX and also supports `--parity-quantize int8`. Ten out of ten transcripts must match. The diagnostic programs verify development manifest hashes without opening heldout.

For logit lens, Zipformer stack wrappers already restore the input time grid. The readout reconstructs the cumulative encoder output using the latest available channel from each stack, pools with the model's final downsampler, and checks that the last readout reproduces the native encoder output. Early stages still require zero padding for channels that do not exist yet. Forced reference alignment locates the target pieces. The primary readout uses the Predictor prefix emitted by greedy decoding before each frame. The report also includes the category from the reference prefix for comparison. Absence of a BPE target from the top ten is evidence about this probe. An unknown BPE piece is recorded separately from a representable acoustic target. A span whose every piece is already in the final-stack top ten is a near miss: the loss happens after the encoder, in Joiner scores or the search. A near-miss share of 30% or more sets `prioritize_score_calibration`. To apply the current categories to a stored report without the model, run `python scripts/diagnostics/fpt_phase2_reclassify.py` (it rewrites `reports/phase2_bug_vs_gap.json` unless `--out` is given).

`scripts/06_eval_blank_penalty.py` sweeps the RNN-T blank penalty for greedy and modified beam decoding. It uses bug mining, clean control and a seeded random sample of 500 other development recordings, whose pool hash is checked against the seal. For each Phase 2 span, it records whether the span changed from wrong to right, split into near-miss and other spans, as a check that the decoder change acts through the near-miss mechanism. The selection rule is fixed in the script docstring. Hypotheses are cached in `artifacts/blank_penalty`. The result is in `reports/phase3b_blank_penalty.md`: no penalty qualified.

`scripts/07_eval_conditional_blank.py` decodes with `gipformer_mech.onnx_greedy`, a NumPy/ONNX greedy decoder that can apply the blank penalty only on frames that meet a condition. Its features come from `gipformer_mech.fbank`, a NumPy implementation of the Kaldi filterbank. It does not depend on `kaldi-native-fbank`. The script caches encoder outputs once, checks parity against sherpa-onnx greedy, then compares a word-continuation gate with a word-start gate, a uniform penalty and rate-matched random gates. It needs `onnxruntime` and uses `--workers` processes. The result is in `reports/phase3c_conditional_blank.md`: no gate qualified.

`scripts/08_joiner_decomposition.py` splits the blank-over-piece margin of the Phase 2 near-miss events over the 512 Joiner hidden units (`gipformer_mech.joiner_probe`). The Joiner is `W @ tanh(enc + dec) + b` with the projections inside the ONNX encoder and decoder, so the split is exact for the fp32 output layer read from `model.pt`. It reuses the encoder outputs cached by script 07 and needs no audio. Units are chosen on one half of the recordings and tested on the other, against random unit sets and against the side effects of zeroing them on a random development sample. The result is in `reports/phase4_joiner_units.md`: the margin is concentrated in a stable set of ten units, but zeroing them changes the argmax on 8% of blank frames and 30% of piece frames, so static unit edits are closed.

## Conditional analysis and acceptance

The activation collector takes 500–1,000 clean utterances from the sealed development pool once the encoder diagnosis threshold is met. It records signed FFN activation percentiles, output weight norms and each target stack's median residual norm in `artifacts/stats/encoder_stats.npz`. Sparse delta helpers use per-tensor flat indices and bind each saved delta to the audited checkpoint. The in-memory patch limit is 0.5% of model parameters. Rollback restores saved parameter values exactly.

The acceptance script takes a plan JSON through `--plan`. Its `systems` object maps each system name to the ONNX `model_dir` and decoding `method`, with optional `quantize`, `hotwords` and `hotwords_score`. Paths in the plan are relative to its directory. The `baseline` and `primary` entries name the matched decoder comparison. A candidate with random-direction evidence supplies its `candidate_fix_rate` and the observed `random_fix_rates` under `candidate_evidence`. The script pins the plan and model-file hashes before reading the heldout and computes the five gates without selecting a new candidate on that set.

A hotword-only comparison can set `evaluation_kind` to `decoder_only`. This verifies that the baseline and candidate use the same ONNX model files and marks the neuron intervention gate G5 as inapplicable. Weight candidates use all five gates.

## Follow-up on inconclusive errors

`scripts/diagnostics/fpt_phase2_review.py prepare` selects a seeded sample from the sealed bug-mining split and writes WAVs and an ASR input manifest in `artifacts/phase2_review`. Use the existing `gipbench.infer.asr_hyps` entry point with that manifest and local model snapshots to produce `qwen.jsonl` and `whisper.jsonl` in the same directory. The `probe` stage uses the Icefall model arguments to replay the rank windows on the original audio. The `report` stage joins the local results by recording ID, verifies the hashes of the WAVs used by each ASR model, and writes `reports/phase2_inconclusive_review.md`.

`scripts/diagnostics/fpt_pool_near_miss.py` applies the Phase 2 logit lens to every other development recording that has a CS error, after checking the pool hash against the lock, and pools the result with the stored bug-mining findings. It needs the Icefall environment, resumes from `artifacts/pool_near_miss/findings.jsonl` and takes about a minute on CPU. The result is in `reports/phase2_pool_near_miss.md`: near misses are 50% of 365 spans, 52.7% in the recordings that bug-mining did not select.

`scripts/diagnostics/fpt_inconclusive_breakdown.py` re-reads the stored Phase 2 ranks (`reports/phase2_bug_vs_gap.json` and `artifacts/pool_near_miss/findings.jsonl`), matches the INCONCLUSIVE spans to the review profiles span by span and describes how far their missing pieces are. It needs the `gipformer_mech` imports (NumPy) but no model or audio. The result is in `reports/phase2_inconclusive_breakdown.md`.

`scripts/diagnostics/fpt_inconclusive_cascade.py` uses the same stored ranks to test, under a prediction written in its docstring, whether the INCONCLUSIVE spans are cascades: an earlier piece of the word lost its frame and the missing piece is reachable once the predictor sees the reference prefix. The result is in `reports/phase2_inconclusive_cascade.md`: the prediction held (110 of 165 spans are chains).

`scripts/diagnostics/fpt_span_lock.py` freezes the 365 development error spans, with their group, as `reports/phase2_span_lock.jsonl` and a checksum file next to it. It refuses to overwrite an existing lock. The docstring holds the definition of a repaired span and the prediction that a later checkpoint (X2) is judged against. `scripts/diagnostics/fpt_gap_profile.py` is an exploratory profile of the gap-like spans against the near misses, written to `reports/phase2_gap_profile.md`. Both read stored ranks only and need NumPy and the `gipformer_mech` imports, no model or audio.

## Reviewed-name decoding preset

`configs/diagnostics/fpt/gip15_mbs4_fpt_reviewed_names.yaml` enables context biasing for ANDY MURRAY, CHAPLIN, CROWELL and SMITH. It uses beam 4, hotword score 3.0 and one recording per batch. It is a diagnostic preset for the FPT development splits, not a benchmark system: the names were chosen from four development recordings, SMITH also occurs in benchmark text, and the bench guard does not inspect plain word lists. Keep it out of `configs/bench/`.

```sh
python scripts/diagnostics/fpt_targeted_names.py --model-dir models/gip15
```

The diagnostic compares the preset with unbiased decoding on the locked development manifests and writes `reports/fpt_targeted_name_probe.md`. The exact-phrase result is in-sample, because the names came from the scored recordings. The checks on newly incorrect clean recordings and introduced name triggers are the informative part. The previous heldout was used for a separate acceptance run and cannot test this preset.
