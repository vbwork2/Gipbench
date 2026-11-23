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

The FPT download has filenames and transcripts, with no verified speaker IDs. The split builder accepts an `utt_id` and `speaker_id` TSV, checks group and recording overlap, and binds the development and heldout manifests to their checksums. It removes the speakers reserved for heldout from the scored development pool.

```sh
python scripts/01_build_manifests.py --speaker-map artifacts/speakers.tsv
python scripts/02_eval_baseline.py --model-dir artifacts/base_model
python scripts/03_diagnose_bug_vs_gap.py --icefall /path/to/icefall --calibration-dir /path/to/calibration_wavs --parity-quantize int8
python scripts/04_eval_hotwords.py
```

Use ten or more calibration WAVs outside the FPT pool. On Windows, use the Ubuntu WSL environment with Icefall and k2 for the checkpoint audit and PyTorch diagnostics if the Windows PyTorch libraries cannot load. The Phase 2 parity check defaults to FP32 ONNX and also supports `--parity-quantize int8`. Ten out of ten transcripts must match. The diagnostic programs verify development manifest hashes without opening heldout.

For logit lens, Zipformer stack wrappers already restore the input time grid. The readout reconstructs the cumulative encoder output using the latest available channel from each stack, pools with the model's final downsampler, and checks that the last readout reproduces the native encoder output. Early stages still require zero padding for channels that do not exist yet. The reported absence of a BPE target from the top ten is evidence about this probe. An unknown BPE piece is recorded separately from a representable acoustic target.

## Conditional analysis and acceptance

The activation collector takes 500–1,000 clean utterances from the sealed development pool once the encoder diagnosis threshold is met. It records signed FFN activation percentiles, output weight norms and each target stack's median residual norm in `artifacts/stats/encoder_stats.npz`. Sparse delta helpers use per-tensor flat indices and bind each saved delta to the audited checkpoint. The in-memory patch limit is 0.5% of model parameters. Rollback restores saved parameter values exactly.

The acceptance script takes a plan JSON through `--plan`. Its `systems` object maps each system name to the ONNX `model_dir` and decoding `method`, with optional `quantize`, `hotwords` and `hotwords_score`. Paths in the plan are relative to its directory. The `baseline` and `primary` entries name the matched decoder comparison. A candidate with random-direction evidence supplies its `candidate_fix_rate` and the observed `random_fix_rates` under `candidate_evidence`. The script pins the plan and model-file hashes before reading the heldout and computes the five gates without selecting a new candidate on that set.
