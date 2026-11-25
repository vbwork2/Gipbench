# GipBench research recovery and training validation

Started 2026-11-25 on branch `feture/rea` at `e488a9c`. This file is the single record for the recovery work.
Only results produced and checked **in this recovery** count as evidence. Earlier reports (TODO.md, TODO_rea.md,
reports/*) are listed as *claims* with their location; a claim is promoted to evidence only when re-run here.

Status (2026-11-25 10:10):

| Phase | Gate | Result |
| --- | --- | --- |
| 0 State | — | done (section 1) |
| 1 Pretrained parity | P1-A … P1-E | **PASSED** (section 2). The export + INT8 path reproduces the official model. |
| 2 Training correctness | P2-0 … P2-5 | **PASSED on CPU / fp32**, two rule exceptions explained. One recipe deviation confirmed (D1, loss warm-up). The fp16 / GradScaler path later ran cleanly in Phase 3 (no inf, scale 2 → 16). (section 3) |
| 3 Tiny-set memorization | registered rule | **PASSED, both arms** (GPU, 600 batches): train CS-WER 40.48 → 1.79 (A) / 2.68 (B), WER 5.65 → 0.27 / 0.33, term recall 69.2 → 98.7 / 97.3; 0 / 194 frozen tensors changed. Arm B (no loss warm-up) is **not** clearly faster than A. (section 5.1) |
| 4 Vietnamese control (X0) | X0's own tolerance | **FAILED on independent dev** (dev_v2_vi ΔWER +4.83 [4.38, 5.25]); cause not isolated (section 6). **This is the first failed gate.** |
| 5 CS adaptation | — | blocked by 4 |

**Single next action:** section 10.

---

## 1. Verified project state (Phase 0)

### 1.1 Code and environment

| Item | Value | How verified |
| --- | --- | --- |
| Branch / commit | `feture/rea` @ `e488a9c` (clean except untracked `.worktrees/`) | `git status`, `git log` |
| Training code | Icefall `egs/librispeech/ASR/zipformer/finetune.py`, patched by `src/gipbench/model/train/patch_finetune.py` into `finetune_cs.py`; driven by `src/gipbench/model/run.py` stage 4 through `train/icefall_runner.py` | read |
| Export code | Icefall `zipformer/export-onnx.py --enable-int8-quantization 1 --avg 1 --use-averaged-model 0` (`src/gipbench/model/export/onnx_export.py`) | read |
| Inference | sherpa-onnx, `src/gipbench/infer/adapters/sherpa_transducer.py`, via `python -m gipbench.infer.decode` | read |
| Icefall checkout | WSL `~/icefall` @ `f8c8578` | `git log -1` |
| FT venv | WSL `~/.venvs/gipbench-ft`: Python 3.11.18, torch 2.15.0+cu132, k2 1.33.0, lhotse (version printed in Phase 1 log) | runtime import |
| GPU | **Not available right now**: WSL reports `Failed to initialize NVML` and torch `cuda.is_available() == False`. Earlier `wsl --shutdown` restored it (TODO.md 1d). Phase 1 runs on CPU. | `nvidia-smi` |
| RAM | WSL 5–6 GB total, 2 GB free at check time | `free -g` |

### 1.2 Base model files (`models/gip15`)

| File | sha256 |
| --- | --- |
| `model.pt` (official Icefall checkpoint, with CTC head) | `7315ea2e15addfe796e673f0b4c9c5200e80371c029524fcc52f96161620eee2` |
| `init_ft.pt` (`wrap_ckpt --drop-ctc` of model.pt; provenance re-checked in Phase 1) | `3ff7bc5f08c1e75baca839505cd3b2048922a5d54f099ab4150409d579f6e5e9` |
| `encoder.int8.onnx` (official) | `b528768939c7711a889be81a718ea7f2ee50d0d2d384d53f399e15b44bd9408c` |
| `decoder.int8.onnx` (official) | `e0a156b5454722a524230f9e35d5d928cfcdd3723437b418e5bd5e04f1c3a101` |
| `joiner.int8.onnx` (official) | `12636559d135315f002a1e1b477077d415e888477378db0fd450aee5b21ac551` |
| `bpe.model` | `289dbb44527c13c419ae3a4d8ce6a349f01a97f8777e69934a77e3692d2f10db` |
| `tokens.txt` | `f536d03c2e95ebd2930cf0abec88e823bd17d3c1933da7ae6a82db3b80605e15` |

Architecture from the checkpoint shapes (`configs/model_map.json`): encoder dims 192,256,384,512,384,256; downsampling
1,2,4,8,4,2; layers 2,2,3,4,3,2; FFN 512,768,1024,1536,1024,768; decoder/joiner dim 512; context 2; vocab 2000; blank 0;
CTC head present. These equal Icefall `train.py` defaults, which is why the train / export commands pass no architecture flags.

### 1.3 Training runs

| Run | Where | Status | Evidence on this machine |
| --- | --- | --- | --- |
| A4 trial (16/11) | local GPU | finished, 194 steps; draft bench only | `exp/a4_trial/{log,onnx}`, `reports/history/a4_trial_report.md`. Reported N-WER +2.8 (claim) |
| E3 v1 (16/11) | local | finished; 4 checkpoints `checkpoint-{4132,8264,12396,16528}.pt` | `C:/Users/goose/gipbench_exp/e3_v1/` (checkpoints, log, tensorboard, wise). Reported C1 CS-WER +21.6 for ck4132 (claim; TODO.md part 4) |
| **X0 seed 0** (19/11) | **Colab** | finished 2 400 batches, artifact exported and scored once on G0 | `results/e4_x0/` (ONNX artifact, model card, hyps, summary). **Checkpoint-2400.pt, finetune.log and tensorboard are NOT on this machine** (only on Colab / Drive, not verified). |
| X0 seed 1 | — | never run | none |
| `e4_x0_lr1e4_vi` (LR 1e-4, dev_v2_vi) | local | stage 0 only (leak gate inputs) | `data/work/model/e4_x0_lr1e4_vi/seed-0/` |
| X1–X6 | — | never run (only stage-0 smoke for X1/X4) | `data/work/model/e4_x{1,4}_interim_smoke` |

### 1.4 X0 result (existing artifact, read not re-run)

X0 = Vietnamese-only replay (`pool/vivos_train@v3`, 40 h mixed cuts, speed perturb + `phone_sim 0.3` + SpecAugment),
LR 3e-4, freeze `encoder_embed`, stacks 0–1, no CTC, fp16, 2 400 batches × 60 s, `fixed_last` checkpoint, int8 ONNX,
modified beam search 4, batch 1. Code commit `c8065b6`. Gate: ΔWER on G0 ≤ +0.3 and CI95 upper ≤ +0.5.

| G0 subset | Baseline (official ONNX) | X0 | Δ |
| --- | ---: | ---: | ---: |
| overall (4 400) | 15.21 | 20.17 | **+4.96 [4.39, 5.56] → FAIL** |
| vivos (read; same corpus as X0 train) | 4.51 | 4.57 | +0.06 |
| commonvoice_vi_27 (read) | 7.47 | 7.74 | +0.27 |
| ggroup_edu (natural) | 23.52 | 33.01 | +9.49 |
| ggroup_finance (natural) | 23.11 | 30.80 | +7.69 |
| ggroup_pubadmin (natural) | 15.28 | 22.03 | +6.75 |
| ggroup_tech (natural) | 27.70 | 39.07 | +11.37 |

Sources: `results/e4_x0/seed-0/acceptance.json`, `report.md`; baseline `results/general_asr/general_v1/gip15_mbs4_open/report.md`.

**Confound:** the baseline is the *official* int8 ONNX. The X0 artifact went through this repo's Icefall export + int8
quantization. The untouched checkpoint has never been exported through that path and scored. So +4.96 mixes training
damage with any export / quantization difference. Phase 1 resolves this before X0 is interpreted.

### 1.5 Data and contamination (claims from TODO.md, not re-run here)

- 6 locked benches (G0 `general_v1`, C0 `code_switch_v1` / `_v1_1`, C1 `cs_ggroup_v1`, E0 `external_v1` / `_v1_1`).
- Final training pools: `vimedcss_train@v4`, `distill@v5`, `vivos_train@v3`, `confusable_train@v3`, `cv_other@v3`,
  `vss_cs@v4`. Leak gate (`model.run` stage 0, `audio_sha: hash`) reported 0 critical for X0 (`model_card.json`
  `leak_gate`, report sha `b9a27623…`).
- Dev sets: `dev/dev_interim_cs@1` (323 utts, natural CS, human-reviewed, *same distribution as vss_cs train*),
  `dev/dev_interim_vi@1` (213 utts), `dev/dev_v2_vi@1` (ViMD valid, 1 691 utts, natural VI, human labels, manifest sha
  `208f8e18…8345`, flags `independent: true`). `dev_v2_cs` does not exist (not enough non-medical natural CS).
- `pool/vivos_train@v3` manifest sha `13528dbd…7185`.

### 1.6 Which historical conclusions still stand

| Claim | Status |
| --- | --- |
| Every FT so far (A4, E3, X0) degraded Vietnamese | X0: **confirmed** on independent dev in one environment (section 6); export path ruled out (section 2). A4 / E3: not re-run. |
| E3 "improved" only on a contaminated dev | Plausible, documented in TODO.md part 4; not re-run. |
| Direction A (decode-time / static weight edits) closed | Accepted as recorded (`TODO_rea.md`); irrelevant to the FT pipeline. |
| "Errors are mostly post-encoder → X2 first" | A diagnosis on greedy INT8 ranks; a hypothesis for Phase 5, not evidence that FT works. |

### 1.7 Blocked experiments

- X1–X5: blocked by this recovery's gates (Phases 1–4) and by the lack of `dev_v2_cs`.
- X0 interpretation: the export confound (1.4) was resolved in Phase 1. X0's damage is a training effect (section 6).

---

## 2. Phase 1: pretrained model parity

### 2.1 Pre-registered checks (written 2026-11-25 00:25, before any parity decode)

Arms. *official* = `models/gip15/*.int8.onnx`. *reexport* = `init_ft.pt` exported with the repo's exact export command
(`onnx_export.export_cmd`: `export-onnx.py --use-transducer 1 --use-ctc 0 --epoch 1 --avg 1 --use-averaged-model 0
--enable-int8-quantization 1`), kept in fp32 and int8. *torch* = Icefall `get_model` with default args, strict-loaded.
All arms run in one environment: WSL `~/.venvs/gipbench-ft`, CPU. Audio is never taken from a locked bench.

| ID | Check | PASS rule |
| --- | --- | --- |
| P1-A | State dict | `init_ft.pt` equals `model.pt` minus `ctc_output.*`, every tensor bit-equal. Strict load of `model.pt` (use_ctc 1) and `init_ft.pt` (use_ctc 0) into the default-arg Icefall model gives 0 missing and 0 unexpected keys. |
| P1-B | Tokenizer | `tokens.txt` agrees with `bpe.model` on all 2 000 ids; `<blk>` = 0 is the blank in every arm (ONNX metadata and Icefall params). |
| P1-C | fp32 export numerics | On K = 100 utterances (60 `dev_v2_vi@1`, 40 `dev_interim_cs@1`, picked by sha256(id)): reexport-fp32 encoder vs torch (after `joiner.encoder_proj`), median relative L2 error < 1e-4. ONNX greedy hyps from reexport-fp32 equal torch greedy hyps on 100% of K. |
| P1-D | **INT8 functional parity (the gate)** | sherpa-onnx, `modified_beam_search`, beam 4, batch 1, int8, on all of `dev_v2_vi@1` (1 691) and `dev_interim_cs@1` (323): paired ΔWER (reexport − official) with \|Δ\| ≤ 0.3 and the CI95 inside [−0.5, +0.5] on **each** set. Identical-hyp rate is reported. |
| P1-E | Train vs inference features | Lhotse `Fbank(FbankConfig(num_mel_bins=80))` (what `finetune_cs.py --on-the-fly-feats` uses) vs kaldi-native-fbank (what sherpa-onnx and `rnnt_loader` use), same waveform, on K: max / mean abs difference reported. Torch greedy with Lhotse features vs with kaldi-native-fbank features: identical hyps on ≥ 90% of K and \|ΔWER\| ≤ 0.5. |

Descriptive only (no pass rule): official-int8 and reexport-int8 encoder error vs torch; ONNX metadata and op-type
counts of both exports; file sizes (reexport int8 encoder 72 828 389 B vs official 70 876 359 B, so the official
files were not produced by this exact export path).

If P1-D fails, X0's +4.96 is not interpretable as training damage, and Phase 2 does not start.

### 2.2 Results

Evidence: `data/work/recovery/p1/parity.json` (+ `parity.log`), produced by `scripts/recovery_p1_parity.py`
(K = 100: 60 `dev_v2_vi@1` + 40 `dev_interim_cs@1`); decodes in `data/work/recovery/p1/decode/<set>/{official,reexport}/`
from `scripts/recovery_p1_decode.sh`. Versions: torch 2.15.0+cu132, lhotse 1.33.0, onnxruntime 1.32.0, sherpa-onnx 2.0.1.

| ID | Result | Numbers |
| --- | --- | --- |
| P1-A | **PASS** | `model.pt` 745 tensors = `init_ft.pt` 743 + `ctc_output.1.{weight,bias}`; 0 tensors differ bitwise. Strict load into the default-arg Icefall model: 0 missing / 0 unexpected for both (69 651 511 params with CTC, 68 625 511 without). |
| P1-B | **PASS** | 2 000 / 2 000 ids agree between `tokens.txt` and `bpe.model`; `<blk>` = 0, `<unk>` = 2. Neither ONNX export stores `blank_id` in its metadata, so sherpa-onnx relies on the `<blk>` = 0 convention of `tokens.txt` (true here). |
| P1-C | **PASS** | reexport-fp32 encoder vs torch: median relative L2 6.0e-7, max 8.9e-7. Greedy ids identical on 100 / 100. |
| P1-E | **PASS** | Lhotse `Fbank` vs kaldi-native-fbank: same frame count on 100 / 100, max abs diff 1.2e-3, median of per-utt mean abs diff 2.8e-6 (log-mel units). Torch greedy hyps identical on 100 / 100, ΔWER 0.00. **Training and inference features match.** |
| P1-D | **PASS** (both sets) | dev_interim_cs@1 (323 utts, beam 4): reexport vs official **323 / 323 hyps identical**, WER 4.95 vs 4.95, Δ 0.00 [0.00, 0.00]. dev_v2_vi@1 (1 691 utts, 9.06 h natural VI): **WER 8.82 vs 8.81, Δ +0.01 [−0.01, +0.03]**, 1 585 / 1 691 hyps identical. Scored with `gipbench.bench.evaluate --baseline-hyps` (diagnostic mode, speaker bootstrap): `data/work/recovery/p1/decode/<set>/score_reexport_vs_official/`. |

**Phase 1 gate: PASSED (2026-11-25 02:16).** The repo's export + INT8 path reproduces the official model within
±0.03 WER on independent natural Vietnamese. Consequence: **X0's +4.96 on G0 is not an export artifact; it is a
training effect** (one remaining difference, platform: the G0 baseline was decoded in WSL on 16/11 and X0 on Windows
on 20/11; section 6 measures X0 vs base in one environment).

Descriptive findings:

- **INT8 itself costs accuracy.** On K = 100, greedy WER (simple normalizer, relative use only): torch / reexport-fp32
  7.51, reexport-int8 7.76, official-int8 7.76. INT8 hyps differ from fp32 on 57 / 100 utterances; INT8 encoder relative
  error is 5.4% median (13% max) for **both** INT8 exports. Comparisons between a fine-tuned model and the base must
  therefore be INT8 vs INT8 (as the pipeline already does), and an fp32 dev score is not the deployed score.
- The two INT8 exports are numerically equivalent but not identical: greedy ids agree on 94 / 100 (the beam decode on
  dev_interim_cs agrees on 323 / 323). Both use dynamic QInt8 `MatMulInteger` (264 `DynamicQuantizeLinear`, 290
  `MatMulInteger`, opset 13, same metadata). The official encoder graph has 9 133 nodes vs 8 856 for the re-export and is
  1.95 MB smaller, i.e. it was exported with another Icefall / torch version. This explains the different file sizes.
- `export-onnx.py` loads `epoch-N.pt` with Icefall's `load_checkpoint(strict=False)`; P1-A shows the keys match exactly
  for the base model, but a future checkpoint with renamed keys would load silently. The `--iter` path used for X0
  (`average_checkpoints` + `load_state_dict`) is strict.

---

## 3. Phase 2: training correctness

### 3.1 Static audit against the Icefall recipe (read, not yet run)

| Item | This repo (X0) | Icefall `finetune.py` / recipe | Consequence |
| --- | --- | --- | --- |
| Loss-scale warm-up | `compute_loss` uses `params.batch_idx_train`, which restarts at 0 in a fine-tune, with `warm_step = 2000` hard-coded in `get_params` | Same code: the recipe never offsets it (unlike `get_adjusted_batch_count`, which adds +100 000 for the module schedules) | Fine-tuning starts with **simple (trivial-joiner) loss × 1.0 and pruned RNN-T loss × 0.1**, reaching 0.5 / 1.0 only at batch 2 000. X0 ran 2 400 batches, so ~83% of its updates had the pruned loss down-weighted. Candidate cause of drift; measured in P2-2. |
| Module schedules (dropout, skip rates, bypass clamps, balancers) | `set_batch_count(model, batch_idx·max_dur/600 + 100000)` | same | At final values from step 1. No issue. |
| Freezing | `requires_grad_(False)` by prefix (patch); frozen tensors **stay in ScaledAdam's param groups** | `get_parameter_groups_with_lrs(..., freeze_modules=...)` exists and removes them | ScaledAdam substitutes zeros for `None` grads (`optim.py:111`) and clamps parameter RMS to `param_max_rms` (`optim.py:234`), so frozen tensors *could* move. Measured in P2-3. |
| CTC head | dropped (`--use-ctc 0`, `wrap_ckpt --drop-ctc`) | recipe trains with CTC if the checkpoint has it | Removes one regularizer of the pretrained alignment; documented A4 fp16 divergence motivated it. |
| LR | `--base-lr 3e-4`, Eden `lr_batches 1e5`, `lr_epochs 100`, warm-up 500 batches from 0.5× | finetune default 0.0045 | 15× below the recipe's fine-tune default; not an obvious cause of forgetting. |
| Batch | `--max-duration 60` (s) | recipe 1 000 | 16× smaller batches, noisier gradients; schedules are rescaled via `ref_duration 600`. |
| Data | VIVOS only, speed perturb ×3, `phone_sim 0.3`, SpecAugment, no MUSAN | recipe mixes the original training data (`--use-mux`) | Narrow read-speech replay vs a model trained on broad natural speech: a domain-forgetting explanation for X0 that needs no bug. |
| Checkpoint load | `load_model_params(strict=True)` | same | P1-A: 0 missing / 0 unexpected. |

### 3.2 Pre-registered runtime checks (written before running `scripts/recovery_p2_train_checks.py`)

CPU, fp32, real `finetune_cs.py` functions, X0 freeze prefixes, LR 3e-4, 6 utterances of `vivos_train@v3`, no
augmentation, the same batch repeated for 12 steps. fp16 / GradScaler cannot be tested here (no GPU), so they stay
**unverified**.

| ID | Check | PASS rule |
| --- | --- | --- |
| P2-0 | Training text vs tokenizer | `model_case_text` + BPE over all 10 555 X0 training rows: 0 `<unk>` tokens; no characters outside the vocabulary. |
| P2-1 | Forward + loss | Finite simple and pruned losses; pruned loss per frame of the pretrained model on its own replay data is low (reported). |
| P2-2 | Backward / gradient flow | Every trainable group gets a finite non-zero gradient; no frozen tensor has a gradient. Report the pruned-loss share of the encoder gradient at batch 1 vs after warm-up and the cosine between the two loss gradients (diagnostic for the warm-up issue; no pass rule). |
| P2-3 | Optimizer, scheduler, updates, frozen invariance | LR follows Eden (≈ 0.5·3e-4 at batch 1, rising); every trainable tensor changes; **every frozen tensor is bit-identical** after 12 steps; eval-mode pruned loss on the batch decreases. |
| P2-4 | Checkpoint save / reload | `save_checkpoint_with_global_batch_idx` → fresh model: 0 missing / unexpected keys, every tensor equal, encoder output max abs diff = 0. |
| P2-5 | Post-reload export parity | Export the saved checkpoint through the X0 path (`--iter N --avg 1`): fp32 ONNX encoder relative error < 1e-4 and greedy ids identical to torch on all 6 utterances. |

### 3.3 Results

Evidence: `data/work/recovery/p2/p2_checks.json`, log `data/work/recovery/p2.log`, saved checkpoint and its ONNX export in
`data/work/recovery/p2/{exp,onnx}`. Follow-up probe: `data/work/recovery/p2/p2_followup.py` with its output in
`p2_followup_output.txt`. The harness ran while the P1-D decode was still running; its results are not used for any decision until
P1-D passes.

| ID | Result | Numbers |
| --- | --- | --- |
| P2-0 | **PASS** | 10 555 rows → 161 827 BPE tokens, **0 `<unk>`**, 0 characters outside the vocabulary, 0 rows with lowercase after `model_case_text`. |
| P2-1 | **PASS** | Pretrained model on 6 replay utterances (eval mode): simple loss 0.417 / frame, pruned loss 0.106 / frame. Finite. |
| P2-2 | **PASS** | No frozen tensor receives a gradient. Every trainable group receives one from the combined loss (simple loss reaches `simple_*_proj`, decoder, encoder; pruned loss reaches joiner, decoder, encoder). **Pruned-loss share of the encoder gradient: 12.9% at batch 1, 80.5% with post-warm-up scales**; cosine(simple, pruned) on the encoder = 0.58. |
| P2-3 | **PASS, one rule exception explained** | LR 1.503e-4 at batch 1 → 1.536e-4 at batch 12 (Eden warm-up from 0.5×). Frozen groups (`encoder_embed`, stacks 0–1, 6.50 M params): **0 / 194 tensors changed** (bit-identical; no frozen tensor has RMS > 3, so the `param_max_rms` path is not exercised). Training-mode pruned loss / frame 0.105 → 0.053, eval-mode 0.106 → 0.042. Trainable tensors changed: all except **12** `encoder.encoders.{2..5}.encoder.layers.*.bypass_scale`. Those are exactly 0.5 (their initial value) in `model.pt`, i.e. unused leftovers that pretraining never updated either; 12 = number of layers in stacks 2–5. Not a defect. Relative update after 12 steps 1–6e-4 per group. |
| P2-4 | **PASS, one rule exception explained** | `save_checkpoint_with_global_batch_idx` → fresh model: 0 missing / 0 unexpected, **every tensor equal**. In-memory trained model vs reloaded model, encoder output max abs diff **2.6e-6** (rule said 0). Probe: two models reloaded from the same checkpoint give 0.0; changing only the thread count gives 1.2e-6. The difference is float32 kernel-path noise, not a save/load defect. |
| P2-5 | **PASS** | Export with `--iter 12 --avg 1` (X0 path): fp32 ONNX encoder relative error max 5.6e-7; greedy ids identical on 6 / 6. |

Not tested here: fp16 autocast and GradScaler (later exercised in Phase 3, section 5.1), `CutSet.mux` mixing, multi-segment
resume (`icefall_runner`), speed-perturb / `phone_sim` audio.

**Confirmed deviation (runtime-measured, consequence not yet measured): loss-scale warm-up restarts in fine-tuning.**
For the first 2 000 batches the encoder is driven mainly by the simple (trivial-joiner) loss: the pruned RNN-T loss,
which is what decoding uses, supplies 13% of the encoder gradient at batch 1 instead of 80%. The two gradients
are only partly aligned (cos 0.58). This matches upstream Icefall `finetune.py`, so it is a recipe property rather
than a repo bug, but it is wrong for continuing a converged model. Whether it explains the X0 regression is **not
shown**. Phase 3/4 must test it as a single-factor change (warm-up disabled vs default).

---

## 4. Confirmed bugs

| ID | Bug | Evidence | Root cause | Impact | Status |
| --- | --- | --- | --- | --- | --- |
| B1 | `pool/vss_cs@v4` (1 087 / 1 087 rows) and `pool/distill@v5` (1 394 / 1 394 rows) store `audio` with **Windows separators** (`..\..\..\dl\…`). All other releases are clean. | Scan of every `data/releases/*/*/manifest.jsonl`; on WSL, `lhotse.Recording.from_file` fails on every row and `to_lhotse` exits "Không còn utterance nào sau khi lọc" (first Phase 3 smoke, `data/work/recovery/p3/smoke/run.log` of the failed attempt). | Written on Windows on 19/11 when the pools were carved for the dev-interim hold-out (`clean_pool` → `register`). `register`'s audio-exists check also ran on Windows, where backslashes resolve, so it passed. The exact writer is not yet localized (`core/release.py:182` converts separators, so the backslashes came in another way). | These are the final CS / replay pools for X1–X5 and the default contents of the Colab bundle. Any Linux / Colab run that includes them stops at stage 2 (loud failure, not silent). X0 is unaffected (VIVOS only). | **Open.** Not fixed in the registered manifests (immutable). Phase 3 works around it locally. Fix options for the user: a v5 / v6 pool re-registered from WSL, or separator normalization in `core.manifest.read_jsonl`. |
| D1 | Loss-scale warm-up restarts at fine-tune start (section 3.3). | P2-2 + real CLI log: batch 0 `loss 0.3607 = 1.0 × simple 0.3562 + 0.1 × pruned 0.0462` (`data/work/recovery/p3/smoke/run.log`). | Icefall `finetune.py` keeps `warm_step = 2000` and `batch_idx_train` restarts at 0. | Down-weights the decoding loss for the first 2 000 batches of every FT here. Effect on WER unmeasured. | Deviation, not yet a proven bug. Phase 3 tests it as a single factor. |

---

## 5. Phase 3: tiny-set memorization (registered before the run; results in 5.1)

Note: the Phase 3 smoke run's checkpoint `data/work/recovery/p3/smoke/exp/` is a throwaway (4 batches); delete it
freely.

**Set.** The 163 utterances of the training pool `pool/vss_cs@v4` that the user listened to in 2d
(`data/listen/vss_cs_v1/listen_A_reviewed.csv`: 133 `ok` with unchanged text, 30 `fix` with the corrected text);
all 163 contain `<en>` (299 spans), 0.60 h, 100 videos; none is in a dev or bench set (they are in the training pool,
which excludes the dev-interim videos and passed the leak gate). Built by `scripts/recovery_p3_overfit.py`, manifest sha
in `<work>/meta.json`.

**Recipe.** X0 recipe (init `init_ft.pt`, tokenizer, freeze `encoder_embed` + stacks 0–1, LR 3e-4, no CTC,
max-duration 60) with **all augmentation off** (no speed perturb, no `phone_sim`, no SpecAugment, no MUSAN). The 163
cuts are repeated into one virtual epoch (unique ids) so Icefall does not write a 1 GB `epoch-N.pt` per pass.
Checkpoints at 100, 200, 400, 600 batches; each is exported through the X0 path, decoded with beam 4 INT8, and scored
against the same 163 references (WER, CS-WER, term recall, decoded examples). Gradient norms come from ScaledAdam's
log lines; parameter updates come from checkpoint − init.

**Arms (single factor).** A = pipeline as-is (loss warm-up restarts). B = `warm_step 0` (scales 0.5 / 1.0 from batch 1),
the only change.

**Pre-registered reading.** Training until batch 600 (≈ 2 200 utterance-passes, ≈ 13 passes over the set).

- *Can memorize*: at batch 600, train CS-WER ≤ 50% of its batch-0 value (40.18 → ≤ 20.1) **and** WER does not rise.
- *Cannot memorize* (stop, find the root cause): CS-WER reduction < 20% relative at batch 600 in both arms.
- In between: extend to 1 200 batches once, same rule.
- Arm B vs A is reported, not used to pass or fail Phase 3.

**Recorded per checkpoint** (`<work>/results.json`): WER / CS-WER / term recall summary, 10 ref–hyp examples, per-group
relative update vs `init_ft.pt`, bit-equality of every frozen tensor, and from `run.log` the loss / simple / pruned
lines, ScaledAdam grad-norm quartiles and fp16 `grad_scale`. On the smoke `checkpoint-4.pt`, which the real
`finetune_cs.py` CLI wrote, **194 / 194 frozen tensors are bit-identical to `init_ft.pt`** and trainable groups moved
by ~7e-5 relative. This confirms P2-3 through the script itself, not only through the exec'd snippet. Use
`--prune-checkpoints` (each checkpoint is ~1.1 GB; D: had 57 GB free on 25/11).

**Smoke test (4 batches, CPU, A).** `data/work/recovery/p3/smoke/`: 163 / 163 cuts kept, training ran, checkpoint-4
exported and scored. Batch 0: WER 5.66 [4.86, 6.51], CS-WER 40.18, term recall 69.57; batch 4: WER 5.64, CS-WER 40.18.
The driver works. **4 batches took 11 min on CPU** while competing with a decode, so the 2 × 600-batch run needs the GPU.

### 5.1 Results (run 2026-11-25 08:46–10:09, WSL, RTX 3050 Laptop 4 GB, fp16)

The user restored the WSL GPU (`nvidia-smi` and `torch.cuda.is_available()` OK). Both arms ran with the section 8
commands and exited 0; training took about 8 min per arm and export + decode + score about 4 min per checkpoint.
Tiny manifest `tiny.jsonl` sha256 `f9ed852a…`, the same in smoke, A and B. Ids sha `a5b50e51…`. Driver `c4c8c85c…`.
Training script `finetune_cs.py` `45a680aa…` (A), `finetune_cs_nowarm.py` `3259383a…` (B; the only diff is
`warm_step 0`). Evidence: `data/work/recovery/p3/{default,nowarm}/results.json`, `run.log`, `eval/step*/`, `exp/`.

Train-set metrics (163 utterances, 299 `<en>` spans; beam 4, INT8, batch 1, 16 threads):

| Batch | A WER | A CS-WER | A term recall | B WER | B CS-WER | B term recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (base) | 5.65 | 40.48 | 69.23 | 5.65 | 40.48 | 69.23 |
| 100 | 4.33 | 33.63 | 72.91 | 4.33 | 34.82 | 72.58 |
| 200 | 2.48 | 17.56 | 86.96 | 2.30 | 14.29 | 90.97 |
| 400 | 0.89 | 5.36 | 96.32 | 0.76 | 5.36 | 96.32 |
| 600 | **0.27** | **1.79** | **98.66** | **0.33** | **2.68** | **97.32** |

N-WER (the Vietnamese words) also falls, 4.29 → 0.21 (A) and 0.23 (B), and the boundary insertion rate falls
7.74 → 0.30 / 0.00.

**Reading against the registered rule.** Can memorize means CS-WER at batch 600 is at most 50% of batch 0 (≤ 20.2)
and WER does not rise. A: 1.79, B: 2.68, and WER falls in both. **Phase 3 PASSES in both arms.** No extension to
1 200 batches is needed.

**Arm B vs A (reported only).** B is ahead at batch 200 (CS-WER 14.29 vs 17.56) and level or behind afterwards
(1.79 vs 2.68 at 600, under 1 point and within a few span errors on 163 utterances). The doc's own condition was "if B memorizes
clearly faster than A, the X0 rerun should use `warm_step 0`". **That condition is not met.** On clean data with
augmentation off, D1 does not stop learning or measurably slow it. Whether D1 contributes to X0's loss of Vietnamese
generalization is a different question, and this run does not answer it.

**Training dynamics** (from `run.log`; `loss` is the scale-weighted sum, so compare `simple` and `pruned` across arms):

| | Batch 0 | 100 | 300 | 550 |
| --- | --- | --- | --- | --- |
| A simple / pruned | 0.297 / 0.044 | 0.217 / 0.031 | 0.192 / 0.0025 | 0.190 / 0.0025 |
| B simple / pruned | 0.297 / 0.044 | 0.218 / 0.026 | 0.195 / 0.0018 | 0.192 / 0.0022 |

- **ScaledAdam grad-norm median** falls in A from 589 to 397 and in B from 360 to 229. B is lower because the pruned
  loss carries weight 1.0 instead of 0.1 and its gradient is smaller.
- **Clipping** stays at 0–2% of batches in both arms.
- **fp16 `grad_scale`** rises 2 → 16 in both, with no inf and no skipped steps. This is the first run of the fp16 /
  GradScaler path in this recovery (Phase 2 was fp32 only).

**Parameter updates vs `init_ft.pt`** (relative L2, batch 600, A; B is within 10% on every group except `encoder.downsample_output`, 0.0154 vs 0.0096):

| Group | Update | Tensors changed |
| --- | --- | --- |
| `encoder_embed`, stacks 0–1 (frozen) | 0 | 0 / 194, bit-identical at every checkpoint in both arms |
| stacks 2–5 | 0.007–0.010 | 131/134, 174/178, 131/134, 88/90; the unchanged ones are the 12 unused `bypass_scale` (section 3.3) |
| `decoder`, `joiner` | 0.010, 0.013 | 2/2, 6/6 |
| `simple_lm_proj` | 0.038 | 2/2 |

Updates grow roughly linearly with steps (joiner 0.0027 → 0.0050 → 0.0087 → 0.0134 at 100 / 200 / 400 / 600).

**Decoded examples.** Ten ref–hyp pairs per checkpoint are in `results.json` (`step*.examples`). At batch 600 the
English spans (`profile`, `instagram`, `make up`, `view`, `viral`, `trend`, `partner`) come out as in the reference.

**What Phase 3 shows.** On clean, human-verified CS data, the pipeline (data → Lhotse → `finetune_cs.py` with freeze
→ checkpoint → `export-onnx --iter` → INT8 → sherpa → scorer) can fit the training targets, including the English
spans. The loss, gradients, optimizer, freezing and export work end to end. It says nothing about generalization; that
is Phase 4.

**Decoding thread count changes INT8 hyps.** The batch-0 decode in this run used 16 threads and gave CS-WER 40.48.
The smoke run used 4 threads and gave 40.18. The model (official `models/gip15`) and the manifest are the same. 18 /
163 hyps differ. At a fixed thread count the output is deterministic: A's and B's batch-0 hyps are 163 / 163
identical. **Rule:** besides batch 1, keep `--threads` fixed in any paired comparison. Phases 1 and 4 used 16 threads
on both sides.

---

## 6. Phase 4: Vietnamese control (X0), re-measured on independent dev

Registered 2026-11-25 02:20, before the X0 decode finished. The existing X0 seed-0 artifact
(`results/e4_x0/artifacts/seed-0/onnx`, encoder sha `4d0072e8…`) and the official base are decoded in the **same
environment and config** as Phase 1 (WSL `gipbench-ft`, beam 4, batch 1, INT8) on `dev_v2_vi@1` (natural VI, independent,
not used by X0 for anything) and `dev_interim_cs@1`. The base hyps are the Phase 1 `official` hyps. No locked bench is
read. Reading uses X0's own registered tolerance: X0 preserves Vietnamese if ΔWER(dev_v2_vi) ≤ +0.3 and CI95 upper ≤ +0.5.

### 6.1 Results

Evidence: decodes `data/work/recovery/p4/decode/<set>/x0_seed0/hyps.jsonl` (log `data/work/recovery/p4/decode.log`),
scores `data/work/recovery/p4/score/{dev_v2_vi,dev_interim_cs}/` (`--baseline-hyps` = Phase 1 official hyps).

| Set | Base (official) | X0 seed 0 | Δ [CI95, speaker bootstrap] | Verdict |
| --- | ---: | ---: | --- | --- |
| dev_v2_vi@1 WER (1 691, natural VI, 19.3 s median) | 8.81 | 13.64 | **+4.83 [4.38, 5.25]** | **FAIL** (tolerance +0.3 / +0.5) |
| dev_v2_vi@1 CER | 6.76 | 11.50 | +4.74 [4.30, 5.17] | |
| dev_interim_cs@1 WER (323, natural CS) | 4.95 | 9.22 | +4.28 [3.64, 5.04] | |
| dev_interim_cs@1 CS-WER | 42.18 | 89.09 | (not paired) | English output collapses |

The G0 regression (+4.96) reproduces on an independent set, in one environment, against the same base. The
"different platform" confound from section 1.4 is therefore not the explanation.

**What the damage looks like (descriptive, not causal):**

| Set | Base S / D / I (% of ref words) | X0 S / D / I |
| --- | --- | --- |
| dev_v2_vi@1 | 2.83 / 1.00 / 4.98 | 2.94 / **8.24** / 2.47 |
| G0 ggroup_edu (3.8 s mean) | 14.22 / 4.37 / 4.93 | 17.20 / 10.66 / 5.14 |
| G0 ggroup_tech (3.5 s) | 18.28 / 3.48 / 5.94 | 24.22 / 7.65 / 7.21 |
| G0 vivos (read, same corpus as X0 train) | 3.66 / 0.43 / 0.41 | 3.56 / 0.69 / 0.32 |
| G0 commonvoice (read) | 6.45 / 0.45 / 0.57 | 6.47 / 0.91 / 0.36 |

- On long natural speech X0 **deletes whole stretches** (dev_v2_vi deletion rate ×8; utterances ≥ 15 s carry +7.8
  points of extra deletions; worst cases drop 40–60 words of a 25–30 s clip). X0 trained on VIVOS clips of 4.8 s mean
  (max 17 s).
- On short natural speech (G-Group, ~3.5 s) deletions roughly double and substitutions rise 3–6 points, so
  utterance length is not the whole story.
- Read speech is untouched. Pattern: the model narrows to the replay domain (clean read speech, short, no English).

**Cause: not isolated.** X0 changed several things together relative to "no training": VIVOS-only data
(read, short, Vietnamese-only), loss warm-up restart (D1), `phone_sim 0.3` + speed perturb + SpecAugment, freeze
depth, LR 3e-4 × 2 400 batches. X0's checkpoint and training log are not on this machine, so no post-hoc analysis
(WiSE interpolation, loss curve) is possible. Separating these needs controlled reruns (section 10).

**Phase 4 verdict: continued Vietnamese-only training with the X0 recipe does NOT preserve pretrained quality.**
Per TODO.md part 5 ("Không đạt thì lỗi nằm ở recipe … dừng mọi thí nghiệm CS"), all CS experiments stay stopped.

---

## 7. Experiment registry

| ID | Commit | Data (sha) | Config | Checkpoint | Seed | Metric |
| --- | --- | --- | --- | --- | --- | --- |
| X0-s0 (historical) | `c8065b6` | vivos_train@v3 `13528dbd…` | `e4_x0.yaml` sha `d7b27bbb…` | ck2400 (Colab, missing); ONNX encoder `4d0072e8…` | 0 | G0 WER 20.17 (Δ +4.96); **dev_v2_vi 13.64 (Δ +4.83 [4.38, 5.25])**; dev_interim_cs WER 9.22, CS-WER 89.09 |
| R1-base-reexport | `e488a9c` (+ uncommitted recovery scripts) | — | `export-onnx.py --epoch 1 --avg 1 --use-averaged-model 0 --enable-int8-quantization 1 --use-ctc 0` | `init_ft.pt` `3ff7bc5f…` → `data/work/recovery/p1/models/reexport` | — | dev_v2_vi 8.82 (Δ +0.01 vs official); dev_interim_cs 4.95 (Δ 0.00) |
| R1-official | — | dev_v2_vi@1 `208f8e18…`, dev_interim_cs@1 `745e512b…` | sherpa-onnx 2.0.1, beam 4, batch 1, INT8 | official ONNX `b5287689…` | — | dev_v2_vi WER 8.81, dev_interim_cs WER 4.95 / CS-WER 42.18 |
| R2-p2checks | same | vivos_train@v3, 6 rows (ids in `p2_checks.json`) | X0 recipe, CPU fp32, no aug, 12 steps on one batch | `data/work/recovery/p2/exp/checkpoint-12.pt` | 0 | eval pruned loss / frame 0.106 → 0.042 |
| R3-smoke | same | tiny set (163 rows) `f9ed852a…` | X0 recipe, no aug, 4 batches, CPU fp32 | `data/work/recovery/p3/smoke/exp/checkpoint-4.pt` | 0 | tiny-set WER 5.66 → 5.64, CS-WER 40.18 → 40.18 (4 decode threads) |
| R3-A (Phase 3 arm A) | same | tiny set `f9ed852a…` | X0 recipe, no aug, 600 batches, GPU fp16, `finetune_cs.py` `45a680aa…` | `data/work/recovery/p3/default/exp/checkpoint-600.pt` (100/200/400 pruned; ONNX kept in `eval/step*/onnx`) | 0 | tiny-set WER 5.65 → 0.27, CS-WER 40.48 → 1.79, term recall 69.2 → 98.7 (16 threads) |
| R3-B (Phase 3 arm B) | same | tiny set `f9ed852a…` | as R3-A + `warm_step 0`, `finetune_cs_nowarm.py` `3259383a…` | `data/work/recovery/p3/nowarm/exp/checkpoint-600.pt` | 0 | tiny-set WER 5.65 → 0.33, CS-WER 40.48 → 2.68, term recall 69.2 → 97.3 |

Script hashes (sha256 prefix): `recovery_p1_parity.py` `1b19aadd…`, `recovery_p1_decode.sh` `8752c054…`,
`recovery_p2_train_checks.py` `84552218…`, `recovery_p3_overfit.py` `c4c8c85c…`. The scripts are committed together with this file (`git log -- scripts/recovery_*`).

---

## 8. Reproduction (WSL, from the repo root `/mnt/d/code/ZAIC/gipbench`)

```bash
# Phase 1: export the untouched checkpoint the same way the pipeline exports a fine-tuned one
W=data/work/recovery/p1/base_exp; mkdir -p $W; cp models/gip15/init_ft.pt $W/epoch-1.pt
(cd ~/icefall/egs/librispeech/ASR && PYTHONPATH=~/icefall ~/.venvs/gipbench-ft/bin/python ./zipformer/export-onnx.py \
   --tokens /mnt/d/code/ZAIC/gipbench/models/gip15/tokens.txt --exp-dir /mnt/d/code/ZAIC/gipbench/$W \
   --use-transducer 1 --use-ctc 0 --epoch 1 --avg 1 --use-averaged-model 0 --enable-int8-quantization 1)
bash scripts/recovery_p1_decode.sh                       # P1-D decodes (official + reexport, two dev sets)
PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p1_parity.py \
   --icefall ~/icefall --reexport data/work/recovery/p1/models/reexport --out data/work/recovery/p1/parity.json
PYTHONPATH=src ~/.venvs/gipbench/bin/python -m gipbench.bench.evaluate --manifest data/releases/dev/dev_v2_vi@1/manifest.jsonl \
   --hyps data/work/recovery/p1/decode/dev_v2_vi@1/reexport/hyps.jsonl \
   --baseline-hyps data/work/recovery/p1/decode/dev_v2_vi@1/official/hyps.jsonl --system reexport \
   --out data/work/recovery/p1/decode/dev_v2_vi@1/score_reexport_vs_official --mode diagnostic --inference-mode open_vocab

# Phase 2
PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p2_train_checks.py \
   --icefall ~/icefall --work data/work/recovery/p2

# Phase 4 (X0 artifact on independent dev; then the same evaluate call with --hyps .../x0_seed0/hyps.jsonl)
PYTHONPATH=src ~/.venvs/gipbench-ft/bin/python -m gipbench.infer.decode --bench data/releases/dev/dev_v2_vi@1/manifest.jsonl \
   --model-dir results/e4_x0/artifacts/seed-0/onnx --system x0_seed0 --results data/work/recovery/p4/decode/dev_v2_vi@1 \
   --quantize int8 --decoding-method modified_beam_search --beam 4 --threads 16 --batch 1 --case upper

# Phase 3 (needs GPU): arm A, then arm B
PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p3_overfit.py \
   --icefall ~/icefall --work data/work/recovery/p3/default --steps 600 --eval-at 100 200 400 600 --prune-checkpoints
PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p3_overfit.py \
   --icefall ~/icefall --work data/work/recovery/p3/nowarm --steps 600 --eval-at 100 200 400 600 --no-loss-warmup --prune-checkpoints
```

Notes: `kaldi_native_fbank` comes from `~/jiwer_target` (not installed in `gipbench-ft`). Keep batch 1 for every
comparative decode (batch composition changes sherpa hyps). The decode script prints progress without newlines.

## 9. Missing evidence and remaining blockers

| # | Blocker / gap | Blocks | Owner |
| --- | --- | --- | --- |
| 1 | ~~WSL GPU unavailable~~ **Resolved 2026-11-25** by the user (WSL restart). RTX 3050 Laptop, **4 GB**: Phase 3 fit at max-duration 60 in fp16. Colab stays **not drop-in** for the Phase 3 driver (it reads `data/listen/vss_cs_v1/listen_A_reviewed.csv` and `vss_cs@v4` audio, not in the Drive bundle, and writes absolute `/mnt/d/…` paths). | — | — |
| 2 | X0 `checkpoint-2400.pt`, `finetune.log`, tensorboard exist only on Colab / Drive (not verified there). | Post-hoc X0 analysis (loss curve, WiSE) | 👤 copy them back if they still exist |
| 3 | fp16 autocast + GradScaler now exercised (Phase 3: no inf, scale 2 → 16). Still not tested: `CutSet.mux`, multi-segment resume, augmentation (speed perturb, `phone_sim`). | Full Phase 2 sign-off; X0 diagnosis | GPU (now available) |
| 4 | Bug B1: `vss_cs@v4`, `distill@v5` audio paths use backslashes. | X1–X5 on Linux / Colab | 👤 choose the fix (new pool version vs reader normalization) |
| 5 | `dev_v2_cs` does not exist; `dev_interim_cs` is same-distribution as `vss_cs` train and its labels started from Gipformer-family pseudo-labels (base WER 4.95). | Phase 5 claims | data (TODO.md 2e) |
| 6 | No checkpoint ever tested on the G0 baseline platform vs Windows; resolved for X0 by section 6, but the G0 baseline hyps remain from 16/11 WSL. | Nothing now | — |

## 10. The single next action

Phases 1–3 pass. **The first failed gate is Phase 4**: X0 loses Vietnamese on independent natural speech. Under the
"stop at the first failed gate" rule, Phase 5 stays blocked, and the next work is finding X0's root cause.

**Next action: reproduce X0 seed 0 locally** with the unchanged `configs/model/experiments/e4_x0.yaml`
(`gipbench.model.run`, all stages). Keep every 300-batch checkpoint and score each one on `dev_v2_vi@1` (beam 4,
INT8, batch 1, 16 threads, the Phase 1 official hyps as baseline).

Why this first:
- X0's checkpoint and training log exist only on Colab. A local run gives the loss curve, the checkpoints and the
  batch at which the Vietnamese regression appears.
- It tests whether the failure reproduces outside Colab. Every single-factor follow-up (augmentation off, replay data
  closer to natural speech) needs that baseline to compare against.
- Phase 3 removes the obvious explanations: the pipeline trains correctly, freezing holds, and export is faithful.
  D1 does not measurably change memorization.

Not yet done, and to be registered before running:
- The exact pass rule for "reproduces": dev_v2_vi ΔWER at batch 2400 inside the CI of X0's +4.83, or a new value.
- Whether the run fits in 6 GB RAM. Stage 2 builds 2 × 20 h with speed perturbation and `phone_sim`.
