---
name: asr-mechanistic-interpretability
description: Plan and run mechanistic-interpretability work on Gipformer 1.5 (Zipformer RNN-T): localise Vietnamese-English code-switching errors to encoder, predictor, joiner, decoding, tokenizer or data; compare base and fine-tuned checkpoints; test causal hypotheses with patching, ablation and low-rank edits; gate any weight change on causal evidence. Use before touching hooks, nnsight, pyvene or captum on this model, and before recommending any weight edit.
---

# ASR mechanistic interpretability for Gipformer 1.5

Gipformer is an icefall **Zipformer2 transducer** (6 encoder stacks + stateless predictor + additive joiner, 2000 BPE pieces, blank id 0). It is not a decoder-only LLM: there is no residual stream, no token position, no attention-head circuit to read off. "Position" means an acoustic frame. Everything below was checked against the real checkpoint with `scripts/diagnostics/interp_gipformer_probe.py` (see `docs/research/COMPATIBILITY_REPORT.md`).

## Hard rules

1. **Causal evidence before any weight recommendation.** A weight edit may be proposed only when (a) the component was localised by an intervention (patch or ablation), not by correlation, probing accuracy or a gradient map alone, (b) the effect replicates on a split not used to find it, (c) a matched control of the same norm/rate does not reproduce it, and (d) clean-speech regression is measured with a stated budget. Probes and attribution generate hypotheses; they never justify an edit.
2. **Three disjoint data roles.** *Discovery* (find where), *validation* (choose thresholds, directions, ranks), *held-out test* (touched once, last). Held-out never designs or selects anything. Thresholds and predictions are written down **before** the run. Every number carries its sample size (spans, utterances, recordings).
3. **Not every code-switching error lives in the weights.** Rule out, in this order: reference/label error and text normalisation; tokenizer/BPE coverage; features and resampling; INT8-ONNX vs FP32-PyTorch parity; decoding (beam, blank handling, hotwords); only then model components; then "gap" (needs data, not editing).
4. **Never write to a checkpoint.** All edits happen on an in-memory copy (`copy.deepcopy`). Saving an accepted edit goes through the repo's reversible-delta format (`gipformer_mech/patch_io.py`), never `torch.save` over `models/`.
5. **No external transmission.** The Mechanist `llm-chat` and `mechanic-db` MCP tools and its autonomous `auto*`/`run-experiment`/`monitor-experiment`/`experiment-queue`/`notify` skills are denied in `.claude/settings.json`. Do not lift that without explicit user approval; never put project data, transcripts or checkpoints into them.

## Environments (do not mix)

| Use | Environment | Notes |
|---|---|---|
| Real Gipformer (icefall + k2, torch 2.15 cu132) | WSL Ubuntu `~/.venvs/gipbench-ft`, icefall `~/icefall` @ `f8c8578` | CPU is enough for hooks and short clips. Never `pip install` into it. |
| nnsight / pyvene / captum on the real model | same env + overlay `.venv/interp-overlay-linux` **appended** to `sys.path` | appended, so core torch/numpy/scipy win |
| Toy-model and library tests (no checkpoint) | Windows `.venv/interp` (torch 2.15 CPU) | `scripts/diagnostics/interp_env_smoke.py` |

`torch.load` of a checkpoint under `/mnt/d` fails with `ENOMEM` in WSL; copy it read-only to `/tmp` first and delete it afterwards. Use `MSYS_NO_PATHCONV=1` when calling `wsl` from Git Bash and run multi-line commands from a script file.

## Model facts that change how tools behave

- Layout is **sequence-first** `(T, N, C)`. Stack outputs `model.encoder.encoders[k]` (k = 0..5) have channels 192/256/384/512/384/256 and downsampling 1/2/4/8/4/2 internally, but **every stack output is restored to the same frame grid**. `encoder_embed` gives `(N, T, 192)`; the final `forward_encoder` output is `(N, T', 512)`. For 400 fbank frames (10 ms hop): 196 frames in the stacks (20 ms), 98 at the output (40 ms). The 512-d output is a concatenation of channel slices from stacks 5, 4 and 3, not a residual sum.
- Tokenizer: `bpe.model` / `tokens.txt`, upper-case vocabulary, `<unk>` and blank are distinct. Predictor context size 2, so the predictor only sees the last two emitted pieces.
- Joiner is `W · tanh(enc + dec) + b`; the repo already decomposes it exactly per unit (`joiner_probe.py`).
- **Gradients through icefall are not clean by default.** `Balancer` and `Whiten` rewrite the backward pass at random even in `eval()` (their only gate is `x.requires_grad`): measured 10–38 % change in a mid-stack gradient between draws on real speech. `SwooshLForward/RForward` (used in eval) overflow in `exp()` and return NaN gradients on out-of-distribution input although the forward is finite. Before any gradient or attribution work, in-process: set `scaling.Balancer.forward` and `scaling.Whiten.forward` to identity, and replace `scaling.SwooshLForward/RForward` with the `softplus` form (forward changes < 2e-6). Do not edit icefall files.
- Frame ↔ time ↔ piece alignment must come from a forced alignment of the reference (as in `logit_lens.align_reference`), not from index arithmetic alone.

## Tool routing

| Need | Use | Gotcha on this model |
|---|---|---|
| Read activations, any module | native `register_forward_hook` (what `gipformer_mech/logit_lens.py` does) or `nnsight` | nnsight must wrap an `EncoderOnly(nn.Module)` that calls `model.forward_encoder`; `AsrModel.forward` needs k2 targets. Paths read `wrapped.model.encoder.encoders[k].output`. Save with `nnsight.save([])` + loop, not a list comprehension. |
| Patch / ablate an activation | nnsight (`output[:] = ...`) or a forward hook that returns the replacement | Hooks must return the tensor or `None`; returning a tuple replaces the output with a tuple. |
| Gradient / attribution | forward hooks with `retain_grad()`, nnsight with `retain_grad()` + `loss.backward()` after the trace, captum `LayerGradientXActivation` | nnsight's interleaved `with metric.backward():` raised `OutOfOrderError` on Zipformer; use the retain-grad form. Apply the gradient-hygiene patches above first. |
| Frame-level interchange intervention | pyvene on a **batch-first tap** | pyvene's `pos` unit indexes dim 1, which is the batch axis here and fails. Wrap the stack in `TappedStack` (a `Tap` submodule that transposes to `(N,T,C)`) on a copy, target `...encoders.2.tap.output`, and add the three shims `model.config`, `model.device`, `pv.type_to_dimension_mapping[type(net)] = {}`. |
| Hypothesis, design, audit, claim checking | Mechanist skills (`mechanist:experiment-plan`, `ablation-planner`, `mechanism-audit`, `experiment-audit`, `result-to-claim`, `data-rule`, `impact-check`) | They assume LLM interpretability and Hugging Face/TransformerLens. Use their planning and audit checklists; ignore their model-specific code. Autonomous skills are denied. |
| Technique references | `nnsight:nnsight` guides (ablation, activation-patching, attribution-patching, causal-tracing, logit-lens, probing, model-editing-and-lora) and `pyvene` | Examples are token-position based; map "position" to frame and "layer" to stack. |

When a library gets in the way, fall back to native PyTorch hooks. They are the reference every other tool is compared against.

## Playbooks

**1. ASR failure localisation.** Start from error spans, not whole-utterance WER. Reuse the locked span lists (`reports/phase2_span_lock.jsonl`) and the `logit_lens.classify` categories (near miss / encoder bug / predictor override / true gap / inconclusive). State the operational definition used: these are rank-based labels, not causal proof. Split every claim by error type (English CS span, Vietnamese word inside a CS sentence, clean sentence).

**2. Encoder vs predictor vs joiner.** Logit lens per stack via `logit_lens.cumulative_outputs`; then three counterfactual runs on the same frames: (a) true predictor context vs greedy-emitted prefix (isolates the predictor), (b) patch the clean-run stack output into the failing run (isolates the encoder), (c) edit or ablate a joiner unit set (isolates the joiner). Report which component's patch recovers the target piece rank, with a same-size random-patch control.

**3. Tokenizer and vocabulary audit.** Encode every CS term with `bpe.model`; count pieces per term, `<unk>` rate, and whether the first piece is a word-initial piece. Check `tokens.txt` ↔ `bpe.model` id agreement (`model_audit`). A term the BPE splits into rare fragments is a data/tokenizer gap, not a weight bug; do not edit weights for it.

**4. Beam-search and decoding error analysis.** Re-decode the same features with greedy, beam 4 and the blank-penalty variants in `onnx_greedy.py`. If the target piece is in the top-10 at the final layer but loses to blank, the failure is calibration/search, and the repo's phases 3–6 already tested static fixes (all rejected). Never conclude from one decoder.

**5. Layer-wise representation analysis.** Per-stack statistics over a fixed clip set: norm, rank (SVD energy), CKA or linear predictability between stacks. Compare like with like (same frames, same fbank, FP32 PyTorch). INT8 ONNX activations are not comparable to PyTorch activations without a measured parity check.

**6. Language and phoneme probing.** Frame-level linear probes (language ID of the aligned word; phone class if a lexicon exists) on stack outputs, trained on discovery, scored on validation. Always report a control: shuffled labels, and a probe on `encoder_embed`. A decodable feature is not a used feature; only a patch or ablation shows use.

**7. Temporal activation alignment.** Map frames to time (see Model facts) and to BPE pieces via forced alignment. Check the alignment on a few clips by listening or by comparing to word boundaries before using it at scale. Edge frames and the final downsampled frame need explicit handling.

**8. Base vs fine-tuned comparison.** Load both state dicts read-only (`torch.load(..., mmap=True)` from local disk), compare by name and shape (the base and `init_ft.pt` currently share 743/745 tensors and are identical). Record which keys are missing before reading deltas.

**9. Weight-delta analysis and SVD.** Per tensor: `‖Δ‖/‖W‖`, effective rank of Δ, alignment of Δ's top singular vectors with W's. Aggregate by stack and by module type (`feed_forward1/2/3`, `self_attn`, `conv`, joiner, predictor). A delta that is large is not necessarily causal: confirm with patching Δ-restricted weights back into the base.

**10. Causal activation patching.** Corrupted (failing) input as base, clean input as source, frame-level and stack-level, on matched utterance pairs. Metric: rank or logit margin of the target piece at the aligned frame, not WER. Include identity patches (must be no-ops) and random-source patches (must not help).

**11. Selective layer transplantation.** `load_state_dict` of one stack's tensors from donor into a **copy** of the base. Verify the original is untouched (`torch.equal` on a tensor sample) and that transplanting an identical donor is a no-op. Measure clean-speech regression before judging gains.

**12. Low-rank weight editing.** Only after rule 1 is met. Edit on a copy, with the repo's tools (`weight_surgeon.py`, `patch_io.py`), restricted to a subspace chosen on discovery data; register the stop rule and the damage budget first. Compare against random same-norm directions in the same subspace (phase 6 did this: the edits fixed training-fold spans but reached the target in only 8–12% of held-out-fold events and damaged 3.9–14.6% of clean rows; read `reports/phase6_joiner_null_edit.md` before proposing another).

**13. Catastrophic forgetting and regression.** After any edit or transplant: Vietnamese-only WER, English-only, CS, plus the clean-row damage rate (fraction of previously-correct rows that break). Use paired bootstrap from `gipformer_mech/acceptance.py`. Report absolute and relative change with CIs and n.

**14. Data leakage prevention.** Before each run: verify the pool hash against `heldout_lock.json` (`development_pool_sha256`), assert disjointness by recording/speaker (`splits.py`), and never open the held-out manifest in discovery code. Keep recordings of one speaker on one side. Mechanist's `data-rule` skill may help write the contract; the repo's `docs/CONTRACTS.md` and leak gates are authoritative. Do not tune on the inputs used to demonstrate a result.

## Workflow

1. Write the question and the prediction that would refute it. Register thresholds.
2. Rule out non-weight causes (rule 3). Stop if one explains the errors.
3. Localise with lens and counterfactual runs on discovery data.
4. Confirm causally on validation with controls.
5. Only then design an edit (rule 1), apply to a copy, measure damage.
6. Run the held-out test once. Report negatives. Update `TODO_rea.md` or the relevant report; do not rewrite past results.

## Verification of the toolchain

`interp_env_smoke.py` (toy, any machine) and `interp_gipformer_probe.py` (real checkpoint, WSL) re-check extraction, patching, gradients, parameter inspection and transplantation. Re-run them after upgrading nnsight, pyvene, torch or icefall.
