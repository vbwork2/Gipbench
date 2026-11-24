# Compatibility report: nnsight, pyvene, captum, Mechanist vs Gipformer 1.5

24/11/2026. All real-model results come from `scripts/diagnostics/interp_gipformer_probe.py` (CPU, read-only, one dev-pool clip of 400 fbank frames, batch 1). Toy results come from `scripts/diagnostics/interp_env_smoke.py`. n is tiny by design: these are tooling checks, not scientific findings.

## Verdict

Ready for **CPU-scale, hook-based mechanistic analysis** of the PyTorch checkpoint, using the documented patches. Not ready for autonomous research. Mechanist is usable for planning and audit only. Several things are untested (below).

## Target model, as inspected

Icefall `zipformer` recipe at commit `f8c8578`, loaded through `gipformer_mech.rnnt_loader.load_model` (sha256-checked `model.pt`, 745 state tensors, 70 M parameters).

| Part | Facts |
|---|---|
| Features | 80-dim kaldi fbank, 10 ms hop; `encoder_embed` (Conv2dSubsampling) gives `(N, 196, 192)` for 400 frames |
| Encoder | `model.encoder.encoders[0..5]`, channels 192/256/384/512/384/256, internal downsampling 1/2/4/8/4/2; every stack output restored to the same 196-frame grid; tensors **sequence-first** `(T, N, C)` |
| Encoder output | `(N, 98, 512)`: concatenation of channel slices from stacks 5, 4 and 3, then a final 2x downsample (40 ms/frame) |
| Predictor | stateless: embedding + conv, context size 2 |
| Joiner | `output_linear` weight `(2000, 512)`; 90 % of spectral energy in 65 singular directions |
| Heads | transducer + CTC (`ctc_output`) + simple_am/lm projections |
| Custom autograd | `Balancer`, `Whiten`, `SwooshL/R`, `BiasNorm`, `ActivationDropoutAndLinear` in `scaling.py` |
| Environment | WSL: Python 3.11.18, torch 2.15.0+cu132, numpy 2.4.6, scipy 1.17.1. Windows venv: torch 2.15.0+cpu (no k2/icefall) |

## Results

### Toy RNN-T (strided conv front end, 3 multi-rate stacks, predictor, joiner): 13/13

Native hook extraction; nnsight extraction equal to native; identity patch no-op and clean-activation patch restores the clean output exactly; nnsight gradient equals autograd; single-channel ablation; pyvene import, API and interchange on a custom module; captum `Saliency` and `LayerGradientXActivation`; state-dict inspection with SVD and no mutation; stack transplant with original untouched.

### Real Gipformer: 15/15 passing, after the workarounds listed below

| Capability asked for | Result |
|---|---|
| Custom PyTorch modules | nnsight works on a wrapper module; pyvene works only with shims |
| Zipformer encoder stacks | nnsight extraction of all 6 stacks and the final output identical to native hooks (atol 1e-5) |
| Acoustic features, subsampling | fbank built with the repo's kaldi settings; shapes 400 -> 196 -> 98 confirmed; tools hook the PyTorch graph after features, they do not process audio |
| Multi-resolution representations | all stacks share one grid on output; internal downsampled activations are reachable only through submodules |
| Prediction network, token embeddings | hookable like any module (`decoder.embedding`, `decoder.conv`); not individually exercised |
| RNN-T joiner | rank-1 edit of `joiner.output_linear` on a copy changes logits, original untouched |
| Intermediate activations | pass (nnsight and native) |
| Gradient extraction | pass, but only after gradient hygiene (below) |
| Checkpoint parameter inspection | pass: 745 tensors, per-component parameter counts, joiner SVD |
| Selective parameter replacement | pass on an in-memory deep copy; base and `init_ft.pt` share 743/745 tensors (the 2 missing are `ctc_output.1.*`) and **are identical**, so the transplant was a verified no-op, not a test of a real delta |

## Issues found, with evidence and handling

1. **Balancer and Whiten corrupt gradients in eval mode.** Their only gate is `x.requires_grad`. A mid-stack (stack 3) gradient of a single output frame changed by 9.7e-2, 3.2e-1, 3.8e-1 (relative) over three draws. Handling: bypass both in-process; after that two consecutive gradients are bit-identical.
2. **NaN gradients from SwooshL/R in eval.** `SwooshLForward` computes `log(1 + exp(x - 4))` and masks `inf` with `torch.where`; the forward is fine, the backward is `0 * inf = NaN`. With random N(0,1) features, the gradient was NaN for stacks 0-4 (found with anomaly detection: `ExpBackward0`). With real speech it was finite. Handling: `softplus` form, forward diff 1.7e-6, gradient diff 1.2e-6.
3. **nnsight interleaved backward fails.** `with metric.backward(): x.grad.save()` raises `OutOfOrderError` for stack 3 and for the last stack on this model (works on the toy). `retain_grad()` + `loss.backward()` after the trace matches autograd exactly.
4. **pyvene `pos` unit indexes dim 1.** On sequence-first tensors that is the batch axis: `index 3 is out of bounds for dimension 1 with size 1`. Handling: on a copy, wrap a stack in `TappedStack` with a transposing `tap` submodule and target `...tap.output`; the result equals a manual native frame patch (max diff 0.0). A custom `intervention=` callable does not help because pyvene gathers units first.
5. **pyvene on an unregistered class needs three shims**: `model.config`, `model.device`, `pv.type_to_dimension_mapping[type(net)] = {}` (found by probing: `no attribute config`, then `KeyError`, then `no attribute device`).
6. **nnsight traps carried over**: list comprehensions inside a trace lose saved names (use `nnsight.save([])` and a loop); forward hooks must return `None` or the replacement tensor, not a tuple.
7. **`AsrModel.forward` needs k2 targets**, so tracing the whole model is not possible; trace `forward_encoder` through a wrapper. Joiner and predictor checks therefore call those modules directly.
8. **WSL `torch.load` from `/mnt/d` fails with `ENOMEM`** (reproduced twice, with 1.3 and 2.1 GB available). Loading from `/tmp` works. Checkpoints were copied read-only and the copies deleted.
9. **Install-name mismatch**: the task's `nnsight@skills` does not exist; the marketplace is named `ndif-team`. Use `nnsight@ndif-team`.
10. **Orchestra's pyvene skill is written for Hugging Face LLMs.** Component names and the GPT-2 causal-tracing recipe do not apply; use dotted module paths plus `.output`.

## Dependency conflicts

- Mechanist `requirements.txt` (`torch==2.7.1`, `numpy==1.26.4`, `transformers==4.57.0`, `vllm`, `flash-attn`, `evo2`, `autoawq`) conflicts with the core env (torch 2.15) and with nnsight 0.8 (needs transformers >= 5). **Not installed**; it is only for experiments Mechanist would run itself, which are disabled.
- nnsight and pyvene pull `transformers 5.24.0`, `pydantic`, `plotnine`/`matplotlib`, `tokenizers` etc. None of those exist in the core env, hence the isolated venv and the appended overlay (core packages win on import).
- Overlay wheels for Linux/cp311 had to drop exact pins on plotting packages (`contourpy 1.4.0` requires Python >= 3.12). They are not used by the probe.
- Not modified: `.venv/gipbench-win`, WSL `gipbench-ft` (torch, CUDA, k2, icefall, numpy, scipy), `models/`, `data/`, any benchmark file. A core-env hash comparison was not possible (no pip in the WSL env); the guarantee rests on installing only into `.venv/interp` and `.venv/interp-overlay-linux`.
- `python` on PATH is a miniconda 3.13 with none of these packages; do not use it.

## Not supported or not tested

- **INT8 ONNX decoding path** (`encoder/decoder/joiner.int8.onnx`, sherpa-onnx): none of these tools can hook it. Conclusions from PyTorch FP32 activations need a measured parity check against the INT8 deployment.
- **Not tested**: GPU (RTX 3050, 4 GB), streaming/chunked mode, the CTC head, batch > 1 with padding masks, long utterances, a genuinely different fine-tuned checkpoint, Colab, `ndif`/remote execution (not wanted).
- **Mechanist runtime**: skills and MCP servers were inventoried and statically reviewed but never run; its deny-listed tools and skills were not exercised. Its method skills assume LLM interpretability (TransformerLens/Hugging Face).
- Only one speech clip was used for numeric checks (a dev bug-mining clip; heldout untouched). Gradient and attribution numbers are tooling checks, not results.
- Skill recognition was confirmed with a headless `claude -p` listing (`pyvene`, `nnsight:nnsight`, 36 `mechanist:*`); behaviour of the skills in use was not evaluated.

## Known limitations of the approach

LLM-style assumptions do not transfer: there is no residual stream, "position" is a 20-40 ms frame, the 512-d output is assembled from stack slices, and the predictor sees two previous pieces only. Rank-based labels in the existing near-miss analysis are operational, not causal. Any weight modification still requires the causal-evidence rule in the project skill.
