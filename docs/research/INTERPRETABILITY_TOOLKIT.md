# Interpretability toolkit for Gipformer 1.5

What to use for which question, how to reproduce the environment, and the recommended workflow. Evidence behind every statement is in [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md); package versions and sources are in [SKILLS_INVENTORY.md](SKILLS_INVENTORY.md). The agent-facing version is `.claude/skills/asr-mechanistic-interpretability/SKILL.md`.

## Layers of the toolkit

1. **Native PyTorch hooks** (`register_forward_hook`, `retain_grad`, `load_state_dict` on copies). Always available, already used by `src/gipformer_mech/logit_lens.py`. The reference every other tool is checked against.
2. **nnsight 0.8.4**: readable traces, saved activations, in-trace patching. Wrap the encoder in a small `nn.Module` calling `model.forward_encoder`.
3. **pyvene 0.1.8**: declarative interchange interventions. Needs a batch-first tap and three shims on this model.
4. **captum 0.9.0**: layer attribution (`LayerGradientXActivation`, `Saliency`) once the gradient patches are on.
5. **Claude Code skills**: `nnsight:nnsight` (technique guides), `pyvene`, `mechanist:*` planning and audit skills, and `asr-mechanistic-interpretability` (routing and rules).
6. **Repo code**: `gipformer_mech` (logit lens, joiner decomposition, trigger, weight surgeon, reversible patches, paired bootstrap, split and lock checks).

## Which tool for which job

| Job | First choice | Notes |
|---|---|---|
| Per-stack activations | native hooks or nnsight | 6 stacks, tensors `(T, N, C)` |
| Logit lens / near-miss classification | `logit_lens.py` | existing, validated |
| Frame-level patching | nnsight `output[:] = ...` | identity patch must be a no-op |
| Interchange by frame with pyvene | `TappedStack` adapter | see probe script for the exact code |
| Gradient maps | retain-grad form + gradient patches | interleaved `metric.backward()` fails on Zipformer |
| Joiner analysis | `joiner_probe.py` | exact per-unit decomposition |
| Checkpoint comparison / SVD | `torch.load(mmap=True)` + `torch.linalg` | read-only, from local disk in WSL |
| Weight edit trial | `weight_surgeon.py`, `patch_io.py` on a deep copy | only after causal evidence |
| Experiment design, claim audit | `mechanist:experiment-plan`, `mechanism-audit`, `result-to-claim` | planning only; autonomous skills are denied |

## Gradient hygiene (required before any gradient or attribution work)

Applied in-process; icefall files are never edited. Importing `scaling` works after `load_model` has put the icefall recipe on `sys.path`.

```python
import torch.nn.functional as F, scaling
scaling.Balancer.forward = lambda self, t: t     # random backward rewrites, active even in eval()
scaling.Whiten.forward = lambda self, t: t
scaling.SwooshLForward = lambda t: F.softplus(t - 4.0) - 0.08 * t - 0.035          # exp() overflow -> NaN grads
scaling.SwooshRForward = lambda t: F.softplus(t - 1.0) - 0.08 * t - 0.313261687
```

Measured: forward output changes by 1.7e-6, gradient by 1.2e-6 versus the shipped form on real speech (finite case); Balancer/Whiten alone moved a mid-stack gradient by 9.7e-2, 3.2e-1 and 3.8e-1 over three draws.

## Reproduce the environment

Run from the repo root. Nothing here touches `.venv/gipbench-win` or the WSL `gipbench-ft` env.

```bash
# 1. Claude Code plugins, project scope
claude plugin marketplace add ndif-team/skills --scope project
claude plugin marketplace add zjunlp/Mechanist --scope project
claude plugin install nnsight@ndif-team --scope project      # marketplace is "ndif-team", not "skills"
claude plugin install mechanist@mechanist --scope project
claude plugin list                                           # expect nnsight fb71ca667ec9, mechanist 0.1.0

# 2. PyVene skill pinned to the reviewed commit
git clone https://github.com/Orchestra-Research/AI-Research-SKILLs.git /tmp/orch
git -C /tmp/orch checkout 773a52944ba4747a18bd4ae9ade53fff041adcbc
mkdir -p .claude/skills && cp -r /tmp/orch/04-mechanistic-interpretability/pyvene .claude/skills/pyvene

# 3. Isolated Windows venv (toy tests, library work)
uv venv -p 3.12 .venv/interp
uv pip install -p .venv/interp/Scripts/python.exe "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/interp/Scripts/python.exe nnsight==0.8.4 pyvene==0.1.8 captum numpy scipy scikit-learn
#    exact versions: docs/research/interp-env.lock.txt

# 4. Linux overlay for the WSL Icefall env (wheels only, appended to sys.path, core env untouched)
uv pip install --python-platform x86_64-manylinux_2_28 --python-version 3.11 \
    --target .venv/interp-overlay-linux --no-deps -r docs/research/interp-overlay-linux.txt
```

Also add the deny rules from `.claude/settings.json` (see SKILLS_INVENTORY) before enabling Mechanist; the plugin install alone does not restrict anything.

Verify:

```bash
# toy model, any machine (13 checks)
.venv/interp/Scripts/python.exe -I scripts/diagnostics/interp_env_smoke.py

# real checkpoint, WSL, CPU, read-only (15 checks). Stage read-only copies on the WSL disk first:
# torch.load from /mnt/d fails with ENOMEM. Use MSYS_NO_PATHCONV=1 when calling wsl from Git Bash.
mkdir -p /tmp/gip15_probe && cp models/gip15/{model.pt,init_ft.pt,bpe.model} /tmp/gip15_probe/
~/.venvs/gipbench-ft/bin/python scripts/diagnostics/interp_gipformer_probe.py \
    --model-dir /tmp/gip15_probe --icefall ~/icefall --overlay .venv/interp-overlay-linux \
    --wav <one dev-pool clip, never heldout>
rm -rf /tmp/gip15_probe
```

Colab was not tested. To use it, install the same pins into a fresh runtime venv and do not reinstall torch, k2 or icefall there either.

## Recommended research workflow

1. **Question and refutation first.** Write the hypothesis, the prediction that would kill it, thresholds and sample sizes before running.
2. **Rule out non-weight causes**: labels and normalisation, tokenizer/BPE coverage, features, INT8-vs-FP32 parity, decoding.
3. **Localise** with the logit lens and counterfactual runs (predictor context, clean-stack patch, joiner-unit edit) on *discovery* data.
4. **Confirm causally** on *validation* data with identity and random-source controls and matched-norm random edits.
5. **Only then** design a weight edit, on a deep copy, with a pre-registered damage budget; measure clean-speech regression with paired bootstrap.
6. **Held-out test once.** Report negatives.

Current project context (from `TODO_rea.md`, 24/11/2026): on FPT-based diagnostics the dominant failure class is after the encoder (Joiner calibration and cascade); every static decoding and weight fix tried (phases 3–6) was rejected. New interpretability work should be aimed at what those phases left open, not at repeating them.

## Known limits

See COMPATIBILITY_REPORT, section "Not supported or not tested". Short version: INT8 ONNX models cannot be hooked by these tools; streaming, CTC head, GPU, batch>1 with padding and a genuinely fine-tuned checkpoint were not tested; Mechanist is LLM-oriented and its runtime was not exercised.
