"""Read-only compatibility probe: do nnsight / pyvene / captum / native hooks work on the real Gipformer 1.5 module?

CPU only, random synthetic fbank input, no dataset, nothing written, no checkpoint modified. Run it in the WSL
environment that already has Icefall + k2 (the repo's existing mech code needs it), with the interp overlay
appended to sys.path so the core packages (torch, numpy, scipy) are never shadowed:

    python scripts/diagnostics/interp_gipformer_probe.py --model-dir models/gip15 --icefall ~/icefall \
        --overlay .venv/interp-overlay-linux

Prints PASS / FAIL / NOTE per check and exits non-zero on any FAIL.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, str, str]] = []


def record(name, status, detail=""):
    RESULTS.append((name, status, str(detail)))


def check(name):
    def wrap(fn):
        try:
            record(name, "PASS", fn() or "")
        except Exception as error:  # noqa: BLE001 - report every failure
            record(name, "FAIL", f"{type(error).__name__}: {str(error)[:300]}")
            traceback.print_exc()
        return fn

    return wrap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/gip15")
    parser.add_argument("--icefall", type=Path, default=Path.home() / "icefall")
    parser.add_argument("--overlay", type=Path, default=ROOT / ".venv/interp-overlay-linux")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--wav", type=Path, default=None, help="optional speech clip (dev pool only, never heldout) for realistic fbank")
    args = parser.parse_args()
    if args.overlay.is_dir():
        sys.path.append(str(args.overlay))  # appended, so the core env's torch/numpy/scipy always win

    import torch
    from torch import nn

    from gipformer_mech.rnnt_loader import load_model

    torch.manual_seed(0)
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    model, _ = load_model(args.model_dir, args.icefall, model_map, "cpu")
    model.eval()
    n_stacks = len(model.encoder.encoders)
    record("load real Gipformer 1.5 through Icefall (sha256-checked)", "PASS", f"torch {torch.__version__}, {n_stacks} encoder stacks")

    class EncoderOnly(nn.Module):
        """nnsight / pyvene trace a module's forward; AsrModel.forward needs k2 targets, so wrap forward_encoder."""

        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, x, x_lens):
            encoder_out, _ = self.model.forward_encoder(x, x_lens)
            return encoder_out

    wrapper = EncoderOnly(model).eval()
    if args.wav is not None:
        import kaldi_native_fbank as kaldi
        import numpy as np
        import soundfile

        from gipformer_mech.rnnt_loader import resample

        wave, rate = soundfile.read(args.wav, dtype="float32")
        wave = resample(wave if wave.ndim == 1 else wave.mean(axis=1), rate, model_map["sample_rate"])
        opts = kaldi.FbankOptions()
        opts.frame_opts.samp_freq = model_map["sample_rate"]
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.high_freq = -400
        opts.mel_opts.num_bins = model_map["feature_dim"]
        fbank = kaldi.OnlineFbank(opts)
        fbank.accept_waveform(model_map["sample_rate"], wave.tolist())
        fbank.input_finished()
        frames = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])[: args.frames]
        x = torch.from_numpy(frames).unsqueeze(0)
        record("input features", "NOTE", f"real fbank from one dev-pool clip, {tuple(x.shape)}, range [{float(x.min()):.1f}, {float(x.max()):.1f}]")
    else:
        x = torch.randn(1, args.frames, 80)
        record("input features", "NOTE", f"random N(0,1) fbank {tuple(x.shape)} (out of distribution; pass --wav for real speech)")
    x_lens = torch.tensor([x.shape[1]])
    state = {}

    def mid_gradient(layer=3, frame=None):
        frame = x.shape[1] // 8 if frame is None else frame
        store = {}

        def keep(_m, _i, out):
            out.retain_grad()
            store["h"] = out

        handle = model.encoder.encoders[layer].register_forward_hook(keep)
        wrapper(x, x_lens)[0, frame].pow(2).sum().backward()
        handle.remove()
        model.zero_grad(set_to_none=True)
        return store["h"].grad.clone()

    @check("gradient hygiene: Icefall Balancer/Whiten and SwooshL/R backward in eval mode (measure, bypass, verify)")
    def _():
        import torch.nn.functional as F

        import scaling

        # (a) Balancer / Whiten rewrite the backward pass at random even in eval mode (x.requires_grad is their only gate)
        random.seed(0)
        draws = [mid_gradient() for _ in range(3)]
        finite = all(torch.isfinite(g).all() for g in draws)
        record("NOTE gradient as shipped (Balancer/Whiten/Swoosh active)", "NOTE", "finite" if finite else "NaN/inf present")
        scaling.Balancer.forward = lambda self, t: t
        scaling.Whiten.forward = lambda self, t: t
        bypassed = [mid_gradient() for _ in range(2)]
        assert torch.equal(bypassed[0], bypassed[1]), "gradient is not deterministic after bypass"
        if finite:
            rel = [float((g - bypassed[0]).norm() / bypassed[0].norm()) for g in draws]
            record("NOTE relative gradient change caused by Balancer/Whiten (3 random draws)", "NOTE", [f"{r:.2e}" for r in rel])
        # (b) SwooshL/R eval path: exp() overflows for large pre-activations; torch.where hides it in the forward only
        with torch.no_grad():
            before = wrapper(x, x_lens)
        shipped = bypassed[0]
        scaling.SwooshLForward = lambda t: F.softplus(t - 4.0) - 0.08 * t - 0.035
        scaling.SwooshRForward = lambda t: F.softplus(t - 1.0) - 0.08 * t - 0.313261687
        with torch.no_grad():
            after = wrapper(x, x_lens)
        forward_diff = float((after - before).abs().max())
        assert forward_diff < 1e-4, f"stable Swoosh changes the forward by {forward_diff:.2e}"
        stable = mid_gradient()
        assert torch.isfinite(stable).all()
        swoosh = f"rel diff {float((shipped - stable).norm() / stable.norm()):.1e}" if torch.isfinite(shipped).all() else "shipped Swoosh gradient NaN, stable finite"
        return f"bypass makes the gradient deterministic; stable Swoosh forward diff {forward_diff:.1e}, gradient {swoosh}. Patches (in-process only) stay active for later checks"

    # ---- native hooks: the fallback every other tool is compared with ----
    @check("native hooks: per-stack outputs (layout, rates, channels)")
    def _():
        captured = {}
        handles = [
            stack.register_forward_hook(lambda _m, _i, out, k=k: captured.__setitem__(k, out.detach().clone()))
            for k, stack in enumerate(model.encoder.encoders)
        ]
        embed_out = {}
        handles.append(model.encoder_embed.register_forward_hook(lambda _m, _i, out: embed_out.__setitem__("o", out[0].detach())))
        with torch.no_grad():
            final = wrapper(x, x_lens)
        for handle in handles:
            handle.remove()
        state["native"], state["final"] = captured, final
        shapes = {k: tuple(v.shape) for k, v in captured.items()}
        record("NOTE stack output shapes (T_k, N, C_k): sequence-first, channels differ per stack", "NOTE", shapes)
        record("NOTE encoder_embed output (N,T,C)", "NOTE", tuple(embed_out["o"].shape))
        record("NOTE final encoder_out (N,T',C)", "NOTE", tuple(final.shape))
        assert all(isinstance(v, torch.Tensor) for v in captured.values())
        return f"{n_stacks} stacks captured"

    # ---- nnsight ----
    try:
        import nnsight
        from nnsight import NNsight
    except Exception as error:  # noqa: BLE001
        nnsight = None
        record("import nnsight", "FAIL", repr(error))
    else:
        record("import nnsight", "PASS", nnsight.__version__)

    if nnsight is not None:
        wrapped = NNsight(wrapper)

        @check("nnsight: extraction at every Zipformer stack equals native hooks")
        def _():
            with wrapped.trace(x, x_lens):
                saved = nnsight.save([])
                for stack in wrapped.model.encoder.encoders:
                    saved.append(stack.output)
                final = wrapped.output.save()
            for k, tensor in enumerate(saved):
                assert tensor.shape == state["native"][k].shape and torch.allclose(tensor, state["native"][k], atol=1e-5), k
            assert torch.allclose(final, state["final"], atol=1e-5)
            return "6/6 stacks and final output identical to native hooks"

        @check("nnsight: identity patch is a no-op, zeroing a stack changes the encoder output")
        def _():
            with wrapped.trace(x, x_lens):
                wrapped.model.encoder.encoders[2].output[:] = state["native"][2]
                same = wrapped.output.save()
            with wrapped.trace(x, x_lens):
                wrapped.model.encoder.encoders[2].output[:] = 0.0
                zeroed = wrapped.output.save()
            assert torch.allclose(same, state["final"], atol=1e-5)
            assert not torch.allclose(zeroed, state["final"], atol=1e-3)
            return f"|zeroed - clean| / |clean| = {float((zeroed - state['final']).norm() / state['final'].norm()):.3f}"

        @check("nnsight: gradient w.r.t. a mid-stack activation equals autograd (retain_grad + backward after the trace)")
        def _():
            frame = x.shape[1] // 8
            try:  # the documented interleaved form
                with wrapped.trace(x, x_lens):
                    hidden = wrapped.model.encoder.encoders[3].output
                    with wrapped.output[0, frame].pow(2).sum().backward():
                        _unused = hidden.grad.save()
                record("NOTE nnsight interleaved `with metric.backward():` on Zipformer", "NOTE", "worked")
            except Exception as error:  # noqa: BLE001
                record("NOTE nnsight interleaved `with metric.backward():` on Zipformer", "NOTE", f"fails: {type(error).__name__}; use retain_grad + backward after the trace")
            with wrapped.trace(x, x_lens):
                held = nnsight.save([])
                hidden = wrapped.model.encoder.encoders[3].output
                hidden.retain_grad()
                held.append(hidden)
                loss = wrapped.output[0, frame].pow(2).sum().save()
            loss.backward()
            grad = held[0].grad.clone()
            model.zero_grad(set_to_none=True)
            assert torch.allclose(grad, mid_gradient(), atol=1e-5, rtol=1e-4)
            return f"grad {tuple(grad.shape)} matches autograd, norm {float(grad.norm()):.4e}"

    # ---- pyvene ----
    try:
        import pyvene as pv
    except Exception as error:  # noqa: BLE001
        pv = None
        record("import pyvene", "FAIL", repr(error))
    else:
        record("import pyvene", "PASS", "0.1.8")

    if pv is not None:
        source = torch.randn_like(x) if args.wav is None else x.roll(shifts=37, dims=1)  # a different input of the same length

        def run_pyvene(net, component):
            net.config = types.SimpleNamespace()
            net.device = torch.device("cpu")
            pv.type_to_dimension_mapping[type(net)] = {}
            representation = pv.RepresentationConfig(layer=0, component=component, unit="pos", max_number_of_units=1, intervention_type=pv.VanillaIntervention)
            intervenable = pv.IntervenableModel(pv.IntervenableConfig(representations=[representation]), net)
            with torch.no_grad():
                _, out = intervenable({"x": x, "x_lens": x_lens}, [{"x": source, "x_lens": x_lens}], {"sources->base": ([[[3]]], [[[3]]])})
            return out

        class Tap(nn.Module):
            """Transposes (T, N, C) -> (N, T, C) so a pyvene 'pos' unit indexes time; the stack restores the layout."""

            def forward(self, t):
                return t.transpose(0, 1)

        class TappedStack(nn.Module):
            def __init__(self, stack):
                super().__init__()
                self.stack = stack
                self.tap = Tap()

            def forward(self, *a, **k):
                return self.tap(self.stack(*a, **k)).transpose(0, 1)

            def __getattr__(self, name):  # Zipformer2 reads attributes of its stacks
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.stack, name)

        @check("pyvene: documented 'pos' unit directly on a stack output fails on sequence-first tensors")
        def _():
            try:
                run_pyvene(wrapper, "model.encoder.encoders.2.output")
            except RuntimeError as error:
                return f"fails as predicted ({str(error)[:90]}): 'pos' indexes dim 1 = batch axis"
            raise AssertionError("pyvene unexpectedly accepted a sequence-first tensor; re-check frame semantics")

        @check("pyvene: batch-first Tap adapter on a copy lets 'pos' address one time frame; equals a manual native patch")
        def _():
            trial = copy.deepcopy(model)
            trial.encoder.encoders[2] = TappedStack(trial.encoder.encoders[2])
            net = EncoderOnly(trial).eval()
            out = run_pyvene(net, "model.encoder.encoders.2.tap.output")
            with torch.no_grad():  # manual reference through native hooks on the unmodified model
                taken = {}
                handle = model.encoder.encoders[2].register_forward_hook(lambda _m, _i, o: taken.__setitem__("o", o.detach().clone()))
                wrapper(source, x_lens)
                handle.remove()
                handle = model.encoder.encoders[2].register_forward_hook(lambda _m, _i, o: o.index_copy(0, torch.tensor([3]), taken["o"][3:4]))
                manual = wrapper(x, x_lens)
                handle.remove()
            assert out.shape == state["final"].shape and not torch.allclose(out, state["final"])
            assert torch.allclose(out, manual, atol=1e-5), float((out - manual).abs().max())
            return f"matches manual frame-3 patch (max diff {float((out - manual).abs().max()):.1e}); |patched-clean|/|clean| = {float((out - state['final']).norm() / state['final'].norm()):.4f}"

    # ---- captum ----
    try:
        import captum
        from captum.attr import LayerGradientXActivation
    except Exception as error:  # noqa: BLE001
        captum = None
        record("import captum", "FAIL", repr(error))
    else:
        record("import captum", "PASS", captum.__version__)

    if captum is not None:

        @check("captum: LayerGradientXActivation runs through a Zipformer stack (shape + non-zero check only)")
        def _():
            frame = x.shape[1] // 8

            def forward(inputs):
                return wrapper(inputs, x_lens)[0, frame].pow(2).sum().reshape(1)

            attributions = LayerGradientXActivation(forward, model.encoder.encoders[3]).attribute(x.clone())
            attr = attributions[0] if isinstance(attributions, tuple) else attributions
            assert attr.shape == state["native"][3].shape
            assert float(attr.abs().sum()) > 0
            return f"attribution {tuple(attr.shape)}, sum {float(attr.sum()):.4e}"

    # ---- parameters: inspection, SVD, base-vs-other checkpoint delta, selective replacement ----
    @check("parameter inspection: state_dict walk by component, SVD of joiner.output_linear")
    def _():
        sd = model.state_dict()
        groups = {}
        for key, tensor in sd.items():
            top = ".".join(key.split(".")[:3]) if key.startswith("encoder.encoders") else key.split(".")[0]
            groups[top] = groups.get(top, 0) + tensor.numel()
        weight = model.joiner.output_linear.weight.detach()
        singular = torch.linalg.svdvals(weight)
        energy = (singular**2).cumsum(0) / (singular**2).sum()
        rank90 = int((energy < 0.9).sum()) + 1
        record("NOTE parameter counts by component", "NOTE", {k: f"{v / 1e6:.2f}M" for k, v in groups.items()})
        return f"{len(sd)} tensors; joiner.output_linear {tuple(weight.shape)}, rank for 90% energy = {rank90}, top singular {float(singular[0]):.2f}"

    @check("weight-delta + selective transplant against init_ft.pt (read-only, on an in-memory copy)")
    def _():
        other = torch.load(args.model_dir / "init_ft.pt", map_location="cpu", weights_only=True, mmap=True)
        other = other.get("model", other)
        base = model.state_dict()
        shared = [k for k in base if k in other and base[k].shape == other[k].shape]
        missing = [k for k in base if k not in other][:3]
        deltas = {k: float((base[k].float() - other[k].float()).norm() / (base[k].float().norm() + 1e-12)) for k in shared}
        changed = {k: v for k, v in deltas.items() if v > 0}
        record("NOTE checkpoint keys", "NOTE", f"{len(shared)}/{len(base)} shared by name+shape, {len(changed)} differ, missing e.g. {missing}")
        before = {k: v.clone() for k, v in list(base.items())[:50]}
        trial = copy.deepcopy(model)
        stack_keys = [k for k in shared if k.startswith("encoder.encoders.2.")]
        donor = {k.removeprefix("encoder.encoders.2."): other[k].clone() for k in stack_keys}
        trial.encoder.encoders[2].load_state_dict(donor, strict=True)
        with torch.no_grad():
            out_trial = EncoderOnly(trial).eval()(x, x_lens)
        note = "donor stack identical to base -> transplant is a verified no-op" if not any(deltas[k] > 0 for k in stack_keys) else f"transplant changed encoder output by {float((out_trial - state['final']).norm() / state['final'].norm()):.4f}"
        assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items()), "original model mutated"
        return f"{note}; original untouched"

    @check("low-rank edit on a copy: rank-1 update of joiner.output_linear changes logits, original untouched")
    def _():
        trial = copy.deepcopy(model)
        original = model.joiner.output_linear.weight.detach().clone()
        u, s, vh = torch.linalg.svd(trial.joiner.output_linear.weight.detach(), full_matrices=False)
        with torch.no_grad():
            trial.joiner.output_linear.weight.copy_(trial.joiner.output_linear.weight - 0.5 * s[0] * u[:, :1] @ vh[:1])
            enc = state["final"][:, :4].unsqueeze(2)  # (N, T, 1, C_enc)
            dec = torch.zeros(1, 1, model.joiner.decoder_proj.in_features).unsqueeze(1)  # (N, 1, U, C_dec)
            base_logits = model.joiner(enc, dec, project_input=True)
            edit_logits = trial.joiner(enc, dec, project_input=True)
        assert torch.equal(model.joiner.output_linear.weight, original)
        assert not torch.allclose(base_logits, edit_logits)
        return f"joiner logits {tuple(base_logits.shape)} changed by {float((edit_logits - base_logits).norm()):.3f}; original untouched"

    width = max(len(name) for name, _, _ in RESULTS)
    for name, status, detail in RESULTS:
        print(f"{status:<5} {name:<{width}}  {detail}")
    failed = [name for name, status, _ in RESULTS if status == "FAIL"]
    scored = [r for r in RESULTS if r[1] != "NOTE"]
    print(f"\n{len(scored) - len(failed)}/{len(scored)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
