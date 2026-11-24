"""Smoke test for the interpretability toolkit on a toy Zipformer-style RNN-T.

Non-destructive: builds a tiny random model in memory, never reads or writes a checkpoint, never touches data.
Run it with the isolated interpretability venv:

    .venv/interp/Scripts/python.exe scripts/diagnostics/interp_env_smoke.py

Each check prints PASS / FAIL and the script exits non-zero when any check fails.
The toy mirrors the parts of Gipformer that matter for tooling: a strided conv front end, encoder stacks that run
at different temporal rates and are restored to the input grid, a stateless predictor (embedding + context conv)
and an additive joiner with a blank class.
"""

import sys
import traceback
import types

import torch
from torch import nn

RESULTS: list[tuple[str, str, str]] = []


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((name, "PASS", str(detail or "")))
        except Exception as error:  # noqa: BLE001 - a smoke test reports every failure
            RESULTS.append((name, "FAIL", f"{type(error).__name__}: {error}"))
            traceback.print_exc()
        return fn

    return wrap


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return x + self.out_proj(torch.relu(self.in_proj(x)))


class Stack(nn.Module):
    """Encoder stack at a fixed temporal rate; output is restored to the input grid like Zipformer stacks."""

    def __init__(self, dim, factor, blocks):
        super().__init__()
        self.factor = factor
        self.blocks = nn.ModuleList(FeedForward(dim) for _ in range(blocks))

    def forward(self, x):  # x: (B, T, D)
        t = x.shape[1]
        y = x
        if self.factor > 1:
            y = nn.functional.avg_pool1d(x.transpose(1, 2), self.factor, self.factor, ceil_mode=True).transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        if self.factor > 1:
            y = y.repeat_interleave(self.factor, dim=1)[:, :t]
        return x + y


class Encoder(nn.Module):
    def __init__(self, dim=16, factors=(1, 2, 4), blocks=2):
        super().__init__()
        self.embed = nn.Conv1d(80, dim, kernel_size=3, stride=4, padding=1)  # acoustic frames -> subsampled
        self.stacks = nn.ModuleList(Stack(dim, f, blocks) for f in factors)

    def forward(self, feats):  # (B, T, 80)
        x = self.embed(feats.transpose(1, 2)).transpose(1, 2)
        for stack in self.stacks:
            x = stack(x)
        return x


class Predictor(nn.Module):
    def __init__(self, vocab, dim=16, context=2):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab, dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=context, groups=dim, bias=False)

    def forward(self, y):  # (B, U)
        e = self.embedding(y).transpose(1, 2)
        e = nn.functional.pad(e, (self.context - 1, 0))
        return torch.relu(self.conv(e).transpose(1, 2))


class Joiner(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.enc_proj = nn.Linear(dim, dim)
        self.dec_proj = nn.Linear(dim, dim)
        self.output_linear = nn.Linear(dim, vocab)

    def forward(self, enc, dec):  # (B,T,D), (B,U,D) -> (B,T,U,V)
        return self.output_linear(torch.tanh(self.enc_proj(enc).unsqueeze(2) + self.dec_proj(dec).unsqueeze(1)))


class ToyTransducer(nn.Module):
    def __init__(self, vocab=32):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Predictor(vocab)
        self.joiner = Joiner(16, vocab)

    def forward(self, feats, y):
        return self.joiner(self.encoder(feats), self.decoder(y))


def make_inputs():
    torch.manual_seed(1)
    return torch.randn(2, 64, 80), torch.randint(1, 32, (2, 5))


def make_model():
    torch.manual_seed(0)
    return ToyTransducer().eval()


# ---------- baseline: native PyTorch hooks, the reference every tool is checked against ----------
def native_stack_outputs(model, feats, y):
    captured = {}
    handles = [
        stack.register_forward_hook(lambda _m, _i, out, k=k: captured.__setitem__(k, out.detach().clone()))
        for k, stack in enumerate(model.encoder.stacks)
    ]
    with torch.no_grad():
        logits = model(feats, y)
    for handle in handles:
        handle.remove()
    return captured, logits


@check("native hooks: activation extraction at 3 stack rates")
def _():
    model = make_model()
    feats, y = make_inputs()
    acts, logits = native_stack_outputs(model, feats, y)
    assert set(acts) == {0, 1, 2} and all(a.shape == (2, 16, 16) for a in acts.values())
    return f"stack outputs {tuple(acts[0].shape)}, logits {tuple(logits.shape)}"


# ---------- nnsight ----------
try:
    import nnsight
    from nnsight import NNsight
except Exception as error:  # noqa: BLE001
    nnsight = None
    RESULTS.append(("import nnsight", "FAIL", repr(error)))
else:
    RESULTS.append(("import nnsight", "PASS", nnsight.__version__))


if nnsight is not None:

    @check("nnsight: activation extraction matches native hooks")
    def _():
        model = make_model()
        feats, y = make_inputs()
        native, _ = native_stack_outputs(model, feats, y)
        wrapped = NNsight(model)
        with wrapped.trace(feats, y):
            saved = nnsight.save([])  # a list comprehension inside a trace loses the saved names (nnsight skill trap 1)
            for stack in wrapped.encoder.stacks:
                saved.append(stack.output)
            joiner_in = wrapped.joiner.output_linear.input.save()
        for k, tensor in enumerate(saved):
            assert torch.allclose(tensor, native[k], atol=1e-6), k
        return f"3 stacks identical to native hooks; joiner input {tuple(joiner_in.shape)}"

    @check("nnsight: activation patching (clean -> corrupted) changes logits, identity patch does not")
    def _():
        model = make_model()
        feats, y = make_inputs()
        corrupt = feats + torch.randn_like(feats)
        wrapped = NNsight(model)
        with wrapped.trace(feats, y):
            clean_act = wrapped.encoder.stacks[1].output.save()
            clean_logits = wrapped.output.save()
        with wrapped.trace(corrupt, y):
            base_logits = wrapped.output.save()
        with wrapped.trace(corrupt, y):  # patch stack 1 with the clean activation
            wrapped.encoder.stacks[1].output[:] = clean_act
            patched = wrapped.output.save()
        with wrapped.trace(feats, y):  # identity patch: must be a no-op
            wrapped.encoder.stacks[1].output[:] = clean_act
            identity = wrapped.output.save()
        assert torch.allclose(identity, clean_logits, atol=1e-6)
        assert not torch.allclose(patched, base_logits)
        # stacks run in sequence with no bypass around stack 1 here, so patching its full output restores the clean run
        assert torch.allclose(patched, clean_logits, atol=1e-6)
        return (
            f"|patched-clean|={float((patched - clean_logits).detach().norm()):.3f} "
            f"vs |corrupt-clean|={float((base_logits - clean_logits).detach().norm()):.3f}"
        )

    @check("nnsight: gradient attribution on a mid-stack activation matches autograd")
    def _():
        model = make_model()
        feats, y = make_inputs()
        wrapped = NNsight(model)
        with wrapped.trace(feats, y):
            hidden = wrapped.encoder.stacks[1].output
            metric = wrapped.output[0, 3, 2, 7]
            with metric.backward():
                grad = hidden.grad.save()
        # reference: plain autograd with a retained hidden state
        store = {}

        def keep(_module, _inputs, out):  # must return None, otherwise the hook replaces the module output
            out.retain_grad()
            store["h"] = out

        handle = model.encoder.stacks[1].register_forward_hook(keep)
        model(feats, y)[0, 3, 2, 7].backward()
        handle.remove()
        assert torch.allclose(grad, store["h"].grad, atol=1e-6)
        return f"grad {tuple(grad.shape)} equals autograd, norm={float(grad.norm()):.4f}"

    @check("nnsight: single-channel ablation (zero a hidden unit) alters joiner output")
    def _():
        model = make_model()
        feats, y = make_inputs()
        wrapped = NNsight(model)
        with wrapped.trace(feats, y):
            base = wrapped.output.save()
        with wrapped.trace(feats, y):
            wrapped.joiner.enc_proj.output[..., 3] = 0.0
            ablated = wrapped.output.save()
        assert not torch.allclose(base, ablated)
        return f"max |delta| = {float((base - ablated).abs().max()):.4f}"


# ---------- pyvene ----------
try:
    import pyvene as pv
except Exception as error:  # noqa: BLE001
    pv = None
    RESULTS.append(("import pyvene", "FAIL", repr(error)))
else:
    RESULTS.append(("import pyvene", "PASS", getattr(pv, "__version__", "?")))


if pv is not None:

    def build_pyvene(model, component):
        # pyvene resolves an unregistered architecture only after three shims (found by probing, see COMPATIBILITY_REPORT):
        model.config = types.SimpleNamespace()  # it reads model.config for the embedding dimension
        model.device = torch.device("cpu")  # it reads model.device when moving sources
        pv.type_to_dimension_mapping[type(model)] = {}  # no registered dimensions for a custom class
        config = pv.IntervenableConfig(
            representations=[
                pv.RepresentationConfig(
                    layer=0,
                    component=component,  # dotted module path + .output works on unregistered architectures
                    unit="pos",
                    max_number_of_units=1,
                    intervention_type=pv.VanillaIntervention,
                )
            ]
        )
        return pv.IntervenableModel(config, model)

    @check("pyvene: skill's documented pv.RepresentationConfig / IntervenableModel / VanillaIntervention exist")
    def _():
        assert hasattr(pv, "RepresentationConfig") and hasattr(pv, "IntervenableModel") and hasattr(pv, "VanillaIntervention")

    @check("pyvene: interchange intervention on a custom module (encoder.stacks.1.output), position-wise")
    def _():
        model = make_model()
        feats, y = make_inputs()
        base_in, src_in = (feats, y), (feats + torch.randn_like(feats), y)
        intervenable = build_pyvene(model, "encoder.stacks.1.output")
        with torch.no_grad():
            base_logits = model(*base_in)
            _, patched = intervenable(
                {"feats": base_in[0], "y": base_in[1]},
                [{"feats": src_in[0], "y": src_in[1]}],
                {"sources->base": ([[[3]]], [[[3]]])},  # copy frame 3 of the source run into the base run
            )
        assert patched.shape == base_logits.shape
        assert not torch.allclose(patched, base_logits)
        return f"patched logits differ from base by {float((patched - base_logits).norm()):.4f}"


# ---------- captum ----------
try:
    import captum
    from captum.attr import LayerGradientXActivation, Saliency
except Exception as error:  # noqa: BLE001
    captum = None
    RESULTS.append(("import captum", "FAIL", repr(error)))
else:
    RESULTS.append(("import captum", "PASS", captum.__version__))


if captum is not None:

    @check("captum: input saliency and LayerGradientXActivation through the transducer")
    def _():
        model = make_model()
        feats, y = make_inputs()

        def forward(f):
            return model(f, y)[:, 3, 2, 7]

        feats = feats.clone().requires_grad_(True)
        saliency = Saliency(forward).attribute(feats)
        layer_attr = LayerGradientXActivation(forward, model.encoder.stacks[1]).attribute(feats)
        assert saliency.shape == feats.shape and layer_attr.shape == (2, 16, 16)
        assert float(saliency.abs().sum()) > 0
        return f"saliency {tuple(saliency.shape)}, layer attr {tuple(layer_attr.shape)}"


# ---------- parameter inspection + selective replacement (read-only on the toy, in-memory) ----------
@check("parameter inspection: state_dict walk, SVD of a joiner weight, no mutation")
def _():
    model = make_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    weight = model.joiner.output_linear.weight.detach()
    singular = torch.linalg.svdvals(weight)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items() if "stacks.1" in k}
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    return f"{len(before)} tensors, joiner SVD top-3 {[round(float(s), 3) for s in singular[:3]]}, {len(shapes)} stack-1 tensors"


@check("selective parameter replacement: transplant one stack between two toy models, base untouched")
def _():
    base = make_model()
    torch.manual_seed(7)
    donor = ToyTransducer().eval()
    feats, y = make_inputs()
    with torch.no_grad():
        before = base(feats, y)
        patched = ToyTransducer().eval()
        patched.load_state_dict(base.state_dict())
        patched.encoder.stacks[1].load_state_dict(donor.encoder.stacks[1].state_dict())
        after = patched(feats, y)
        assert torch.equal(base(feats, y), before), "original model mutated"
    assert not torch.allclose(before, after)
    return "stack-1 transplant changes logits; original unchanged (done on a copy, never on a checkpoint)"


def main() -> int:
    width = max(len(name) for name, _, _ in RESULTS)
    for name, status, detail in RESULTS:
        print(f"{status:<5} {name:<{width}}  {detail}")
    failed = [name for name, status, _ in RESULTS if status == "FAIL"]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
