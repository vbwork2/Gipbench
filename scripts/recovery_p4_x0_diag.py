#!/usr/bin/env python3
"""Phase 4 X0 reproduction: per-checkpoint evaluation and diagnostics (RESEARCH_RECOVERY.md section 6.2, registered
before the run).

For the base model and each checkpoint N of the X0 run (in the order given, ck2400 first):
  1. export  `export-onnx.py --iter N --avg 1 --use-averaged-model 0 --enable-int8-quantization 1` (the X0 artifact
             path, gipbench.model.export.onnx_export), base = models/gip15 as shipped
  2. decode  dev_v2_vi@1, dev_interim_cs@1 and the 200 `valid_from_train` VIVOS rows with gipbench.infer.decode
             (modified_beam_search, beam 4, INT8, batch 1, 16 threads)
  3. score   gipbench.bench.evaluate (diagnostic, open_vocab); paired vs the Phase 1 official hyps on the dev sets and
             vs this script's base decode on the train rows
  4. strata  S / D / I and WER by duration bin from utt_scores.jsonl; output BPE tokens per second of audio
  5. blank   INT8 ONNX greedy (gipformer_mech.onnx_greedy.OnnxTransducer) on a fixed subset: softmax blank
             probability at every greedy step, by relative-position decile
  6. grads   (torch, CPU fp32) simple / pruned encoder gradient on the 6 fixed P2 VIVOS rows, cosine and pruned share at
             the loss scales in force at batch N; per-layer relative update vs init_ft.pt; frozen bit-equality
Plus the finetune.log curves (loss / simple / pruned, lr, grad-norm quartiles, grad_scale).

  PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p4_x0_diag.py --icefall ~/icefall \
      --exp-dir data/work/recovery/p4x0/exp/e4_x0/seed-0 --work data/work/model/e4_x0/seed-0 \
      --out data/work/recovery/p4x0/diag --steps base 2400 300 600 900 1200 1500 1800 2100

Each step writes <out>/steps/<name>.json; every stage skips work whose output already exists.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
GIP = REPO / "models/gip15"
DEV = {"dev_v2_vi": REPO / "data/releases/dev/dev_v2_vi@1/manifest.jsonl",
       "dev_interim_cs": REPO / "data/releases/dev/dev_interim_cs@1/manifest.jsonl"}
BASELINE = {"dev_v2_vi": REPO / "data/work/recovery/p1/decode/dev_v2_vi@1/official/hyps.jsonl",
            "dev_interim_cs": REPO / "data/work/recovery/p1/decode/dev_interim_cs@1/official/hyps.jsonl"}
MODEL_ARGS = "--use-transducer 1 --use-ctc 0"
FREEZE = ("encoder_embed.", "encoder.encoders.0.", "encoder.encoders.1.")
BINS = ((0, 10), (10, 15), (15, 20), (20, 25), (25, 1e9))
LR = re.compile(r"batch (\d+), loss\[.*?lr:\s+([\d.e+-]+)")
LAYER = re.compile(r"^(encoder\.encoders\.\d+\.(?:encoder\.)?layers\.\d+)\.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, log, **kw):
    cmd = [str(c) for c in cmd]
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **kw)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(cmd)}; see {log}")


def bin_name(d):
    lo, hi = next(b for b in BINS if b[0] <= d < b[1])
    return f"{lo}-{hi}s" if hi < 1e9 else f">={lo}s"


def read_rows(path):
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------- 1-3 export, decode, score
def export(args, step, log):
    from gipbench.model.export import onnx_export

    if step == "base":
        return GIP
    md = args.out / "models" / f"ck{step}"
    if all((md / f"{p}.int8.onnx").is_file() for p in onnx_export.PARTS):
        return md
    egs = args.icefall / "egs/librispeech/ASR"
    env = dict(os.environ, PYTHONPATH=f"{args.icefall}:{os.environ.get('PYTHONPATH', '')}", PYTHONIOENCODING="utf-8")
    run(onnx_export.export_cmd(sys.executable, args.exp_dir, GIP, MODEL_ARGS, ["--iter", str(step)]), log, cwd=egs, env=env)
    onnx_export.collect(args.exp_dir, md, GIP)
    return md


def decode_and_score(args, name, md, sets, log):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONIOENCODING="utf-8")
    out = {}
    for s, manifest in sets.items():
        res = args.out / "decode" / s
        hyps = res / name / "hyps.jsonl"
        if not (hyps.is_file() and (res / name / "decode_summary.json").is_file()):
            run([sys.executable, "-m", "gipbench.infer.decode", "--bench", manifest, "--model-dir", md, "--system", name,
                 "--results", res, "--quantize", "int8", "--decoding-method", "modified_beam_search", "--beam", 4,
                 "--threads", 16, "--batch", 1, "--case", "upper"], log, cwd=REPO, env=env)
        base = BASELINE.get(s) or (args.out / "decode" / s / "base" / "hyps.jsonl")
        score = args.out / "score" / s / name
        if not (score / "summary.json").is_file():
            cmd = [sys.executable, "-m", "gipbench.bench.evaluate", "--manifest", manifest, "--hyps", hyps, "--system", name,
                   "--out", score, "--mode", "diagnostic", "--inference-mode", "open_vocab"]
            if name != "base" or s in BASELINE:
                cmd += ["--baseline-hyps", base]
            run(cmd, log, cwd=REPO, env=env)
        out[s] = {"hyps": str(hyps), "hyps_sha256": sha(hyps), "score_dir": str(score)}
    return out


def summarize_score(score_dir, manifest, hyps, sp):
    summ = json.loads((Path(score_dir) / "summary.json").read_text(encoding="utf-8"))
    prof = summ["profiles"]["gipbench_strict_v1"]["overall"]
    keep = ("utt", "hours", "wer", "wer_ci95", "cer", "cs_wer", "n_wer", "term_recall", "boundary_ins_rate",
            "en_hallucination_per100")
    res = {k: prof[k] for k in keep if k in prof}
    paired = (summ.get("paired") or {}).get("overall") or []
    res["paired"] = {p["metric"]: {"a": p["a"], "b": p["b"], "delta": p["delta"], "ci95": p["ci95"]} for p in paired}
    utts = read_rows(Path(score_dir) / "utt_scores.jsonl")
    dur = {r["id"]: float(r["duration"]) for r in read_rows(manifest)}
    hyp = {h["id"]: h.get("hyp") or "" for h in read_rows(hyps)}

    def agg(rows):
        n = sum(r["n"] for r in rows)
        sec = sum(dur[r["id"]] for r in rows)
        tok = sum(len(sp.encode(hyp.get(r["id"], ""))) for r in rows)
        return {"utts": len(rows), "ref_words": n, "wer": 100 * sum(r["err"] for r in rows) / max(n, 1),
                **{k: 100 * sum(r[k] for r in rows) / max(n, 1) for k in ("S", "D", "I")},
                "tokens_per_sec": tok / max(sec, 1e-9)}
    res["sdi"] = agg(utts)
    by = {}
    for r in utts:
        by.setdefault(bin_name(dur[r["id"]]), []).append(r)
    order = [bin_name(lo) for lo, _ in BINS]
    res["by_duration"] = {b: agg(by[b]) for b in order if b in by}
    return res


# ---------------------------------------------------------------- 5 blank statistics
def blank_subset(work):
    rows = read_rows(DEV["dev_v2_vi"])
    rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    picked, count = [], {}
    for r in rows:
        b = bin_name(float(r["duration"]))
        if count.get(b, 0) < 30:
            count[b] = count.get(b, 0) + 1
            picked.append(("dev_v2_vi", DEV["dev_v2_vi"].parent, r))
    train = read_rows(work / "inputs" / "valid_from_train.jsonl")
    train.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    picked += [("train", work / "inputs", r) for r in train[:50]]
    return picked


def blank_stats(md, subset):
    import numpy as np
    import soundfile as sf
    from gipformer_mech.onnx_greedy import OnnxTransducer

    t = OnnxTransducer(Path(md), quantize="int8", threads=4)
    groups = {}
    for kind, root, r in subset:
        wav, rate = sf.read(str((Path(root) / r["audio"]).resolve()), dtype="float32", always_2d=True)
        enc = t.encode(t.features(wav.mean(axis=1), rate))
        context = (-1,) * (t.context_size - 1) + (t.blank,)
        dec = t.decoder_out(context)
        pb, emitted = [], 0
        for f in range(len(enc)):
            logit = t.joiner.run(None, {"encoder_out": enc[f:f + 1], "decoder_out": dec})[0][0].astype(np.float64)
            p = np.exp(logit - logit.max())
            p /= p.sum()
            pb.append(p[t.blank])
            tok = int(logit.argmax())
            if tok not in (t.blank, t.unk):
                emitted += 1
                context = context[1:] + (tok,)
                dec = t.decoder_out(context)
        pb = np.array(pb)
        key = "train" if kind == "train" else f"dev_v2_vi:{bin_name(float(r['duration']))}"
        g = groups.setdefault(key, {"utts": 0, "frames": 0, "sum_pb": 0.0, "blank_argmax": 0, "emitted": 0, "sec": 0.0,
                                    "decile_sum": [0.0] * 10, "decile_n": [0] * 10})
        g["utts"] += 1
        g["frames"] += len(pb)
        g["sum_pb"] += float(pb.sum())
        g["blank_argmax"] += int(len(pb) - emitted)
        g["emitted"] += emitted
        g["sec"] += float(r["duration"])
        for i, v in enumerate(pb):
            d = min(9, int(10 * i / len(pb)))
            g["decile_sum"][d] += float(v)
            g["decile_n"][d] += 1
    out = {}
    for k, g in sorted(groups.items()):
        out[k] = {"utts": g["utts"], "frames": g["frames"], "mean_blank_posterior": g["sum_pb"] / g["frames"],
                  "blank_argmax_share": g["blank_argmax"] / g["frames"], "greedy_tokens_per_sec": g["emitted"] / g["sec"],
                  "blank_posterior_by_decile": [s / max(n, 1) for s, n in zip(g["decile_sum"], g["decile_n"])]}
    return out


# ---------------------------------------------------------------- 6 gradients and parameter deltas
class GradProbe:
    """P2 harness (scripts/recovery_p2_train_checks.py) on the 6 fixed vivos_train@v3 rows, CPU fp32, no augmentation."""

    def __init__(self, icefall, work):
        import torch
        import recovery_p2_train_checks as P2

        torch.set_num_threads(8)
        P2.setup_icefall(icefall)
        from gipbench.model.train import patch_finetune as PF
        subprocess.run([sys.executable, "-m", "gipbench.model.train.patch_finetune", "--icefall", str(icefall)], check=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "src")), stdout=subprocess.DEVNULL)
        import finetune_cs as F
        from lhotse import load_manifest_lazy

        work.mkdir(parents=True, exist_ok=True)
        rows = P2.sample_rows(6)
        P2.build_cuts(rows[:6], work / "train_cuts.jsonl.gz")
        P2.build_cuts(rows[6:10], work / "valid_cuts.jsonl.gz")
        self.P2, self.F = P2, F
        self.fargs = P2.make_args(F, work, 3e-4, max_duration=200)
        self.params, self.sp, self.model, _, _ = P2.build(F, PF, self.fargs)
        self.init = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}  # parameters + buffers
        dm = F.LibriSpeechAsrDataModule(self.fargs)
        self.batch = next(iter(dm.train_dataloaders(load_manifest_lazy(work / "train_cuts.jsonl.gz"))))
        self.ids = [r["id"] for r in rows[:6]]

    def scales(self, n):
        w, s = self.params.warm_step, self.params.simple_loss_scale
        if n >= w:
            return s, 1.0
        return 1.0 - (n / w) * (1.0 - s), 0.1 + 0.9 * (n / w)

    def __call__(self, ckpt, n):
        import random

        import numpy as np
        import torch

        P2, F, model = self.P2, self.F, self.model
        sd = None
        if ckpt is None:
            model.load_state_dict(self.init_state, strict=True)
        else:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                sys.exit(f"{ckpt}: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.params.batch_idx_train = n
        F.set_batch_count(model, F.get_adjusted_batch_count(self.params))
        model.train()
        grads, losses = {}, {}
        for term in ("simple", "pruned"):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1)  # dropout; zipformer also draws from python random (layer skip, whitening)
            random.seed(1)
            np.random.seed(1)
            s, p, frames = P2.losses_of(F, self.params, model, self.sp, self.batch)
            (s if term == "simple" else p).backward()
            grads[term] = P2.flat_grad(model, "encoder.")
            losses[term] = float((s if term == "simple" else p).detach()) / frames
        model.zero_grad(set_to_none=True)
        gs, gp = grads["simple"], grads["pruned"]

        def share(ws, wp):
            return float((wp * gp).norm() / (ws * gs + wp * gp).norm())
        ws, wp = self.scales(n)
        grad = {"batch_idx": n, "train_loss_per_frame": losses,
                "encoder_grad_norm": {"simple": float(gs.norm()), "pruned": float(gp.norm())},
                "cosine": float(gs @ gp / (gs.norm() * gp.norm())), "scales_in_force": {"simple": ws, "pruned": wp},
                "pruned_share_in_force": share(ws, wp),
                "pruned_share_post_warmup": share(self.params.simple_loss_scale, 1.0)}
        return grad, (None if sd is None else self.deltas(sd))

    def deltas(self, sd):
        import torch

        acc, frozen_changed, n_frozen = {}, [], 0
        for n, b in self.init.items():
            a = sd[n].to(b.dtype)
            m = LAYER.match(n)
            key = m.group(1) if m else self.P2.group_of(n)
            e = acc.setdefault(key, [0.0, 0.0])
            e[0] += float((a.double() - b.double()).pow(2).sum())
            e[1] += float(b.double().pow(2).sum())
            if n.startswith(FREEZE):
                n_frozen += 1
                if not torch.equal(a, b):
                    frozen_changed.append(n)
        return {"rel_update": {k: (d / max(r, 1e-30)) ** 0.5 for k, (d, r) in sorted(acc.items())},
                "frozen_tensors": n_frozen, "frozen_changed": frozen_changed}


# ---------------------------------------------------------------- training log
def train_log(work):
    import recovery_p3_overfit as P3

    log = work / "finetune.log"
    st = P3.log_stats(log)
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    st["lr"] = [{"batch": int(b), "lr": float(v)} for b, v in LR.findall(text)]
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icefall", type=Path, required=True)
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="X0 work dir (inputs/valid_from_train.jsonl, finetune.log)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", nargs="+", required=True, help="'base' and/or checkpoint batch numbers")
    ap.add_argument("--skip", nargs="*", default=[], choices=["blank", "grad"])
    args = ap.parse_args()
    args.icefall, args.exp_dir = args.icefall.expanduser().resolve(), args.exp_dir.resolve()
    args.work, args.out = args.work.resolve(), args.out.resolve()
    (args.out / "steps").mkdir(parents=True, exist_ok=True)
    log = args.out / "run.log"
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(GIP / "bpe.model"))
    train_rows = args.work / "inputs" / "valid_from_train.jsonl"
    sets = {**DEV, "train200": train_rows}
    meta = {"script_sha256": sha(__file__), "exp_dir": str(args.exp_dir), "train200_sha256": sha(train_rows),
            "dev_sha256": {k: sha(v) for k, v in DEV.items()}, "baseline_sha256": {k: sha(v) for k, v in BASELINE.items()}}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    subset = blank_subset(args.work)
    probe = None
    if "base" not in args.steps and not (args.out / "steps" / "base.json").is_file():
        args.steps = ["base", *args.steps]  # the train200 paired baseline needs the base decode first
    for step in args.steps:
        name = "base" if step == "base" else f"ck{step}"
        dst = args.out / "steps" / f"{name}.json"
        res = json.loads(dst.read_text(encoding="utf-8")) if dst.is_file() else {"name": name}
        t0 = time.time()
        ckpt = None if step == "base" else args.exp_dir / f"checkpoint-{step}.pt"
        if ckpt is not None:
            if not ckpt.is_file():
                sys.exit(f"missing {ckpt}")
            res["checkpoint_sha256"] = res.get("checkpoint_sha256") or sha(ckpt)
        md = export(args, step, log)
        res["model_dir"] = str(md)
        res["onnx_sha256"] = {p: sha(Path(md) / f"{p}.int8.onnx") for p in ("encoder", "decoder", "joiner")}
        dec = decode_and_score(args, name, md, sets, log)
        res["sets"] = {s: {**d, **summarize_score(d["score_dir"], sets[s], d["hyps"], sp)} for s, d in dec.items()}
        dst.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        if "blank" not in args.skip and "blank" not in res:
            res["blank"] = blank_stats(md, subset)
            dst.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        if "grad" not in args.skip and "grad" not in res:
            probe = probe or GradProbe(args.icefall, args.out / "grad_work")
            res["grad"], res["deltas"] = probe(ckpt, 0 if step == "base" else int(step))
            res["grad_rows"] = probe.ids
        res["seconds"] = round(time.time() - t0, 1)
        dst.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        v = res["sets"]["dev_v2_vi"]
        print(f"{name}: dev_v2_vi WER {v['wer']:.2f} paired {v['paired'].get('wer')} D {v['sdi']['D']:.2f}", flush=True)
    (args.out / "train_log.json").write_text(json.dumps(train_log(args.work), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
