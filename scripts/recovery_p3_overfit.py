#!/usr/bin/env python3
"""Phase 3 tiny-set memorisation run (RESEARCH_RECOVERY.md section 5; the pass rule is registered there).

Data: the 163 utterances of pool/vss_cs@v4 that the user listened to in 2d (data/listen/vss_cs_v1/listen_A_reviewed.csv,
decision ok or fix; the fixed text is used for fix). No augmentation: no speed perturb, no phone_sim, no SpecAugment,
no MUSAN. Same model, tokenizer, init checkpoint, freeze, LR and loss as the X0 recipe unless an option says otherwise.
The 163 cuts are repeated into one virtual epoch (unique cut ids) so Icefall does not write a 1 GB epoch-N.pt per pass.

  PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p3_overfit.py \
      --icefall ~/icefall --work data/work/recovery/p3/default --steps 600 --eval-at 0 100 200 400 600
  # single-factor arm: loss-scale warm-up disabled (warm_step 0 -> scales 0.5 / 1.0 from batch 1)
  ... --work data/work/recovery/p3/nowarm --no-loss-warmup

Each evaluated checkpoint is exported through the X0 path (export-onnx --iter N --avg 1, int8) and decoded with the repo
decoder (modified beam search 4, batch 1), then scored with gipbench.bench.evaluate against the same 163 references.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GIP = REPO / "models/gip15"
POOL = REPO / "data/releases/pool/vss_cs@v4/manifest.jsonl"
REVIEW = REPO / "data/listen/vss_cs_v1/listen_A_reviewed.csv"
G0 = REPO / "data/benchmarks/general_v1/manifest.jsonl"
CATALOG = REPO / "data/catalogs/train_v1/catalog.tsv"
X0_FREEZE = "encoder_embed.,encoder.encoders.0.,encoder.encoders.1."


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, log, cwd=None, env=None):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        rc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.exit(f"failed ({rc}): {' '.join(map(str, cmd))} -> see {log}")


def tiny_manifest(out):
    pool = {json.loads(ln)["id"]: json.loads(ln) for ln in POOL.read_text(encoding="utf-8").splitlines() if ln.strip()}
    rows = []
    for r in csv.DictReader(open(REVIEW, encoding="utf-8-sig")):
        if r["decision"] not in ("ok", "fix") or r["id"] not in pool:
            continue
        p = dict(pool[r["id"]])
        p["text_tagged"] = r["text_tagged_fixed"] if r["decision"] == "fix" else r["text_tagged"]
        p["label_source"] = f"human:listen_2d:{r['decision']}"
        # pool/vss_cs@v4 stores Windows separators ("..\..\dl\..."), unreadable on Linux (RESEARCH_RECOVERY.md bug B1)
        p["audio"] = str((POOL.parent / p["audio"].replace("\\", "/")).resolve())
        rows.append(p)
    rows.sort(key=lambda r: r["id"])
    if not 100 <= len(rows) <= 200 or not all("<en>" in r["text_tagged"] for r in rows):
        sys.exit(f"tiny set has {len(rows)} rows or rows without <en>")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def repeat_cuts(src, dst, times):
    from lhotse import CutSet, load_manifest

    cuts = load_manifest(src)
    CutSet.from_cuts(c.with_id(f"{c.id}_r{k}") for k in range(times) for c in cuts).to_file(dst)


GROUPS = ("encoder_embed.", *(f"encoder.encoders.{i}." for i in range(6)), "encoder.downsample_output.",
          "decoder.", "joiner.", "simple_am_proj.", "simple_lm_proj.")
LOSS = re.compile(r"batch (\d+), loss\[loss=([\d.e+-]+), simple_loss=([\d.e+-]+), pruned_loss=([\d.e+-]+)")
QUART = re.compile(r"grad-norm quartiles ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+), threshold=([\d.e+-]+), "
                   r"percent-clipped=([\d.e+-]+)")
SCALE = re.compile(r"grad_scale: ([\d.e+-]+)")


def ckpt_stats(path, init, freeze):
    """Per-group relative update vs init_ft.pt and bit-equality of every frozen tensor (from the real CLI checkpoint)."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    groups, frozen_changed, n_frozen = {}, [], 0
    for n, t in sd.items():
        if n not in init:
            continue
        a, b = t.double(), init[n].double()
        g = next((x for x in GROUPS if n.startswith(x)), "other")
        e = groups.setdefault(g, [0.0, 0.0, 0, 0])
        e[0] += float((a - b).pow(2).sum())
        e[1] += float(b.pow(2).sum())
        e[2] += 1
        e[3] += not torch.equal(t, init[n])
        if any(n.startswith(x) for x in freeze):
            n_frozen += 1
            if not torch.equal(t, init[n]):
                frozen_changed.append(n)
    return {"rel_update": {g: (d / max(r, 1e-30)) ** 0.5 for g, (d, r, _, _) in groups.items()},
            "tensors_changed": {g: f"{c}/{k}" for g, (_, _, k, c) in groups.items()},
            "frozen_tensors": n_frozen, "frozen_changed": frozen_changed}


def log_stats(log):
    """Training-loss, ScaledAdam grad-norm quartile and fp16 grad_scale lines of finetune_cs.py, in order."""
    text = Path(log).read_text(encoding="utf-8", errors="replace") if Path(log).is_file() else ""
    loss = [{"batch": int(m[0]), "loss": float(m[1]), "simple": float(m[2]), "pruned": float(m[3])}
            for m in LOSS.findall(text)]
    quart = [{"q": [float(x) for x in m[:5]], "threshold": float(m[5]), "pct_clipped": float(m[6])}
             for m in QUART.findall(text)]
    return {"loss": loss, "grad_norm_quartiles": quart, "grad_scale": [float(x) for x in SCALE.findall(text)]}


def examples(manifest, hyps, k=10):
    refs = {json.loads(ln)["id"]: json.loads(ln)["text_tagged"] for ln in Path(manifest).read_text(encoding="utf-8").splitlines()}
    out = []
    for ln in Path(hyps).read_text(encoding="utf-8").splitlines():
        h = json.loads(ln)
        out.append({"id": h["id"], "ref": refs.get(h["id"]), "hyp": h.get("hyp")})
    return sorted(out, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icefall", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-at", type=int, nargs="+", default=[0, 100, 200, 400, 600])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-duration", type=float, default=60)
    ap.add_argument("--freeze", default=X0_FREEZE)
    ap.add_argument("--no-loss-warmup", action="store_true", help="single-factor arm: warm_step 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--exp-dir", type=Path, help="checkpoint directory (default <work>/exp); ~1.1 GB per checkpoint")
    ap.add_argument("--prune-checkpoints", action="store_true",
                    help="delete each checkpoint-N.pt after its export and stats (keeps the last one)")
    args = ap.parse_args()
    import torch

    icefall = args.icefall.expanduser()
    asr = icefall / "egs/librispeech/ASR"
    work = args.work.resolve()
    exp = (args.exp_dir or work / "exp").resolve()
    exp.mkdir(parents=True, exist_ok=True)
    log = work / "run.log"
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}")
    ienv = dict(os.environ, PYTHONPATH=f"{icefall}")
    cuda = torch.cuda.is_available()
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": {k: str(v) for k, v in vars(args).items()},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
            "device": "cuda" if cuda else "cpu", "fp16": cuda, "script_sha256": sha256(__file__)}

    rows = tiny_manifest(work / "tiny.jsonl")
    meta["tiny"] = {"rows": len(rows), "hours": sum(float(r["duration"]) for r in rows) / 3600,
                    "manifest_sha256": sha256(work / "tiny.jsonl"),
                    "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()}
    lh = work / "lhotse"
    run([py, "-m", "gipbench.model.features.to_lhotse", "--manifest", work / "tiny.jsonl", "--name", "tiny",
         "--bench", G0, "--catalog", CATALOG, "--out", lh, "--seed", args.seed], log, env=env)
    n_cuts = sum(1 for _ in gzip.open(lh / "tiny_cuts_raw.jsonl.gz", "rt"))
    if n_cuts != len(rows):
        sys.exit(f"to_lhotse kept {n_cuts} / {len(rows)} cuts; see {log}")
    per_epoch = max(1.0, sum(float(r["duration"]) for r in rows) / (args.max_duration * 0.93))
    times = int(args.steps / per_epoch) + 3
    repeat_cuts(lh / "tiny_cuts_raw.jsonl.gz", lh / "train_cuts.jsonl.gz", times)
    meta["repeats"] = times

    run([py, "-m", "gipbench.model.train.patch_finetune", "--icefall", icefall], log, env=env)
    script = "./zipformer/finetune_cs.py"
    if args.no_loss_warmup:
        src = (asr / "zipformer/finetune_cs.py").read_text(encoding="utf-8")
        old = '            "warm_step": 2000,\n'
        if src.count(old) != 1:
            sys.exit("cannot find warm_step in finetune_cs.py")
        (asr / "zipformer/finetune_cs_nowarm.py").write_text(src.replace(old, '            "warm_step": 0,\n'),
                                                             encoding="utf-8")
        script = "./zipformer/finetune_cs_nowarm.py"
    meta["train_script"] = script
    meta["train_script_sha256"] = sha256(asr / script.lstrip("./"))
    evals = sorted(set(a for a in args.eval_at if 0 < a <= args.steps) | {args.steps})
    save_every = evals[0] if len(evals) == 1 else min(b - a for a, b in zip([0] + evals, evals))
    cmd = [py, script, "--world-size", "1", "--use-fp16", "1" if cuda else "0", "--use-transducer", "1", "--use-ctc", "0",
           "--do-finetune", "1", "--finetune-ckpt", GIP / "init_ft.pt", "--start-epoch", "1", "--num-epochs", "1",
           "--seed", args.seed, "--exp-dir", exp, "--bpe-model", GIP / "bpe.model", "--freeze-modules", args.freeze,
           "--train-cuts", lh / "train_cuts.jsonl.gz", "--valid-cuts", lh / "tiny_cuts_raw.jsonl.gz",
           "--on-the-fly-feats", "True", "--num-workers", "2", "--num-buckets", "10", "--mp-start-method", "spawn",
           "--shuffle", "True", "--base-lr", args.lr, "--max-duration", args.max_duration, "--enable-musan", "False",
           "--enable-spec-aug", "False", "--max-batches", args.steps, "--save-every-n", save_every,
           "--average-period", min(200, save_every), "--keep-last-k", 100, "--manifest-dir", lh]
    meta["train_cmd"] = list(map(str, cmd))
    (work / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    if not list(exp.glob(f"checkpoint-{args.steps}.pt")):
        run(cmd, log, cwd=asr, env=dict(ienv, OMP_NUM_THREADS=str(args.threads)))

    init = torch.load(GIP / "init_ft.pt", map_location="cpu", weights_only=False)["model"]
    results = {}
    for step in [0] + evals:
        name = f"step{step}"
        md = work / "eval" / name / "onnx"
        if step == 0:
            md = GIP
        elif not (md / "encoder.int8.onnx").is_file():
            run([py, "./zipformer/export-onnx.py", "--tokens", GIP / "tokens.txt", "--exp-dir", exp, "--use-transducer",
                 "1", "--use-ctc", "0", "--iter", step, "--avg", "1", "--use-averaged-model", "0",
                 "--enable-int8-quantization", "1"], log, cwd=asr, env=ienv)
            md.mkdir(parents=True, exist_ok=True)
            for part in ("encoder", "decoder", "joiner"):
                (exp / f"{part}-iter-{step}-avg-1.int8.onnx").rename(md / f"{part}.int8.onnx")
                (exp / f"{part}-iter-{step}-avg-1.onnx").unlink()
            for f in ("tokens.txt", "bpe.model"):
                (md / f).write_bytes((GIP / f).read_bytes())
        res_dir = work / "eval" / name / "decode"
        if not (res_dir / name / "hyps.jsonl").is_file():
            run([py, "-m", "gipbench.infer.decode", "--bench", work / "tiny.jsonl", "--model-dir", md, "--system", name,
                 "--results", res_dir, "--decoding-method", "modified_beam_search", "--beam", 4, "--batch", 1,
                 "--threads", args.threads, "--case", "upper"], log, cwd=REPO, env=env)
        score = work / "eval" / name / "score"
        run([py, "-m", "gipbench.bench.evaluate", "--manifest", work / "tiny.jsonl", "--catalog", CATALOG,
             "--hyps", res_dir / name / "hyps.jsonl", "--system", name, "--out", score, "--mode", "diagnostic",
             "--inference-mode", "open_vocab"], log, cwd=REPO, env=env)
        results[name] = {"summary": json.loads((score / "summary.json").read_text(encoding="utf-8")),
                         "examples": examples(work / "tiny.jsonl", res_dir / name / "hyps.jsonl")}
        ck = exp / f"checkpoint-{step}.pt"
        if step and ck.is_file():
            results[name]["checkpoint"] = ckpt_stats(ck, init, [x for x in args.freeze.split(",") if x])
            if args.prune_checkpoints and step != args.steps:
                ck.unlink()
    results["train_log"] = log_stats(log)
    (work / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"done -> {work / 'results.json'}")


if __name__ == "__main__":
    main()
