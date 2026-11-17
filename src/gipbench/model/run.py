#!/usr/bin/env python3
"""Chạy một experiment fine-tune theo stage (thay cho ft/run_e3_v1*.sh, xem docs/history/).

  python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 0-6
  python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 1-3 --set curate.max_domain_share=0.4
  python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 0-6 --dry-run     # chỉ in lệnh

Stage (đầu vào chỉ là release + experiment YAML; không đọc data/raw, không import dataset / bench):
  0 prepare     chép pool + dev từ release ra <work>/inputs/ (audio tuyệt đối), báo cáo rò rỉ so với leak_guard
                (<work>/leak_report.md, chỉ báo cáo như `validate || true` cũ), ghi experiment đã phân giải
  1 curate      gipbench.model.recipe.curate -> <work>/train_curated.jsonl
  2 features    tách nhóm theo pools.*.datasets -> <work>/group_<nhóm>.jsonl; to_lhotse từng nhóm (speed perturb, mô phỏng
                8 kHz) + dev -> <work>/lhotse/
  3 mix         mix_cuts theo trọng số nhóm (ghi đè bằng MIX_* hoặc --set pools.<nhóm>.weight=...)
  4 finetune    icefall finetune_cs.py theo đoạn + watchdog (gipbench.model.train.icefall_runner)
  5 candidates  model gốc + checkpoint mỗi epoch ảo -> export int8 -> decode dev
  6 select      chọn checkpoint trên dev, WiSE-FT các alpha, chọn lại -> artifact (onnx + model_card.json) + system
                YAML cho bench (configs/bench/systems/<system>.yaml, không ghi đè file đã có)
Chấm bench KHÔNG nằm ở đây: python -m gipbench.bench.run --bench <bench release> --system configs/bench/systems/<system>.yaml
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core import release as R
from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO
from gipbench.core.textnorm import Catalog
from gipbench.model.export import onnx_export
from gipbench.model.recipe import experiment as X
from gipbench.model.recipe.split_groups import split
from gipbench.model.train import icefall_runner as IR

DEFAULT_DECODE = {"decoding_method": "modified_beam_search", "beam": 4, "quantize": "int8", "threads": 16, "batch": 32,
                  "blank_penalty": 0.0, "case": "upper"}


def safe(ref):
    return re.sub(r"[^\w.-]+", "_", ref)


def opt_args(d):
    out = []
    for k, v in (d or {}).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out += [flag] if v else []
        elif isinstance(v, list):
            out += [flag, *map(str, v)]
        elif v is not None:
            out += [flag, str(v)]
    return out


class Ctx:
    def __init__(self, exp, dry):
        self.exp, self.dry = exp, dry
        self.py = str(paths.path("python_cpu"))
        self.py_ft = str(paths.path("python_ft"))
        self.icefall = X.abspath(paths.get("icefall"))
        self.model = X.abspath(exp["base_model"])
        o = exp["output"]
        self.work = X.abspath(o["work_dir"])
        self.exp_dir = X.abspath(o["exp_dir"])
        self.rd = X.abspath(o["dev_results"])
        self.artifact = X.abspath(o["artifact"])
        self.cat = R.release_dir(exp["catalog"]) / "catalog.tsv"
        self.bench = R.release_dir(exp["leak_guard"]) / "manifest.jsonl"
        self.inputs = [(ref, self.work / "inputs" / f"{safe(ref)}.jsonl") for ref in X.input_refs(exp)]
        self.dev = {k: self.work / "inputs" / f"dev_{k}.jsonl" for k in ("cs", "vi")}
        self.lhotse = self.work / "lhotse"
        self.egs = self.icefall / "egs" / "librispeech" / "ASR"

    def run(self, cmd, cwd=None, env=None, log=None):
        cmd = [str(c) for c in cmd]
        print(f"$ {'(cd ' + str(cwd) + ') ' if cwd else ''}{' '.join(cmd)}{' > ' + str(log) if log else ''}")
        if self.dry:
            return 0
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w", encoding="utf-8") as f:
                return subprocess.run(cmd, cwd=cwd or REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True).returncode
        return subprocess.run(cmd, cwd=cwd or REPO, env=env, check=True).returncode

    def icefall_env(self):
        return dict(os.environ, PYTHONPATH=f"{self.icefall}:{os.environ.get('PYTHONPATH', '')}",
                    PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")


# ------------------------------------------------------------------ stage 0
def stage_prepare(c):
    print("== 0. prepare: pool + dev từ release -> inputs/ (audio tuyệt đối), báo cáo rò rỉ")
    jobs = c.inputs + [(c.exp["dev"][k], c.dev[k]) for k in ("cs", "vi")]
    for ref, out in jobs:
        print(f"  {ref} -> {out}")
        if not c.dry:
            R.materialize(ref, out)
    if c.dry:
        return
    c.work.mkdir(parents=True, exist_ok=True)
    (c.work / "experiment.resolved.yaml").write_text(yaml.safe_dump(dict(c.exp), allow_unicode=True, sort_keys=False),
                                                     encoding="utf-8")
    from gipbench.core.leak import LeakGuard
    cat = Catalog(str(c.cat))
    guard = LeakGuard(read_jsonl(str(c.bench)), cat)
    lines = [f"# Báo cáo rò rỉ: {c.exp['name']} so với {c.exp['leak_guard']}", "",
             "Chỉ báo cáo (như `validate.py ... || true` của bản shell). to_lhotse (stage 2) loại các câu này khỏi cut.", "",
             "| Đầu vào | Số câu | Rò rỉ | Lý do |", "| --- | --- | --- | --- |"]
    for ref, out in jobs:
        rows = read_jsonl(str(out))
        why = {}
        for r in rows:
            w = guard.reason(r, r.get("text_tagged") or r.get("text") or "")
            if w:
                why[w] = why.get(w, 0) + 1
        lines.append(f"| {ref} | {len(rows)} | {sum(why.values())} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(why.items(), key=lambda x: -x[1])) or '—'} |")
    (c.work / "leak_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:]))


# ------------------------------------------------------------------ stage 1
def stage_curate(c):
    print("== 1. curate")
    c.run([c.py, "-m", "gipbench.model.recipe.curate", "--manifest", *[p for _, p in c.inputs], "--catalog", c.cat,
           "--out", c.work / "train_curated.jsonl", *opt_args(c.exp.get("curate"))])


# ------------------------------------------------------------------ stage 2
def stage_features(c):
    print("== 2. tách nhóm + to_lhotse (chặn rò rỉ bench, speed perturb, mô phỏng 8 kHz)")
    groups = X.groups(c.exp)
    if c.dry:
        print(f"  split_groups {c.work / 'train_curated.jsonl'} -> {c.work}/group_<nhóm>.jsonl")
        files = {g["group"]: c.work / f"group_{g['group']}.jsonl" for g in groups if g["refs"]}
    else:
        files = split(c.work / "train_curated.jsonl", groups, c.work)
    f = c.exp.get("features", {})
    aug = (["--speed-perturb"] if f.get("speed_perturb") else []) + (["--phone-sim", f["phone_sim"]] if f.get("phone_sim") else [])
    tail = ["--bench", c.bench, "--catalog", c.cat, "--out", c.lhotse]
    for g, p in sorted(files.items()):
        c.run([c.py_ft, "-m", "gipbench.model.features.to_lhotse", "--manifest", p, "--name", g, *tail, *aug])
    for k in ("cs", "vi"):
        c.run([c.py_ft, "-m", "gipbench.model.features.to_lhotse", "--manifest", c.dev[k], "--name", f"dev_{k}", *tail])


# ------------------------------------------------------------------ stage 3
def stage_mix(c):
    m = c.exp["mix"]
    print(f"== 3. mix_cuts: {m['epochs']} epoch ảo × {m['epoch_hours']} h cut")
    args = []
    for g in X.groups(c.exp):
        p = c.lhotse / f"{g['group']}_cuts_raw.jsonl.gz"
        if p.exists() or (c.dry and g["refs"]):
            args += ["--group", f"{g['group']}={p}:{g['weight']}"]
    c.run([c.py_ft, "-m", "gipbench.model.features.mix_cuts", *args, "--epoch-hours", m["epoch_hours"],
           "--epochs", m["epochs"], "--out", c.lhotse / "train_mix_cuts.jsonl.gz"])


# ------------------------------------------------------------------ stage 4
def finetune_cmd(c, seg):
    t, lh = c.exp["train"], c.lhotse
    return lambda res: [c.py_ft, "./zipformer/finetune_cs.py", "--world-size", "1", "--use-fp16", "1",
                        *t["model_args"].split(), *res, "--exp-dir", str(c.exp_dir), "--bpe-model", str(c.model / "bpe.model"),
                        "--freeze-modules", t["freeze"], "--train-cuts", str(lh / "train_mix_cuts.jsonl.gz"),
                        "--valid-cuts", str(lh / "dev_cs_cuts_raw.jsonl.gz"), str(lh / "dev_vi_cuts_raw.jsonl.gz"),
                        "--on-the-fly-feats", "True", "--num-workers", str(t.get("workers", 2)),
                        "--num-buckets", str(t.get("buckets", 10)), "--mp-start-method", "spawn", "--shuffle", "True",
                        "--base-lr", str(t["lr"]), "--max-duration", str(t["max_duration"]), "--enable-musan", "False",
                        "--enable-spec-aug", "True", "--save-every-n", str(seg), "--keep-last-k", str(t.get("keep_last_k", 40)),
                        "--manifest-dir", str(lh)]


def stage_finetune(c):
    t, m = c.exp["train"], c.exp["mix"]
    stats = c.lhotse / "train_mix_cuts.jsonl.gz.json"
    eh = json.loads(stats.read_text(encoding="utf-8"))["epoch_hours"] if stats.exists() else m["epoch_hours"]
    steps = IR.steps_per_epoch(eh, t["max_duration"])
    seg, eps = IR.segment_plan(steps, t.get("segments_per_epoch", 4))
    print(f"== 4. finetune (LR {t['lr']}, freeze {t['freeze']}) -> {c.exp_dir}")
    print(f"  ước tính {steps} step / epoch ảo -> checkpoint mỗi {seg} step, epoch ảo = {eps} step")
    c.run([c.py_ft, "-m", "gipbench.model.train.wrap_ckpt", c.model / "model.pt", c.model / "init_ft.pt"]
          + (["--drop-ctc"] if t.get("drop_ctc") else []))
    c.run([c.py_ft, "-m", "gipbench.model.train.patch_finetune", "--icefall", c.icefall])
    cmd = finetune_cmd(c, seg)
    if c.dry:
        print(f"$ (cd {c.egs}) " + " ".join(cmd(IR.resume_args(c.exp_dir, c.model / "init_ft.pt"))) + f" >> {c.work}/finetune.log")
        print(f"  watchdog {t.get('watchdog')}; resume: --do-finetune 0 --start-epoch 2 --num-epochs 1 --start-batch <N>")
        return
    wd = t.get("watchdog", {})
    runner = IR.Runner(cmd, cwd=c.egs, env=c.icefall_env(), exp_dir=c.exp_dir, eps=eps, log=c.work / "finetune.log",
                       seg_log=c.work / "segments.log", watchdog=wd, disk_path=wd.get("disk_path", str(c.work)))
    res = runner.run(c.model / "init_ft.pt")
    if res != "done":
        raise SystemExit(f"finetune dừng: {res}")


# ------------------------------------------------------------------ stage 5–6
def export(c, ed, name, select_args):
    od = c.exp_dir / "cand" / name / "onnx"
    c.run(onnx_export.export_cmd(c.py_ft, ed, c.model, c.exp["train"]["model_args"], select_args), cwd=c.egs,
          env=c.icefall_env(), log=c.exp_dir / "cand" / name / "export.log")
    if not c.dry:
        onnx_export.collect(ed, od, c.model)
    return od


def decode_dev(c, md, name):
    d = c.exp.get("decode_dev", {})
    for k in ("cs", "vi"):
        s = f"dev_{k}"
        if (c.rd / name / s / "hyps.jsonl").exists():
            continue
        c.run([c.py, "-m", "gipbench.infer.decode", "--bench", c.dev[k], "--model-dir", md, "--system", s,
               "--results", c.rd / name, "--threads", d.get("threads", 16), "--batch", d.get("batch", 32)],
              log=c.rd / name / f"{s}_decode.log")


def stage_candidates(c):
    print("== 5. ứng viên: model gốc + checkpoint mỗi epoch ảo -> export int8 -> decode dev")
    decode_dev(c, c.model, "base")
    nums = IR.ckpt_nums(c.exp_dir) or (["<N>"] if c.dry else [])
    for n in nums:
        name = f"ck{n}"
        if not (c.exp_dir / "cand" / name / "onnx" / "encoder.int8.onnx").exists():
            export(c, c.exp_dir, name, ["--iter", str(n)])
        decode_dev(c, c.exp_dir / "cand" / name / "onnx", name)


def natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(p).name)]


def select(c, cands, out):
    c.run([c.py, "-m", "gipbench.model.select.select_ckpt", "--catalog", c.cat, "--dev-cs", c.dev["cs"], "--dev-vi", c.dev["vi"],
           "--base", c.rd / "base", "--cand", *cands, "--max-vi-delta", c.exp.get("select", {}).get("max_vi_delta", 0.3),
           "--out", out])
    return None if c.dry else json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))


def stage_select(c):
    alphas = c.exp.get("select", {}).get("wise_alphas", [0.3, 0.5, 0.7, 1.0])
    print(f"== 6. chọn checkpoint trên dev, WiSE-FT alpha {' / '.join(map(str, alphas))} của checkpoint đó, chọn lại")
    cks = sorted(c.rd.glob("ck*"), key=natural) if not c.dry else [c.rd / "ck<N>"]
    r = select(c, cks, c.rd / "select_ckpt")
    if c.dry:
        pick, wck = "<pick>", "ck<N>"
    else:
        pick = r["pick"]
        best = min((k for k in r["results"] if k.startswith("ck")), key=lambda k: r["results"][k]["cs_wer"])
        wck = best if pick == "base" else pick
    print(f"  checkpoint theo luật: {pick}; WiSE-FT trên: {wck}")
    c.run([c.py_ft, "-m", "gipbench.model.select.interpolate_ckpt", "--base", c.model / "init_ft.pt",
           "--ft", c.exp_dir / f"checkpoint-{wck[2:]}.pt", "--ft-key", "model", "--alphas", *map(str, alphas),
           "--out-dir", c.exp_dir / "wise", "--epoch", "1"])
    wise = []
    for a in alphas:
        an = f"{float(a):g}"
        if not (c.exp_dir / "cand" / f"wise_{an}" / "onnx" / "encoder.int8.onnx").exists():
            export(c, c.exp_dir / "wise" / f"alpha_{an}", f"wise_{an}", ["--epoch", "1"])
        decode_dev(c, c.exp_dir / "cand" / f"wise_{an}" / "onnx", f"wise_{an}")
        if not c.dry:
            (c.exp_dir / "wise" / f"alpha_{an}" / "epoch-1.pt").unlink(missing_ok=True)
        wise.append(c.rd / f"wise_{an}")
    rw = select(c, wise, c.rd / "select_wise")
    final = "<pick>" if c.dry else rw["pick"]
    md = c.model if final == "base" else c.exp_dir / "cand" / final / "onnx"
    print(f"  model chọn: {final} ({md}) -> artifact {c.artifact}")
    if c.dry:
        return
    card = {"system": c.exp["name"], "experiment": {"path": c.exp["_path"], "overrides": c.exp["_overrides"],
                                                     "sha256": hashlib.sha256(Path(c.exp["_path"]).read_bytes()).hexdigest()},
            "base_model": {"path": str(c.model), "sha256": onnx_export.sha256(c.model / "model.pt")
                           if (c.model / "model.pt").exists() else None},
            "releases_used": {ref: R.resolve(ref).meta["files"] for ref in
                              [*X.input_refs(c.exp), c.exp["dev"]["cs"], c.exp["dev"]["vi"], c.exp["catalog"], c.exp["leak_guard"]]},
            "checkpoint_picked": final, "wise_from": wck, "dev_metrics": rw["results"].get(final),
            "git_commit": R.git_commit()}
    onnx_export.write_artifact(c.artifact, md, card)
    write_systems(c)


def write_systems(c):
    exp_root = str(paths.get("exp_root"))
    art = str(c.artifact)
    art = art.replace(exp_root, "${exp_root}") if art.startswith(exp_root) else os.path.relpath(art, REPO)
    for s in c.exp.get("systems", []):
        p = REPO / "configs" / "bench" / "systems" / f"{s['system']}.yaml"
        if p.exists():
            print(f"  giữ nguyên {p} (đã có)")
            continue
        dec = dict(DEFAULT_DECODE, **(s.get("decode") or {}))
        hw = s.get("hotwords")
        if hw:
            dec.update({"hotwords_catalog": hw["catalog"], "hotwords_exclude_b3": bool(hw.get("exclude_b3")),
                        "hotwords_score": hw.get("score", 1.5)})
        p.write_text(f"# Sinh bởi gipbench.model.run ({c.exp['_path']}). Chạy: python -m gipbench.bench.run --bench <bench> "
                     f"--system {p.relative_to(REPO).as_posix()}\n"
                     + yaml.safe_dump({"system": s["system"], "model": art, "decode": dec}, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        print(f"  đã ghi {p}")


STAGES = [stage_prepare, stage_curate, stage_features, stage_mix, stage_finetune, stage_candidates, stage_select]


def parse_stages(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--stages", default="0-6", help="vd 0-6, 2-3, 4")
    ap.add_argument("--set", action="append", default=[], metavar="KHOÁ=GIÁ_TRỊ", help="ghi đè config, vd train.lr=0.001")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    exp = X.load(args.experiment, args.set)
    c = Ctx(exp, args.dry_run)
    for i in parse_stages(args.stages):
        if not 0 <= i < len(STAGES):
            sys.exit(f"stage {i} không có (0–{len(STAGES) - 1})")
        STAGES[i](c)


if __name__ == "__main__":
    main()
