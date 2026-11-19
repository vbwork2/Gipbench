#!/usr/bin/env python3
"""Chạy một experiment fine-tune theo stage (thay cho ft/run_e3_v1*.sh, xem docs/history/).

  python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 0-6
  python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 1-3 --set curate.max_domain_share=0.4
  python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 0-6 --dry-run     # chỉ in lệnh

Stage (đầu vào chỉ là release + experiment YAML; không đọc data/raw, không import dataset / bench):
  0 prepare     chép pool + dev từ release ra <work>/inputs/ (audio tuyệt đối), CỔNG RÒ RỈ đóng khi lỗi: kiểm MỌI bench khoá
                (configs/dataset/releases.yaml, flags.locked) hợp với leak_guard so với mọi pool + dev + text; có critical chưa
                miễn trừ thì thoát mã 3 trước khi curate / train (<work>/leak_report.{json,md}; experiment: leak_gate.*).
                Catalog có span của bench (from_bench) bị từ chối ngay từ đầu (exit 3) trừ catalog_use: oracle_diagnostic
  1 curate      gipbench.model.recipe.curate -> <work>/train_curated.jsonl
  2 features    tách nhóm theo pools.*.datasets -> <work>/group_<nhóm>.jsonl; to_lhotse từng nhóm (speed perturb, mô phỏng
                8 kHz) + dev -> <work>/lhotse/
  3 mix         mix_cuts theo trọng số nhóm (ghi đè bằng MIX_* hoặc --set pools.<nhóm>.weight=...)
  4 finetune    icefall finetune_cs.py theo đoạn + watchdog (gipbench.model.train.icefall_runner)
  5 candidates  model gốc + checkpoint mỗi epoch ảo -> export int8 -> decode dev
  6 select      chọn checkpoint trên dev, WiSE-FT các alpha, chọn lại -> artifact (onnx + model_card.json + leak_report.*)
                select.rule: fixed_last + dev: null = luật cố định đăng ký trước, không cần dev: checkpoint cuối, WiSE-FT
                một alpha (select.fixed_wise_alpha, mặc định 1.0); stage 5 bỏ qua; valid loss của icefall trên
                select.valid_rows câu train (chỉ theo dõi)
                + system YAML cho bench (configs/bench/systems/<system>.yaml, không ghi đè file đã có). Cần báo cáo cổng
                rò rỉ của stage 0 (không có hoặc blocking -> thoát mã 3)
Chấm bench KHÔNG nằm ở đây: python -m gipbench.bench.run --bench <bench release> --system configs/bench/systems/<system>.yaml
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core import leakgate as LG
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
    def __init__(self, exp, dry, seed=None):
        self.exp, self.dry, self.seed = exp, dry, seed
        self.py = str(paths.path("python_cpu"))
        self.py_ft = str(paths.path("python_ft"))
        self.icefall = X.abspath(paths.get("icefall"))
        self.model = X.abspath(exp["base_model"])
        o = exp["output"]
        subdir = f"seed-{seed}" if seed is not None else ""
        self.work = X.abspath(o["work_dir"]) / subdir
        self.exp_dir = X.abspath(o["exp_dir"]) / subdir
        self.rd = X.abspath(o["dev_results"]) / subdir
        self.artifact = X.abspath(o["artifact"]) / subdir
        self.seed_args = [] if seed is None else ["--seed", str(seed)]
        self.init_ckpt = (self.model if seed is None else self.work) / "init_ft.pt"
        self.cat = R.release_dir(exp["catalog"]) / "catalog.tsv"
        self.guard_refs = LG.guard_refs(exp)
        if not self.guard_refs:
            sys.exit("LỖI: leak_guard rỗng (cần ít nhất một bench release)")
        self.bench = R.release_dir(self.guard_refs[0]) / "manifest.jsonl"  # LeakGuard cũ (lọc câu trong to_lhotse): bench đầu tiên
        self.oracle = X.catalog_use(exp) == "oracle_diagnostic"
        self.inputs = [(ref, self.work / "inputs" / f"{safe(ref)}.jsonl") for ref in X.input_refs(exp)]
        self.has_dev = bool(exp.get("dev"))
        self.dev = {k: self.work / "inputs" / f"dev_{k}.jsonl" for k in ("cs", "vi")} if self.has_dev else {}
        self.fixed = (exp.get("select") or {}).get("rule") == X.FIXED_RULE
        # không có dev: valid loss của icefall tính trên mẫu nhỏ CỦA TRAIN (chỉ để theo dõi, không dùng chọn checkpoint)
        self.valid = self.work / "inputs" / "valid_from_train.jsonl"
        self.lhotse = self.work / "lhotse"
        self.egs = self.icefall / "egs" / "librispeech" / "ASR"
        check_catalogs(self)

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


def check_catalogs(c):
    """Catalog (và catalog hotwords của system) có span lấy từ chính bench -> thoát mã 3 (audit R05: cải thiện trên bench khi
    đó là ghi nhớ). Cố ý dùng làm chẩn đoán: catalog_use: oracle_diagnostic (hoặc system khai báo inference_mode: oracle_bias)."""
    errs = []
    uses = [("catalog của experiment", c.exp["catalog"], c.oracle)] + [
        (f"hotwords của system {s}", ref, c.oracle or declared) for s, ref, declared in X.system_hotword_catalogs(c.exp)]
    for what, ref, allow in uses:
        try:
            R.forbid_bench_catalog(ref, what, allow=allow)
        except R.ReleaseError as e:
            if not (c.dry and "chưa có release" in str(e)):  # dry-run trên máy chưa có data/releases: không biết cờ, bỏ qua
                errs.append(str(e))
    if errs:
        print("\n".join(f"LỖI {e}" for e in errs))
        sys.exit(3)


# ------------------------------------------------------------------ stage 0
def stage_prepare(c):
    print("== 0. prepare: pool + dev từ release -> inputs/ (audio tuyệt đối), cổng rò rỉ đóng khi lỗi")
    jobs = c.inputs + [(c.exp["dev"][k], c.dev[k]) for k in c.dev]
    if not c.has_dev:
        print(f"  dev: không có (select.rule: {X.FIXED_RULE}, luật cố định đăng ký trước)")
    try:
        g = LG.gate_config(c.exp)
    except ValueError as e:
        sys.exit(f"LỖI {e}")
    refs, extra = LG.plan_refs(c.guard_refs, R.locked_bench_refs())
    print(f"  cổng rò rỉ ({g['mode']}) sẽ kiểm {len(refs)} bench: {', '.join(refs)}"
          + (f"\n  bench khoá không có trong leak_guard (vẫn được kiểm): {', '.join(extra)}" if extra else ""))
    dev_refs = list(c.exp["dev"].values()) if c.has_dev and g["check_dev"] else []
    if not c.dry:
        (c.work / "leak_report.json").unlink(missing_ok=True)
    try:
        if not c.dry:
            for ref in dev_refs:
                if not (R.resolve(ref).meta.get("flags") or {}).get("independent"):
                    raise R.ReleaseError(f"{ref}: flags.independent must be true")
        for ref, out in jobs:
            print(f"  {ref} -> {out}")
            if not c.dry:
                R.materialize(ref, out)
    except R.ReleaseError as exc:
        print(exc)
        sys.exit(3)
    if c.dry:
        return
    c.work.mkdir(parents=True, exist_ok=True)
    resolved = dict(c.exp)
    if c.seed is not None:
        resolved["run_seed"] = c.seed
    (c.work / "experiment.resolved.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
                                                     encoding="utf-8")
    try:
        res = LG.run_gate(c.exp["name"], c.guard_refs, jobs, mode=g["mode"],
                          exemptions=str(X.abspath(g["exemptions"])) if g["exemptions"] else None,
                          reserved_terms=str(X.abspath(g["reserved_terms"])) if g["reserved_terms"] else None,
                          text_files=[str(X.abspath(f)) for f in g["text_files"]], audio_sha=g["audio_sha"],
                          require_audio_sha=g["require_audio_sha"], near=g["near"], dev_refs=dev_refs, out_dir=c.work)
    except FileNotFoundError as exc:
        print(f"Missing leakage-gate input: {exc.filename}")
        sys.exit(3)
    print(f"  cổng rò rỉ: {'ĐẠT' if res['passed'] else 'KHÔNG ĐẠT'}: {res['critical']} critical, {res['review']} cần xem, "
          f"{res['exempted']} miễn trừ ({res['sec']} s); audio_sha={res['audio_sha']}")
    for ref, why in res["unresolved"].items():
        print(f"    CHƯA KIỂM {ref}: {why}")
    for ref, b in {**res["benches"], **res["dev"]}.items():
        for f in b["findings"]:
            if f["severity"] == "critical":
                print(f"    {ref}: {f['kind']} {f['count']} giá trị / {f.get('test_rows', '–')} câu test")
    from gipbench.core.leak import LeakGuard
    cat = Catalog.load(str(c.cat))  # (bản cũ gọi Catalog(str(...)): truyền đường dẫn vào trường `terms`, stage 0 vỡ khi chạy thật)
    guard = LeakGuard(read_jsonl(str(c.bench)), cat)
    lines = [f"## Bộ lọc câu cũ (LeakGuard) so với {c.guard_refs[0]}", "",
             "Không phải cổng: to_lhotse (stage 2) loại các câu này khỏi cut.", "",
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
    md = c.work / "leak_report.md"
    md.write_text(md.read_text(encoding="utf-8") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:]))
    if res["blocking"]:
        print(f"LỖI cổng rò rỉ: {res['critical']} critical chưa được miễn trừ -> DỪNG trước khi curate / train. "
              f"Báo cáo: {md}. Sửa dữ liệu (loại câu / video trùng) hoặc miễn trừ có ghi chép (leak_gate.exemptions); "
              "mode: report chỉ cho chẩn đoán có chủ ý.")
        sys.exit(3)


# ------------------------------------------------------------------ stage 1
def stage_curate(c):
    print("== 1. curate")
    curate = dict(c.exp.get("curate") or {})
    if c.seed is not None:
        curate["seed"] = c.seed
    c.run([c.py, "-m", "gipbench.model.recipe.curate", "--manifest", *[p for _, p in c.inputs], "--catalog", c.cat,
           "--out", c.work / "train_curated.jsonl", *(["--allow-bench-catalog"] if c.oracle else []),
           *opt_args(curate)])


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
        c.run([c.py_ft, "-m", "gipbench.model.features.to_lhotse", "--manifest", p, "--name", g, *tail, *aug, *c.seed_args])
    for k in c.dev:
        c.run([c.py_ft, "-m", "gipbench.model.features.to_lhotse", "--manifest", c.dev[k], "--name", f"dev_{k}", *tail])
    if not c.has_dev:
        n = int((c.exp.get("select") or {}).get("valid_rows", 200))
        print(f"  valid (chỉ theo dõi loss, KHÔNG chọn checkpoint): {n} câu train theo hash id -> {c.valid}")
        if not c.dry:
            write_valid_sample(c.work / "train_curated.jsonl", c.valid, n)
        c.run([c.py_ft, "-m", "gipbench.model.features.to_lhotse", "--manifest", c.valid, "--name", "valid", *tail])


def write_valid_sample(src, out, n):
    """n câu của train_curated chọn theo sha256(id) (ổn định, không phụ thuộc thứ tự). Chỉ cho valid loss của icefall."""
    rows = read_jsonl(str(src))
    rows.sort(key=lambda r: hashlib.sha256(str(r["id"]).encode("utf-8")).hexdigest())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows[:n]:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ stage 3
def stage_mix(c):
    m = c.exp["mix"]
    print(f"== 3. mix_cuts: {m['epochs']} epoch ảo × {m['epoch_hours']} h cut")
    args = []
    for g in X.groups(c.exp):
        if not g["refs"] or not g["weight"]:
            continue
        p = c.lhotse / f"{g['group']}_cuts_raw.jsonl.gz"
        if p.exists() or c.dry:
            args += ["--group", f"{g['group']}={p}:{g['weight']}"]
        elif c.exp["train"].get("max_batches"):
            sys.exit(f"Missing cuts for the configured E4 group {g['group']}: {p}")
    c.run([c.py_ft, "-m", "gipbench.model.features.mix_cuts", *args, "--epoch-hours", m["epoch_hours"],
           "--epochs", m["epochs"], *(["--max-views", m["max_views"]] if m.get("max_views") else []),
           *c.seed_args, "--out", c.lhotse / "train_mix_cuts.jsonl.gz"])


# ------------------------------------------------------------------ stage 4
def finetune_cmd(c, seg):
    t, lh = c.exp["train"], c.lhotse
    valid = ([str(lh / "dev_cs_cuts_raw.jsonl.gz"), str(lh / "dev_vi_cuts_raw.jsonl.gz")] if c.has_dev
             else [str(lh / "valid_cuts_raw.jsonl.gz")])
    return lambda res: [c.py_ft, "./zipformer/finetune_cs.py", "--world-size", "1", "--use-fp16", "1",
                        *t["model_args"].split(), *res, *c.seed_args, "--exp-dir", str(c.exp_dir), "--bpe-model", str(c.model / "bpe.model"),
                        "--freeze-modules", t["freeze"], "--train-cuts", str(lh / "train_mix_cuts.jsonl.gz"),
                        "--valid-cuts", *valid,
                        "--on-the-fly-feats", "True", "--num-workers", str(t.get("workers", 2)),
                        "--num-buckets", str(t.get("buckets", 10)), "--mp-start-method", "spawn", "--shuffle", "True",
                        "--base-lr", str(t["lr"] if t["lr"] is not None else "<X1-selected-LR>"),
                        "--max-duration", str(t["max_duration"]), "--enable-musan", "False",
                        *opt_args({"max_batches": t.get("max_batches")}),
                        "--enable-spec-aug", "True", "--save-every-n", str(seg), "--keep-last-k", str(t.get("keep_last_k", 40)),
                        "--manifest-dir", str(lh)]


def stage_finetune(c):
    t, m = c.exp["train"], c.exp["mix"]
    max_batches = t.get("max_batches")
    if not c.dry and t["lr"] is None:
        sys.exit("Set train.lr to the learning rate selected in X1 before training.")
    stats = c.lhotse / "train_mix_cuts.jsonl.gz.json"
    if not c.dry and max_batches:
        checked_gate(c)
        if not stats.is_file():
            sys.exit("Run stages 1-3 before the E4 training run.")
    eh = json.loads(stats.read_text(encoding="utf-8"))["epoch_hours"] if stats.exists() else m["epoch_hours"]
    steps = max_batches // m["epochs"] if max_batches else IR.steps_per_epoch(eh, t["max_duration"])
    seg, eps = IR.segment_plan(steps, t.get("segments_per_epoch", 4))
    previous = IR.ckpt_nums(c.exp_dir)
    if not c.dry and max_batches and previous and previous[-1] >= max_batches:
        if previous[-1] != max_batches:
            sys.exit(f"Existing checkpoint {previous[-1]} exceeds the configured {max_batches} batches.")
        print(f"  Training already reached {max_batches} batches.")
        return
    print(f"== 4. finetune (LR {t['lr']}, freeze {t['freeze']}) -> {c.exp_dir}")
    print(f"  ước tính {steps} step / epoch ảo -> checkpoint mỗi {seg} step, epoch ảo = {eps} step")
    c.run([c.py_ft, "-m", "gipbench.model.train.wrap_ckpt", c.model / "model.pt", c.init_ckpt]
          + (["--drop-ctc"] if t.get("drop_ctc") else []))
    c.run([c.py_ft, "-m", "gipbench.model.train.patch_finetune", "--icefall", c.icefall])
    cmd = finetune_cmd(c, seg)
    if c.dry:
        print(f"$ (cd {c.egs}) " + " ".join(cmd(IR.resume_args(c.exp_dir, c.init_ckpt))) + f" >> {c.work}/finetune.log")
        print(f"  watchdog {t.get('watchdog')}; resume: --do-finetune 0 --start-epoch 2 --num-epochs 1 --start-batch <N>")
        return
    wd = t.get("watchdog", {})
    runner = IR.Runner(cmd, cwd=c.egs, env=c.icefall_env(), exp_dir=c.exp_dir, eps=eps, log=c.work / "finetune.log",
                       seg_log=c.work / "segments.log", watchdog=wd, disk_path=wd.get("disk_path", str(c.work)))
    res = runner.run(c.init_ckpt)
    if res != "done":
        raise SystemExit(f"finetune dừng: {res}")
    nums = IR.ckpt_nums(c.exp_dir)
    if max_batches and (not nums or nums[-1] != max_batches):
        sys.exit(f"Training data was exhausted before {max_batches} batches. Check the realized mix and batch size.")


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
    for k in c.dev:
        s = f"dev_{k}"
        if (c.rd / name / s / "hyps.jsonl").exists():
            continue
        c.run([c.py, "-m", "gipbench.infer.decode", "--bench", c.dev[k], "--model-dir", md, "--system", s,
               "--results", c.rd / name, "--threads", d.get("threads", 16), "--batch", d.get("batch", 32)],
              log=c.rd / name / f"{s}_decode.log")


def stage_candidates(c):
    if c.fixed:
        print(f"== 5. ứng viên: bỏ qua (select.rule: {X.FIXED_RULE}, không có dev để decode)")
        return
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
    sel = c.exp.get("select", {})
    limits = {key: sel.get(key) for key in ("max_vi_ci_upper", "max_vi_hallucination_delta", "max_cs_domain_delta")}
    gate_config = LG.gate_config(c.exp)
    reserved = gate_config["reserved_terms"] if gate_config["check_dev"] else None
    c.run([c.py, "-m", "gipbench.model.select.select_ckpt", "--catalog", c.cat, "--dev-cs", c.dev["cs"], "--dev-vi", c.dev["vi"],
           "--base", c.rd / "base", "--cand", *cands, "--max-vi-delta", sel.get("max_vi_delta", 0.3),
           "--rule", sel.get("rule", "term_acc_macro"), "--dev-ref-cs", c.exp["dev"]["cs"], "--dev-ref-vi", c.exp["dev"]["vi"],
           *(["--allow-legacy-dev"] if sel.get("allow_legacy_dev") else []), *(["--allow-bench-catalog"] if c.oracle else []),
           *opt_args({**limits, "reserved_terms": X.abspath(reserved) if reserved else None}), "--out", out])
    return None if c.dry else json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))


def checked_gate(c):
    path = c.work / "leak_report.json"
    if not path.is_file():
        print(f"LỖI chưa có báo cáo cổng rò rỉ {path}: chạy stage 0 trước.")
        sys.exit(3)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate["blocking"]:
        print(f"LỖI báo cáo cổng rò rỉ {path} còn {gate['critical']} critical chưa miễn trừ: không tạo artifact.")
        sys.exit(3)
    config = LG.gate_config(c.exp)
    if c.exp.get("train", {}).get("max_batches"):
        texts = [str(X.abspath(path)) for path in config["text_files"]]
        dev = [c.exp["dev"][kind] for kind in c.dev]
        refs = [*X.input_refs(c.exp), *dev, *[f"text:{Path(path).name}" for path in texts]]
        if (not gate["passed"] or gate["mode"] != "fail" or refs != [entry["label"] for entry in gate["inputs"]]
                or set(gate["benches"]) != set(c.guard_refs) | set(R.locked_bench_refs())
                or gate.get("text_file_sha256", {}) != {path: R.sha256(path) for path in texts}):
            sys.exit("Run stage 0 with the exact E4 inputs in fail mode before training or export.")
    if config["check_dev"] and not set(c.exp["dev"].values()) <= set(gate.get("dev", {})):
        sys.exit("Run stage 0 with the exact development releases before training or export.")
    return gate


def stage_select(c):
    alphas = c.exp.get("select", {}).get("wise_alphas", [0.3, 0.5, 0.7, 1.0])
    gate = None
    if not c.dry:
        gate = checked_gate(c)
        max_batches = c.exp["train"].get("max_batches")
        nums = IR.ckpt_nums(c.exp_dir)
        if max_batches and (not nums or nums[-1] != max_batches):
            sys.exit(f"Stage 4 must reach the configured {max_batches} batches before export.")
    if c.fixed:
        final, md, wck = select_fixed(c)
        rw = {"results": {}}
    else:
        final, md, wck, rw = select_on_dev(c, alphas)
    print(f"  model chọn: {final} ({md}) -> artifact {c.artifact}")
    if c.dry:
        return
    sel = c.exp.get("select", {})
    dev_refs = [c.exp["dev"][k] for k in ("cs", "vi")] if c.has_dev else []
    card = {"system": c.exp["name"], "experiment": {"path": c.exp["_path"], "overrides": c.exp["_overrides"],
                                                     "sha256": hashlib.sha256(Path(c.exp["_path"]).read_bytes()).hexdigest()},
            "base_model": {"path": str(c.model), "sha256": onnx_export.sha256(c.model / "model.pt")
                           if (c.model / "model.pt").exists() else None},
            "releases_used": {ref: R.resolve(ref).meta["files"] for ref in
                              [*X.input_refs(c.exp), *dev_refs, c.exp["catalog"], *c.guard_refs]},
            "catalog": {"ref": c.exp["catalog"], "flags": R.catalog_flags(c.exp["catalog"]), "use": X.catalog_use(c.exp)},
            "select": {"rule": sel.get("rule", "term_acc_macro"), "dev": c.exp["dev"],
                       "dev_flags": {k: R.resolve(c.exp["dev"][k]).meta.get("flags") or {} for k in ("cs", "vi")}
                       if c.has_dev else None,
                       **({"fixed_wise_alpha": float(sel.get("fixed_wise_alpha", 1.0))} if c.fixed else {})},
            "leak_gate": LG.card_summary(gate, c.work / "leak_report.json"),
            "leak_report_sha256": hashlib.sha256((c.work / "leak_report.json").read_bytes()).hexdigest(),
            "checkpoint_picked": final, "wise_from": wck, "dev_metrics": rw["results"].get(final),
            "git_commit": R.git_commit()}
    if c.oracle:
        card.update(inference_mode="oracle_bias", rankable=False)
    if c.seed is not None:
        card["seed"] = c.seed
        card["mix"] = json.loads((c.lhotse / "train_mix_cuts.jsonl.gz.json").read_text(encoding="utf-8"))
        card["experiment"]["sources"] = {str(Path(path).name): R.sha256(path) for path in c.exp["_sources"]}
        path = c.exp_dir / f"checkpoint-{nums[-1]}.pt"
        sampler_path = c.work / "sampler_state.json"
        c.run([c.py_ft, "-m", "gipbench.model.train.sampler_state", path, sampler_path])
        card["sampler_state"] = json.loads(sampler_path.read_text(encoding="utf-8"))
        card["train"] = c.exp["train"]
    onnx_export.write_artifact(c.artifact, md, card)
    for ext in ("json", "md"):
        shutil.copy(c.work / f"leak_report.{ext}", c.artifact / f"leak_report.{ext}")
    write_systems(c)


def select_fixed(c):
    """Luật cố định đăng ký trước (không nhìn dev / bench): checkpoint cuối; WiSE-FT đúng một alpha nếu alpha < 1."""
    alpha = float((c.exp.get("select") or {}).get("fixed_wise_alpha", 1.0))
    nums = IR.ckpt_nums(c.exp_dir)
    last = nums[-1] if nums else ("<N>" if c.dry else None)
    if last is None:
        sys.exit(f"LỖI không có checkpoint-*.pt trong {c.exp_dir}: chạy stage 4 trước")
    wck = f"ck{last}"
    print(f"== 6. luật cố định ({X.FIXED_RULE}): checkpoint cuối {wck}, WiSE-FT alpha {alpha:g} (không dùng dev)")
    if alpha >= 1.0:
        return wck, export(c, c.exp_dir, wck, ["--iter", str(last)]), wck
    an = f"{alpha:g}"
    c.run([c.py_ft, "-m", "gipbench.model.select.interpolate_ckpt", "--base", c.init_ckpt,
           "--ft", c.exp_dir / f"checkpoint-{last}.pt", "--ft-key", "model", "--alphas", an,
           "--out-dir", c.exp_dir / "wise", "--epoch", "1"])
    md = export(c, c.exp_dir / "wise" / f"alpha_{an}", f"wise_{an}", ["--epoch", "1"])
    if not c.dry:
        (c.exp_dir / "wise" / f"alpha_{an}" / "epoch-1.pt").unlink(missing_ok=True)
    return f"wise_{an}", md, wck


def select_on_dev(c, alphas):
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
    c.run([c.py_ft, "-m", "gipbench.model.select.interpolate_ckpt", "--base", c.init_ckpt,
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
    return final, md, wck, rw


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
        body = {"system": s["system"], "model": art}
        if c.oracle or s.get("inference_mode") == "oracle_bias":  # catalog có span bench: không được xếp hạng
            body["inference_mode"] = "oracle_bias"
        body["decode"] = dec
        p.write_text(f"# Sinh bởi gipbench.model.run ({c.exp['_path']}). Chạy: python -m gipbench.bench.run --bench <bench> "
                     f"--system {p.relative_to(REPO).as_posix()}\n"
                     + yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    ap.add_argument("--seed", type=int, help="Run a single configured seed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    exp = X.load(args.experiment, args.set)
    stages = parse_stages(args.stages)
    for i in stages:
        if not 0 <= i < len(STAGES):
            sys.exit(f"stage {i} không có (0–{len(STAGES) - 1})")
    for seed in X.run_seeds(exp, args.seed, args.dry_run):
        c = Ctx(exp, args.dry_run, seed=seed)
        if seed is not None:
            print(f"== {exp['name']} seed {seed}")
        for i in stages:
            STAGES[i](c)


if __name__ == "__main__":
    main()
