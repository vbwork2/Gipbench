"""Đọc experiment YAML (configs/model/experiments/<exp>.yaml): công thức một lần FT, thay cho các file shell run_e3_*.sh.

  exp = load("configs/model/experiments/e3_v1b.yaml", overrides=["train.lr=0.001", "pools.synth.weight=0.3"])
  groups(exp)       -> [{"group", "weight", "env", "datasets", "note", "refs"}]  (cùng dạng mix_plan.load_mix cũ)
  input_refs(exp)   -> pool ref theo thứ tự nhóm (= thứ tự --manifest của curate)

Chuỗi ${exp_root}, ${icefall}, ${name}, ${repo} được thay bằng giá trị trong gipbench.core.paths / chính file. Trọng số
nhóm vẫn ghi đè được bằng biến môi trường ở khoá `env` (MIX_CS_REAL, MIX_SYNTH, ... như configs/train_mix.tsv cũ).
"""
import os
import re
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core.paths import REPO

REQUIRED = ["name", "base_model", "catalog", "leak_guard", "dev", "pools", "output"]


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _expand(o, ctx):
    if isinstance(o, dict):
        return {k: _expand(v, ctx) for k, v in o.items()}
    if isinstance(o, list):
        return [_expand(v, ctx) for v in o]
    if isinstance(o, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: str(ctx.get(m.group(1), m.group(0))), o)
    return o


def load(path, overrides=()):
    path = Path(path)
    exp = yaml.safe_load(path.read_text(encoding="utf-8"))
    for ov in overrides:
        k, sep, v = ov.partition("=")
        if not sep:
            raise SystemExit(f"--set cần dạng khoá.con=giá_trị: {ov!r}")
        _set(exp, k.strip(), yaml.safe_load(v))
    missing = [k for k in REQUIRED if k not in exp]
    if missing:
        raise SystemExit(f"{path}: thiếu khoá {missing}")
    ctx = {"exp_root": paths.get("exp_root"), "icefall": paths.get("icefall"), "repo": str(REPO), "name": exp["name"]}
    exp = _expand(exp, ctx)
    exp["_path"] = str(path)
    exp["_overrides"] = list(overrides)
    return exp  # tổng trọng số không bắt buộc = 1 (quét MIX_SYNTH=0 ...); mix_plan kiểm, mix_cuts chuẩn hoá


def groups(exp):
    out = []
    for name, g in exp["pools"].items():
        env = g.get("env", "")
        w = float(os.environ.get(env, g["weight"])) if env else float(g["weight"])
        out.append({"group": name, "weight": w, "env": env, "datasets": list(g.get("datasets", [])),
                    "note": g.get("note", ""), "refs": list(g.get("refs", []))})
    return out


def input_refs(exp):
    return [r for g in groups(exp) for r in g["refs"]]


def abspath(p):
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO / p
