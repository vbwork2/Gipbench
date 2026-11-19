"""Đọc experiment YAML (configs/model/experiments/<exp>.yaml): công thức một lần FT, thay cho các file shell run_e3_*.sh.

  exp = load("configs/model/experiments/e3_v1b.yaml", overrides=["train.lr=0.001", "pools.synth.weight=0.3"])
  groups(exp)       -> [{"group", "weight", "env", "datasets", "note", "refs"}]  (cùng dạng mix_plan.load_mix cũ)
  input_refs(exp)   -> pool ref theo thứ tự nhóm (= thứ tự --manifest của curate)

Chuỗi ${exp_root}, ${results_root}, ${icefall}, ${name}, ${repo} được thay bằng giá trị trong gipbench.core.paths / chính file. Trọng số
nhóm vẫn ghi đè được bằng biến môi trường ở khoá `env` (MIX_CS_REAL, MIX_SYNTH, ... như configs/train_mix.tsv cũ).
"""
import os
import re
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core.paths import REPO

REQUIRED = ["name", "base_model", "catalog", "leak_guard", "dev", "pools", "output"]
# Luật chọn không cần dev (đăng ký trước): checkpoint cuối, WiSE-FT một alpha cố định (select.fixed_wise_alpha, mặc định
# 1.0 = không nội suy). dev: null. Checkpoint vẫn được giữ (keep_last_k) để chọn lại trên dev sau này nếu cần.
FIXED_RULE = "fixed_last"


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


def _merge(base, changes):
    result = dict(base)
    for key, value in changes.items():
        previous = result.get(key)
        result[key] = _merge(previous, value) if isinstance(previous, dict) and isinstance(value, dict) else value
    return result


def _read(path, chain=()):
    path = path.resolve()
    if path in chain:
        raise SystemExit(f"Cyclic experiment inheritance: {path}")
    exp = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = exp.pop("extends", None)
    if parent is None:
        return exp, [path]
    base, sources = _read(path.parent / parent, (*chain, path))
    return _merge(base, exp), [*sources, path]


def load(path, overrides=()):
    path = Path(path)
    exp, sources = _read(path)
    for ov in overrides:
        k, sep, v = ov.partition("=")
        if not sep:
            raise SystemExit(f"--set cần dạng khoá.con=giá_trị: {ov!r}")
        _set(exp, k.strip(), yaml.safe_load(v))
    missing = [k for k in REQUIRED if k not in exp]
    if missing:
        raise SystemExit(f"{path}: thiếu khoá {missing}")
    rule = (exp.get("select") or {}).get("rule")
    if not exp["dev"] and rule != FIXED_RULE:
        raise SystemExit(f"{path}: dev rỗng chỉ dùng được với select.rule: {FIXED_RULE} (luật chọn cố định đăng ký trước)")
    if exp["dev"] and rule == FIXED_RULE:
        raise SystemExit(f"{path}: select.rule {FIXED_RULE} không dùng dev; đặt dev: null cho rõ")
    ctx = {"exp_root": paths.get("exp_root"), "results_root": paths.get("results_root"),
           "icefall": paths.get("icefall"), "repo": str(REPO), "name": exp["name"]}
    exp = _expand(exp, ctx)
    exp["_path"] = str(path)
    exp["_overrides"] = list(overrides)
    exp["_sources"] = [str(source) for source in sources]
    return exp  # tổng trọng số không bắt buộc = 1 (quét MIX_SYNTH=0 ...); mix_plan kiểm, mix_cuts chuẩn hoá


def run_seeds(exp, selected, dry):
    seeds = exp.get("seeds")
    if seeds is None and selected is None:
        return [None]
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise SystemExit("Experiment seeds must be a list of distinct non-negative integers")
    if selected is not None:
        if selected not in seeds:
            raise SystemExit(f"--seed must be one of {seeds}")
        return [selected]
    if not dry and len(seeds) > 1:
        raise SystemExit(f"Select one run with --seed from {seeds}")
    return seeds


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


CATALOG_USES = ("train", "oracle_diagnostic")


def catalog_use(exp):
    """`catalog_use`: train (mặc định; catalog dựng từ text train, không có span của bench) | oracle_diagnostic (cố ý dùng
    catalog có span bench làm chẩn đoán: artifact + system YAML gắn inference_mode oracle_bias, không được xếp hạng)."""
    u = exp.get("catalog_use", "train")
    if u not in CATALOG_USES:
        raise SystemExit(f"catalog_use phải thuộc {CATALOG_USES}, không phải {u!r}")
    return u


def system_hotword_catalogs(exp):
    """[(tên system, catalog ref, đã khai báo inference_mode oracle_bias)] của các system có hotwords."""
    return [(s["system"], (s.get("hotwords") or {})["catalog"], s.get("inference_mode") == "oracle_bias")
            for s in exp.get("systems", []) if (s.get("hotwords") or {}).get("catalog")]


def abspath(p):
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO / p
