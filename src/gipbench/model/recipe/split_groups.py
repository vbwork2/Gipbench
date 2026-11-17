#!/usr/bin/env python3
"""Tách manifest train đã curate thành từng nhóm trộn (trước đây là heredoc stage 2 của ft/run_e3_v1*.sh).

  python -m gipbench.model.recipe.split_groups --experiment configs/model/experiments/e3_v1b.yaml \
      --manifest data/e3_v1b/train_curated.jsonl --out-dir data/e3_v1b

Nhóm của một dòng: tiền tố trước ':' của trường `dataset` (không phân biệt hoa thường) so với `pools.<nhóm>.datasets`
của experiment; không khớp mà dòng có `distill: true` thì vào nhóm distill; còn lại: lỗi. Ghi <out-dir>/group_<nhóm>.jsonl
(giữ nguyên từng dòng, đúng thứ tự). --mix nhận cả configs/train_mix.tsv cũ.
"""
import argparse
import json
from pathlib import Path

from gipbench.model.recipe.mix_plan import load_mix


def dataset_to_group(groups):
    m = {}
    for g in groups:
        for d in g["datasets"]:
            m[d.strip().lower()] = g["group"]
    return m


def split(manifest, groups, out_dir):
    owner = dataset_to_group(groups)
    out = {}
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        d = r["dataset"].split(":")[0].lower()
        g = owner.get(d) or ("distill" if r.get("distill") else None)
        if g is None:
            raise SystemExit(f"không biết nhóm của dataset {r['dataset']}")
        out.setdefault(g, []).append(line)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for g, ls in out.items():
        open(Path(out_dir) / f"group_{g}.jsonl", "w", encoding="utf-8").writelines(ls)
        print(g, len(ls), "câu", round(sum(json.loads(x)["duration"] for x in ls) / 3600, 2), "h")
    return {g: Path(out_dir) / f"group_{g}.jsonl" for g in out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--experiment", help="experiment YAML")
    src.add_argument("--mix", help="configs/train_mix.tsv (định dạng cũ)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    split(args.manifest, load_mix(args.experiment or args.mix), args.out_dir)


if __name__ == "__main__":
    main()
