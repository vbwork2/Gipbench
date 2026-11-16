#!/usr/bin/env python3
"""Kiểm kế hoạch trộn dữ liệu train theo configs/model/train_mix.tsv (không ghi dữ liệu, chỉ báo cáo).

  python -m gipbench.model.recipe.mix_plan --manifest data/raw/vss_full/vss_cs_clean.jsonl data/raw/vivos/train_vivos.jsonl \
      --assume ViMedCSS:medical:24.30 --out reports/mix_plan.md

1. Đọc nhóm + tỉ trọng (ghi đè bằng biến môi trường ở cột env); tỉ trọng phải cộng lại = 1.00 (lệch > 1e-6: lỗi).
2. Gán mỗi câu vào nhóm theo trường `dataset` (tiền tố trước ':') so với cột datasets. Câu không thuộc nhóm nào được
   liệt kê riêng. --assume DATASET:DOMAIN:GIỜ thêm số giờ cho bộ chưa convert hết (vd ViMedCSS train theo meta CSV).
3. Phân bố domain DỰ KIẾN trong phần CS của batch (--cs-groups, mặc định cs_real + cs_pseudo): domain d chiếm
   sum_g w_g * giờ_g(d) / giờ_g  chia  sum_g w_g, chỉ tính nhóm có dữ liệu (nhóm rỗng không lấy mẫu được).
4. Trần domain (--domain-cap 0.35): domain nào vượt trần thì CẢNH BÁO (riêng medical ghi rõ là ViMedCSS).
5. --fit-cap (decisions §12.2, 2 biến thể E3 với trần 35% / 50%): đưa ra PHƯƠNG ÁN đạt trần bằng trọng số lấy mẫu theo
   domain trong phần CS: domain vượt trần bị cắt về đúng trần, phần dư chia cho các domain còn lại theo tỉ lệ tự nhiên
   (lặp tới khi không domain nào vượt; --cap-exempt mặc định 'other' không bị cắt vì không phải một domain thật).
   Báo hệ số lặp tương đối (1 giờ audio domain d được lấy mẫu gấp bao nhiêu lần so với trung bình phần CS) và ghi
   trọng số theo (dataset, domain) ra --out-json để pipeline train dùng sau.
Exit 0 kể cả khi có cảnh báo; exit 1 nếu tổng tỉ trọng sai.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.manifest import read_jsonl

from gipbench.core.paths import REPO as G


def load_mix(path):
    groups = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            w = float(os.environ.get(r["env"], r["weight"])) if r.get("env") else float(r["weight"])
            groups.append({"group": r["group"], "weight": w, "env": r["env"],
                           "datasets": [d.strip() for d in r["datasets"].split(",") if d.strip()], "note": r.get("note", "")})
    return groups


def plan(groups, rows_by_dataset, cs_groups, cap):
    """rows_by_dataset: {dataset_prefix: Counter(domain -> giờ)}. Trả về (giờ theo nhóm/domain, phân bố CS, cảnh báo)."""
    owner = {d: g["group"] for g in groups for d in g["datasets"]}
    gh = defaultdict(Counter)
    orphan = Counter()
    for ds, doms in rows_by_dataset.items():
        if ds in owner:
            gh[owner[ds]].update(doms)
        else:
            orphan[ds] += sum(doms.values())
    w = {g["group"]: g["weight"] for g in groups}
    live = [g for g in cs_groups if sum(gh[g].values()) > 0]
    tot_w = sum(w[g] for g in live)
    share = Counter()
    for g in live:
        H = sum(gh[g].values())
        for d, h in gh[g].items():
            share[d] += w[g] * h / H / tot_w
    warn = []
    for g in cs_groups:
        if w.get(g, 0) > 0 and g not in live:
            warn.append(f"nhóm {g} (tỉ trọng {w[g]:.2f}) chưa có dữ liệu: phần CS chỉ lấy từ {', '.join(live) or 'không nhóm nào'}")
    for d, s in share.most_common():
        if s > cap + 1e-12:
            extra = " (ViMedCSS)" if d == "medical" else ""
            warn.append(f"domain {d}{extra} chiếm {100 * s:.1f}% phần CS ({' + '.join(cs_groups)}) > trần {100 * cap:.0f}%")
    return gh, share, warn, orphan


def fit_cap(share, cap, exempt):
    """Water-filling: -> (tỉ lệ mục tiêu theo domain, các domain bị cắt, khả thi?)."""
    tgt = dict(share)
    capped = set()
    for _ in range(len(tgt) + 1):
        over = [d for d, v in tgt.items() if v > cap + 1e-12 and d not in exempt and d not in capped]
        if not over:
            break
        capped |= set(over)
        free = [d for d in tgt if d not in capped]
        rest = 1.0 - cap * len(capped)
        base = sum(share[d] for d in free)
        if not free or base <= 0 or rest < 0:
            return tgt, capped, False
        for d in capped:
            tgt[d] = cap
        for d in free:
            tgt[d] = rest * share[d] / base
    return tgt, capped, all(v <= cap + 1e-9 for d, v in tgt.items() if d not in exempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", default=str(G / "configs/model/train_mix.tsv"))
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--assume", nargs="*", default=[], metavar="DATASET:DOMAIN:GIỜ")
    ap.add_argument("--cs-groups", nargs="+", default=["cs_real", "cs_pseudo"])
    ap.add_argument("--domain-cap", type=float, default=0.35)
    ap.add_argument("--fit-cap", action="store_true", help="đưa ra phương án trọng số theo domain đạt --domain-cap")
    ap.add_argument("--cap-exempt", nargs="*", default=["other"])
    ap.add_argument("--out-json")
    ap.add_argument("--out")
    args = ap.parse_args()
    groups = load_mix(args.mix)
    total = sum(g["weight"] for g in groups)
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"LỖI: tổng tỉ trọng = {total:.4f} (phải = 1.00): "
                 + ", ".join(f"{g['group']}={g['weight']:.2f}" for g in groups))
    by_ds = defaultdict(Counter)
    for m in args.manifest:
        for r in read_jsonl(m):
            by_ds[str(r.get("dataset", "?")).split(":")[0]][r.get("domain") or "other"] += float(r["duration"]) / 3600
    for a in args.assume:
        ds, dom, h = a.rsplit(":", 2)
        by_ds[ds] = Counter({dom: float(h)})
    gh, share, warn, orphan = plan(groups, by_ds, args.cs_groups, args.domain_cap)
    md = ["# Kế hoạch trộn dữ liệu train", "", f"Cấu hình: `{args.mix}`. Tổng tỉ trọng = {total:.2f}.", "",
          "| Nhóm | Tỉ trọng | Datasets | Giờ có | Domain (giờ) |", "| --- | --- | --- | --- | --- |"]
    for g in groups:
        h = gh[g["group"]]
        md.append(f"| {g['group']} | {g['weight']:.2f} | {', '.join(g['datasets'])} | {sum(h.values()):.2f} | "
                  + (", ".join(f"{d} {v:.2f}" for d, v in h.most_common()) or "—") + " |")
    md += ["", f"Phân bố domain dự kiến trong phần CS của batch ({' + '.join(args.cs_groups)}), trần {100 * args.domain_cap:.0f}%:",
           "", "| Domain | Tỉ lệ |", "| --- | --- |"] + [f"| {d} | {100 * s:.1f}% |" for d, s in share.most_common()]
    if orphan:
        md += ["", "Dữ liệu không thuộc nhóm nào (không được trộn): "
               + ", ".join(f"{d} {h:.2f} h" for d, h in orphan.most_common())]
    if args.assume:
        md += ["", "Giả định (--assume): " + ", ".join(f"`{a}`" for a in args.assume)]
    md += [""] + ([f"**CẢNH BÁO:** {x}" for x in warn] or ["Không có cảnh báo."])
    if args.fit_cap and share:
        tgt, capped, ok = fit_cap(share, args.domain_cap, set(args.cap_exempt))
        cs_h = Counter()
        for g in args.cs_groups:
            cs_h.update(gh[g])
        tot_h = sum(cs_h.values())
        rep = {d: (tgt[d] / cs_h[d]) / (1.0 / tot_h) if cs_h[d] else float("inf") for d in tgt}
        md += ["", f"## Phương án đạt trần {100 * args.domain_cap:.0f}% (--fit-cap)", "",
               ("Đạt trần." if ok else "**KHÔNG đạt được trần** với dữ liệu hiện có.")
               + f" Domain bị cắt: {', '.join(sorted(capped)) or 'không'}; miễn trần: {', '.join(args.cap_exempt) or 'không'}.",
               "", "| Domain | Giờ có (phần CS) | Tỉ lệ tự nhiên | Tỉ lệ mục tiêu | Hệ số lặp tương đối |",
               "| --- | --- | --- | --- | --- |"]
        md += [f"| {d} | {cs_h[d]:.2f} | {100 * share[d]:.1f}% | {100 * tgt[d]:.1f}% | {rep[d]:.2f}× |"
               for d in sorted(tgt, key=lambda d: -tgt[d])]
        md += ["", "Hệ số lặp tương đối = (tỉ lệ mục tiêu / giờ có của domain) chia (1 / tổng giờ phần CS): 1× = lấy mẫu "
               "đều theo giờ; số lớn = audio domain đó bị lặp nhiều (dễ overfit)."]
        if args.out_json:
            cell = {}
            for g in args.cs_groups:
                for d, hh in gh[g].items():
                    cell[f"{g}:{d}"] = {"hours": round(hh, 3), "target_share": round(tgt[d] * hh / cs_h[d], 5)}
            Path(args.out_json).write_text(json.dumps({"domain_cap": args.domain_cap, "feasible": ok,
                                                       "target_share": {d: round(v, 5) for d, v in tgt.items()},
                                                       "cells": cell}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
