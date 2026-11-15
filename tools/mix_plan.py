#!/usr/bin/env python3
"""Kiểm kế hoạch trộn dữ liệu train theo configs/train_mix.tsv (không ghi dữ liệu, chỉ báo cáo).

  python tools/mix_plan.py --manifest data/raw/vss_full/vss_cs_clean.jsonl data/raw/vivos/train_vivos.jsonl \
      --assume ViMedCSS:medical:24.30 --out reports/mix_plan.md

1. Đọc nhóm + tỉ trọng (ghi đè bằng biến môi trường ở cột env); tỉ trọng phải cộng lại = 1.00 (lệch > 1e-6: lỗi).
2. Gán mỗi câu vào nhóm theo trường `dataset` (tiền tố trước ':') so với cột datasets. Câu không thuộc nhóm nào được
   liệt kê riêng. --assume DATASET:DOMAIN:GIỜ thêm số giờ cho bộ chưa convert hết (vd ViMedCSS train theo meta CSV).
3. Phân bố domain DỰ KIẾN trong phần CS của batch (--cs-groups, mặc định cs_real + cs_pseudo): domain d chiếm
   sum_g w_g * giờ_g(d) / giờ_g  chia  sum_g w_g, chỉ tính nhóm có dữ liệu (nhóm rỗng không lấy mẫu được).
4. Trần domain (--domain-cap 0.35): domain nào vượt trần thì CẢNH BÁO (riêng medical ghi rõ là ViMedCSS).
Exit 0 kể cả khi có cảnh báo; exit 1 nếu tổng tỉ trọng sai.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.manifest import read_jsonl  # noqa: E402

G = Path(__file__).resolve().parents[1]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", default=str(G / "configs/train_mix.tsv"))
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--assume", nargs="*", default=[], metavar="DATASET:DOMAIN:GIỜ")
    ap.add_argument("--cs-groups", nargs="+", default=["cs_real", "cs_pseudo"])
    ap.add_argument("--domain-cap", type=float, default=0.35)
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
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
