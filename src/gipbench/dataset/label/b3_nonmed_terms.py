#!/usr/bin/env python3
"""Đề xuất thuật ngữ reserved_b3 NGOÀI y khoa (decisions §12.6). CHỈ ĐỀ XUẤT, không sửa catalog / manifest.

  python -m gipbench.dataset.label.b3_nonmed_terms --ggroup data/raw/ggroup/b1_candidates.jsonl \
      --vss-cs data/raw/vss_full/vss_cs_clean_dom.jsonl --vss-pool data/raw/vss_full/vss_cs_candidates.jsonl \
      --exclude-manifest data/raw/consensus/b2_consensus_candidates.jsonl data/raw/consensus/b2_consensus_overcap.jsonl \
      --vimedcss-train data/dl/vimedcss_meta/train_set.csv --lexicons data/dl/lexicons \
      --catalog data/raw/_proxy/catalog_proxy.tsv --out reports/history/b3_nonmed_terms.csv --report reports/history/b3_nonmed_report.md

Nguồn thuật ngữ: span <en> của G-Group tech / edu / pubadmin (KHÔNG lấy finance) và của vss_cs (--vss-cs). Không lấy
consensus: thuật ngữ có trong --exclude-manifest (consensus, chỉ dùng để test, giữ ở B2) bị loại.
Điều kiện (tham số, chưa kiểm chứng):
  - tổng số câu (G-Group + vss_cs) trong [--min-sent 3, --max-sent 20], và có >= --min-gg 2 câu G-Group
    (để B3 có câu chấm);
  - "nhiều người nói": số nguồn khác nhau (mỗi câu G-Group một nguồn, vss_cs theo speaker = video) >= --min-speakers 3;
  - bỏ từ quá phổ biến: nằm trong --common-top (500) từ đầu của google-10000, hoặc có trong catalog (2000 -> 32 thuật
    ngữ, 1000 -> 38, 500 -> 45 trên dữ liệu 15/11, nên chọn 500 để đạt 40–80);
  - bỏ thuật ngữ đã có quyết định vi / drop_tag (configs/dataset/labeling/term_decisions.tsv, hoặc được đề xuất như vậy trong --ask),
    và mục 1 chữ <= 2 ký tự.
Chọn tối đa --n-max (80) thuật ngữ theo số nguồn rồi số câu, xoay vòng theo domain G-Group (tech / edu / pubadmin) để
cân domain. Báo trước (CHƯA áp dụng): số câu / giờ vss_cs (--vss-cs và cả kho --vss-pool) chứa ít nhất một thuật ngữ
được chọn (sẽ bị loại khỏi train), số câu ViMedCSS train chứa chúng, và số câu G-Group sẽ chuyển sang B3.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.term_decisions import NO_TAG, load_term_decisions
from gipbench.core.textnorm import Catalog, basic_tokens, normalize_ref

NONMED = ("tech", "edu", "pubadmin")


def spans(tagged, cat):
    _, sp = normalize_ref(tagged or "", cat)
    return {" ".join(s) for s in sp}


def jl(path):
    for line in open(path, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def contains(toks, term):
    t = term.split()
    return any(toks[i:i + len(t)] == t for i in range(len(toks) - len(t) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ggroup", required=True)
    ap.add_argument("--vss-cs", required=True)
    ap.add_argument("--vss-pool", help="kho vss_cs_candidates để báo số giờ có thể mất ở các lần lọc sau")
    ap.add_argument("--exclude-manifest", nargs="*", default=[])
    ap.add_argument("--ask", help="reports/history/term_decisions_ask.csv: bỏ cả thuật ngữ được ĐỀ XUẤT vi / drop_tag (chưa chốt)")
    ap.add_argument("--vimedcss-train")
    ap.add_argument("--lexicons")
    ap.add_argument("--catalog")
    ap.add_argument("--min-sent", type=int, default=3)
    ap.add_argument("--max-sent", type=int, default=20)
    ap.add_argument("--min-gg", type=int, default=2)
    ap.add_argument("--min-speakers", type=int, default=3)
    ap.add_argument("--common-top", type=int, default=500)
    ap.add_argument("--n-max", type=int, default=80)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()
    cat = Catalog.load(args.catalog) if args.catalog else Catalog()
    cat_keys = {" ".join(k) for k in cat.strict_map} if cat.strict_map else set()
    dec = dict(load_term_decisions())
    if args.ask and Path(args.ask).exists():
        for r in csv.DictReader(open(args.ask, encoding="utf-8-sig")):
            d = (r.get("decision") or "").strip()
            if d in NO_TAG:
                dec.setdefault(r["term"], d)
    common = set()
    if args.lexicons:
        p = Path(args.lexicons) / "google-10000-english-no-swears.txt"
        if p.exists():
            common = {l.strip().lower() for l in list(open(p, encoding="utf-8"))[:args.common_top]}
    excl = set()
    for m in args.exclude_manifest:
        for r in jl(m):
            excl |= spans(r.get("text_tagged"), cat)
    gg = defaultdict(lambda: {"n": 0, "dom": Counter(), "ex": ""})
    gg_fin = Counter()
    for r in jl(args.ggroup):
        for t in spans(r.get("text_tagged"), cat):
            if r.get("domain") in NONMED:
                e = gg[t]
                e["n"] += 1
                e["dom"][r["domain"]] += 1
                e["ex"] = e["ex"] or f"{r['id']}: {r['text_tagged']}"
            else:
                gg_fin[t] += 1
    vs = defaultdict(lambda: {"n": 0, "spk": set(), "h": 0.0, "ex": ""})
    vss_rows = list(jl(args.vss_cs))
    for r in vss_rows:
        for t in spans(r.get("text_tagged"), cat):
            e = vs[t]
            e["n"] += 1
            e["spk"].add(r.get("speaker"))
            e["h"] += float(r["duration"]) / 3600
            e["ex"] = e["ex"] or f"{r['id']}: {r['text_tagged'][:140]}"
    cands, why = [], Counter()
    for t in set(gg) | set(vs):
        g, v = gg.get(t), vs.get(t)
        ng, nv = (g["n"] if g else 0), (v["n"] if v else 0)
        nspk = ng + (len(v["spk"]) if v else 0)
        toks = t.split()
        if t in excl:
            why["bỏ: có trong consensus (giữ ở B2)"] += 1
            continue
        if t in cat_keys:
            why["bỏ: có trong catalog"] += 1
            continue
        if dec.get(t) in NO_TAG:
            why["bỏ: term_decisions = vi / drop_tag"] += 1
            continue
        if len(toks) == 1 and (t in common or len(t) <= 2):
            why["bỏ: quá phổ biến (google-10000 top) hoặc <= 2 ký tự"] += 1
            continue
        if ng < args.min_gg:
            why[f"bỏ: < {args.min_gg} câu G-Group tech/edu/pubadmin"] += 1
            continue
        if not (args.min_sent <= ng + nv <= args.max_sent):
            why[f"bỏ: tổng số câu ngoài [{args.min_sent}, {args.max_sent}]"] += 1
            continue
        if nspk < args.min_speakers:
            why[f"bỏ: < {args.min_speakers} nguồn / người nói"] += 1
            continue
        cands.append({"term": t, "domain_gg": g["dom"].most_common(1)[0][0], "n_ggroup": ng,
                      "n_ggroup_by_domain": " ".join(f"{d}:{c}" for d, c in g["dom"].most_common()),
                      "n_ggroup_finance": gg_fin[t], "n_vss_cs": nv, "n_speakers": nspk,
                      "hours_vss_cs": round(v["h"], 3) if v else 0.0,
                      "example_ggroup": g["ex"], "example_vss_cs": v["ex"] if v else ""})
    cands.sort(key=lambda c: (-c["n_speakers"], -(c["n_ggroup"] + c["n_vss_cs"]), c["term"]))
    by_dom = defaultdict(list)
    for c in cands:
        by_dom[c["domain_gg"]].append(c)
    chosen, k = [], 0
    while len(chosen) < args.n_max and any(by_dom.values()):  # xoay vòng theo domain
        d = NONMED[k % len(NONMED)]
        k += 1
        if by_dom.get(d):
            chosen.append(by_dom[d].pop(0))
    sel = {c["term"] for c in chosen}
    for c in cands:
        c["selected"] = int(c["term"] in sel)
    cols = ["term", "selected", "domain_gg", "n_ggroup", "n_ggroup_by_domain", "n_ggroup_finance", "n_vss_cs",
            "n_speakers", "hours_vss_cs", "example_ggroup", "example_vss_cs"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(cands, key=lambda c: (-c["selected"], -c["n_speakers"], c["term"])))

    # ---- báo trước nếu áp dụng
    def lost(rows, text_key="text_tagged"):
        n, h = 0, 0.0
        for r in rows:
            if spans(r.get(text_key), cat) & sel:
                n += 1
                h += float(r["duration"]) / 3600
        return n, h
    n_cs, h_cs = lost(vss_rows)
    tot_cs = sum(float(r["duration"]) for r in vss_rows) / 3600
    pool = (lost(jl(args.vss_pool)) if args.vss_pool else None)
    n_vm = 0
    if args.vimedcss_train:
        for r in csv.DictReader(open(args.vimedcss_train, encoding="utf-8")):
            toks = basic_tokens(r["segment_text"])
            if any(contains(toks, t) for t in sel):
                n_vm += 1
    gg_move = sum(1 for r in jl(args.ggroup) if spans(r.get("text_tagged"), cat) & sel)
    md = ["# Đề xuất reserved_b3 ngoài y khoa (CHƯA áp dụng)", "",
          f"Ứng viên đạt điều kiện: {len(cands)}; chọn **{len(chosen)}** (`selected = 1` trong `{args.out}`), theo domain "
          f"G-Group: " + ", ".join(f"{d} {sum(c['domain_gg'] == d for c in chosen)}" for d in NONMED) + ".",
          "", "Lý do bỏ: " + ", ".join(f"{k} {v}" for k, v in why.most_common()), "",
          "## Nếu áp dụng (đưa vào catalog với reserved_b3 = 1)", "",
          f"- vss_cs (`{args.vss_cs}`): mất **{n_cs} câu, {h_cs:.2f} h** / {tot_cs:.2f} h ({100 * h_cs / max(tot_cs, 1e-9):.1f}%).",
          (f"- kho vss_cs_candidates (`{args.vss_pool}`): {pool[0]} câu, {pool[1]:.2f} h chứa thuật ngữ được chọn "
           "(bị chặn ở bước lọc rẻ các lần sau)." if pool else ""),
          f"- ViMedCSS train (cs_real): {n_vm} câu chứa thuật ngữ được chọn (cũng phải loại khỏi train).",
          f"- G-Group: {gg_move} câu B1 chứa thuật ngữ được chọn -> chuyển sang B3 (thuật ngữ không có trong train).", "",
          "| Thuật ngữ | Domain | Câu G-Group | Câu vss_cs | Nguồn / người nói | Giờ vss_cs |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {c['term']} | {c['domain_gg']} | {c['n_ggroup']} | {c['n_vss_cs']} | {c['n_speakers']} | {c['hours_vss_cs']:.3f} |"
           for c in chosen]
    if args.report:
        Path(args.report).write_text("\n".join(l for l in md if l is not None) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
