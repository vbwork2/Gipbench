#!/usr/bin/env python3
"""Câu tiếng Việt chứa âm tiết GẦN ÂM với thuật ngữ tiếng Anh (tim / team, ga / gas, xốp / shop) — chống chèn bừa.

  python tools/make_confusables.py --manifest data/raw/vss_full/train_vss.jsonl data/raw/vivos/train_vivos.jsonl \
      --catalog configs/catalog_v0.tsv --out-dir data/raw/confusable

Dạng gần âm lấy từ: --pairs (configs/confusables.tsv: term, vi_forms "a|b") + cột phonetic của --catalog
(vd "mít tinh" của meeting). Chỉ quét câu KHÔNG có span <en> (nhãn coi là tiếng Việt thuần); câu nào có dạng gần âm
(chuỗi token liền, có dấu đúng) là ứng viên. Máy không biết chắc từ đó mang nghĩa tiếng Việt: phần bench phải qua
người duyệt (phiếu *_review.csv); phần train dùng làm ví dụ âm.

Tách hai file KHÔNG trùng người nói và nguồn: câu được gom thành nhóm liên thông theo speaker + source (union-find),
mỗi nhóm đi trọn vào một phía. Duyệt nhóm theo thứ tự ngẫu nhiên (--seed): nhóm vào bench khi bench chưa đủ
--bench-n câu (200–300) VÀ nhóm góp được ít nhất một câu (mỗi dạng tối đa --max-per-form-bench câu, mỗi nhóm tối đa
--max-per-group-bench câu; câu còn lại của nhóm đó bị bỏ); nhóm còn lại vào train (mỗi dạng tối đa --max-per-form-train).
--bench: manifest bench hiện có; câu trùng speaker / source / text với bench bị loại khỏi CẢ hai file.
Ghi: <out-dir>/b5_confusable.jsonl + b5_confusable_review.csv (split B5, domain confusable) và
     <out-dir>/train_confusable.jsonl (dataset confusable) + make_confusables.report.md.
Lưu ý: nếu ứng viên bench lấy từ manifest cũng dùng cho replay train, khi nhận vào bench phải loại người nói / nguồn
đó khỏi train (validate.py --train sẽ báo).
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from converters.common import REVIEW_COLS  # noqa: E402
from lib.leak import LeakGuard  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402
from lib.textnorm import Catalog, basic_tokens, plain_text  # noqa: E402


def load_forms(pairs_path, cat):
    """{dạng (tuple token): (thuật ngữ, nguồn)}"""
    forms = {}
    if pairs_path and Path(pairs_path).exists():
        with open(pairs_path, encoding="utf-8") as f:
            for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
                for v in (r.get("vi_forms") or "").split("|"):
                    t = tuple(basic_tokens(v))
                    if t:
                        forms.setdefault(t, (r["term"].strip().lower(), "pairs"))
    for k, canon in cat.lenient_map.items():  # cột phonetic của catalog = khoá chỉ có ở lenient_map
        if k not in cat.strict_map:
            forms.setdefault(k, (" ".join(canon), "catalog phonetic"))
    return forms


def find_form(toks, forms, max_n):
    for n in range(max_n, 0, -1):
        for i in range(len(toks) - n + 1):
            k = tuple(toks[i:i + n])
            if k in forms:
                return k
    return None


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, help="transcript tiếng Việt (câu có <en> bị bỏ qua)")
    ap.add_argument("--catalog")
    ap.add_argument("--pairs", default=str(G / "configs/confusables.tsv"))
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--bench-n", type=int, default=250)
    ap.add_argument("--max-per-form-bench", type=int, default=10)
    ap.add_argument("--max-per-form-train", type=int, default=50)
    ap.add_argument("--max-per-group-bench", type=int, default=5, help="số câu bench tối đa từ một nhóm người nói / nguồn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    forms = load_forms(args.pairs, cat)
    if not forms:
        sys.exit("Không có dạng gần âm nào (--pairs / cột phonetic của --catalog)")
    max_n = max(len(k) for k in forms)
    guard = LeakGuard((r for b in args.bench for r in read_jsonl(b)), cat) if args.bench else None
    cands, stat = [], Counter()
    for m in args.manifest:
        for r in read_jsonl(m):
            t = r.get("text_tagged") or r.get("text") or ""
            if "<en" in t:
                stat["bỏ: câu có span <en>"] += 1
                continue
            k = find_form(basic_tokens(plain_text(t)), forms, max_n)
            if not k:
                continue
            if guard and guard.reason(r, plain_text(t)):
                stat["bỏ: trùng bench"] += 1
                continue
            cands.append((r, k))
    uf = UF()
    for r, _ in cands:
        uf.u(("spk", r.get("speaker")), ("src", r.get("source")))
    groups = defaultdict(list)
    for r, k in cands:
        groups[uf.f(("spk", r.get("speaker")))].append((r, k))
    order = sorted(groups)
    random.Random(args.seed).shuffle(order)
    bench, train, per_b, per_t = [], [], Counter(), Counter()
    for g in order:  # cả nhóm vào một phía: bench nếu bench chưa đủ VÀ nhóm góp được ít nhất 1 câu, ngược lại train
        take = []
        if len(bench) < args.bench_n:
            seen = Counter()
            for r, k in groups[g]:
                if (len(bench) + len(take) < args.bench_n and len(take) < args.max_per_group_bench
                        and per_b[k] + seen[k] < args.max_per_form_bench):
                    seen[k] += 1
                    take.append((r, k))
        if take:
            for r, k in take:
                per_b[k] += 1
            bench += take
            continue
        for r, k in groups[g]:
            if per_t[k] < args.max_per_form_train:
                per_t[k] += 1
                train.append((r, k))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def mk(r, k, split):
        o = {kk: v for kk, v in r.items() if not kk.startswith("_")}
        o.update(id=f"cf_{r['id']}", split=split, domain="confusable", dataset=f"confusable:{r.get('dataset', '?')}",
                 audio=os.path.relpath(r["_audio_path"], out.resolve()) if r.get("_audio_path") else r.get("audio"),
                 text_tagged=plain_text(r.get("text_tagged") or r.get("text") or ""),
                 confusable={"form": " ".join(k), "term": forms[k][0], "from": forms[k][1]})
        return o
    b_rows = [mk(r, k, "B5") for r, k in bench]
    t_rows = [mk(r, k, "train") for r, k in train]
    write_jsonl(str(out / "b5_confusable.jsonl"), b_rows)
    write_jsonl(str(out / "train_confusable.jsonl"), t_rows)
    with open(out / "b5_confusable_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
        w.writeheader()
        for o in b_rows:
            w.writerow({"id": o["id"], "split": "B5", "audio": o["audio"], "duration": o["duration"],
                        "text_tagged": o["text_tagged"], "confidence": "confusable",
                        "flags": f"confusable:{o['confusable']['form']}~{o['confusable']['term']}",
                        "region": o.get("region", ""), "gender": o.get("gender", ""),
                        "note": "ok nếu từ mang nghĩa tiếng Việt; drop nếu thực ra là từ tiếng Anh"})
    spk_b = {o.get("speaker") for o in b_rows}
    src_b = {o.get("source") for o in b_rows}
    overlap = sum(1 for o in t_rows if o.get("speaker") in spk_b or o.get("source") in src_b)
    h = lambda rows: sum(float(o["duration"]) for o in rows) / 3600  # noqa: E731
    md = ["# Câu gần âm (confusable)", "",
          f"Dạng gần âm: {len(forms)} ({sum(v[1] == 'pairs' for v in forms.values())} từ --pairs, "
          f"{sum(v[1] != 'pairs' for v in forms.values())} từ cột phonetic của catalog). Ứng viên: {len(cands)} câu, "
          f"{len(groups)} nhóm người nói / nguồn.", "",
          f"- bench (B5 confusable, cần duyệt): {len(b_rows)} câu, {h(b_rows):.2f} h, {len(spk_b)} người nói",
          f"- train (ví dụ âm): {len(t_rows)} câu, {h(t_rows):.2f} h",
          f"- câu train trùng người nói / nguồn với bench: {overlap} (phải = 0)"]
    md += [f"- {k}: {v}" for k, v in stat.items()]
    md += ["", "| Dạng | Thuật ngữ | Bench | Train |", "| --- | --- | --- | --- |"]
    md += [f"| {' '.join(k)} | {forms[k][0]} | {per_b[k]} | {per_t[k]} |"
           for k in sorted(set(per_b) | set(per_t), key=lambda k: -(per_b[k] + per_t[k]))]
    (out / "make_confusables.report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8]))
    if overlap:
        sys.exit("LỖI: bench và train trùng người nói / nguồn")


if __name__ == "__main__":
    main()
