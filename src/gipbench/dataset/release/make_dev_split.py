#!/usr/bin/env python3
"""Tách bench thành dev (chọn checkpoint) / test (báo cáo) theo CỤM NGƯỜI NÓI, không để đơn vị nào ở cả hai phía.

  python -m gipbench.dataset.release.make_dev_split --bench data/bench_v0_draft/bench_v0_draft.jsonl --splits B1 --dev-frac 0.3 --seed 0 \
      --extra-dev data/raw/vimedcss_full/dev_vimedcss.jsonl data/e3_v1/dev_cs_vss.jsonl \
      --train data/e3_v1/train_curated.jsonl \
      --out-dev data/e3_v1/dev_bench.jsonl --out-test data/bench_v0_draft/bench_test.jsonl
  # report: <out-dev>.report.md; thoát mã 1 nếu dev-test trùng đơn vị / speaker / source, hoặc dev-train trùng speaker / source

Vì sao: dev cũ (data/e3_v1/dev_cs.jsonl) trùng speaker (= video) với train, 87% y khoa và chỉ ~45 thuật ngữ chung với bench,
nên chọn checkpoint theo nó thưởng cho việc học thuộc y khoa. Dev mới = một phần bench (cụm người nói riêng) + dev ngoài
(--extra-dev) đã loại speaker / source có trong train và trong test.

Quy tắc
  - Đơn vị tách = speaker_cluster nếu có, không thì speaker; các đơn vị cùng có dòng thuộc một source được gộp làm một
    (để dev-test cũng không chung source). Chỉ các dòng thuộc --splits (mặc định B1) được tách; dòng split
    khác luôn vào test, và đơn vị nào có dòng ở split khác thì cũng vào test (để không có đơn vị ở hai phía).
  - Phân tầng theo domain (domain của đơn vị = domain nhiều giờ nhất). Mỗi domain: xếp đơn vị theo hash ổn định sha256(seed, đơn vị),
    đi từ đầu danh sách, nhận vào dev cho tới khi số giờ dev gần mục tiêu dev-frac nhất. Hash chỉ phụ thuộc (seed, đơn vị) nên chạy
    lại ra đúng kết quả cũ; thêm câu vào đơn vị cũ không đổi thứ tự, thêm đơn vị mới chỉ xê dịch ranh giới trong domain đó.
  - --extra-dev: chỉ nhận dòng mà speaker và source KHÔNG có trong --train và không có trong phần test. --max-domain-share
    giới hạn tỉ trọng giờ của một domain trong dev bằng cách cắt bớt NGẪU NHIÊN (có seed) dòng của --extra-dev; dòng bench giữ nguyên.
Ra: dev có split = "dev", split gốc ở bench_split, dev_origin = "bench" hoặc "extra:<tên file>". Test giữ nguyên mọi trường.
"""
import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.manifest import read_jsonl, write_jsonl
from gipbench.core.textnorm import Catalog, normalize_ref


def unit_of(r):
    """Đơn vị tách: speaker_cluster nếu có, không thì speaker."""
    return str(r.get("speaker_cluster") or r.get("speaker") or r.get("id"))


def stable_hash(seed, key):
    """Hash ổn định (không phụ thuộc PYTHONHASHSEED) của (seed, key)."""
    return int(hashlib.sha256(f"{seed}\x1f{key}".encode("utf-8")).hexdigest()[:16], 16)


def domain_of(r):
    return r.get("domain") or "unk"


def hours_of(r):
    return float(r.get("duration") or 0.0) / 3600


def merge_by_source(rows):
    """Gộp các đơn vị cùng có dòng thuộc một source (union-find) -> {đơn vị: đại diện của nhóm}.

    Để một source không bị chia ra hai phía dù nó nối hai đơn vị khác nhau (vd cùng video, hai cụm người nói).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for r in rows:
        u = unit_of(r)
        find(u)
        src = r.get("source")
        if src in first:
            a, b = find(first[src]), find(u)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[src] = u
    return {u: find(u) for u in list(parent)}


def split_units(rows, dev_frac, seed, forced_test=frozenset(), key=unit_of):
    """Chọn tập đơn vị (theo key(dòng)) vào dev. rows = dòng được phép tách. Trả về set đơn vị dev.

    Mỗi domain độc lập: đơn vị xếp theo hash, nhận từ đầu danh sách tới khi giờ dev gần dev_frac * giờ domain nhất.
    Domain có >= 2 đơn vị luôn để lại ít nhất 1 đơn vị cho dev và 1 cho test; domain 1 đơn vị -> test.
    """
    by_unit = defaultdict(lambda: defaultdict(float))
    for r in rows:
        by_unit[key(r)][domain_of(r)] += hours_of(r)
    by_dom = defaultdict(list)
    for u, dh in by_unit.items():
        if u in forced_test:
            continue
        by_dom[max(sorted(dh), key=lambda d: dh[d])].append(u)
    dev = set()
    for d, us in by_dom.items():
        us.sort(key=lambda u: (stable_hash(seed, u), u))
        hs = [sum(by_unit[u].values()) for u in us]
        target, cum, chosen = dev_frac * sum(hs), 0.0, []
        for u, h in zip(us, hs):
            if abs(cum + h - target) < abs(cum - target):
                chosen.append(u)
                cum += h
            else:
                break
        if len(us) >= 2:
            if not chosen:
                chosen = [us[0]]
            if len(chosen) == len(us):
                chosen = chosen[:-1]
        elif len(us) == 1:
            chosen = []
        dev.update(chosen)
    return dev


def filter_extra(extras, train_spk, train_src, test_spk, test_src, test_units, seen_ids):
    """Loại dòng extra-dev trùng train / test. extras = [(tên nguồn, dòng)]. Trả về (giữ lại, Counter lý do loại)."""
    keep, why = [], Counter()
    for name, r in extras:
        if r["id"] in seen_ids:
            why["id trùng"] += 1
        elif r.get("speaker") in train_spk:
            why["speaker có trong train"] += 1
        elif r.get("source") in train_src:
            why["source có trong train"] += 1
        elif r.get("speaker") in test_spk or unit_of(r) in test_units:
            why["speaker / đơn vị có trong test"] += 1
        elif r.get("source") in test_src:
            why["source có trong test"] += 1
        else:
            seen_ids.add(r["id"])
            keep.append((name, r))
    return keep, why


def cap_domains(fixed, extras, share, seed):
    """Giới hạn tỉ trọng giờ của mỗi domain trong dev <= share bằng cách cắt dòng extras (cắt ngẫu nhiên, có seed).

    fixed = dòng dev lấy từ bench (không cắt); extras = [(tên, dòng)]. Trả về (extras còn lại, log, cảnh báo).
    Không khả thi khi số domain * share < 1 (không thể chia đều dưới trần): bỏ qua và cảnh báo.
    """
    log, warn = [], []
    cur = list(extras)
    doms = {domain_of(r) for r in fixed} | {domain_of(r) for _, r in cur}
    if share >= 1 or not doms:
        return cur, log, warn
    if len(doms) * share < 1 - 1e-9:
        return cur, log, [f"{len(doms)} domain x trần {share} < 1: không thể đạt trần, bỏ qua"]
    for _ in range(50):
        hrs = Counter()
        for r in fixed:
            hrs[domain_of(r)] += hours_of(r)
        for _, r in cur:
            hrs[domain_of(r)] += hours_of(r)
        total = sum(hrs.values())
        over = [d for d in hrs if hrs[d] > share * total + 1e-9]
        if not over:
            break
        d = max(over, key=lambda x: hrs[x])
        cap = share / (1 - share) * (total - hrs[d])
        budget = cap - sum(hours_of(r) for r in fixed if domain_of(r) == d)
        mine = sorted((x for x in cur if domain_of(x[1]) == d), key=lambda x: (stable_hash(seed, x[1]["id"]), x[1]["id"]))
        if budget <= 0 and not mine:
            warn.append(f"domain {d}: riêng phần bench đã vượt trần {share}, không có dòng extra để cắt")
            break
        keep, cum = [], 0.0
        for x in mine:
            h = hours_of(x[1])
            if cum + h > budget:
                break
            keep.append(x)
            cum += h
        log.append(f"domain {d}: giữ {len(keep)}/{len(mine)} dòng extra ({cum:.3f} h) để tỉ trọng giờ <= {share}")
        kept_ids = {x[1]["id"] for x in keep}
        cur = [x for x in cur if domain_of(x[1]) != d or x[1]["id"] in kept_ids]
        if budget <= 0:
            warn.append(f"domain {d}: riêng phần bench đã vượt trần {share}; đã bỏ hết dòng extra của domain này")
    return cur, log, warn


def rel_audio(r, out_dir):
    """Đường dẫn audio của dòng, viết lại tương đối theo thư mục file ra (đường tuyệt đối kiểu /mnt/... hoặc C:\\ giữ nguyên)."""
    a = str(r.get("audio") or "")
    if not a or a.startswith("/") or re.match(r"[A-Za-z]:[\\/]", a) or "_audio_path" not in r:
        return a
    try:
        return os.path.relpath(r["_audio_path"], out_dir).replace("\\", "/")
    except ValueError:  # khác ổ đĩa
        return str(Path(r["_audio_path"]).resolve()).replace("\\", "/")


def side_stats(rows, cat):
    """-> {domain: {n, h, units, spans, terms(set)}} (+ khoá '__all__')."""
    out = defaultdict(lambda: {"n": 0, "h": 0.0, "units": set(), "spans": 0, "terms": set()})
    for r in rows:
        _, sp = normalize_ref(r.get("text_tagged") or "", cat)
        for k in (domain_of(r), "__all__"):
            s = out[k]
            s["n"] += 1
            s["h"] += hours_of(r)
            s["units"].add(unit_of(r))
            s["spans"] += len(sp)
            s["terms"].update(" ".join(t) for t in sp)
    return out


def write_report(path, args, sides, notes, checks, shared):
    names = list(sides)
    doms = sorted({d for s in sides.values() for d in s if d != "__all__"})
    L = ["# Tách dev / test theo cụm người nói", "",
         f"- bench: `{args.bench}` (split tách: {', '.join(args.splits)}), dev-frac {args.dev_frac}, seed {args.seed}",
         f"- dev: `{args.out_dev}`; test: `{args.out_test}`", ""]
    L += ["## Tổng theo phía", "", "| phía | câu | giờ | đơn vị | span | thuật ngữ khác nhau |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        s = sides[n].get("__all__")
        if s:
            L.append(f"| {n} | {s['n']} | {s['h']:.2f} | {len(s['units'])} | {s['spans']} | {len(s['terms'])} |")
    L += ["", "## Theo domain", "", "| phía | domain | câu | giờ | đơn vị | span | thuật ngữ khác nhau |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        for d in doms:
            s = sides[n].get(d)
            if s:
                L.append(f"| {n} | {d} | {s['n']} | {s['h']:.2f} | {len(s['units'])} | {s['spans']} | {len(s['terms'])} |")
    L += ["", "## Thuật ngữ chung dev - test", ""]
    for name, k, tot in shared:
        L.append(f"- {name}: {k} / {tot} thuật ngữ dev cũng có trong test")
    if notes:
        L += ["", "## Ghi chú"] + [""] + [f"- {x}" for x in notes]
    L += ["", "## Kiểm tra", ""] + [f"- {'ĐẠT' if ok else 'VI PHẠM'}: {msg}" for ok, msg in checks]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--splits", nargs="+", default=["B1"], help="split của bench được tách dev / test (mặc định B1)")
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-dev", nargs="*", default=[], help="manifest dev ngoài (vd ViMedCSS validation, vss_cs held-out)")
    ap.add_argument("--train", nargs="*", default=[], help="manifest train: loại dòng extra-dev trùng speaker / source")
    ap.add_argument("--max-domain-share", type=float, default=0.4, help="trần tỉ trọng giờ của một domain trong dev")
    ap.add_argument("--catalog", help="catalog để gộp biến thể khi đếm thuật ngữ (tuỳ chọn)")
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--out-test", required=True)
    args = ap.parse_args(argv)
    if not 0 < args.dev_frac < 1:
        sys.exit("--dev-frac phải nằm trong (0, 1)")
    cat = Catalog.load(args.catalog) if args.catalog else Catalog()

    bench = read_jsonl(args.bench)
    sel = [r for r in bench if r.get("split") in set(args.splits)]
    if not sel:
        sys.exit(f"bench không có dòng nào thuộc split {args.splits}")
    comp = merge_by_source(bench)  # đơn vị chung source gộp làm một: dev-test không bao giờ chung source
    key = lambda r: comp[unit_of(r)]  # noqa: E731
    forced = {key(r) for r in bench if r.get("split") not in set(args.splits)}
    dev_units = split_units(sel, args.dev_frac, args.seed, forced, key)
    bench_dev = [r for r in sel if key(r) in dev_units]
    dev_ids = {r["id"] for r in bench_dev}
    test = [r for r in bench if r["id"] not in dev_ids]

    train_spk, train_src = set(), set()
    for p in args.train:
        for r in read_jsonl(p):
            train_spk.add(r.get("speaker"))
            train_src.add(r.get("source"))
    extras = [(Path(p).stem, r) for p in args.extra_dev for r in read_jsonl(p)]
    n_extra_in = len(extras)
    seen = {r["id"] for r in bench}
    extras, why = filter_extra(extras, train_spk, train_src, {r.get("speaker") for r in test},
                               {r.get("source") for r in test}, {unit_of(r) for r in test}, seen)
    extras, cap_log, cap_warn = cap_domains(bench_dev, extras, args.max_domain_share, args.seed)

    out_dev_dir = str(Path(args.out_dev).resolve().parent)
    out_test_dir = str(Path(args.out_test).resolve().parent)
    dev_rows = []
    for r in bench_dev:
        o = dict(r)
        o.update(split="dev", bench_split=r["split"], dev_origin="bench", audio=rel_audio(r, out_dev_dir))
        dev_rows.append(o)
    for name, r in sorted(extras, key=lambda x: x[1]["id"]):
        o = dict(r)
        o.update(split="dev", bench_split=r.get("split"), dev_origin=f"extra:{name}", audio=rel_audio(r, out_dev_dir))
        dev_rows.append(o)
    test_rows = [dict(r, audio=rel_audio(r, out_test_dir)) for r in test]
    write_jsonl(args.out_dev, dev_rows)
    write_jsonl(args.out_test, test_rows)

    checks = []
    du, tu = {unit_of(r) for r in dev_rows}, {unit_of(r) for r in test_rows}
    checks.append((not (du & tu), f"đơn vị trùng dev-test: {len(du & tu)} (cần 0)"))
    ds, ts = {r.get("speaker") for r in dev_rows}, {r.get("speaker") for r in test_rows}
    checks.append((not (ds & ts), f"speaker trùng dev-test: {len(ds & ts)} (cần 0)"))
    dsrc, tsrc = {r.get("source") for r in dev_rows}, {r.get("source") for r in test_rows}
    checks.append((not (dsrc & tsrc), f"source trùng dev-test: {len(dsrc & tsrc)} (cần 0)"))
    if args.train:
        a, b = ds & train_spk, dsrc & train_src
        checks.append((not a, f"speaker trùng dev-train: {len(a)} (cần 0)"))
        checks.append((not b, f"source trùng dev-train: {len(b)} (cần 0)"))
    else:
        checks.append((True, "không có --train: bỏ qua kiểm tra dev-train"))

    sides = {"test": side_stats(test_rows, cat), "dev (bench)": side_stats([r for r in dev_rows if r["dev_origin"] == "bench"], cat),
             "dev (extra)": side_stats([r for r in dev_rows if r["dev_origin"] != "bench"], cat),
             "dev (tổng)": side_stats(dev_rows, cat)}
    t_terms = sides["test"].get("__all__", {"terms": set()})["terms"]
    shared = []
    for name in ("dev (bench)", "dev (extra)", "dev (tổng)"):
        dt = sides[name].get("__all__", {"terms": set()})["terms"]
        shared.append((name, len(dt & t_terms), len(dt)))
    n_forced = sum(1 for r in sel if key(r) in forced)
    notes = [f"{len(sel)} dòng bench thuộc split tách -> {len(bench_dev)} dòng dev, {len(sel) - len(bench_dev)} dòng test; "
             f"{len(forced)} đơn vị có dòng ở split khác (luôn vào test), trong đó {n_forced}/{len(sel)} dòng của split tách "
             f"bị ép vào test vì đơn vị của chúng còn có dòng ở split khác (muốn tách luôn các split đó thì thêm vào --splits)",
             f"extra-dev: nhận {len(extras)}/{n_extra_in} dòng; loại: "
             + (", ".join(f"{k} = {v}" for k, v in why.most_common()) or "không")] + cap_log + [f"CẢNH BÁO: {w}" for w in cap_warn]
    if not args.train:
        notes.append("CẢNH BÁO: không truyền --train nên extra-dev chưa được lọc theo train")
    rep = Path(str(args.out_dev) + ".report.md")
    write_report(rep, args, sides, notes, checks, shared)
    print(f"dev: {len(dev_rows)} câu ({len(bench_dev)} bench + {len(extras)} extra) -> {args.out_dev}")
    print(f"test: {len(test_rows)} câu -> {args.out_test}; report -> {rep}")
    for ok, msg in checks:
        print(("ĐẠT   " if ok else "LỖI   ") + msg)
    return 0 if all(ok for ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(run())
