#!/usr/bin/env python3
"""Lọc dữ liệu train chống overfit (Spec mục 10): đọc nhiều manifest train, xuất MỘT manifest đã lọc + báo cáo.

  python -m gipbench.model.recipe.curate --manifest data/raw/vimedcss/train_vimedcss.jsonl data/raw/vss_full/vss_cs_clean.jsonl \
      data/raw/vivos/train_vivos.jsonl --catalog configs/catalog_v0.tsv --out data/train_curated.jsonl

"Câu CS" = câu có ít nhất một span <en>; các phần (giờ CS) dưới đây tính trên câu CS. Câu không phải CS chỉ chịu
luật --holdout-domain và bước 0. Bước 0 luôn chạy trên MỌI câu:
  0. chặn dữ liệu chỉ được test: loại câu license / source / dataset test_only (lib.leak.is_test_only, KHÔNG tắt
     được) và câu có text (chuẩn hoá) trùng danh sách chặn --block-text (mặc định data/raw/consensus/leak_block.txt
     do src/gipbench/dataset/ingest/consensus.py ghi; truyền --block-text không kèm file để tắt riêng phần text)
  0b. dedupe audio (mặc định BẬT; --no-dedupe-audio để tắt): hai câu cùng audio (audio_sha256 nếu có, không thì đường
     dẫn audio tuyệt đối đã chuẩn hoá; thêm mã clip gốc CV / VIVOS) chỉ giữ một. Câu giữ lại thuộc dataset đứng trước
     trong --dedupe-priority (khớp tiền tố trước ':' của dataset); dataset không có trong danh sách xếp sau, theo thứ tự
     --manifest. Báo cáo số câu trùng theo cặp (dataset giữ <- dataset bỏ). Lý do (audit R14): 590/590 câu confusable
     cũng có trong vivos_train, mỗi audio được thấy gấp đôi.
Mỗi luật sau tắt được độc lập (0 / bỏ trống) và áp theo thứ tự:
  1. --holdout-domain D        loại MỌI câu thuộc domain D (domain CS hoàn toàn ngoài train)
  2. --agreement-hyps A B      câu CS có nhãn máy (label_source bắt đầu bằng pseudo/machine, hoặc dataset thuộc
                               --agreement-datasets) chỉ được giữ khi mọi span <en> có trong >= --min-agree hyp
                               (lib.agreement, so sau normalize_hyp); không có hyp = không xác nhận = bỏ
  3. --max-per-term 50         mỗi thuật ngữ catalog có trong tối đa N câu CS. Khi phải cắt, chọn xoay vòng theo
                               người nói và ưu tiên ngữ cảnh (từ trước + từ sau thuật ngữ) chưa gặp; câu có nhiều
                               thuật ngữ vượt trần chỉ được giữ khi được MỌI thuật ngữ đó chọn
  4. --max-speaker-share 0.02  một người nói tối đa 2% số giờ CS (lấy mẫu xuống, --seed)
  5. --max-domain-share 0.35   một domain tối đa 35% số giờ CS (lấy mẫu xuống; ViMedCSS = medical sẽ bị cắt mạnh)
  6. --max-synthetic-share 0.5 câu synthetic: true (TTS / ghép) tối đa 50% số giờ CS
  --min-speakers-per-term 3    chỉ BÁO CÁO thuật ngữ có ít hơn N người nói (không loại)
Báo cáo <out>.report.md: giờ trước/sau theo domain, nguồn (dataset) và người nói; số câu/giờ bị từng luật loại;
thuật ngữ bị cắt; thuật ngữ thiếu người nói.
"""
import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.agreement_rules import agree
from gipbench.core.decontam import clip_key
from gipbench.core.sampling import pick_diverse
from gipbench.infer.cache import HashIndex, HypLookup
from gipbench.core.leak import add_block_arg, is_test_only, load_block_texts, norm_text
from gipbench.core.manifest import read_jsonl, write_jsonl
from gipbench.core.textnorm import Catalog, basic_tokens, plain_text


def is_cs(r):
    return "<en" in (r.get("text_tagged") or "")


def hours(rows):
    return sum(float(r["duration"]) for r in rows) / 3600


def dataset(r):
    return str(r.get("dataset") or "?").split(":")[0]


def term_hits(r, cat):
    """[(thuật ngữ, từ trước, từ sau)] — thuật ngữ catalog trong câu và ngữ cảnh lần xuất hiện đầu."""
    toks = cat.apply(basic_tokens(plain_text(r.get("text_tagged") or r.get("text") or "")))
    out, seen = [], set()
    for t in cat.find_terms(toks):
        if t in seen:
            continue
        seen.add(t)
        n = len(t)
        pos = next((i for i in range(len(toks) - n + 1) if tuple(toks[i:i + n]) == tuple(t)), 0)
        out.append((" ".join(t), toks[pos - 1] if pos > 0 else "<s>", toks[pos + n] if pos + n < len(toks) else "</s>"))
    return out


def audio_keys(r):
    """Các khoá định danh audio của một câu: sha256, đường dẫn chuẩn hoá, mã clip gốc (None nếu không có)."""
    keys = []
    if r.get("audio_sha256"):
        keys.append("sha:" + r["audio_sha256"])
    p = r.get("_audio_path") or r.get("audio")
    if p:
        keys.append("path:" + os.path.normcase(os.path.normpath(os.path.abspath(str(p)))))
    ck = clip_key(r)
    if ck:
        keys.append(ck)
    return keys


def dedupe_audio(rows, priority=()):
    """Giữ một câu cho mỗi audio. Thứ tự ưu tiên: dataset theo `priority`, rồi thứ tự xuất hiện.
    Trả (rows giữ theo thứ tự cũ, Counter {(dataset giữ, dataset bỏ): số câu})."""
    rank = {d: i for i, d in enumerate(priority)}
    order = sorted(range(len(rows)), key=lambda i: (rank.get(dataset(rows[i]), len(rank)), i))
    owner, drop, pairs = {}, set(), Counter()
    for i in order:
        ks = audio_keys(rows[i])
        hit = next((owner[k] for k in ks if k in owner), None)
        if hit is not None:
            drop.add(i)
            pairs[(dataset(rows[hit]), dataset(rows[i]))] += 1
            continue
        for k in ks:
            owner[k] = i
    return [r for i, r in enumerate(rows) if i not in drop], pairs


def downsample(rows, key, cap_share, rnd, exempt=()):
    """Giữ mỗi nhóm key(r) <= cap_share * tổng giờ (tổng tính SAU khi cắt: giải đúng cho tập nhóm vượt trần).
    Nhóm trong `exempt` (vd domain other = chưa biết domain) không bị cắt nhưng vẫn tính vào tổng.
    Trả về (rows giữ, {nhóm: (giờ trước, giờ sau)}, cảnh báo)."""
    h = Counter()
    for r in rows:
        h[key(r)] += float(r["duration"]) / 3600
    total = sum(h.values())
    over = {g for g, v in h.items() if v > cap_share * total + 1e-12 and g not in exempt}
    warn = ""
    for _ in range(len(h) + 1):  # tìm tập nhóm vượt trần ổn định
        if not over:
            break
        if cap_share * len(over) >= 1:
            warn = f"không thể đưa mọi nhóm về <= {cap_share:.0%}: chỉ có {len(h)} nhóm"
            over = set()
            break
        others = sum(v for g, v in h.items() if g not in over)
        x = cap_share * others / (1 - cap_share * len(over))
        new_over = {g for g, v in h.items() if v > x + 1e-12 and g not in exempt}
        if new_over == over:
            break
        over = new_over
    if not over:
        return rows, {}, warn
    others = sum(v for g, v in h.items() if g not in over)
    x = cap_share * others / (1 - cap_share * len(over))
    acc, drop = Counter(), set()
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    for i in idx:
        g = key(rows[i])
        if g in over:
            d = float(rows[i]["duration"]) / 3600
            if acc[g] + d <= x + 1e-9:
                acc[g] += d
            else:
                drop.add(i)
    return [r for i, r in enumerate(rows) if i not in drop], {g: (h[g], acc[g]) for g in over}, warn


def cap_synthetic(rows, share, rnd):
    """Chỉ cắt câu synthetic: giờ synthetic <= share * tổng mới  <=>  <= share * giờ thật / (1 - share)."""
    real = sum(float(r["duration"]) for r in rows if not r.get("synthetic")) / 3600
    syn = [i for i, r in enumerate(rows) if r.get("synthetic")]
    h_syn = sum(float(rows[i]["duration"]) for i in syn) / 3600
    x = share * real / (1 - share) if share < 1 else float("inf")
    if h_syn <= x + 1e-12:
        return rows, {}, ""
    rnd.shuffle(syn)
    acc, drop = 0.0, set()
    for i in syn:
        d = float(rows[i]["duration"]) / 3600
        if acc + d <= x + 1e-9:
            acc += d
        else:
            drop.add(i)
    return [r for i, r in enumerate(rows) if i not in drop], {"synthetic": (h_syn, acc)}, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--allow-bench-catalog", action="store_true",
                    help="cho phép catalog có span lấy từ chính bench (from_bench): chỉ để chẩn đoán oracle, kết quả không xếp hạng")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-domain", nargs="*", default=[])
    ap.add_argument("--agreement-hyps", nargs="*", default=[], help="file / thư mục hyps của các ASR khác")
    ap.add_argument("--agreement-datasets", nargs="*", default=["sea_speech"],
                    help="dataset coi là nhãn máy (ngoài label_source pseudo/machine)")
    ap.add_argument("--min-agree", type=int, default=1)
    ap.add_argument("--hash-index", help="tra hyps theo id + hash audio (src/gipbench/infer/cache.py)")
    ap.add_argument("--audio-root")
    ap.add_argument("--max-per-term", type=int, default=50, help="0 = tắt")
    ap.add_argument("--min-speakers-per-term", type=int, default=3, help="chỉ báo cáo")
    ap.add_argument("--max-speaker-share", type=float, default=0.02, help="0 = tắt")
    ap.add_argument("--max-domain-share", type=float, default=0.35, help="0 = tắt")
    ap.add_argument("--cap-exempt", nargs="*", default=[],
                    help="domain không bị cắt bởi --max-domain-share (vd other = chưa biết domain, như src/gipbench/model/recipe/mix_plan.py)")
    ap.add_argument("--max-synthetic-share", type=float, default=0.5, help="0 = tắt")
    ap.add_argument("--no-dedupe-audio", action="store_true", help="tắt bước 0b (dedupe audio)")
    ap.add_argument("--dedupe-priority", nargs="*", default=[],
                    help="dataset (tiền tố trước ':') ưu tiên giữ khi trùng audio, vd vivos commonvoice ViMedCSS")
    add_block_arg(ap)
    args = ap.parse_args()
    if args.catalog:  # --max-per-term đếm thuật ngữ theo catalog: catalog có span bench làm lệch chọn dữ liệu (audit R05)
        try:
            R.forbid_bench_catalog(args.catalog, "curate", allow=args.allow_bench_catalog)
        except R.ReleaseError as e:
            print("LỖI", e)
            sys.exit(3)
    rnd = random.Random(args.seed)
    cat = Catalog.load(args.catalog)
    rows = [r for m in args.manifest for r in read_jsonl(m)]
    before = rows[:]
    steps = []

    def step(name):
        cs = [r for r in rows if is_cs(r)]
        steps.append((name, len(rows), hours(rows), len(cs), hours(cs)))

    def apply_cs(keep_cs):
        ids = {id(r) for r in keep_cs}
        return [r for r in rows if not is_cs(r) or id(r) in ids]

    step("đầu vào")
    notes = []
    block = load_block_texts(args.block_text, cat)
    why0 = Counter()
    kept0 = []
    for r in rows:
        w = is_test_only(r)
        if not w and block and norm_text(r.get("text_tagged") or r.get("text"), cat) in block:
            w = "text trùng danh sách chặn"
        if w:
            why0[w.split(" (")[0]] += 1
            continue
        kept0.append(r)
    rows = kept0
    step("chặn dữ liệu test_only")
    notes.append(f"chặn dữ liệu test_only (danh sách chặn {len(block)} câu): "
                 + (", ".join(f"{k}: {v} câu" for k, v in why0.most_common()) if why0 else "không câu nào bị loại"))
    if not args.no_dedupe_audio:
        rows, pairs = dedupe_audio(rows, args.dedupe_priority)
        step("dedupe audio")
        notes.append("dedupe audio (giữ <- bỏ): " + (", ".join(f"{a} <- {b}: {n} câu" for (a, b), n in pairs.most_common())
                                                      if pairs else "không có audio trùng")
                     + (f"; ưu tiên {' > '.join(args.dedupe_priority)}" if args.dedupe_priority else "; ưu tiên theo thứ tự --manifest"))
    if args.holdout_domain:
        rows = [r for r in rows if (r.get("domain") or "other") not in set(args.holdout_domain)]
        step(f"--holdout-domain {' '.join(args.holdout_domain)}")
    if args.agreement_hyps:
        look = HypLookup(args.agreement_hyps, HashIndex(args.hash_index) if args.hash_index else None, args.audio_root)
        mset = set(args.agreement_datasets)

        def machine(r):
            return str(r.get("label_source") or "").startswith(("pseudo", "machine")) or dataset(r) in mset
        kept, n_bad = [], 0
        for r in rows:
            if is_cs(r) and machine(r):
                if not agree(r["text_tagged"], [h or "" for h in look.get(r)], cat, args.min_agree)["keep"]:
                    n_bad += 1
                    continue
            kept.append(r)
        rows = kept
        step(f"--agreement-hyps (min-agree {args.min_agree})")
        notes.append(f"--agreement-hyps: bỏ {n_bad} câu CS nhãn máy có span không được xác nhận.")
    terms_cut = {}
    if args.max_per_term:
        hits = {i: term_hits(r, cat) for i, r in enumerate(rows) if is_cs(r)}
        by_term = defaultdict(list)
        for i, hs in hits.items():
            for t, a, b in hs:
                by_term[t].append((i, rows[i].get("speaker"), (a, b)))
        over = {t: c for t, c in by_term.items() if len(c) > args.max_per_term}
        drop = set()
        for t, c in sorted(over.items()):
            chosen = pick_diverse(c, args.max_per_term, rnd)
            drop |= {i for i, _, _ in c if i not in chosen}
        after = Counter(t for i, hs in hits.items() if i not in drop for t, _, _ in hs)
        terms_cut = {t: (len(c), after[t]) for t, c in over.items()}
        rows = [r for i, r in enumerate(rows) if i not in drop]
        step(f"--max-per-term {args.max_per_term}")
    for flag, share, key, what in (
            ("--max-speaker-share", args.max_speaker_share, lambda r: r.get("speaker") or "?", "người nói"),
            ("--max-domain-share", args.max_domain_share, lambda r: r.get("domain") or "other", "domain"),
            ("--max-synthetic-share", args.max_synthetic_share,
             lambda r: "synthetic" if r.get("synthetic") else "thật", "synthetic")):
        if not share:
            continue
        cs = [r for r in rows if is_cs(r)]
        if what == "synthetic":
            keep_cs, cut, w = cap_synthetic(cs, share, rnd)
        else:
            keep_cs, cut, w = downsample(cs, key, share, rnd, args.cap_exempt if what == "domain" else ())
        rows = apply_cs(keep_cs)
        step(f"{flag} {share:g}")
        shown = list(cut.items())[:10]
        notes.append(f"{flag}: " + (", ".join(f"{g} {a:.2f} -> {b:.2f} h" for g, (a, b) in shown)
                                    + (f" … (+{len(cut) - 10} {what})" if len(cut) > 10 else "")
                                    if cut else f"không {what} nào vượt trần") + (f"; CẢNH BÁO: {w}" if w else ""))
    write_jsonl(args.out, rows)

    spk_terms = defaultdict(set)
    for r in rows:
        if is_cs(r):
            for t, _, _ in term_hits(r, cat):
                spk_terms[t].add(r.get("speaker"))
    few = sorted((t, len(s)) for t, s in spk_terms.items() if len(s) < args.min_speakers_per_term)

    def table(title, key, top=20):
        hb, ha = Counter(), Counter()
        for src, acc in ((before, hb), (rows, ha)):
            for r in src:
                if is_cs(r):
                    acc[key(r)] += float(r["duration"]) / 3600
        t = ["", f"### Giờ CS theo {title} ({len(hb)} -> {len(ha)})", "", f"| {title} | Trước | Sau |", "| --- | --- | --- |"]
        return t + [f"| {k} | {v:.2f} | {ha[k]:.2f} |" for k, v in hb.most_common(top)]

    md = [f"# Lọc train — {len(args.manifest)} manifest -> `{args.out}`", "", f"Seed {args.seed}.", "",
          "| Bước | Câu | Giờ | Câu CS | Giờ CS |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {n} | {a} | {b:.2f} | {c} | {d:.2f} |" for n, a, b, c, d in steps]
    md += [""] + [f"- {x}" for x in notes]
    md += table("domain", lambda r: r.get("domain") or "other") + table("nguồn", dataset)
    md += table("người nói", lambda r: r.get("speaker") or "?")
    md += ["", f"### Thuật ngữ bị cắt (--max-per-term {args.max_per_term}): {len(terms_cut)}", "",
           "| Thuật ngữ | Câu trước | Câu sau |", "| --- | --- | --- |"]
    md += [f"| {t} | {a} | {b} |" for t, (a, b) in sorted(terms_cut.items(), key=lambda x: -x[1][0])[:50]]
    md += ["", f"### Thuật ngữ có ít hơn {args.min_speakers_per_term} người nói: {len(few)}", "",
           ", ".join(f"{t} ({n})" for t, n in few[:200]) or "không có"]
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for n, a, b, c, d in steps:
        print(f"  {n}: {a} câu ({b:.2f} h), CS {c} câu ({d:.2f} h)")
    print(f"Đã ghi {args.out} và {args.out}.report.md; {len(terms_cut)} thuật ngữ bị cắt, "
          f"{len(few)} thuật ngữ thiếu người nói")


if __name__ == "__main__":
    main()
