#!/usr/bin/env python3
"""Bước 1 của chuỗi TTS: sinh KỊCH BẢN (văn bản) cho dữ liệu code-switching tổng hợp.

  python -m gipbench.dataset.synth.make_text --catalog data/releases/catalog/train-v1/catalog.tsv \
      --reserved-terms configs/dataset/reserved_terms_v1.txt \
      --bench data/bench_v0_draft/bench_v0_draft.jsonl --out-dir data/tts/script \
      [--terms extra_terms.tsv] [--sentences cau_viet_san.jsonl] [--contexts-per-term 5] [--readings-per-term 2] \
      [--holdout-frac 0.3] [--seed 0] [--require-lexicon]

Mỗi câu có HAI văn bản: `text_tagged` (nhãn ASR: chính tả Anh trong <en>…</en>, số viết dạng đọc, cùng quy ước với bench)
và `tts_text` (thứ đưa cho máy đọc: phần tiếng Anh thay bằng CÁCH ĐỌC lấy từ configs/dataset/synth/tts_lexicon.tsv, vì TTS tiếng Việt
đọc sai chính tả Anh, vd `meeting` -> "mê e tinh"). Một thuật ngữ có nhiều cách đọc (Việt hoá mạnh / trung gian / gần bản
ngữ; viết tắt kiểu Anh hay kiểu Việt), mỗi câu dùng một cách, xoay vòng; mức đọc ghi vào en_meta.viet_level ở bước synth.
Thuật ngữ KHÔNG có trong lexicon: sinh cách đọc dự phòng bằng luật đơn giản (viết tắt toàn phụ âm -> đánh vần chữ cái Anh
và Việt; còn lại đưa nguyên chính tả) và gắn cờ `reading_fallback`; --require-lexicon bỏ hẳn các thuật ngữ đó.

Chống rò rỉ / giữ đối chứng:
  - --catalog có span lấy từ chính bench (release flags.from_bench, vd catalog/v0-draft) bị TỪ CHỐI (thoát mã 1; --allow-bench-catalog
    chỉ để chẩn đoán oracle): lexicon E3 v2 dựng từ catalog đó có 65 / 107 thuật ngữ là thuật ngữ C1 (audit R05);
  - --reserved-terms (BẮT BUỘC, không rỗng) = thuật ngữ giữ riêng của dự án (configs/dataset/reserved_terms_v1.txt): coi như
    reserved_b3, không bao giờ có trong kịch bản, kể cả trong câu mẫu / câu --sentences; kiểm lại ở cuối, lọt thì thoát mã 1;
  - thuật ngữ reserved_b3 của catalog KHÔNG BAO GIỜ có trong kịch bản;
  - một phần thuật ngữ (--holdout-frac, chọn theo hash ỔN ĐỊNH của chính tả thuật ngữ + --holdout-salt, không phụ thuộc --seed)
    không bao giờ được tổng hợp: ghi ra tts_holdout_terms.txt; bench còn lát "chưa thấy" để đo TTS có giúp thật không;
  - câu trùng text (sau chuẩn hoá + catalog) với bench bị loại và được thay bằng mẫu khác.
Vào thêm: --sentences FILE = JSONL {"text": "...", "terms": ["meeting"], "domain": "tech"} do người / LLM viết sẵn (công cụ
KHÔNG gọi LLM). Câu này được ưu tiên hơn mẫu; công cụ tự gắn <en> vào các thuật ngữ khai báo và KIỂM: đủ thuật ngữ khai báo,
không có thuật ngữ catalog nào khác (đặc biệt reserved_b3 / đối chứng), không còn chữ số, tag đúng. Câu sai bị loại, lý do ghi trong report.
Ra: <out-dir>/tts_script.jsonl, tts_holdout_terms.txt, tts_text.txt (text thuần, 1 câu / dòng, cho `leak_audit --text-files` và
leak_gate.text_files), tts_script.report.md. Cùng --seed -> cùng kịch bản (byte-by-byte).
Dòng kịch bản: {"id","text_tagged","tts_text","terms":[...],"readings":[{"term","reading","viet_level"}],"domain","category","flags"}.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.decontam import find_terms_in_texts, read_lines
from gipbench.core.paths import REPO as G
from gipbench.core.leak import norm_text
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import Catalog, basic_tokens, parse_tagged, plain_text, tagged_well_formed
from gipbench.dataset.synth.backends import (TermIndex, build_tts_text, fallback_readings, load_lexicon, term_key)

CAT2GROUP = {"tech": "tech", "finance": "business", "business": "business", "edu": "edu", "daily": "daily",
             "brand": "brand", "acronym": "acronym"}
CAT2DOMAIN = {"tech": "tech", "finance": "finance", "business": "product", "edu": "edu", "daily": "daily",
              "brand": "product", "acronym": "tech", "medical": "medical", "pubadmin": "pubadmin", "name": "daily"}
BUCKETS = ["start", "mid", "end", "multi"]
_EDGE = re.compile(r"^([^\w<>]*)(.*?)([^\w<>]*)$", re.S)


@dataclass
class Term:
    key: str
    canon: tuple
    category: str = ""
    source: str = "catalog"
    reserved: bool = False
    held: bool = False


def stable_frac(key: str, salt: str) -> float:
    """Số trong [0, 1) ổn định theo (salt, thuật ngữ): quyết định thuật ngữ đối chứng."""
    return int(hashlib.sha1(f"{salt}|{key}".encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32


def load_templates(path):
    tpls = []
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for n, line in enumerate(lines[1:], 1):  # dòng đầu = header
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 3:
            sys.exit(f"{path}: dòng dữ liệu {n} thiếu cột (cần category, domain, template)")
        cat, dom, t = cols[0].strip(), cols[1].strip(), cols[2].strip()
        if "{term}" not in t:
            sys.exit(f"{path}: dòng dữ liệu {n} không có {{term}}: {t}")
        rest = t.replace("{term2}", "").replace("{term}", "")
        if "{" in rest or "}" in rest or "<" in rest:
            sys.exit(f"{path}: dòng dữ liệu {n} có ngoặc / tag lạ: {t}")
        body = re.sub(r"[\s.,?!]+$", "", t)
        pos = "multi" if "{term2}" in t else "start" if t.startswith("{term}") else "end" if body.endswith("{term}") else "mid"
        tpls.append(dict(category=cat, domain=dom or "*", template=t, pos=pos, i=n))
    return tpls


def strip_edge(w):
    return _EDGE.match(w).groups()


def tag_sentence(text0, declared):
    """Gắn <en> vào mọi lần xuất hiện (khớp dài nhất, theo ranh giới từ) của các thuật ngữ khai báo.
    declared: {term_key: canon tuple}. Trả về text_tagged hoặc None khi không khớp ranh giới từ."""
    words = text0.split()
    toks, w_of = [], []
    for wi, w in enumerate(words):
        for t in basic_tokens(strip_edge(w)[1]):
            toks.append(t)
            w_of.append(wi)
    idx = TermIndex()
    for k, c in declared.items():
        idx.add(c, k)
    marks = {}  # từ đầu -> từ cuối
    for s, e, _ in idx.find(toks):
        ws, we = w_of[s], w_of[e - 1]
        if s > 0 and w_of[s - 1] == ws or e < len(toks) and w_of[e] == we:  # khớp giữa từ (vd "check-in" cho "check")
            continue
        marks[ws] = we
    out, i = [], 0
    while i < len(words):
        if i in marks:
            j = marks[i]
            pre, _, _ = strip_edge(words[i])
            _, _, post = strip_edge(words[j])
            inner = " ".join(([strip_edge(words[i])[1]] if i == j else
                              [strip_edge(words[i])[1]] + [words[k] for k in range(i + 1, j)] + [strip_edge(words[j])[1]]))
            out.append(f"{pre}<en>{inner}</en>{post}")
            i = j + 1
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="catalog.tsv: thuật ngữ + category + reserved_b3")
    ap.add_argument("--reserved-terms", required=True,
                    help="BẮT BUỘC: thuật ngữ giữ riêng (FT-unseen; 1 dòng / thuật ngữ, bỏ dòng '#'): không bao giờ vào kịch bản, "
                         "kể cả khi xuất hiện trong câu mẫu hay câu --sentences (audit R05 / R23)")
    ap.add_argument("--allow-bench-catalog", action="store_true",
                    help="cho phép catalog có span lấy từ chính bench (from_bench): chỉ để chẩn đoán oracle")
    ap.add_argument("--terms", help="file thuật ngữ thêm: mỗi dòng `term` hoặc `term<TAB>category` (bỏ dòng '#')")
    ap.add_argument("--bench", nargs="*", default=[], help="manifest bench: chặn câu trùng text")
    ap.add_argument("--sentences", help="JSONL câu viết sẵn {text, terms, domain?}: ưu tiên hơn mẫu")
    ap.add_argument("--lexicon", default=str(G / "configs/dataset/synth/tts_lexicon.tsv"))
    ap.add_argument("--templates", default=str(G / "configs/dataset/synth/tts_templates.tsv"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contexts-per-term", type=int, default=5, help="số câu khác nhau cho mỗi thuật ngữ")
    ap.add_argument("--readings-per-term", type=int, default=2, help="số cách đọc khác nhau cho mỗi thuật ngữ")
    ap.add_argument("--holdout-frac", type=float, default=0.3, help="tỉ lệ thuật ngữ đối chứng (không bao giờ được tổng hợp)")
    ap.add_argument("--holdout-salt", default="tts-holdout-v1", help="muối của hash chọn đối chứng (đổi = đổi nhóm đối chứng)")
    ap.add_argument("--holdout-terms", help="file thuật ngữ BẮT BUỘC là đối chứng (hợp với nhóm chọn bằng hash)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-lexicon", action="store_true", help="bỏ hẳn thuật ngữ không có trong lexicon (không dùng dự phòng)")
    ap.add_argument("--include-lexicon-terms", action="store_true", help="thêm thuật ngữ của lexicon mà catalog không có")
    ap.add_argument("--no-templates", action="store_true", help="chỉ dùng câu của --sentences")
    ap.add_argument("--min-term-chars", type=int, default=2, help="bỏ thuật ngữ ngắn hơn N ký tự")
    ap.add_argument("--max-term-tokens", type=int, default=3, help="bỏ thuật ngữ dài hơn N token (catalog nháp có cụm rác)")
    ap.add_argument("--max-terms", type=int, default=0, help="chỉ lấy N thuật ngữ đầu (theo thứ tự hash ổn định); 0 = hết")
    args = ap.parse_args()

    try:  # kịch bản TTS dựng từ catalog có span bench = dạy model đúng thuật ngữ sẽ bị chấm (60.7% lexicon E3 v2 là thuật ngữ C1)
        R.forbid_bench_catalog(args.catalog, "make_text", allow=args.allow_bench_catalog)
    except R.ReleaseError as e:
        sys.exit(f"LỖI {e}")
    reserved_list = read_lines(args.reserved_terms)
    if not reserved_list:
        sys.exit(f"LỖI --reserved-terms {args.reserved_terms} rỗng: danh sách thuật ngữ giữ riêng phải có ít nhất 1 mục")
    cat = Catalog.load(args.catalog)
    lex, lex_problems = load_lexicon(args.lexicon)
    templates = [] if args.no_templates else load_templates(args.templates)

    # ---- tập thuật ngữ ứng viên
    terms = {}

    def add_term(raw, category="", source="catalog", reserved=False):
        canon = tuple(basic_tokens(raw))
        if not canon:
            return
        k = " ".join(canon)
        t = terms.get(k)
        if t is None:
            terms[k] = Term(k, canon, category, source, reserved)
        else:
            t.category = t.category or category
            t.reserved = t.reserved or reserved

    for canon, v in cat.terms.items():
        add_term(" ".join(canon), v.category, "catalog", v.reserved_b3)
    for raw in reserved_list:  # thuật ngữ giữ riêng: như reserved_b3 (cấm tuyệt đối), kể cả khi không có trong catalog
        add_term(raw, "", "reserved_terms", True)
    if args.terms:
        for line in open(args.terms, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if cols[0].strip() == "term":
                continue
            add_term(cols[0], cols[1].strip() if len(cols) > 1 else "", "terms")
    if args.include_lexicon_terms:
        for k, rds in lex.items():
            kinds = {r.kind for r in rds}
            guess = "brand" if "brand" in kinds else "acronym" if kinds & {"acronym_en", "acronym_vi"} else ""
            add_term(k, guess, "lexicon")

    # ---- câu viết sẵn: đọc trước để thuật ngữ của chúng vào tập ứng viên (và được xét đối chứng như mọi thuật ngữ)
    sent_in = []
    if args.sentences:
        for ln, line in enumerate(open(args.sentences, encoding="utf-8"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{args.sentences}:{ln}: JSON lỗi: {e}")
            rec["_ln"] = ln
            sent_in.append(rec)
            for t in rec.get("terms") or []:
                add_term(t, "", "sentences")

    # ---- đối chứng (hash ổn định) + reserved
    forced = set()
    if args.holdout_terms:
        forced = {term_key(l.split("\t")[0]) for l in open(args.holdout_terms, encoding="utf-8")
                  if l.strip() and not l.startswith("#")}
    for t in terms.values():
        t.held = (not t.reserved) and (stable_frac(t.key, args.holdout_salt) < args.holdout_frac or t.key in forced)
    short = {k for k in terms if len(k) < args.min_term_chars or len(terms[k].canon) > args.max_term_tokens}
    reserved = {k for k, t in terms.items() if t.reserved}
    held = {k for k, t in terms.items() if t.held and k not in short}
    pool_all = sorted((k for k, t in terms.items() if not t.reserved and not t.held and k not in short),
                      key=lambda k: stable_frac(k, "order|" + args.holdout_salt))
    if args.max_terms:
        pool_all = pool_all[:args.max_terms]
    dropped_lex = [k for k in pool_all if k not in lex] if args.require_lexicon else []
    pool = sorted(k for k in pool_all if not (args.require_lexicon and k not in lex))
    pool_set = set(pool)

    status_idx = TermIndex()  # mọi thuật ngữ đã biết, để dò thuật ngữ lạ lọt vào câu
    for k, t in terms.items():
        status_idx.add(t.canon, k)
    for k in lex:
        status_idx.add(tuple(k.split()), k)

    # ---- bench
    bench_txt = set()
    for b in args.bench:
        for r in read_jsonl(b):
            bench_txt.add(norm_text(r.get("text_tagged") or r.get("text"), cat))
    stat = Counter()
    reject = Counter()
    lines, seen_text, seen_norm = [], set(), set()
    ctx = Counter()  # số câu mỗi thuật ngữ
    reading_cache, cursor = {}, Counter()

    def readings_of(key):
        if key in reading_cache:
            return reading_cache[key]
        rds = list(lex.get(key) or fallback_readings(key.split()))
        rng = random.Random(f"{args.seed}|read|{key}")
        by = defaultdict(list)
        for r in rds:
            by[r.viet_level].append(r)
        order = list(by)
        rng.shuffle(order)
        pick = [by[lv].pop(rng.randrange(len(by[lv]))) for lv in order]
        rest = [r for lv in order for r in by[lv]]
        rng.shuffle(rest)
        chosen = (pick + rest)[:max(1, args.readings_per_term)]
        cursor[key] = rng.randrange(len(chosen))
        reading_cache[key] = chosen
        return chosen

    def next_reading(key):
        ch = readings_of(key)
        r = ch[cursor[key] % len(ch)]
        cursor[key] += 1
        return r

    def domain_of(category):
        return CAT2DOMAIN.get(category, "other")

    def emit(text_tagged, domain, flags):
        """Dựng dòng kịch bản; trả về False nếu trùng / trùng bench / lỗi tag."""
        why = tagged_well_formed(text_tagged)
        if why:
            reject[f"tag lỗi: {why}"] += 1
            return False
        if text_tagged in seen_text:
            reject["trùng câu đã sinh"] += 1
            return False
        n = norm_text(text_tagged, cat)
        if n in bench_txt:
            stat["bench_dup"] += 1
            return False
        if n in seen_norm:
            reject["trùng câu đã sinh (sau chuẩn hoá)"] += 1
            return False
        order = []
        for seg, is_en in parse_tagged(text_tagged):
            k = term_key(seg) if is_en else None
            if k and k not in order:
                order.append(k)
        rds, rmap, fl = [], {}, set(flags)
        for k in order:
            r = next_reading(k)
            rds.append({"term": k, "reading": r.reading, "viet_level": r.viet_level})
            rmap[k] = r.reading
            if r.fallback:
                fl.add("reading_fallback")
                fl.add(f"reading_fallback:{k}")
        if len(order) > 1:
            fl.add("two_terms")
        cats = [terms[k].category for k in order if k in terms and terms[k].category]
        category = cats[0] if cats else ""
        tts = build_tts_text(text_tagged, rmap)
        sid = "tts_" + hashlib.sha1(f"{text_tagged}|{tts}".encode("utf-8")).hexdigest()[:10]
        lines.append({"id": sid, "text_tagged": text_tagged, "tts_text": tts, "terms": order, "readings": rds,
                      "domain": domain or domain_of(category), "category": category, "flags": sorted(fl)})
        seen_text.add(text_tagged)
        seen_norm.add(n)
        for k in order:
            ctx[k] += 1
        return True

    def find_unwanted(text_tagged, allowed):
        """Thuật ngữ đã biết (reserved / đối chứng / khác) có trong câu mà không thuộc `allowed`. [(key, lý do)]"""
        toks = basic_tokens(plain_text(text_tagged))
        bad = []
        for _, _, k in status_idx.find(toks):
            key = status_idx.d[k]
            if key in allowed:
                continue
            bad.append((key, "reserved_b3" if key in reserved else "thuật ngữ đối chứng" if key in held else "thuật ngữ không khai báo"))
        return bad

    # ---- 1) câu viết sẵn
    srej = Counter()
    n_sent_in = 0
    for rec in sent_in:
        n_sent_in += 1
        text = unicodedata.normalize("NFC", str(rec.get("text") or "")).strip()
        decl_raw = rec.get("terms") or []
        if not text or not decl_raw:
            srej["thiếu text hoặc terms"] += 1
            continue
        declared = {term_key(t): tuple(basic_tokens(t)) for t in decl_raw if term_key(t)}
        if re.search(r"\d", text):
            srej["còn chữ số (phải viết dạng đọc)"] += 1
            continue
        bad = [k for k in declared if k in reserved or k in held or k in short]
        if bad:
            srej["thuật ngữ khai báo là " + ("reserved_b3" if bad[0] in reserved else "đối chứng / quá ngắn")] += 1
            continue
        if args.require_lexicon and any(k not in lex for k in declared):
            srej["thuật ngữ khai báo không có trong lexicon (--require-lexicon)"] += 1
            continue
        had_tags = "<en" in text
        orig = {term_key(s) for s, e in parse_tagged(text) if e} if had_tags else set()
        tagged = tag_sentence(plain_text(text), declared)
        got = {term_key(sg) for sg, e in parse_tagged(tagged) if e}
        miss = [k for k in declared if k not in got]
        if miss:
            srej[f"thuật ngữ khai báo không có trong câu / sai ranh giới từ: {miss[0]}"] += 1
            continue
        if had_tags and orig != set(declared):
            srej["tag <en> có sẵn không khớp thuật ngữ khai báo"] += 1
            continue
        unwanted = find_unwanted(tagged, set(declared))
        if unwanted:
            srej[f"{unwanted[0][1]}: {unwanted[0][0]}"] += 1
            continue
        if emit(tagged, rec.get("domain") or "", ["from_sentences"]):
            stat["from_sentences"] += 1
        else:
            srej["trùng (bench / câu đã có)"] += 1

    # ---- 2) mẫu câu
    by_group = defaultdict(lambda: defaultdict(list))
    for t in templates:
        by_group[t["category"]][t["pos"]].append(t)

    def groups_of(key):
        t = terms.get(key)
        gs = []
        g = CAT2GROUP.get(t.category if t else "")
        if g:
            gs.append(g)
        kinds = {r.kind for r in lex.get(key, [])}
        if kinds & {"acronym_en", "acronym_vi"} and "acronym" not in gs:
            gs.append("acronym")
        if "brand" in kinds and "brand" not in gs:
            gs.append("brand")
        gs.append("general")
        return gs

    terms_short = []
    if templates:
        for key in pool:
            need = args.contexts_per_term - ctx[key]
            if need <= 0:
                continue
            rng = random.Random(f"{args.seed}|tpl|{key}")
            buckets = {b: [] for b in BUCKETS}
            for g in groups_of(key):
                for b in BUCKETS:
                    buckets[b] += by_group[g][b]
            for b in BUCKETS:
                rng.shuffle(buckets[b])
            order = [b for b in BUCKETS if buckets[b]]
            rng.shuffle(order)
            surface = key
            tries = 0
            while need > 0 and any(buckets[b] for b in order) and tries < 400:
                for b in order:
                    if need <= 0:
                        break
                    if not buckets[b]:
                        continue
                    tries += 1
                    tpl = buckets[b].pop()
                    text = tpl["template"].replace("{term}", f"<en>{surface}</en>")
                    allowed = {key}
                    if "{term2}" in text:
                        if len(pool) < 2:
                            continue
                        p2 = rng.choice(pool)
                        while p2 == key:
                            p2 = rng.choice(pool)
                        text = text.replace("{term2}", f"<en>{p2}</en>")
                        allowed.add(p2)
                    extra = find_unwanted(text, allowed)
                    if extra:
                        reject[f"mẫu #{tpl['i']} chứa {extra[0][1]} ({extra[0][0]})"] += 1
                        continue
                    if emit(text, tpl["domain"] if tpl["domain"] != "*" else "", []):
                        need -= 1
            if need > 0:
                terms_short.append((key, args.contexts_per_term - need))

    # ---- ghi kết quả
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines.sort(key=lambda r: (r["terms"][0] if r["terms"] else "", r["id"]))
    with open(out / "tts_script.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "tts_holdout_terms.txt", "w", encoding="utf-8", newline="\n") as f:
        for k in sorted(held):
            f.write(k + "\n")
    # text thuần của kịch bản (1 câu / dòng) để `leak_audit --text-files` / cổng rò rỉ kiểm lại với bench và reserved_terms
    with open(out / "tts_text.txt", "w", encoding="utf-8", newline="\n") as f:
        for r in lines:
            f.write(plain_text(r["text_tagged"]) + "\n")
    leaked = find_terms_in_texts(reserved_list, (plain_text(r["text_tagged"]) for r in lines))
    if leaked:  # kiểm lại độc lập với bộ lọc lúc sinh (đóng khi lỗi)
        sys.exit(f"LỖI {len(leaked)} thuật ngữ giữ riêng lọt vào kịch bản: {sorted(leaked)[:10]}")

    used = {k for r in lines for k in r["terms"]}
    n_lex = sum(1 for k in used if k in lex)
    fb_terms = sorted(k for k in used if k not in lex)
    lv = Counter(rd["viet_level"] + (" (dự phòng, giá trị giữ chỗ)" if f"reading_fallback:{rd['term']}" in r["flags"] else "")
                 for r in lines for rd in r["readings"])
    cats = Counter(r["category"] or "(trống)" for r in lines)
    doms = Counter(r["domain"] for r in lines)
    kinds = Counter()
    for r in lines:
        for rd in r["readings"]:
            for x in (lex.get(rd["term"]) or fallback_readings(rd["term"].split())):
                if x.reading == rd["reading"]:
                    kinds[x.kind + (" (dự phòng)" if x.fallback else "")] += 1
                    break
    per_term = Counter()
    for r in lines:
        for k in r["terms"]:
            per_term[k] += 1
    low = sorted(k for k in used if per_term[k] < args.contexts_per_term)
    md = ["# Kịch bản TTS", "",
          f"Catalog `{args.catalog}`, lexicon `{args.lexicon}` ({len(lex)} thuật ngữ), seed {args.seed}, "
          f"contexts/term {args.contexts_per_term}, readings/term {args.readings_per_term}, holdout {args.holdout_frac:g}.", "",
          "## Thuật ngữ", "",
          f"- ứng viên: {len(terms)} (catalog {sum(1 for t in terms.values() if t.source == 'catalog')}, "
          f"thêm {sum(1 for t in terms.values() if t.source != 'catalog')})",
          f"- reserved_b3 (cấm tuyệt đối): {len(reserved)}",
          f"- bỏ vì quá ngắn (< {args.min_term_chars} ký tự) hoặc quá dài (> {args.max_term_tokens} token): {len(short - reserved)}",
          f"- đối chứng (không bao giờ tổng hợp): {len(held)} -> tts_holdout_terms.txt",
          f"- còn lại trong nhóm tổng hợp: {len(pool_all)}"
          + (f", bỏ {len(dropped_lex)} vì --require-lexicon" if args.require_lexicon else ""),
          f"- ĐƯỢC SINH CÂU: {len(used)} thuật ngữ — có cách đọc trong lexicon: {n_lex}, dùng cách đọc DỰ PHÒNG: {len(fb_terms)}",
          f"- thuật ngữ chưa đủ {args.contexts_per_term} câu (hết mẫu / trùng): {len(low)}", "",
          "## Câu", "",
          f"- tổng: {len(lines)} câu ({stat['from_sentences']} từ --sentences, {len(lines) - stat['from_sentences']} từ mẫu); "
          f"hai thuật ngữ: {sum(1 for r in lines if len(r['terms']) > 1)}; có cờ reading_fallback: "
          f"{sum(1 for r in lines if 'reading_fallback' in r['flags'])}",
          f"- bị loại vì TRÙNG text bench (đã thay bằng mẫu khác): {stat['bench_dup']}",
          f"- --sentences: đọc {n_sent_in}, loại {sum(srej.values())}"]
    md += [f"  - {k}: {v}" for k, v in srej.most_common()]
    md += ["- mẫu bị loại (chứa thuật ngữ đã biết ngoài thuật ngữ cần đặt, câu lỗi tag, trùng):"]
    md += [f"  - {k}: {v}" for k, v in reject.most_common(15)] or ["  - không có"]
    md += ["", "### Phân bố category (số câu)", "", "| category | câu |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in cats.most_common()]
    md += ["", "### Phân bố domain (số câu)", "", "| domain | câu |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in doms.most_common()]
    md += ["", "### Mức đọc (số span theo viet_level của cách đọc đã chọn)", "", "| viet_level | span |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in sorted(lv.items())]
    md += ["", "### Loại cách đọc", "", "| kind | span |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in kinds.most_common()]
    md += ["", f"## Thuật ngữ THIẾU cách đọc trong lexicon ({len(fb_terms)}) — đang dùng dự phòng, cần người bổ sung vào "
           f"configs/dataset/synth/tts_lexicon.tsv", "", ", ".join(fb_terms[:300]) + (f" … (+{len(fb_terms) - 300})" if len(fb_terms) > 300 else "")
           if fb_terms else "không có"]
    if low:
        md += ["", f"## Thuật ngữ chưa đủ câu ({len(low)})", "", ", ".join(f"{k} ({per_term[k]})" for k in low[:100])]
    if lex_problems:
        md += ["", "## Cảnh báo lexicon", ""] + [f"- {p}" for p in lex_problems[:50]]
    (out / "tts_script.report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Thuật ngữ: ứng viên {len(terms)}, reserved {len(reserved)}, đối chứng {len(held)}, nhóm tổng hợp {len(pool_all)}, "
          f"được sinh câu {len(used)} (lexicon {n_lex}, dự phòng {len(fb_terms)})")
    print(f"Câu: {len(lines)} (từ --sentences {stat['from_sentences']}); loại do trùng bench {stat['bench_dup']}; "
          f"--sentences bị loại {sum(srej.values())}")
    print(f"Đã ghi {out / 'tts_script.jsonl'}, {out / 'tts_holdout_terms.txt'}, {out / 'tts_text.txt'}, {out / 'tts_script.report.md'}")
    if not lines:
        sys.exit("LỖI: kịch bản rỗng (kiểm --catalog / --holdout-frac / --require-lexicon)")


if __name__ == "__main__":
    main()
