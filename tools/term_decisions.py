#!/usr/bin/env python3
"""Phiếu chốt tag theo thuật ngữ (decisions §12.4): MỘT file duy nhất cho người dùng duyệt.

  # 1) sinh phiếu từ manifest có trusted_text (G-Group, consensus agreed) + gộp reports/top_review.csv
  python tools/term_decisions.py ask \
      --manifests data/raw/ggroup/b1_candidates.jsonl data/raw/ggroup/b5_ggroup.jsonl \
                  data/raw/consensus/b2_consensus_candidates.jsonl data/raw/consensus/b2_consensus_overcap.jsonl \
                  data/raw/consensus/b5_consensus_product.jsonl \
      --top-review reports/top_review.csv --catalog data/raw/_proxy/catalog_proxy.tsv \
      --lexicons data/dl/lexicons --out reports/term_decisions_ask.csv --auto-out reports/term_auto_en.csv
  # 2) người dùng sửa cột decision những dòng sai (en / vi / drop_tag / để trống = chưa chốt), rồi
  python tools/term_decisions.py import --ask reports/term_decisions_ask.csv          # ghi configs/term_decisions.tsv
  # 3) chạy lại conv_ggroup.py / conv_consensus.py, rồi build_bench_v0.py --accept trusted-text

ask: mỗi thuật ngữ còn cờ (lib/term_decisions.needed_terms: abbr_reading, maybe_name, vi_word, suspect, name, span có từ
mơ hồ) mà CHƯA có trong --decisions -> một dòng, kèm số câu / giờ (trong các manifest trusted_text) và 2 câu ví dụ.
Thêm các mục của top_review.csv (vi_abbrev / vi_loanwords xuất hiện nhiều nhất) chưa có dòng, để chỉ phải duyệt 1 file:
với các mục này decision vi = giữ trong danh sách (không EN), en = coi là EN ở mọi bộ (thắng danh sách).
Cột decision điền sẵn ĐỀ XUẤT của Claude (lý do ở cột note; "đoán" = không chắc); trống = Claude không đoán được
(phụ thuộc ngữ cảnh) -> câu chứa nó vẫn chờ. Sắp theo số câu giảm dần, rồi số lần trong corpus.
--auto-out: span heuristic không cờ (vd "deep learning") được coi là EN rõ ràng, KHÔNG cần duyệt; ghi ra để tham khảo.
import: dòng có decision -> --decisions (dòng cùng term bị thay; ghi chú "ask: ..."). --out khác: ghi bản giả định
(vd chấp nhận mọi đề xuất) mà không đụng file thật.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.term_decisions import (DEFAULT_TERM_DECISIONS, load_term_decisions, needed_terms, norm_term,  # noqa: E402
                                read_term_rows, spans_of, write_term_rows)
from lib.textnorm import Catalog  # noqa: E402

# viết tắt tiếng Anh thường được đọc kiểu Anh trong hội thoại tiếng Việt (chỉ để ĐOÁN đề xuất)
EN_ACRONYMS = set("""ai ceo cfo cto coo cmo kpi okr hr pr it seo api ui ux crm erp sql usb cpu gpu ram ssd led lcd oled gps
wifi vpn iot ar vr qr nfc b2b b2c mc dj vip mba phd ielts toefl gmat dna covid hiv fifa nasa ok sms app pc cv faq diy
ootd kol pov fyi asap otp ip id""".split())
# cờ suspect (từ đứng một mình trùng âm tiết Việt): đề xuất theo nghĩa thường gặp khi text có dấu đầy đủ
SUSPECT_GUESS = {"chat": ("en", "chat không dấu = từ Anh (tiếng Việt viết 'chát')"),
                 "bot": ("en", "bot = chatbot"), "net": ("en", "net = internet"), "mic": ("en", "mic = micro"),
                 "chip": ("en", "chip điện tử"), "top": ("en", "top = đứng đầu"), "hot": ("en", "hot = nổi"),
                 "pay": ("en", "pay = thanh toán"), "mac": ("en", "mac = máy Mac"),
                 "pin": ("vi", "pin = pin điện (từ Việt)"), "sim": ("vi", "sim điện thoại (từ mượn đã Việt hoá)"),
                 "ai": ("", "'ai' tiếng Việt hay AI: phụ thuộc ngữ cảnh, để duyệt từng câu")}
VI_GUESS = {"hose": "Sở GDCK TP.HCM (HOSE), thường đọc kiểu Việt", "hnx": "Sở GDCK Hà Nội, đánh vần kiểu Việt",
            "abc": "tên giả định (công ty ABC), thường đọc a-bê-xê"}
C5_VI = {"fpt": "tên công ty Việt (FPT), thường đọc ép-pê-tê", "bic": "Bảo hiểm BIDV, đọc kiểu Việt"}


def read_list(path):
    out = {}
    if not Path(path).exists():
        return out
    for line in open(path, encoding="utf-8"):
        cols = line.rstrip("\n").split("\t")
        if not cols[0].strip() or cols[0].startswith("#") or cols[0] in ("abbr", "word"):
            continue
        out[cols[0].strip().lower()] = cols[1] if len(cols) > 1 else ""
    return out


def read_dropped_c5(path):
    out = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            if line.startswith("#-") and "text không phải tài chính" in line:
                out.add(line[2:].split("\t")[0].strip().lower())
    return out


def lex(path, name):
    p = Path(path) / name if path else None
    return {l.strip().lower() for l in open(p, encoding="utf-8")} if p and p.exists() else set()


def suggest(term, kinds, ctx):
    """-> (decision đề xuất, lý do). ctx: abbrev, loan, g10k, words, c5."""
    toks = term.split()
    if "name" in kinds:
        return "en", "tên người nước ngoài (luật tên người §12, đã duyệt)"
    if "maybe_name" in kinds:
        if len(toks) >= 2:
            return "en", "đoán: cụm tên người (en_names), vd họ + tên"
        return "", "tên trùng âm tiết Việt đứng một mình: phụ thuộc ngữ cảnh"
    if term in ctx["abbrev"]:
        return "vi", f"có trong vi_abbrev ({ctx['abbrev'][term]})"
    if term in ctx["loan"]:
        k = ctx["loan"][term]
        note = "từ mượn trùng mặt chữ Anh, không EN ở mọi bộ (§12)" if k == "same_surface" else "từ mượn chính tả Việt"
        if term == "test":
            note += "; lưu ý: vi-asr-tech có 'unit test' đọc kiểu Anh"
        return "vi", f"vi_loanwords ({k}): {note}"
    if term in ctx["c5"]:
        if term in C5_VI:
            return "vi", f"đoán: {C5_VI[term]}; bị luật §12.3 (>= 20 lần ngoài tài chính) loại khỏi vi_abbrev"
        return "drop_tag", "bị luật §12.3 loại khỏi vi_abbrev; không phải từ Anh rõ ràng"
    if "suspect" in kinds and term in SUSPECT_GUESS:
        return SUSPECT_GUESS[term]
    if "ambiguous_joined" in kinds:
        return "en", "cụm tiếng Anh có từ mơ hồ (to, the, in...): kiểm tra ranh giới span"
    if "vi_word" in kinds:
        return "vi", "đang được danh sách coi là tiếng Việt"
    if term in VI_GUESS:
        return "vi", f"đoán: {VI_GUESS[term]}"
    if term in EN_ACRONYMS:
        return "en", "đoán: viết tắt tiếng Anh thường đọc kiểu Anh"
    has_vowel = any(c in "aeiouy" for c in term)
    if len(toks) == 1 and has_vowel and len(term) >= 3 and term in ctx["g10k"]:
        return "en", ("đoán: " if len(term) <= 3 else "") + "từ tiếng Anh thông dụng (google-10000), không phải viết tắt"
    if len(toks) == 1 and has_vowel and len(term) >= 3 and term in ctx["words"]:
        return "en", "đoán: từ tiếng Anh (words_alpha)"
    if len(toks) == 1 and not has_vowel and len(term) <= 5:
        return "vi", "đoán: viết tắt toàn phụ âm, thường đánh vần kiểu Việt"
    if has_vowel:
        return "en", "đoán: brand / tên riêng chữ Latin = EN (SCHEMA §1)"
    return "", "không đoán được"


def cmd_ask(args):
    dec = load_term_decisions(args.decisions)
    cat = Catalog.load(args.catalog) if args.catalog else Catalog()
    cat_keys = {" ".join(k) for k in cat.strict_map} if cat.strict_map else set()
    ctx = {"abbrev": read_list(G / "configs/vi_abbrev.txt"), "loan": read_list(G / "configs/vi_loanwords.txt"),
           "g10k": lex(args.lexicons, "google-10000-english-no-swears.txt"), "words": lex(args.lexicons, "words_alpha.txt"),
           "c5": read_dropped_c5(G / "configs/vi_abbrev.txt")}
    info = defaultdict(lambda: {"kinds": set(), "n": 0, "dur": 0.0, "src": Counter(), "ex": []})
    auto = defaultdict(lambda: {"n": 0, "src": Counter(), "ex": ""})
    n_rows = 0
    for mp in args.manifests:
        for line in open(mp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("trusted_text"):
                continue
            n_rows += 1
            terms, _ = needed_terms(r)
            src = r.get("dataset", "?") if r.get("dataset") != "consensus" else f"consensus:{r['split']}"
            for t, kinds in terms.items():
                e = info[t]
                e["kinds"] |= kinds
                e["n"] += 1
                e["dur"] += float(r.get("duration") or 0)
                e["src"][src] += 1
                if len(e["ex"]) < 2:
                    e["ex"].append((r["id"], r["text_tagged"]))
            for sp in set(spans_of(r.get("text_tagged", ""))):
                if sp not in terms and sp not in cat_keys and sp not in dec:
                    a = auto[sp]
                    a["n"] += 1
                    a["src"][src] += 1
                    a["ex"] = a["ex"] or f"{r['id']}: {r['text_tagged']}"
    top = []
    if args.top_review and Path(args.top_review).exists():
        top = list(csv.DictReader(open(args.top_review, encoding="utf-8-sig")))
    corpus = {}
    for t in top:
        e = norm_term(t["entry"])
        corpus[e] = (t["file"], t.get("group_or_kind", ""), int(t.get("count") or 0),
                     f"ggroup {t.get('count_ggroup', 0)} / vimedcss {t.get('count_vimedcss', 0)} / vss {t.get('count_vss', 0)}",
                     f"{t.get('example_id', '')}: {t.get('example_text', '')}")
    rows = []
    for t, e in info.items():
        if t in dec:
            continue
        d, why = suggest(t, e["kinds"], ctx)
        c = corpus.get(t)
        rows.append({"term": t, "decision": d, "n_sentences": e["n"], "hours": f"{e['dur'] / 3600:.3f}",
                     "flags": " ".join(sorted(e["kinds"])), "sources": "; ".join(f"{k} {v}" for k, v in e["src"].most_common()),
                     "in_list": "vi_abbrev" if t in ctx["abbrev"] else "vi_loanwords" if t in ctx["loan"] else
                     ("catalog" if t in cat_keys else ""),
                     "corpus_count": c[3] if c else "", "_cc": c[2] if c else 0,
                     "example_1": f"{e['ex'][0][0]}: {e['ex'][0][1]}" if e["ex"] else "",
                     "example_2": f"{e['ex'][1][0]}: {e['ex'][1][1]}" if len(e["ex"]) > 1 else "", "note": why})
    seen = {r["term"] for r in rows}
    for t, c in corpus.items():
        if t in seen or t in dec:
            continue
        d, why = suggest(t, {"top_review"}, ctx)
        rows.append({"term": t, "decision": d, "n_sentences": 0, "hours": "0.000", "flags": "top_review",
                     "sources": "", "in_list": c[0].replace(".txt", "") + (f" ({c[1]})" if c[1] else ""),
                     "corpus_count": c[3], "_cc": c[2], "example_1": c[4], "example_2": "",
                     "note": why + "; (từ top_review.csv: vi = giữ trong danh sách, en = coi là EN)"})
    rows.sort(key=lambda r: (-int(r["n_sentences"]), -r["_cc"], r["term"]))
    cols = ["term", "decision", "n_sentences", "hours", "flags", "sources", "in_list", "corpus_count",
            "example_1", "example_2", "note"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    by = Counter(r["decision"] or "(trống)" for r in rows)
    print(f"Đã ghi {args.out}: {len(rows)} thuật ngữ ({n_rows} câu trusted_text); đề xuất: {dict(by)}")
    if args.auto_out:
        with open(args.auto_out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["term", "n_sentences", "sources", "example"])
            for t, a in sorted(auto.items(), key=lambda x: (-x[1]["n"], x[0])):
                w.writerow([t, a["n"], "; ".join(f"{k} {v}" for k, v in a["src"].most_common()), a["ex"]])
        print(f"Đã ghi {args.auto_out}: {len(auto)} span heuristic không cờ (coi là EN rõ ràng, không cần duyệt)")
    return rows


def cmd_import(args):
    out = args.out or args.decisions
    base = read_term_rows(args.decisions)
    by = {norm_term(r["term"]): r for r in base}
    n = 0
    for r in csv.DictReader(open(args.ask, encoding="utf-8-sig")):
        t, d = norm_term(r["term"]), (r.get("decision") or "").strip().lower()
        if not t or not d:
            continue
        by[t] = {"term": t, "decision": d, "note": f"ask: {(r.get('note') or '').strip()}"}
        n += 1
    write_term_rows(out, by.values(), "Dòng do người dùng chốt ghi 'người dùng chốt'; dòng nhập từ phiếu ghi 'ask: ...'.")
    load_term_decisions(out)  # kiểm tra decision hợp lệ
    print(f"Đã ghi {out}: {len(by)} thuật ngữ ({n} dòng có decision từ {args.ask})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--manifests", nargs="+", required=True)
    a.add_argument("--decisions", default=str(DEFAULT_TERM_DECISIONS))
    a.add_argument("--top-review", default=str(G / "reports/top_review.csv"))
    a.add_argument("--catalog")
    a.add_argument("--lexicons", help="thư mục google-10000 / words_alpha (chỉ để đoán đề xuất)")
    a.add_argument("--out", required=True)
    a.add_argument("--auto-out")
    i = sub.add_parser("import")
    i.add_argument("--ask", required=True)
    i.add_argument("--decisions", default=str(DEFAULT_TERM_DECISIONS))
    i.add_argument("--out", help="mặc định ghi đè --decisions")
    args = ap.parse_args()
    cmd_ask(args) if args.cmd == "ask" else cmd_import(args)


if __name__ == "__main__":
    main()
