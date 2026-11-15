#!/usr/bin/env python3
"""Dựng bản nháp configs/vi_abbrev.txt (viết tắt / mã đọc kiểu Việt) và configs/vi_loanwords.txt (từ mượn có
trong từ điển tiếng Việt) từ các nguồn công khai, có nguyên tắc. auto_tag không gắn <en> cho từ trong hai file này.

  python tools/build_wordlists.py --src data/dl/lexicons --catalog configs/catalog.example.tsv \
      --vimedcss-meta data/dl/vimedcss_meta --out-abbrev configs/vi_abbrev.txt --out-loan configs/vi_loanwords.txt

Nguồn (tải về --src trước, xem lệnh tải trong reports/phase0b_report.md):
  vietcap_symbols.json      danh sách mã của Vietcap (trading.vietcap.com.vn/api/price/symbols/getAll), có thể nén gzip
  wiki_Danh_sách_ngân_hàng_tại_Việt_Nam.txt   wikitext trang Wikipedia tiếng Việt (action=raw)
  wiki_tinh.txt             wikitext "Đơn vị hành chính cấp tỉnh (Việt Nam)"
  wikt_viettat_pages.json   {tiêu đề: wikitext} các trang trong Thể loại:Từ viết tắt tiếng Việt + Chữ viết tắt tiếng Việt
  Viet74K.txt               từ điển tiếng Việt (github.com/duyet/vietnamese-wordlist)
  words_alpha.txt           từ tiếng Anh (github.com/dwyl/english-words)
  google-10000-english-no-swears.txt   10 000 từ tiếng Anh phổ biến (github.com/first20hours/google-10000-english)

vi_abbrev, luật cho từng nhóm:
  mã chứng khoán  mã cổ phiếu sàn HSX/HNX (--boards), chỉ gồm chữ cái
  ngân hàng       tên giao dịch của ngân hàng trong nước dạng chữ cái đầu (2–6 chữ, không chứa "bank");
                  tên dạng brand (vietcombank, vpbank) là EN theo quy ước nên không lấy; ngân hàng nước ngoài không lấy
  tỉnh thành      chữ cái đầu của tên 34 tỉnh thành hiện hành (bỏ "Thành phố"); tên có Đ thì bỏ (đã có dấu)
  viết tắt        mục Wiktionary có dạng đầy đủ toàn tiếng Việt, không có định nghĩa "abbreviation of|en"
Ứng viên bị LOẠI tự động (ghi lại dưới dạng "#-" kèm lý do, người duyệt có thể bật lại):
  trùng catalog (EN), trùng từ tiếng Anh phổ biến (google-10000), hoặc đã là âm tiết tiếng Việt (không cần).
Luật text không phải tài chính (--nonfin-text, decisions §12.3): mã CHỈ thuộc nhóm "mã chứng khoán" (không thêm tay)
gặp >= --nonfin-min (20) lần trong các text đó (VSS, VIVOS) thì bị loại, vd tot, fir (thực ra là từ Anh / nhãn giả).
Quyết định tay: --project-abbrev (mặc định configs/vi_abbrev_project.tsv), áp SAU các nguồn: add (thắng luật
google-10000 / âm tiết, vd tp) và exclude (vd cdn). Luật catalog áp sau cùng: mục nào là term EN trong catalog đều bị
loại, danh sách in ra màn hình.

vi_loanwords, hai loại:
  vi_spelling   mục một chữ trong Viet74K, không phải âm tiết tiếng Việt đơn, không có trong words_alpha, không trùng
                google-10000 (axit, canxi, vaccin): chính tả Việt -> không EN, áp cho mọi dataset
  same_surface  thuật ngữ ViMedCSS (cs_terms_list, mặt chữ khớp trong câu) có trong Viet74K và trùng mặt chữ tiếng Anh
                (vitamin, protein, insulin): không EN ở MỌI bộ (decisions §12)
  --exclude TỪ:LÝ_DO loại tay một mục (vẫn ghi lại thành dòng '#-' kèm lý do)
"""
import argparse
import csv
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converters.common import auto_tag, is_vietnamese_token  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402

TODAY = "2026-11-15"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def read_lines(p):
    return [nfc(l.strip()) for l in open(p, encoding="utf-8") if l.strip()]


def strip_wiki(s):
    s = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return re.sub(r"'''?|<br\s*/?>", "", s).strip()


# ------------------------------------------------------------------ nguồn vi_abbrev
def from_tickers(path, boards):
    raw = Path(path).read_bytes()
    data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    for d in data:
        sym = (d.get("symbol") or "").strip()
        if d.get("board") in boards and d.get("type") == "STOCK" and sym.isalpha():
            yield sym.lower(), "mã chứng khoán", d.get("organShortName") or d.get("organName") or "", \
                f"vietcap symbols/getAll board={d['board']} ({TODAY})"


def from_banks(path):
    text = Path(path).read_text(encoding="utf-8")
    stop = text.find("==Ngân hàng 100% vốn nước ngoài")
    text = text[:stop if stop > 0 else len(text)]
    for block in text.split("\n|-"):
        cells = []
        for line in block.splitlines():
            if line.startswith("|") and not line.startswith(("|}", "|+", "|-")):
                cells += [c.strip() for c in line[1:].split("||")]
        if len(cells) < 3 or not re.fullmatch(r"\d+", cells[0]):
            continue
        name = strip_wiki(cells[1])
        for part in re.split(r"[/,]", strip_wiki(cells[2])):
            t = part.strip().lower()
            # chỉ lấy dạng chữ cái đầu: một token, tối đa 1 nguyên âm (bidv, shb), không phải brand (bac a bank, vikki)
            if t.isalpha() and 2 <= len(t) <= 6 and "bank" not in t and sum(c in "aeiouy" for c in t) <= 1:
                yield t, "ngân hàng", f"Ngân hàng {name}", f"vi.wikipedia: Danh sách ngân hàng tại Việt Nam ({TODAY})"


def from_provinces(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines[:-1]):
        if re.fullmatch(r"\|\d+", line.strip()) and re.match(r"\|(''')?\[\[", lines[i + 1]):
            name = strip_wiki(lines[i + 1][1:])
            words = re.sub(r"^(Thành phố|TP\.|Tỉnh)\s*", "", name).split()
            ini = "".join(w[0] for w in words).lower()
            if len(words) >= 2 and "đ" not in ini:
                yield ini, "tỉnh thành", name, f"vi.wikipedia: Đơn vị hành chính cấp tỉnh (Việt Nam), chữ cái đầu ({TODAY})"


_ABBR_T = re.compile(r"\{\{(?:abbreviation of|abbr of)\|(\w+)\|([^|}]*)")


def from_wiktionary(path):
    pages = json.loads(Path(path).read_text(encoding="utf-8"))
    for title, wt in pages.items():
        t = nfc(title).lower()
        if not re.fullmatch(r"[a-z]{2,8}", t) or "abbreviation of|en" in wt or "abbr of|en" in wt:
            continue
        defs = [l for l in wt.splitlines() if l.startswith("#") and not l.startswith(("#:", "#*"))]
        for d in defs:
            m = _ABBR_T.search(d)
            if m and m.group(1) != "vi":
                continue
            exp = strip_wiki(m.group(2)) if m else strip_wiki(d.lstrip("#").replace("Viết tắt của", ""))
            exp = re.sub(r"[().,;:]", " ", exp)
            words = exp.split()
            if len(words) >= 2 and all(is_vietnamese_token(w) for w in words):
                yield t, "viết tắt", " ".join(words), f"vi.wiktionary: {title} ({TODAY})"
                break


def nonfin_counts(paths, toks):
    """Đếm token (trong toks) trong text không phải tài chính: .json (list / dict bản ghi có 'text', như VSS),
    .jsonl (manifest: 'text' hoặc 'text_tagged'), .txt (mỗi dòng '<id> <text>', như prompts của VIVOS)."""
    from lib.textnorm import basic_tokens
    cnt = Counter()

    def add(t):
        for w in basic_tokens(t or ""):
            if w in toks:
                cnt[w] += 1
    for p in paths:
        p = Path(p)
        if p.suffix == ".json":
            d = json.loads(p.read_text(encoding="utf-8"))
            for x in (d if isinstance(d, list) else d.values()):
                add(x.get("text") if isinstance(x, dict) else str(x))
        elif p.suffix == ".jsonl":
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    x = json.loads(line)
                    add(x.get("text") or re.sub(r"</?en>", "", x.get("text_tagged", "")))
        else:
            for line in open(p, encoding="utf-8"):
                add(line.split(" ", 1)[1] if " " in line.strip() else "")
    return cnt


def build_abbrev(args, cat_keys, g10k):
    entries, dropped = OrderedDict(), OrderedDict()
    gens = [from_tickers(Path(args.src) / "vietcap_symbols.json", set(args.boards)),
            from_banks(Path(args.src) / "wiki_Danh_sách_ngân_hàng_tại_Việt_Nam.txt"),
            from_provinces(Path(args.src) / "wiki_tinh.txt"),
            from_wiktionary(Path(args.src) / "wikt_viettat_pages.json")]
    for gen in gens:
        for tok, group, full, src in gen:
            if is_vietnamese_token(tok):
                continue
            reason = ("trùng catalog (EN)" if tok in cat_keys else
                      "trùng từ tiếng Anh phổ biến (google-10000)" if tok in g10k else "")
            book = dropped if reason else entries
            if tok in book:
                e = book[tok]
                if group not in e["group"]:
                    e["group"].append(group)
                e["src"].append(src)
                e["full"].append(full)
            else:
                book[tok] = {"group": [group], "full": [full], "src": [src], "reason": reason}
    # quyết định tay (configs/vi_abbrev_project.tsv): add thắng luật google-10000 / âm tiết, KHÔNG thắng luật catalog
    added = set()
    if args.project_abbrev and Path(args.project_abbrev).exists():
        with open(args.project_abbrev, encoding="utf-8") as f:
            for p in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
                tok = nfc(p["abbr"].strip().lower())
                e = entries.pop(tok, None) or dropped.pop(tok, None) or {"group": [], "full": [], "src": [], "reason": ""}
                if p["action"] == "exclude":
                    e["reason"] = f"loại tay: {p['source']}"
                    dropped[tok] = e
                    continue
                added.add(tok)
                for k, v in (("group", p["group"]), ("full", p["full"]), ("src", p["source"])):
                    if v and v not in e[k]:
                        e[k].insert(0, v)
                e["reason"] = "trùng catalog (EN)" if tok in cat_keys else ""
                (dropped if e["reason"] else entries)[tok] = e
    if args.nonfin_text:
        only_stock = {k for k, e in entries.items() if e["group"] == ["mã chứng khoán"] and k not in added}
        cnt = nonfin_counts(args.nonfin_text, only_stock)
        nf = [(k, cnt[k]) for k in sorted(only_stock, key=lambda k: -cnt[k]) if cnt[k] >= args.nonfin_min]
        for k, c in nf:
            e = entries.pop(k)
            e["reason"] = f"mã chứng khoán gặp {c} lần trong text không phải tài chính (>= {args.nonfin_min})"
            dropped[k] = e
        print(f"  luật text không phải tài chính: {len(nf)} mã bị loại: " + ", ".join(f"{k} ({c})" for k, c in nf))
    cat_drops = sorted(k for k, e in dropped.items() if e["reason"] == "trùng catalog (EN)")
    print(f"  luật catalog: {len(cat_drops)} mục vi_abbrev trùng term EN trong catalog bị loại: {', '.join(cat_drops)}")
    hdr = ["# vi_abbrev.txt — viết tắt / mã thường đọc kiểu Việt: auto_tag KHÔNG gắn <en> (trừ khi có trong catalog).",
           "# TRẠNG THÁI: NHÁP, chờ duyệt, chưa khoá phiên bản. Sinh bởi tools/build_wordlists.py.",
           "# Quy ước gắn <en> theo cách đọc: SCHEMA.md. Viết tắt bị đọc kiểu Anh trong một câu cụ thể vẫn được",
           "# người duyệt gắn <en> trong câu đó.",
           "# Cột (TAB): viết tắt | nhóm | dạng đầy đủ | nguồn. Dòng '#-' = ứng viên bị loại tự động (lý do ở cột cuối);",
           "# bỏ '#-' để bật lại, thêm '#' đầu dòng để tắt.",
           f"# Số mục: {len(entries)} dùng, {len(dropped)} bị loại.", "abbr\tgroup\tfull\tsource"]

    def row(k, e, pre=""):
        full = " | ".join(OrderedDict.fromkeys(f[:60] for f in e["full"]))
        return f"{pre}{k}\t{', '.join(e['group'])}\t{full}\t" \
               f"{'; '.join(OrderedDict.fromkeys(e['src']))}" + (f"\tLOẠI: {e['reason']}" if pre else "")
    out = hdr + [row(k, e) for k, e in entries.items()] + ["", "# ---- ứng viên bị loại tự động"] + \
        [row(k, e, "#-") for k, e in dropped.items()]
    Path(args.out_abbrev).write_text("\n".join(out) + "\n", encoding="utf-8")
    by = Counter(g for e in entries.values() for g in e["group"])
    print(f"Đã ghi {args.out_abbrev}: {len(entries)} mục ({dict(by)}), {len(dropped)} bị loại")


# ------------------------------------------------------------------ vi_loanwords
def build_loan(args, cat_keys, g10k, exclude):
    src = Path(args.src)
    viet = read_lines(src / "Viet74K.txt")
    viet_lc = {w.lower() for w in viet}
    en = {l.lower() for l in read_lines(src / "words_alpha.txt")}
    rows, dropped = OrderedDict(), OrderedDict()
    for w in viet:
        if " " in w or "-" in w or not w[0].islower() or not re.fullmatch(r"[a-z]+", w) or is_vietnamese_token(w):
            continue
        if w in en or w in cat_keys:
            continue
        reason = exclude.get(w) or ("trùng từ tiếng Anh phổ biến (google-10000)" if w in g10k else "")
        (dropped if reason else rows)[w] = ("vi_spelling", reason or "áp dụng",
                                            "Viet74K, không có trong words_alpha (chính tả Việt)")
    surf = Counter()
    if args.vimedcss_meta:
        for f in sorted(Path(args.vimedcss_meta).glob("*_set.csv")):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                terms = [t.strip() for t in (r.get("cs_terms_list") or "").split(";") if t.strip()]
                # chỉ thuật ngữ do dataset gắn (cs_terms_list), không lấy span heuristic (video, ok, top...)
                tg = auto_tag(r["segment_text"], None, terms, vi_words=set(), heuristic=False, term_dec={})
                for s in tg["spans"]:
                    surf[s] += 1
    for s, c in surf.most_common():
        if s in viet_lc and s not in rows and s not in dropped and s not in cat_keys and all(w in en for w in s.split()):
            (dropped if s in exclude else rows)[s] = (
                "same_surface", exclude.get(s) or "không EN ở mọi bộ (decisions §12)",
                f"Viet74K ∩ words_alpha ∩ thuật ngữ ViMedCSS ({c} lần)")
    n_vs = sum(v[0] == "vi_spelling" for v in rows.values())
    hdr = ["# vi_loanwords.txt — từ mượn có trong từ điển tiếng Việt: auto_tag KHÔNG gắn <en> (trừ khi có trong catalog).",
           "# TRẠNG THÁI: NHÁP, chờ duyệt, chưa khoá phiên bản. Sinh bởi tools/build_wordlists.py.",
           "# kind = vi_spelling: chính tả Việt khác chính tả Anh (axit, canxi) -> không EN theo Spec; áp cho MỌI dataset.",
           "# kind = same_surface: trùng mặt chữ tiếng Anh (vitamin, protein) -> KHÔNG EN ở MỌI bộ (decisions §12).",
           "#   Ngoại lệ theo từng thuật ngữ: configs/term_decisions.tsv (decision = en).",
           "# Muốn coi một từ là EN: xoá dòng hoặc thêm '#' đầu dòng. Dòng '#-' = bị loại (lý do ở cột status).",
           f"# Số mục: {n_vs} vi_spelling, {len(rows) - n_vs} same_surface, {len(dropped)} bị loại.",
           "word\tkind\tstatus\tsource"]
    order = sorted(rows.items(), key=lambda kv: (kv[1][0] != "same_surface", kv[0]))
    body = [f"{w}\t{k}\t{st}\t{so}" for w, (k, st, so) in order]
    body += ["", "# ---- bị loại (trùng từ Anh phổ biến hoặc --exclude)"] + \
        [f"#-{w}\t{k}\tLOẠI: {st}\t{so}" for w, (k, st, so) in dropped.items()]
    Path(args.out_loan).write_text("\n".join(hdr + body) + "\n", encoding="utf-8")
    print(f"Đã ghi {args.out_loan}: {len(rows)} mục, {len(dropped)} bị loại ({', '.join(dropped)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="thư mục chứa các file nguồn đã tải")
    ap.add_argument("--catalog", help="catalog EN: mục trùng catalog không vào danh sách")
    ap.add_argument("--boards", nargs="+", default=["HSX", "HNX"])
    ap.add_argument("--project-abbrev", default=str(Path(__file__).resolve().parents[1] / "configs/vi_abbrev_project.tsv"),
                    help="quyết định tay cho vi_abbrev (add / exclude); '' = bỏ qua")
    ap.add_argument("--nonfin-text", nargs="*", default=[],
                    help="text không phải tài chính (VSS train.json/dev.json, VIVOS prompts) cho luật loại mã chứng khoán")
    ap.add_argument("--nonfin-min", type=int, default=20)
    ap.add_argument("--vimedcss-meta", help="thư mục *_set.csv của ViMedCSS (cho same_surface)")
    ap.add_argument("--out-abbrev")
    ap.add_argument("--out-loan")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="TỪ:LÝ_DO",
                    help="loại tay một mục khỏi vi_loanwords, ghi lý do vào file (vd 'iot:trùng IoT trong vi-asr-tech')")
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    cat_keys = {" ".join(k) for k in cat.strict_map} if cat and cat.strict_map else set()
    g10k = set(l.lower() for l in read_lines(Path(args.src) / "google-10000-english-no-swears.txt"))
    if args.out_abbrev:
        build_abbrev(args, cat_keys, g10k)
    if args.out_loan:
        exclude = {}
        for x in args.exclude:
            w, _, why = x.partition(":")
            exclude[nfc(w.strip().lower())] = f"loại tay: {why.strip() or 'không ghi lý do'}"
        build_loan(args, cat_keys, g10k, exclude)


if __name__ == "__main__":
    main()
