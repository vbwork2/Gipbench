#!/usr/bin/env python3
"""Gom thuật ngữ tiếng Anh GÂY NHIỄU cho hotwords: thuật ngữ KHÔNG có trong catalog bench, lấy từ span <en> của manifest train
và từ danh sách từ tuỳ chọn. decode.py --hotwords-distractors trộn chúng vào danh sách hotwords cùng điểm với thuật ngữ thật,
để E1 không còn "biết trước đáp án" (hotwords chỉ gồm đúng thuật ngữ của bench thì kết quả lạc quan).

  python tools/make_distractors.py --catalog data/bench_v0_draft/catalog_v0_draft.tsv \
      --manifest data/e3_v1/train_curated.jsonl --wordlist data/dl/lexicons/google-10000-english-no-swears.txt \
      --out data/e3_v1/distractors.tsv
  python decode.py ... --hotwords-catalog <catalog> --hotwords-distractors data/e3_v1/distractors.tsv --hotwords-distractors-n 500

Loại: thuật ngữ trùng catalog (sau chuẩn hoá, kể cả `variants`), thuật ngữ reserved_b3 (hoặc chứa một thuật ngữ reserved_b3),
cụm có ký tự ngoài chữ Latin không dấu, cụm quá ngắn (< 2 ký tự). Từ của --wordlist (1 từ / dòng, cột đầu của TSV, bỏ dòng '#').
Ra: TSV, mỗi dòng một thuật ngữ: term, tổng, rồi số lần theo từng nguồn (dòng đầu '# ...' là tiêu đề), xếp theo tổng giảm dần.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.manifest import read_jsonl  # noqa: E402
from lib.textnorm import Catalog, basic_tokens, normalize_ref  # noqa: E402

_LATIN = re.compile(r"[a-z' ]+")


def source_names(paths):
    """Tên nguồn = tên file (thêm số thứ tự nếu hai file trùng tên)."""
    names, used = [], Counter(Path(p).name for p in paths)
    for i, p in enumerate(paths):
        n = Path(p).name
        names.append(n if used[n] == 1 else f"{n}#{i}")
    return names


def wordlist_terms(path, cat):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            toks = cat.apply(basic_tokens(line.split("\t")[0]))
            if toks:
                yield tuple(toks)


def manifest_terms(path, cat):
    for r in read_jsonl(path):
        _, spans = normalize_ref(r.get("text_tagged") or "", cat)
        yield from spans


def collect(manifests, wordlists, cat, min_count=1):
    """-> (rows, nguồn, thống kê). rows = [(term, tổng, [số theo nguồn])] xếp theo tổng giảm dần rồi theo term."""
    srcs = source_names(list(manifests) + list(wordlists))
    counts = {}
    stats = Counter()
    reserved = {t for t, v in cat.terms.items() if v.reserved_b3}
    gens = [manifest_terms(p, cat) for p in manifests] + [wordlist_terms(p, cat) for p in wordlists]
    for si, gen in enumerate(gens):
        for term in gen:
            stats["span / từ đọc được"] += 1
            s = " ".join(term)
            if term in reserved or any(t in reserved for t in cat.find_terms(list(term))):
                stats["loại: reserved_b3"] += 1
            elif term in cat.terms:
                stats["loại: trùng catalog"] += 1
            elif not _LATIN.fullmatch(s):
                stats["loại: ký tự ngoài chữ Latin không dấu"] += 1
            elif len(s.replace(" ", "")) < 2:
                stats["loại: quá ngắn"] += 1
            else:
                counts.setdefault(s, [0] * len(srcs))[si] += 1
    rows = [(t, sum(c), c) for t, c in counts.items() if sum(c) >= min_count]
    rows.sort(key=lambda x: (-x[1], x[0]))
    stats["thuật ngữ gây nhiễu"] = len(rows)
    return rows, srcs, stats


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="catalog bench (thuật ngữ của nó không bao giờ là từ gây nhiễu)")
    ap.add_argument("--manifest", nargs="*", default=[], help="manifest train: lấy span <en> của text_tagged")
    ap.add_argument("--wordlist", nargs="*", default=[], help="danh sách từ tiếng Anh, 1 từ / dòng")
    ap.add_argument("--min-count", type=int, default=1, help="chỉ giữ thuật ngữ xuất hiện >= N lần (mặc định 1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not args.manifest and not args.wordlist:
        sys.exit("cần ít nhất một --manifest hoặc --wordlist")
    rows, srcs, stats = collect(args.manifest, args.wordlist, Catalog.load(args.catalog), args.min_count)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# term\ttotal\t" + "\t".join(srcs) + "\n")
        for t, tot, c in rows:
            f.write(f"{t}\t{tot}\t" + "\t".join(map(str, c)) + "\n")
    for k, v in stats.items():
        print(f"{v:8d}  {k}")
    print(f"-> {args.out} ({len(rows)} thuật ngữ; nguồn: {', '.join(srcs)})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
