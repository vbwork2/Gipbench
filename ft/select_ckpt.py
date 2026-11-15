#!/usr/bin/env python3
"""Chọn checkpoint / alpha WiSE-FT trên DEV (không dùng bench): CS-WER dev CS thấp nhất, với điều kiện WER dev VI tăng
<= --max-vi-delta (điểm tuyệt đối) so với model gốc.

  python ft/select_ckpt.py --catalog data/bench_v0_draft/catalog_v0_draft.tsv \
      --dev-cs data/e3_v1/dev_cs.jsonl --dev-vi data/e3_v1/dev_vi.jsonl \
      --base results/e3_v1_dev/base --cand results/e3_v1_dev/ck4000 results/e3_v1_dev/wise_0.5 ... \
      --out results/e3_v1_dev/select

Mỗi thư mục (base và ứng viên) cần dev_cs/hyps.jsonl và dev_vi/hyps.jsonl (decode.py --system dev_cs / dev_vi).
Reference: text_tagged nếu có, không thì text (dev VI). Ghi <out>.md (bảng) và <out>.json (ứng viên được chọn).
Không ứng viên nào đạt luật thì chọn model gốc và ghi rõ.
"""
import argparse
import json
import sys
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.manifest import read_jsonl  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402
from score import agg, score_utt  # noqa: E402


def evaluate(rows, hyp_path, cat):
    hyps = {h["id"]: h.get("hyp", "") for h in read_jsonl(str(hyp_path))}
    recs = []
    for r in rows:
        r = dict(r)
        r.setdefault("text_tagged", r.get("text", ""))
        recs.append(score_utt(r, hyps.get(r["id"], ""), cat))
    return agg(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--dev-cs", required=True)
    ap.add_argument("--dev-vi", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--max-vi-delta", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    cs_rows, vi_rows = read_jsonl(args.dev_cs), read_jsonl(args.dev_vi)
    res = {}
    for d in [args.base] + args.cand:
        p = Path(d)
        if not (p / "dev_cs/hyps.jsonl").exists() or not (p / "dev_vi/hyps.jsonl").exists():
            print(f"BỎ {p.name}: thiếu hyps dev")
            continue
        cs, vi = evaluate(cs_rows, p / "dev_cs/hyps.jsonl", cat), evaluate(vi_rows, p / "dev_vi/hyps.jsonl", cat)
        res[p.name] = {"cs_wer": cs["cs_wer"], "wer_cs": cs["wer"], "n_wer_cs": cs["n_wer"], "term_recall": cs["term_recall"],
                       "wer_vi": vi["wer"]}
    base = res[Path(args.base).name]
    rows = []
    for name, m in res.items():
        dv = m["wer_vi"] - base["wer_vi"]
        ok = name == Path(args.base).name or dv <= args.max_vi_delta
        rows.append((name, m, dv, ok))
    elig = [(n, m) for n, m, dv, ok in rows if ok and n != Path(args.base).name]
    pick = min(elig, key=lambda x: (x[1]["cs_wer"], x[1]["wer_vi"]))[0] if elig else Path(args.base).name
    L = ["| Ứng viên | CS-WER dev CS | Δ | WER dev CS | N-WER dev CS | Term recall dev CS | WER dev VI | Δ VI | Đạt luật |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, m, dv, ok in rows:
        L.append(f"| {'**' + n + '**' if n == pick else n} | {m['cs_wer']:.2f} | {m['cs_wer'] - base['cs_wer']:+.2f} | "
                 f"{m['wer_cs']:.2f} | {m['n_wer_cs']:.2f} | {m['term_recall']:.2f} | {m['wer_vi']:.2f} | {dv:+.2f} | "
                 f"{'có' if ok else 'KHÔNG'} |")
    L += ["", f"Luật: WER dev VI tăng <= {args.max_vi_delta} điểm so với {Path(args.base).name}; trong các ứng viên đạt, "
          f"chọn CS-WER dev CS thấp nhất (hoà: WER dev VI thấp hơn). Được chọn: **{pick}**"
          + ("" if elig else " (KHÔNG ứng viên nào đạt luật -> giữ model gốc)")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps({"pick": pick, "results": res}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
