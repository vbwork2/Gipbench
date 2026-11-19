#!/usr/bin/env python3
"""Dev TẠM (interim) cắt từ pool train VSS theo VIDEO, nhãn được người nghe kiểm (TODO Phần 3, phương án chọn 18/11).

Không có nguồn CS tự nhiên ngoài y khoa có id người nói (2e), nên dev_v2 đạt chuẩn chưa dựng được. Dev tạm này chỉ để chạy
smoke test / X0 và chọn checkpoint; KHÔNG đại diện 4 domain mục tiêu, đơn vị người nói = video, cùng phân phối với vss_cs
train (giống thiết kế dev_cs@e3_v1 đã báo "tốt lên" sai), và KHÔNG độc lập với checkpoint đã train trên pool cũ
(vss_cs@e3_v1 / @v3, distill@e3_v1 / @v3 / @v4: E3, ck4132...). Không dùng để claim "general".

  # 1. chọn video giữ lại (hash ổn định theo seed) + phiếu nghe (ghi cả danh sách loại cho clean_pool)
  python -m gipbench.dataset.release.holdout_dev select --pool pool/vss_cs@v3 --kind cs --frac 0.3 \
      --reuse data/listen/vss_cs_v1/listen_A_reviewed.csv --out data/listen/dev_interim_cs
  python -m gipbench.dataset.release.holdout_dev select --pool pool/distill@v4 --kind vi --frac 0.2 \
      --out data/listen/dev_interim_vi
  # 2. pool train mới không còn các video đó
  python -m gipbench.dataset.release.clean_pool --pool pool/vss_cs@v3 --exclude data/listen/dev_interim_cs/heldout.jsonl \
      --out data/pools/vss_cs_v4/manifest.jsonl
  # 3. 👤 nghe <out>/listen.csv bằng src/gipbench/dataset/label/review_bench.html (thư mục audio = <out>/audio),
  #    xuất thành <out>/listen_reviewed.csv (ok / fix / drop, như 2d)
  # 4. ghép nhãn đã kiểm -> verified.jsonl, rồi make_dev_split --interim
  python -m gipbench.dataset.release.holdout_dev apply --dir data/listen/dev_interim_cs
  python -m gipbench.dataset.release.make_dev_split --candidates data/listen/dev_interim_cs/verified.jsonl --kind cs \
      --interim --exclude-from <mọi pool train> --release dev/dev_interim_cs@1 --out-dev data/dev/dev_interim_cs.jsonl

Chọn video: xếp video theo sha256(seed, video), nhận lần lượt tới khi số giờ của câu đủ điều kiện (cs: có span <en>; vi:
không span, không token giống tiếng Anh) đạt --frac. Cả video bị bỏ khỏi train (heldout.jsonl = MỌI câu của video đó), nhưng
chỉ câu đủ điều kiện vào phiếu nghe. Câu đã nghe ở phiếu cũ (--reuse) mà nhãn không đổi thì dùng lại quyết định, không nghe lại.
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from gipbench.core.manifest import read_jsonl, write_jsonl
from gipbench.core.scoring import en_like
from gipbench.core.textnorm import Catalog, normalize_ref
from gipbench.dataset.pseudolabel.listen_sample import REVIEW_COLS, load_rows, local_audio, read_csv

_CAT = Catalog()


def video_of(r: Dict) -> str:
    return str(r.get("source") or r.get("speaker") or r["id"])


def eligible(r: Dict, kind: str) -> bool:
    toks, spans = normalize_ref(r.get("text_tagged") or r.get("text") or "", _CAT)
    if not toks:
        return False
    if kind == "cs":
        return bool(spans) and any(t.span < 0 for t in toks)
    return not spans and not any(en_like(t.tok) for t in toks)


def pick_videos(rows: List[Dict], kind: str, frac: float, seed: int):
    """-> (tập video giữ lại, giờ đủ điều kiện giữ lại, tổng giờ đủ điều kiện). Chỉ phụ thuộc (seed, tên video)."""
    hours = defaultdict(float)
    for r in rows:
        if eligible(r, kind):
            hours[video_of(r)] += float(r.get("duration") or 0) / 3600
    total = sum(hours.values())
    order = sorted({video_of(r) for r in rows},
                   key=lambda v: hashlib.sha256(f"{seed}\x1f{v}".encode("utf-8")).hexdigest())
    picked, got = set(), 0.0
    for v in order:
        if got >= frac * total:
            break
        if hours.get(v):
            picked.add(v)
            got += hours[v]
    return picked, got, total


def cmd_select(args):
    rows = load_rows(args.pool)
    picked, got, total = pick_videos(rows, args.kind, args.frac, args.seed)
    held = [r for r in rows if video_of(r) in picked]
    cand = [r for r in held if eligible(r, args.kind)]
    reuse = {}
    for p in args.reuse or []:
        for x in read_csv(Path(p)):
            if (x.get("decision") or "").strip().lower() in ("ok", "fix", "drop"):
                reuse[x["id"]] = x
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    abs_rows = lambda rs: (dict(r, audio=str(local_audio(r) or r.get("audio"))) for r in rs)  # noqa: E731
    write_jsonl(str(out / "heldout.jsonl"), abs_rows(held))
    write_jsonl(str(out / "candidates.jsonl"), abs_rows(cand))
    sheet, reused, missing = [], [], []
    for r in cand:
        old = reuse.get(r["id"])
        if old is not None and (old.get("text_tagged") or "") == (r.get("text_tagged") or ""):
            reused.append({c: old.get(c, "") for c in REVIEW_COLS})
            continue
        src = local_audio(r)
        if src is None:
            missing.append(r["id"])
            continue
        dst = out / "audio" / f"{r['id']}{src.suffix}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        sheet.append({"id": r["id"], "split": "listen", "audio": f"audio/{dst.name}", "duration": r.get("duration", ""),
                      "text_tagged": r.get("text_tagged", ""), "text_tagged_fixed": r.get("text_tagged", ""),
                      "region": r.get("region", "unk"), "gender": r.get("gender", "u")})
    if missing:
        raise SystemExit(f"Thiếu audio local cho {len(missing)} câu, vd {missing[:3]}")
    for name, rs in (("listen.csv", sheet), ("reused_reviews.csv", reused)):
        with open(out / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
            w.writeheader()
            for x in rs:
                w.writerow({c: x.get(c, "") for c in REVIEW_COLS})
    meta = {"pool": args.pool, "kind": args.kind, "frac": args.frac, "seed": args.seed,
            "videos_total": len({video_of(r) for r in rows}), "videos_heldout": len(picked),
            "rows_heldout": len(held), "hours_heldout": round(sum(float(r.get("duration") or 0) for r in held) / 3600, 3),
            "eligible_hours_total": round(total, 3), "eligible_hours_heldout": round(got, 3),
            "candidates": len(cand), "to_listen": len(sheet), "reused": len(reused), "reuse_files": args.reuse or [],
            "heldout_videos": sorted(picked)}
    (out / "holdout_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "heldout_videos"}, ensure_ascii=False, indent=1))
    print(f"Phiếu nghe: {out / 'listen.csv'} ({len(sheet)} câu; {len(reused)} câu dùng lại quyết định cũ). "
          f"Loại khỏi train: clean_pool --exclude {out / 'heldout.jsonl'}")


AUTO_NOTE = "auto:asr_agree"


def cmd_triage(args):
    """Bỏ qua nghe câu mà nhãn khớp TỪNG TOKEN (basic_tokens) với mọi hyp ASR khác họ. Câu khớp nhận luôn (auto_reviews.csv),
    trừ --audit câu ngẫu nhiên vẫn để lại trong phiếu (mù) để đo tỉ lệ sai của luật. Phiếu đầy đủ cũ cất ở listen_full.csv."""
    from gipbench.core.textnorm import basic_tokens
    d = Path(args.dir)
    full = d / "listen_full.csv"
    if not full.exists():
        shutil.copyfile(d / "listen.csv", full)
    sheet = read_csv(full)
    hyps = []
    for p in args.hyps:
        hyps.append({r["id"]: r.get("hyp") for r in read_jsonl(str(p)) if r.get("hyp") is not None})
    missing = [x["id"] for x in sheet if any(x["id"] not in h for h in hyps)]
    if missing:
        raise SystemExit(f"Thiếu hyp cho {len(missing)} câu (vd {missing[:3]}); chạy xong ASR rồi triage")
    agree = [x for x in sheet if all(basic_tokens(detag(x["text_tagged"])) == basic_tokens(h[x["id"]]) for h in hyps)]
    rnd = __import__("random").Random(args.seed)
    audit = set(x["id"] for x in rnd.sample(agree, min(args.audit, len(agree))))
    auto = [x for x in agree if x["id"] not in audit]
    auto_ids = {x["id"] for x in auto}
    keep = [x for x in sheet if x["id"] not in auto_ids]
    for name, rs, extra in (("listen.csv", keep, {}), ("auto_reviews.csv", auto, {"decision": "ok", "note": AUTO_NOTE})):
        with open(d / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
            w.writeheader()
            for x in rs:
                w.writerow({c: extra.get(c, x.get(c, "")) for c in REVIEW_COLS})
    meta = {"hyps": [str(p) for p in args.hyps], "n_sheet": len(sheet), "n_agree": len(agree), "n_auto": len(auto),
            "audit_ids": sorted(audit), "n_to_listen": len(keep), "seed": args.seed}
    (d / "triage_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(agree)} / {len(sheet)} câu khớp mọi hyp; nhận luôn {len(auto)}, để lại {len(audit)} câu kiểm mù. "
          f"Phiếu còn {len(keep)} câu: {d / 'listen.csv'}")


def align(a: List[str], b: List[str]):
    """Căn hai dãy token (Levenshtein, có truy vết) -> danh sách (token a | None, token b | None)."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append((a[i - 1], None))
            i -= 1
        else:
            out.append((None, b[j - 1]))
            j -= 1
    return out[::-1]


def hint_for(label: str, hyps: Dict[str, str]) -> str:
    """Dòng gợi ý cho trang duyệt (cột `hint`, ngăn bằng ||): nhãn với từ mà ít nhất một ASR khác đánh dấu ⟦ ⟧, rồi từng
    bản ASR với từ lệch so với nhãn trong ⟦ ⟧ (⟦∅⟧ = ASR thiếu từ của nhãn)."""
    from gipbench.core.textnorm import basic_tokens
    lab = basic_tokens(detag(label))
    bad = [False] * len(lab)
    lines = []
    for name, hyp in hyps.items():
        pairs, k, toks, nerr = align(lab, basic_tokens(hyp)), 0, [], 0
        for x, y in pairs:
            if x is not None and y is not None and x == y:
                toks.append(y)
            else:
                nerr += 1
                if x is not None or k < len(lab):  # ASR chèn từ: đánh dấu từ nhãn ngay sau chỗ chèn
                    bad[min(k, len(lab) - 1)] = True
                toks.append(f"⟦{y}⟧" if y is not None else "⟦∅⟧")
            k += x is not None
        lines.append(f"{name} (lệch {nerr}): " + " ".join(toks))
    if not any(bad):
        return " || ".join(["mọi ASR khớp nhãn"] + lines)
    return " || ".join([f"nghe kỹ ({sum(bad)} từ): " + " ".join(f"⟦{t}⟧" if b else t for t, b in zip(lab, bad))] + lines)


def cmd_hint(args):
    d = Path(args.dir)
    hyps = {}
    for spec in args.hyps:
        name, _, p = spec.partition("=")
        hyps[name] = {r["id"]: r.get("hyp") for r in read_jsonl(p) if r.get("hyp") is not None}
    sheet = read_csv(d / args.sheet)
    n = 0
    for x in sheet:
        hs = {k: v[x["id"]] for k, v in hyps.items() if x["id"] in v}
        x["hint"] = hint_for(x["text_tagged"], hs) if hs else ""
        n += bool(hs)
    with open(d / args.sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS + ["hint"])
        w.writeheader()
        for x in sheet:
            w.writerow({c: x.get(c, "") for c in REVIEW_COLS + ["hint"]})
    print(f"Gợi ý ASR cho {n} / {len(sheet)} câu -> {d / args.sheet} (cột hint)")


def detag(text_tagged: str) -> str:
    return " ".join(text_tagged.replace("<en>", " ").replace("</en>", " ").split())


def cmd_apply(args):
    d = Path(args.dir)
    cand = {r["id"]: r for r in read_jsonl(str(d / "candidates.jsonl"))}
    reviews = {}
    for name in [args.review, "reused_reviews.csv", "auto_reviews.csv"]:
        p = d / name
        if p.exists():
            for x in read_csv(p):
                reviews.setdefault(x["id"], x)
    tri = d / "triage_meta.json"
    audit = json.loads(tri.read_text(encoding="utf-8"))["audit_ids"] if tri.exists() else []
    todo = [i for i in cand if (reviews.get(i, {}).get("decision") or "").strip().lower() not in ("ok", "fix", "drop")]
    if todo:
        raise SystemExit(f"Còn {len(todo)} / {len(cand)} câu chưa nghe (vd {todo[:3]}); nghe hết rồi chạy lại")
    out, n = [], defaultdict(int)
    for i, r in cand.items():
        x = reviews[i]
        dec = x["decision"].strip().lower()
        n[dec] += 1
        if dec == "drop":
            continue
        fixed = (x.get("text_tagged_fixed") or "").strip() if dec == "fix" else ""
        tt = fixed or r.get("text_tagged") or ""
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        how = "auto:asr_agree" if (x.get("note") or "").strip() == AUTO_NOTE else "human:listen_review"
        row.update(text_tagged=tt, text=detag(tt), label_source=f"{how}(from {r.get('label_source', '?')})",
                   label_review=dec, text_tagged_pseudo=r.get("text_tagged", ""))
        out.append(row)
    write_jsonl(str(d / "verified.jsonl"), out)
    print(f"{len(out)} câu -> {d / 'verified.jsonl'} (ok {n['ok']}, fix {n['fix']}, drop {n['drop']})")
    if audit:
        from gipbench.dataset.pseudolabel.listen_sample import wilson
        bad = sum((reviews[i]["decision"] or "").strip().lower() != "ok" for i in audit if i in reviews)
        p, lo, hi = wilson(bad, len(audit))
        res = {"audit_n": len(audit), "audit_not_ok": bad, "rate_ci95": [round(p, 4), round(lo, 4), round(hi, 4)]}
        (d / "triage_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Kiểm mù luật nhận tự động: {bad} / {len(audit)} câu không ok, CI95 [{100 * lo:.1f}, {100 * hi:.1f}]%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--pool", required=True, help="release hoặc manifest train để cắt video ra")
    s.add_argument("--kind", choices=["cs", "vi"], required=True)
    s.add_argument("--frac", type=float, default=0.3, help="tỉ lệ giờ (câu đủ điều kiện) giữ lại làm dev")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--reuse", nargs="*", help="phiếu đã nghe (review_bench CSV) để dùng lại quyết định")
    s.add_argument("--out", required=True)
    t = sub.add_parser("triage")
    t.add_argument("--dir", required=True)
    t.add_argument("--hyps", nargs="+", required=True, help="hyps.jsonl của các ASR khác họ ({id, hyp})")
    t.add_argument("--audit", type=int, default=20, help="số câu khớp vẫn để lại trong phiếu (mù) để đo luật")
    t.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("hint")
    h.add_argument("--dir", required=True)
    h.add_argument("--hyps", nargs="+", required=True, help="tên=đường_dẫn hyps.jsonl, vd qwen3=hyps_qwen3.jsonl")
    h.add_argument("--sheet", default="listen.csv")
    a = sub.add_parser("apply")
    a.add_argument("--dir", required=True)
    a.add_argument("--review", default="listen_reviewed.csv")
    args = ap.parse_args(argv)
    {"select": cmd_select, "triage": cmd_triage, "hint": cmd_hint, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
