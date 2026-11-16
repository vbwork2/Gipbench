"""Kiểm nhiễm bẩn test -> train / dev / hotwords / text LM (bổ sung cho core.leak, không thay thế).

Mỗi kiểm tra trả về danh sách phát hiện (finding) có `kind`, `severity` và ví dụ. Mức độ:
  critical : chắc chắn rò rỉ (trùng id, trùng sha256 audio, trùng text sau chuẩn hoá, trùng speaker / source / session,
             thuật ngữ FT-unseen có trong text train / TTS / LM / hotwords, hotwords chứa thuật ngữ lấy từ test)
  review   : nghi ngờ, cần người xem (text gần trùng theo Jaccard n-gram ký tự >= ngưỡng)
Gần trùng là heuristic: KHÔNG chứng minh được là sạch hoàn toàn; trùng âm học (cùng audio cắt khác) không kiểm ở đây.
"""
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from gipbench.core.normprofiles import canon_v1
from gipbench.core.textnorm import basic_tokens, plain_text


def sha256_file(path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def dup_key(text: str) -> str:
    """Khoá so trùng text: bỏ tag, chuẩn chính tả v1, token hoá như bench (số -> chữ, bỏ dấu câu)."""
    return " ".join(basic_tokens(canon_v1(plain_text(text or ""))))


def row_text(r: Dict) -> str:
    return r.get("text_tagged") or r.get("text") or ""


def shingles(s: str, n: int = 5) -> set:
    s = f" {s} "
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def _overlap(kind: str, test: Sequence[Dict], train: Sequence[Dict], field: str, max_examples: int) -> List[Dict]:
    tv = defaultdict(list)
    for r in test:
        if r.get(field):
            tv[str(r[field])].append(r["id"])
    hits = defaultdict(list)
    for r in train:
        v = r.get(field)
        if v and str(v) in tv:
            hits[str(v)].append(r.get("id"))
    if not hits:
        return []
    ex = [{"value": v, "test_ids": tv[v][:3], "train_ids": ids[:3]} for v, ids in sorted(hits.items())[:max_examples]]
    return [{"kind": kind, "severity": "critical", "count": len(hits),
             "test_rows": sum(len(tv[v]) for v in hits), "examples": ex}]


def near_duplicates(test: Sequence[Dict], train: Sequence[Dict], threshold: float = 0.8, n: int = 5,
                    min_chars: int = 20, stop_df: int = 2000) -> List[Dict]:
    """Cặp (test, train) có Jaccard shingle ký tự >= threshold nhưng KHÔNG trùng hẳn. Câu ngắn (< min_chars ký tự sau
    chuẩn hoá) bỏ qua (dễ trùng ngẫu nhiên). Shingle xuất hiện trong > stop_df câu train bị bỏ khỏi chỉ mục."""
    tr = []
    for r in train:
        k = dup_key(row_text(r))
        if len(k) >= min_chars:
            tr.append((r.get("id"), k, shingles(k, n)))
    df = Counter(s for _, _, sh in tr for s in sh)
    idx = defaultdict(list)
    for j, (_, _, sh) in enumerate(tr):
        for s in sh:
            if df[s] <= stop_df:
                idx[s].append(j)
    out = []
    for r in test:
        k = dup_key(row_text(r))
        if len(k) < min_chars:
            continue
        sh = shingles(k, n)
        cand = Counter(j for s in sh for j in idx.get(s, ()))
        best = None
        for j, c in cand.most_common(20):
            tid, tk, tsh = tr[j]
            if tk == k:
                continue
            jac = len(sh & tsh) / len(sh | tsh)
            if jac >= threshold and (best is None or jac > best[0]):
                best = (jac, tid, tk)
        if best:
            out.append({"test_id": r["id"], "train_id": best[1], "jaccard": round(best[0], 3),
                        "test_text": k, "train_text": best[2]})
    return out


def find_terms_in_texts(terms: Iterable[str], texts: Iterable[str]) -> Dict[str, int]:
    """Đếm số dòng text chứa từng thuật ngữ (khớp chuỗi token liên tiếp sau dup_key)."""
    tt = {t: tuple(dup_key(t).split()) for t in terms}
    tt = {t: k for t, k in tt.items() if k}
    first = defaultdict(list)
    for t, k in tt.items():
        first[k[0]].append((t, k))
    cnt: Dict[str, int] = Counter()
    for text in texts:
        toks = dup_key(text).split()
        seen = set()
        for i, tok in enumerate(toks):
            for t, k in first.get(tok, ()):
                if t not in seen and tuple(toks[i:i + len(k)]) == k:
                    seen.add(t)
        for t in seen:
            cnt[t] += 1
    return dict(cnt)


def audit(test: Sequence[Dict], train: Sequence[Dict], *, check_audio_sha: bool = True,
          fields: Sequence[str] = ("speaker", "source", "session"), near_threshold: float = 0.8,
          reserved_terms: Sequence[str] = (), runtime_vocab: Sequence[str] = (), test_terms: Sequence[str] = (),
          max_examples: int = 10) -> Dict:
    """Kiểm một tập test với một (gộp nhiều) tập train / dev.

    reserved_terms: thuật ngữ FT-unseen (không được xuất hiện trong text train, TTS, LM, hotwords).
    runtime_vocab : danh sách hotwords / từ vựng ngữ cảnh dùng lúc decode (kiểm với reserved_terms và test_terms).
    test_terms    : thuật ngữ trích từ reference của test (vd span <en>); có trong runtime_vocab = biasing có thông tin test.
    Trả về {"findings": [...], "critical": số, "review": số, "stats": {...}}. audio sha256 lấy từ trường `audio_sha256`
    nếu có, không thì đọc file `_audio_path` (bỏ qua file không tồn tại, đếm vào stats).
    """
    f: List[Dict] = []
    f += _overlap("id_overlap", test, train, "id", max_examples)
    for fld in fields:
        f += _overlap(f"{fld}_overlap", test, train, fld, max_examples)
    stats = {"n_test": len(test), "n_train": len(train), "audio_unreadable": 0}
    if check_audio_sha:
        def sha(r):
            if r.get("audio_sha256"):
                return r["audio_sha256"]
            p = r.get("_audio_path")
            if p and Path(p).exists():
                return sha256_file(p)
            stats["audio_unreadable"] += 1
            return None
        ts = [dict(r, audio_sha256=sha(r)) for r in test]
        trs = [dict(r, audio_sha256=sha(r)) for r in train]
        f += _overlap("audio_sha256_overlap", ts, trs, "audio_sha256", max_examples)
    tk = defaultdict(list)
    for r in test:
        k = dup_key(row_text(r))
        if k:
            tk[k].append(r["id"])
    ex = defaultdict(list)
    for r in train:
        k = dup_key(row_text(r))
        if k in tk:
            ex[k].append(r.get("id"))
    if ex:
        f.append({"kind": "text_exact_overlap", "severity": "critical", "count": len(ex),
                  "test_rows": sum(len(tk[k]) for k in ex),
                  "examples": [{"text": k, "test_ids": tk[k][:3], "train_ids": v[:3]} for k, v in sorted(ex.items())[:max_examples]]})
    nd = near_duplicates(test, train, near_threshold)
    if nd:
        f.append({"kind": "text_near_duplicate", "severity": "review", "count": len(nd), "test_rows": len(nd),
                  "threshold": near_threshold, "examples": nd[:max_examples]})
    if reserved_terms:
        hit = find_terms_in_texts(reserved_terms, (row_text(r) for r in train))
        if hit:
            f.append({"kind": "reserved_term_in_train", "severity": "critical", "count": len(hit),
                      "examples": [{"term": t, "train_rows": c} for t, c in sorted(hit.items(), key=lambda x: -x[1])[:max_examples]]})
    if runtime_vocab:
        rv = {dup_key(v) for v in runtime_vocab} - {""}
        for kind, terms in (("reserved_term_in_runtime_vocab", reserved_terms), ("test_term_in_runtime_vocab", test_terms)):
            hit = sorted({dup_key(t) for t in terms} & rv)
            if hit:
                f.append({"kind": kind, "severity": "critical", "count": len(hit),
                          "share_of_terms": round(len(hit) / max(1, len({dup_key(t) for t in terms} - {""})), 4),
                          "examples": hit[:max_examples]})
    return {"findings": f, "critical": sum(x["severity"] == "critical" for x in f),
            "review": sum(x["severity"] == "review" for x in f), "stats": stats}


def test_terms_from_rows(rows: Iterable[Dict], field: str = "text_tagged") -> List[str]:
    """Thuật ngữ tiếng Anh lấy từ span <en> của reference (dạng dup_key, không trùng)."""
    from gipbench.core.textnorm import parse_tagged
    out = set()
    for r in rows:
        for seg, is_en in parse_tagged(r.get(field) or ""):
            if is_en:
                k = dup_key(seg)
                if k:
                    out.add(k)
    return sorted(out)


def read_lines(path: Optional[str]) -> List[str]:
    if not path:
        return []
    return [l.split("\t")[0].strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]
