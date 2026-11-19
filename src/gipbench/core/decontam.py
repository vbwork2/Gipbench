"""Kiểm nhiễm bẩn test -> train / dev / hotwords / text LM (bổ sung cho core.leak, không thay thế).

Mỗi kiểm tra trả về danh sách phát hiện (finding) có `kind`, `severity` và ví dụ. Mức độ:
  critical : chắc chắn rò rỉ (trùng id, trùng sha256 audio, trùng clip gốc, trùng text sau chuẩn hoá, trùng speaker /
             source / session, trùng nguồn chuẩn hoá (video YouTube, người nói CV / VIVOS), thuật ngữ FT-unseen có trong
             text train / TTS / LM / hotwords, hotwords chứa thuật ngữ lấy từ test). `audio_unverifiable` là critical khi
             require_audio_sha (đóng khi lỗi: không kiểm được thì không coi là sạch).
  review   : nghi ngờ, cần người xem (text gần trùng; trùng text trong cùng bộ đọc có chủ ý như VIVOS; cụm hotwords có
             trong text test; tỉ lệ câu test không xác định được nguồn; audio không kiểm được khi không require)
  exempted : finding đã được miễn trừ có ghi chép (lý do, người duyệt, ngày). Vẫn in, không tính vào critical / review.
Gần trùng là heuristic: KHÔNG chứng minh được là sạch hoàn toàn; trùng âm học (cùng audio cắt khác, không cùng mã clip
hay cùng nguồn) không kiểm ở đây.
"""
import hashlib
import re
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


def audio_check(path) -> Dict:
    """Đọc file audio một lần: {"sha256", "defect"}. defect: "missing" (không có file) | "empty" (WAV không có mẫu, vd
    file 44 byte) | "digital_silence" (WAV mọi mẫu bằng 0) | "unchecked" (không phải WAV: chỉ có sha) | None (có tín hiệu).
    File im lặng / rỗng trùng sha với nhau dù khác video (cùng độ dài là đủ), nên không được coi là bằng chứng rò rỉ;
    câu có nhãn mà audio im lặng là dữ liệu hỏng (train dạy model sinh chữ từ im lặng)."""
    import io
    import wave
    p = Path(path) if path else None
    if p is None or not p.exists():
        return {"sha256": None, "defect": "missing"}
    data = p.read_bytes()
    out = {"sha256": hashlib.sha256(data).hexdigest(), "defect": None}
    if p.suffix.lower() != ".wav":
        out["defect"] = "unchecked"
        return out
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            frames = w.readframes(w.getnframes())
            zero = b"\x80" if w.getsampwidth() == 1 else b"\x00"
    except (wave.Error, EOFError):
        out["defect"] = "unchecked"
        return out
    if not frames:
        out["defect"] = "empty"
    elif frames.count(zero) == len(frames):
        out["defect"] = "digital_silence"
    return out


def dup_key(text: str) -> str:
    """Khoá so trùng text: bỏ tag, chuẩn chính tả v1, token hoá như bench (số -> chữ, bỏ dấu câu)."""
    return " ".join(basic_tokens(canon_v1(plain_text(text or ""))))


def row_text(r: Dict) -> str:
    return r.get("text_tagged") or r.get("text") or ""


def shingles(s: str, n: int = 5) -> set:
    s = f" {s} "
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


# --- khoá định danh chuẩn hoá ---
# Các nguồn ghi cùng một video / người nói theo định dạng khác nhau (vd ViMedCSS train `yt_ID`, E0 vimedcss_test URL
# `https://www.youtube.com/watch?v=ID`), nên so chuỗi chính xác bỏ sót. source_key / clip_key đưa về một khoá chung.
# KHÔNG BAO GIỜ dùng id câu làm khoá: bộ không có id người nói (G-Group: speaker = id câu, source = id câu) cho None.
IDENTITY_FIELDS = ("speaker", "source", "upstream_id", "audio")
_YT = re.compile(r"(?:youtube:|(?<![A-Za-z0-9])yt_|[?&]v=|youtu\.be/|youtube\.com/(?:embed|shorts|live)/)([A-Za-z0-9_-]{11})")
_CV = re.compile(r"^(?:cv\d*[_:])([0-9a-f]{16,})$")
_VIVOS = re.compile(r"(VIVOS(?:SPK|DEV)\d+)")
_CLIP = [(re.compile(r"(common_voice_[a-z]{2,3}_\d+)"), "cv_clip"),
         (re.compile(r"(VIVOS(?:SPK|DEV)\d+_R?\d+)"), "vivos_clip")]


def source_key(r: Dict) -> Optional[str]:
    """Khoá nguồn chuẩn hoá của một dòng (video / người nói), hoặc None nếu không xác định được.

    Thứ tự: YouTube ID (từ speaker / source / upstream_id / audio: `youtube:ID`, `yt_ID`, `watch?v=ID`, `youtu.be/ID`)
    -> "yt:ID"; người nói Common Voice (`cv_<hash>`, `cv17_<hash>`, `cv:<hash>`) -> "cv:" + hash[:16]; người nói
    VIVOS -> "vivos:VIVOSSPKnn" / "vivos:VIVOSDEVnn". Còn lại None (không lấy id câu, kể cả khi speaker = id)."""
    vals = [str(r.get(f) or "") for f in IDENTITY_FIELDS]
    for v in vals:
        m = _YT.search(v)
        if m:
            return "yt:" + m.group(1)
    for v in vals[:2]:  # chỉ speaker / source mang id người nói CV
        m = _CV.match(v)
        if m:
            return "cv:" + m.group(1)[:16]
    for v in vals[:2]:
        m = _VIVOS.search(v)
        if m:
            return "vivos:" + m.group(1)
    return None


def clip_key(r: Dict) -> Optional[str]:
    """Mã clip gốc của upstream (duy nhất xuyên phiên bản bộ dữ liệu): Common Voice `common_voice_vi_N`, VIVOS
    `VIVOSSPKnn_Rmmm`. Bắt được cùng một clip đã resample / đổi container (sha256 khác) ở hai manifest khác nhau."""
    for f in ("upstream_id", "id", "audio"):
        v = str(r.get(f) or "")
        for rx, ns in _CLIP:
            m = rx.search(v)
            if m:
                return f"{ns}:{m.group(1)}"
    return None


# họ bộ dữ liệu (để nhận ra trùng text CÓ CHỦ Ý trong cùng bộ đọc, vd VIVOS train / test đọc cùng kịch bản)
_FAMILY_SUFFIX = re.compile(r"(?:[_-](?:vi|train|test|dev|valid|validation|v?\d+))+$")
FAMILY_ALIASES = {"cv": "commonvoice", "common_voice": "commonvoice"}


def dataset_family(r: Dict) -> str:
    s = str(r.get("dataset") or "").split(":")[0].lower()
    s = _FAMILY_SUFFIX.sub("", s)
    s = re.sub(r"^cv\d*$", "cv", s)
    return FAMILY_ALIASES.get(s, s)


def _exempt_match(exemptions: Sequence[Dict], kind: str, value: Optional[str], test_ids: Sequence[str]) -> Optional[Dict]:
    """Miễn trừ khớp: cùng kind, và (value bằng nhau, hoặc test_id thuộc các câu test của giá trị đó, hoặc miễn trừ không
    ghi value / test_id = miễn cả loại)."""
    for e in exemptions:
        if e.get("kind") != kind:
            continue
        if "value" in e and value is not None and str(e["value"]) == str(value):
            return e
        if "test_id" in e and str(e["test_id"]) in set(map(str, test_ids)):
            return e
        if "value" not in e and "test_id" not in e:
            return e
    return None


def _emit(kind: str, severity: str, hits: Dict[str, Dict], max_examples: int, exemptions: Sequence[Dict] = (),
          extra: Optional[Dict] = None) -> List[Dict]:
    """hits: {value: {"test_ids": [...], "train_ids": [...]}} -> finding (và finding `<kind>:exempted` nếu có miễn trừ)."""
    keep, exm = {}, {}
    for v, h in hits.items():
        e = _exempt_match(exemptions, kind, v, h["test_ids"])
        if e:
            exm[v] = dict(h, exemption=e)
        else:
            keep[v] = h
    out = []
    for k, sev, hs in ((kind, severity, keep), (kind + ":exempted", "exempted", exm)):
        if not hs:
            continue
        ex = [dict({"value": v, "test_ids": h["test_ids"][:3], "train_ids": h["train_ids"][:3]},
                   **({"exemption": h["exemption"]} if "exemption" in h else {})) for v, h in sorted(hs.items())[:max_examples]]
        out.append(dict({"kind": k, "severity": sev, "count": len(hs),
                         "test_rows": len({i for h in hs.values() for i in h["test_ids"]}), "examples": ex}, **(extra or {})))
    return out


def _hits(test: Sequence[Dict], train: Sequence[Dict], key) -> Dict[str, Dict]:
    tv = defaultdict(list)
    for r in test:
        v = key(r)
        if v:
            tv[str(v)].append(r["id"])
    hits: Dict[str, Dict] = {}
    for r in train:
        v = key(r)
        if v and str(v) in tv:
            hits.setdefault(str(v), {"test_ids": tv[str(v)], "train_ids": []})["train_ids"].append(r.get("id"))
    return hits


def _overlap(kind: str, test: Sequence[Dict], train: Sequence[Dict], field: str, max_examples: int,
             exemptions: Sequence[Dict] = ()) -> List[Dict]:
    return _emit(kind, "critical", _hits(test, train, lambda r: r.get(field)), max_examples, exemptions)


def near_duplicates(test: Sequence[Dict], train: Sequence[Dict], threshold: float = 0.8, n: int = 5,
                    min_chars: int = 20, stop_df: int = 2000, best_only: bool = True) -> List[Dict]:
    """Cặp (test, train) có Jaccard shingle ký tự >= threshold nhưng KHÔNG trùng hẳn. Câu ngắn (< min_chars ký tự sau
    chuẩn hoá) bỏ qua (dễ trùng ngẫu nhiên). Shingle xuất hiện trong > stop_df câu train bị bỏ khỏi chỉ mục.
    best_only: mỗi câu test chỉ trả cặp có Jaccard cao nhất (báo cáo); False = mọi câu train đạt ngưỡng (để lọc train)."""
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
        found = []
        for j, c in cand.most_common(20):
            tid, tk, tsh = tr[j]
            if tk == k:
                continue
            jac = len(sh & tsh) / len(sh | tsh)
            if jac >= threshold:
                found.append((jac, tid, tk))
        if best_only and found:
            found = [max(found, key=lambda x: x[0])]
        out += [{"test_id": r["id"], "train_id": tid, "jaccard": round(jac, 3), "test_text": k, "train_text": tk}
                for jac, tid, tk in found]
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
          max_examples: int = 10, require_audio_sha: bool = False, exemptions: Sequence[Dict] = (),
          hash_files: bool = True, near: bool = True) -> Dict:
    """Kiểm một tập test với một (gộp nhiều) tập train / dev.

    hash_files    : False = chỉ dùng trường audio_sha256 có sẵn, KHÔNG đọc file audio (nhanh; dòng không có hash thì tính vào
                    audio_unverifiable). near: False = bỏ kiểm text gần trùng (tốn bộ nhớ khi tập lớn).

    reserved_terms: thuật ngữ FT-unseen (không được xuất hiện trong text train, TTS, LM, hotwords).
    runtime_vocab : danh sách hotwords / từ vựng ngữ cảnh dùng lúc decode (kiểm với reserved_terms, test_terms và text test).
    test_terms    : thuật ngữ trích từ reference của test (vd span <en>); có trong runtime_vocab = biasing có thông tin test.
    require_audio_sha: dòng không xác định được sha256 audio -> `audio_unverifiable` critical (mặc định: review).
    exemptions    : [{kind, value | test_id, reason, approved_by, date}] (xem _exempt_match). Finding được miễn ghi thành
                    `<kind>:exempted`, mức `exempted`.
    Trả về {"findings": [...], "critical": số, "review": số, "exempted": số, "stats": {...}}. audio sha256 lấy từ trường
    `audio_sha256` nếu có, không thì đọc file `_audio_path`; không có cả hai thì đếm vào stats và vào audio_unverifiable.
    """
    f: List[Dict] = []
    f += _overlap("id_overlap", test, train, "id", max_examples, exemptions)
    for fld in fields:
        f += _overlap(f"{fld}_overlap", test, train, fld, max_examples, exemptions)
    f += _emit("source_norm_overlap", "critical", _hits(test, train, source_key), max_examples, exemptions)
    f += _emit("clip_overlap", "critical", _hits(test, train, clip_key), max_examples, exemptions)
    unknown = [r for r in test if source_key(r) is None]
    stats = {"n_test": len(test), "n_train": len(train), "audio_sha_checked": check_audio_sha, "audio_unreadable": 0,
             "audio_unreadable_test": 0, "audio_unreadable_train": 0, "identity_unknown_test": len(unknown)}
    if unknown:
        by = Counter(str(r.get("dataset") or "?") for r in unknown)
        f += _emit("identity_unknown", "review", {"*": {"test_ids": [r["id"] for r in unknown], "train_ids": []}},
                   max_examples, exemptions, {"share": len(unknown) / len(test), "by_dataset": dict(by.most_common()),
                                              "note": "không xác định được video / người nói (source_key) cho các câu này: "
                                                      "trùng nguồn chỉ kiểm được bằng so chuỗi speaker / source thô"})
    if check_audio_sha:
        def sha(r, side):
            if r.get("audio_sha256"):
                return r["audio_sha256"]
            p = r.get("_audio_path")
            if hash_files and p and Path(p).exists():
                return sha256_file(p)
            stats["audio_unreadable"] += 1
            stats[f"audio_unreadable_{side}"] += 1
            bad[side].append(r.get("id"))
            return None
        bad = {"test": [], "train": []}
        ts = [dict(r, audio_sha256=sha(r, "test")) for r in test]
        trs = [dict(r, audio_sha256=sha(r, "train")) for r in train]
        f += _overlap("audio_sha256_overlap", ts, trs, "audio_sha256", max_examples, exemptions)
        if bad["test"] or bad["train"]:
            f += _emit("audio_unverifiable", "critical" if require_audio_sha else "review",
                       {"*": {"test_ids": bad["test"], "train_ids": bad["train"]}}, max_examples, exemptions,
                       {"train_rows": len(bad["train"]),
                        "note": "không có audio_sha256 và không đọc được _audio_path: trùng audio KHÔNG kiểm được"})
    # trùng text sau chuẩn hoá; trong cùng họ bộ đọc có chủ ý (domain read, vd VIVOS train / test đọc cùng kịch bản)
    # là trùng từ vựng, không phải nhìn trộm test -> review (cần miễn trừ / báo lát text_unseen)
    tk = defaultdict(list)
    for r in test:
        k = dup_key(row_text(r))
        if k:
            tk[k].append(r)
    ex = defaultdict(list)
    for r in train:
        k = dup_key(row_text(r))
        if k in tk:
            ex[k].append(r)
    crit, read = {}, {}
    for k, trs in ex.items():
        rows = tk[k] + trs
        fam = {dataset_family(r) for r in rows}
        same_read = len(fam) == 1 and "" not in fam and all(str(r.get("domain") or "") == "read" for r in rows)
        (read if same_read else crit)[k] = {"test_ids": [r["id"] for r in tk[k]], "train_ids": [r.get("id") for r in trs]}
    f += _emit("text_exact_overlap", "critical", crit, max_examples, exemptions)
    f += _emit("lexical_overlap_read_corpus", "review", read, max_examples, exemptions,
               {"note": "trùng câu chữ trong cùng bộ đọc có chủ ý (không trùng audio / người nói); báo lát text_unseen"})
    nd = near_duplicates(test, train, near_threshold) if near else []
    if nd:
        f.append({"kind": "text_near_duplicate", "severity": "review", "count": len(nd), "test_rows": len(nd),
                  "threshold": near_threshold, "examples": nd[:max_examples]})
    if reserved_terms:
        hit = find_terms_in_texts(reserved_terms, (row_text(r) for r in train))
        for kind, sev, sel in (("reserved_term_in_train", "critical",
                                {t: c for t, c in hit.items() if not _exempt_match(exemptions, "reserved_term_in_train", t, [])}),
                               ("reserved_term_in_train:exempted", "exempted",
                                {t: c for t, c in hit.items() if _exempt_match(exemptions, "reserved_term_in_train", t, [])})):
            if sel:
                f.append({"kind": kind, "severity": sev, "count": len(sel),
                          "examples": [{"term": t, "train_rows": c} for t, c in sorted(sel.items(), key=lambda x: -x[1])[:max_examples]]})
    if runtime_vocab:
        rv = {dup_key(v) for v in runtime_vocab} - {""}
        flagged = set()
        for kind, terms in (("reserved_term_in_runtime_vocab", reserved_terms), ("test_term_in_runtime_vocab", test_terms)):
            hit = sorted({dup_key(t) for t in terms} & rv)
            flagged |= set(hit)
            if hit:
                f.append({"kind": kind, "severity": "critical", "count": len(hit),
                          "share_of_terms": round(len(hit) / max(1, len({dup_key(t) for t in terms} - {""})), 4),
                          "examples": hit[:max_examples]})
        # cụm hotwords có trong text test (kể cả bộ không có tag <en>, vd G0): hotwords có thể lấy từ test.
        # review vì cụm phổ biến cũng khớp ngẫu nhiên; xếp cụm hiếm (ít câu test chứa) lên đầu.
        hit = find_terms_in_texts(sorted(rv - flagged), (row_text(r) for r in test))
        if hit:
            f.append({"kind": "test_ngram_in_runtime_vocab", "severity": "review", "count": len(hit),
                      "share_of_vocab": round(len(hit) / len(rv), 4),
                      "examples": [{"phrase": t, "test_rows": c} for t, c in sorted(hit.items(), key=lambda x: (x[1], x[0]))[:max_examples]]})
    return {"findings": f, "critical": sum(x["severity"] == "critical" for x in f),
            "review": sum(x["severity"] == "review" for x in f),
            "exempted": sum(x["severity"] == "exempted" for x in f), "stats": stats}


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


EXEMPTION_FIELDS = ("kind", "reason", "approved_by", "date")


def load_exemptions(path: Optional[str]) -> List[Dict]:
    """Đọc yaml miễn trừ: danh sách (hoặc {exemptions: [...]}) các {kind, value | test_id, reason, approved_by, date}.
    Thiếu reason / approved_by / date thì ValueError: miễn trừ không ghi chép không được chấp nhận."""
    if not path:
        return []
    import yaml
    ex = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if isinstance(ex, dict):
        ex = ex.get("exemptions") or []
    for i, e in enumerate(ex):
        miss = [k for k in EXEMPTION_FIELDS if not e.get(k)]
        if miss:
            raise ValueError(f"miễn trừ #{i} trong {path} thiếu {miss} (bắt buộc: {EXEMPTION_FIELDS})")
    return ex


def overlap_counts(test: Sequence[Dict], train: Sequence[Dict]) -> Dict[str, int]:
    """Số câu TEST trùng với `train` theo từng khoá rẻ (id, nguồn chuẩn hoá, clip gốc, text sau chuẩn hoá). Dùng để báo
    cáo từng pool / dev / text đóng góp bao nhiêu (audit() chỉ liệt kê ví dụ)."""
    def n(key):
        h = _hits(test, train, key)
        return len({i for v in h.values() for i in v["test_ids"]})
    return {"id": n(lambda r: r.get("id")), "source_norm": n(source_key), "clip": n(clip_key),
            "text_exact": n(lambda r: dup_key(row_text(r)) or None)}
