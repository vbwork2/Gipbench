#!/usr/bin/env python3
"""Bộ consensus (data/dl/zac: consensus_clips_anon.csv + audio/) -> ứng viên B2 / B5-product. CHỈ ĐƯỢC DÙNG ĐỂ TEST.

  python converters/conv_consensus.py --root data/dl/zac --out data/raw/consensus \
      --catalog data/raw/_proxy/catalog_proxy.tsv [--max-per-term 10] [--spk-embed auto|on|off]

CSV: id (phiên), turn (u1, u2, ..., final), status, consensus_text, listener_texts ("A: ... || B: ..."), share,
numbers_majority, tier, need, n_blind, source (cách có nhãn), reviewers, confirmed_after_prompt. Không có speaker id.
Mỗi (id, turn) là một clip riêng: id clip = cons_<id>_<turn>.

Audio: tự dò luật ghép (id, turn) -> file trong --root (thử các mẫu tên <id>_<turn>, <id>-<turn>, <id>, <turn> trong
thư mục <id>, ... riêng cho turn final và các turn khác; chọn cặp luật ghép được nhiều clip nhất, mỗi file chỉ cho
một clip). Báo số clip thiếu audio / mơ hồ và số file không dùng. File không phải 16 kHz mono được resample (soxr HQ)
+ gộp kênh, ghi WAV PCM16 vào <out>/audio/; file đã đúng định dạng thì manifest trỏ thẳng vào (không copy).

Trạng thái:
  agreed             -> ứng viên. Câu có span <en> (auto_tag, sau vi_abbrev / vi_loanwords) -> B2 (b2_consensus_candidates),
                        câu Việt thuần -> B5 domain product (b5_consensus_product). KHÔNG tách dev.
  disputed / pending -> disputed_candidates.jsonl + phiếu duyệt disputed_candidates_review.csv (cột hint = listener_texts
                        làm gợi ý; text_tagged = consensus_text, trống thì lấy bản của người nghe đầu tiên). Các dòng có
                        review_required = true: build_bench_v0.py chỉ nhận khi decision = ok / fix. Khi fix,
                        split theo text đã sửa (split_by_tag: còn <en> -> B2, không -> B5), áp cho mọi dòng.
  escalate           -> bỏ.
Mọi dòng: speaker = consensus:<id phiên>, source = consensus, domain = product, license = test_only, dataset = consensus.
Text: số -> chữ, auto_tag, rồi chuẩn hoá như normalize_ref (viết thường, bỏ dấu câu; không áp bảng biến thể catalog).
Tên người nước ngoài (--names, configs/en_names.tsv) được gắn <en> và ghi term_category {"<span>": "name"}
(score.py cắt lát theo category); cờ name:<span> để người duyệt xác nhận, maybe_name:<cụm> khi chỉ nghi là tên.
--max-per-term N: mỗi thuật ngữ (span <en> sau catalog) có trong tối đa N câu B2; khi cắt, xoay vòng theo phiên và ưu tiên
ngữ cảnh chưa gặp (như ft/curate_train.py). Câu bị cắt ghi ra b2_consensus_overcap.jsonl (không vào bench).
Phiếu duyệt B2 = mọi câu B2 giữ lại; phiếu B5 = chỉ câu B5 có cờ (câu không cờ build_bench_v0.py nhận thẳng).
leak_block.txt: text chuẩn hoá của MỌI dòng (mọi trạng thái, cả listener_texts) -> danh sách chặn rò rỉ mặc định của
ft/curate_train.py, tools/vss_cs_prefilter.py / rebuild_vss_cs.sh, validate.py (lib/leak.py DEFAULT_BLOCK_FILES).
--spk-embed (tuỳ chọn, CPU): speaker embedding (sherpa-onnx + wespeaker ResNet34, --spk-model) -> gom các phiên có
khả năng cùng người nói, ghi speaker_cluster (CHỈ dùng cho bootstrap CI của score.py) + speaker_clusters.tsv. Ngưỡng
cosine mặc định = phân vị --spk-pct (5%) của độ giống giữa các clip CÙNG phiên (gom mạnh tay: CI rộng hơn, an toàn hơn).
auto = thử, lỗi / thiếu model thì bỏ qua và ghi lý do vào báo cáo; on = lỗi thì dừng; off = không chạy.
Báo cáo: <out>/consensus_report.md + consensus_stats.json.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from converters.common import REVIEW_COLS, add_vi_words_arg, auto_tag, is_vietnamese_token, load_vi_words  # noqa: E402
from ft.curate_train import pick_diverse  # noqa: E402
from lib.leak import DEFAULT_BLOCK_FILES, TEST_ONLY_LICENSE  # noqa: E402
from lib.manifest import write_jsonl  # noqa: E402
from lib.textnorm import Catalog, basic_tokens, normalize_ref, numbers_to_words  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
TARGET_SR = 16000
EMPTY = Catalog()
REVIEW_EXTRA = ["hint", "status", "turn"]
# luật ghép (id, turn) -> tên file (không đuôi); need_dir: file phải nằm trong thư mục tên <id>
RULES = [("<id>_<turn>", lambda i, t: f"{i}_{t}", False), ("<id>-<turn>", lambda i, t: f"{i}-{t}", False),
         ("<id>.<turn>", lambda i, t: f"{i}.{t}", False), ("<id><turn>", lambda i, t: f"{i}{t}", False),
         ("<id>", lambda i, t: i, False), ("<id>/<turn>", lambda i, t: t, True)]


# ------------------------------------------------------------------ audio
def index_audio(root: Path):
    idx = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXT and p.is_file():
            idx[p.stem.lower()].append(p)
    return idx


def _lookup(rule, sid, turn, idx):
    _, f, need_dir = rule
    cands = idx.get(f(sid, turn).lower(), [])
    if need_dir:
        cands = [p for p in cands if p.parent.name.lower() == sid.lower()]
    if len(cands) > 1:  # cùng tên ở nhiều chỗ: ưu tiên file nằm trong thư mục <id>
        cands = [p for p in cands if sid.lower() in (x.lower() for x in p.parts[:-1])] or cands
    return cands


def detect_mapping(keys, idx):
    """keys: [(id, turn)]. Chọn (luật cho turn final, luật cho turn khác) ghép được nhiều clip nhất, mỗi file 1 clip."""
    best = None
    for rf in RULES:
        for rt in RULES:
            got = {}
            for sid, turn in keys:
                c = _lookup(rf if turn == "final" else rt, sid, turn, idx)
                if len(c) == 1:
                    got[(sid, turn)] = c[0]
            use = Counter(got.values())
            score = sum(1 for p in got.values() if use[p] == 1)
            if best is None or score > best[0]:
                best = (score, rf, rt)
    _, rf, rt = best
    mapping, why = {}, Counter()
    for sid, turn in keys:
        c = _lookup(rf if turn == "final" else rt, sid, turn, idx)
        if len(c) == 1:
            mapping[(sid, turn)] = c[0]
        else:
            why["thiếu audio" if not c else "nhiều file khớp (mơ hồ)"] += 1
    use = Counter(mapping.values())
    for k in [k for k, p in mapping.items() if use[p] > 1]:  # một file cho nhiều clip: không tin được
        del mapping[k]
        why["một file khớp nhiều clip"] += 1
    return mapping, rf[0], rt[0], why


def prepare_audio(src: Path, out_dir: Path, stem: str):
    """-> (đường dẫn, duration, sample rate gốc, đã resample?)."""
    import soundfile as sf
    info = sf.info(str(src))
    if info.samplerate == TARGET_SR and info.channels == 1:
        return src, info.frames / info.samplerate, info.samplerate, False
    import soxr
    wav, sr = sf.read(str(src), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = soxr.resample(wav, sr, TARGET_SR, quality="HQ")
    dst = out_dir / "audio" / f"{stem}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), wav, TARGET_SR, subtype="PCM_16")
    return dst, len(wav) / TARGET_SR, info.samplerate, True


# ------------------------------------------------------------------ text
def load_names(path):
    names, full = {}, set()
    if not path:
        return names, full
    for line in open(path, encoding="utf-8"):
        cols = line.rstrip("\n").split("\t")
        if not cols[0].strip() or cols[0].startswith("#") or cols[0] == "name":
            continue
        toks = tuple(basic_tokens(cols[0]))
        kind = (cols[1].strip() if len(cols) > 1 else "given") or "given"
        if len(toks) == 1:
            names[toks[0]] = kind
        elif toks:
            full.add(toks)
    return names, full


def listener_list(s):
    out = []
    for part in (s or "").split("||"):
        part = part.strip()
        if part:
            m = re.match(r"^([^\s:]{1,8}):\s*(.*)$", part, re.S)
            out.append(m.group(2).strip() if m else part)
    return out


def tag_text(raw, cat, vi_words, names, full):
    """-> (text_tagged đã chuẩn hoá, auto_tag dict, {span: "name"}, cờ thêm)."""
    t = auto_tag(numbers_to_words(raw), cat, vi_words=vi_words)
    toks, _ = normalize_ref(t["text_tagged"], EMPTY)
    words = [x.tok for x in toks]
    lab = [x.span for x in toks]
    n = len(words)
    nid = max(lab + [-1]) + 1
    name_spans, flags = [], []

    def trigger(w):
        return names.get(w) in ("given", "surname") and not is_vietnamese_token(w)
    i = 0
    while i < n:
        if words[i] not in names:
            i += 1
            continue
        j = i
        while j < n and words[j] in names:
            j += 1
        end = j
        while end > i and names[words[end - 1]] == "title":  # danh xưng phải đứng trước tên
            end -= 1
        run = words[i:end]
        ok = run and (any(trigger(w) for w in run) or tuple(run) in full
                      or (len(run) >= 2 and names[run[0]] == "title"))
        if ok:
            for k in range(i, end):
                lab[k] = nid
            nid += 1
            name_spans.append(" ".join(run))
        elif len(run) >= 2 or (len(run) == 1 and names[run[0]] == "given"):
            flags.append(f"maybe_name:{' '.join(run)}")
        i = j
    out, k = [], 0
    while k < n:
        if lab[k] < 0:
            out.append(words[k])
            k += 1
            continue
        m = k
        while m < n and lab[m] == lab[k]:
            m += 1
        out.append("<en>" + " ".join(words[k:m]) + "</en>")
        k = m
    tagged = " ".join(out)
    tcat = {}
    for s in name_spans:
        _, sp = normalize_ref(f"<en>{s}</en>", cat)
        tcat[" ".join(sp[0])] = "name"
        flags.append(f"name:{s}")
    if re.search(r"[\w.+-]+@[\w-]+\.\w+|https?://|www\.", raw or ""):
        flags.append("email_or_url")
    return tagged, t, tcat, flags


def canon_terms(tagged, cat):
    """Thuật ngữ (span sau catalog) và ngữ cảnh (từ trước, từ sau) lần đầu gặp."""
    toks, spans = normalize_ref(tagged, cat)
    out, seen = [], set()
    for si, sp in enumerate(spans):
        term = " ".join(sp)
        if term in seen:
            continue
        seen.add(term)
        pos = [k for k, x in enumerate(toks) if x.span == si]
        a = toks[pos[0] - 1].tok if pos[0] > 0 else "<s>"
        b = toks[pos[-1] + 1].tok if pos[-1] + 1 < len(toks) else "</s>"
        out.append((term, (a, b)))
    return out


def span_category(term, tcat, cat):
    if term in tcat:
        return tcat[term]
    t = cat.terms.get(tuple(term.split()))
    return t.category if t and t.category else "(ngoài catalog)"


# ------------------------------------------------------------------ speaker embedding
def cluster_sessions(clips, model, threads, threshold, pct, seed):
    """clips: [(phiên, đường dẫn)]. -> ({phiên: nhãn cụm}, thống kê). Ném lỗi nếu thiếu thư viện / model."""
    import numpy as np
    import sherpa_onnx
    import soundfile as sf
    from scipy.cluster.hierarchy import fcluster, linkage
    if not Path(model).exists():
        raise FileNotFoundError(f"không có model {model}")
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=threads, provider="cpu"))
    by = defaultdict(list)
    for sess, p in clips:
        w, sr = sf.read(str(p), dtype="float32", always_2d=True)
        st = ex.create_stream()
        st.accept_waveform(sr, w.mean(axis=1))
        st.input_finished()
        e = np.asarray(ex.compute(st), dtype="float64")
        by[sess].append(e / (np.linalg.norm(e) + 1e-12))
    sess = sorted(by)
    within = [float(a @ b) for s in sess for i, a in enumerate(by[s]) for b in by[s][i + 1:]]
    rnd = random.Random(seed)
    flat = [(s, e) for s in sess for e in by[s]]
    between = []
    for _ in range(min(20000, len(flat) * 10)):
        (s1, a), (s2, b) = rnd.choice(flat), rnd.choice(flat)
        if s1 != s2:
            between.append(float(a @ b))
    if threshold is None:
        if len(within) < 10:
            raise ValueError(f"quá ít cặp clip cùng phiên ({len(within)}) để chọn ngưỡng; truyền --spk-threshold")
        threshold = float(np.percentile(within, pct))
    stats = {"model": Path(model).name, "n_sessions": len(sess), "n_clips": len(flat), "threshold": round(threshold, 4),
             "threshold_rule": f"phân vị {pct:g}% độ giống cùng phiên" if within else "--spk-threshold",
             "within": _q(within), "between": _q(between)}
    if len(sess) == 1:
        return {sess[0]: 1}, dict(stats, n_clusters=1, sweep={})
    means = np.stack([np.mean(by[s], axis=0) for s in sess])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    Z = linkage(means, method="average", metric="cosine")
    lab = fcluster(Z, t=1 - threshold, criterion="distance")
    sweep = {f"{th:.2f}": int(len(set(fcluster(Z, t=1 - th, criterion="distance")))) for th in (0.5, 0.6, 0.7, 0.8, 0.9)}
    return {s: int(c) for s, c in zip(sess, lab)}, dict(stats, n_clusters=int(len(set(lab))), sweep=sweep)


def _q(v):
    if not v:
        return {}
    v = sorted(v)
    return {"n": len(v), **{f"p{p}": round(v[min(len(v) - 1, int(p / 100 * len(v)))], 3) for p in (5, 25, 50, 75, 95)}}


# ------------------------------------------------------------------ ghi
def write_review(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS + REVIEW_EXTRA)
        w.writeheader()
        for r in rows:
            at = r.get("auto_tag", {})
            w.writerow({"id": r["id"], "split": r["split"], "audio": r["_review_audio"], "duration": r["duration"],
                        "text_tagged": r["text_tagged"], "text_tagged_orig": "", "confidence": at.get("confidence", ""),
                        "flags": " ".join(at.get("flags", [])), "decision": "", "text_tagged_fixed": "",
                        "viet_levels": "", "reading_note": "", "region": r.get("region", ""), "gender": r.get("gender", ""),
                        "note": "", "hint": r["_hint"], "status": r["consensus"]["status"], "turn": r["consensus"]["turn"]})


def hours(rows):
    return sum(float(r["duration"]) for r in rows) / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/dl/zac")
    ap.add_argument("--csv", help="mặc định <root>/consensus_clips_anon.csv")
    ap.add_argument("--out", default="data/raw/consensus")
    ap.add_argument("--catalog")
    ap.add_argument("--names", default=str(_ROOT / "configs" / "en_names.tsv"), help="danh sách tên (rỗng = tắt)")
    ap.add_argument("--max-per-term", type=int, default=10, help="0 = tắt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="chỉ đọc N dòng CSV đầu (thử nhanh)")
    ap.add_argument("--spk-embed", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--spk-model", default=str(_ROOT / "models" / "spk" / "wespeaker_en_voxceleb_resnet34.onnx"))
    ap.add_argument("--spk-threshold", type=float, help="ngưỡng cosine gom phiên (mặc định: tự chọn, xem --spk-pct)")
    ap.add_argument("--spk-pct", type=float, default=5.0)
    ap.add_argument("--spk-threads", type=int, default=2)
    add_vi_words_arg(ap)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = Catalog.load(args.catalog)
    vw = load_vi_words(args.vi_words)
    names, full = load_names(args.names)
    with open(args.csv or root / "consensus_clips_anon.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    status = Counter(r["status"] for r in rows)
    stats = {"csv_rows": len(rows), "sessions": len({r["id"] for r in rows}), "status": dict(status)}
    print(f"Đọc {len(rows)} dòng, {stats['sessions']} phiên: {dict(status)}")

    # ---- audio
    keys = [(r["id"], r["turn"]) for r in rows]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    if dup:
        print(f"  CẢNH BÁO: {len(dup)} cặp (id, turn) trùng, vd {dup[:3]}: giữ dòng đầu")
    idx = index_audio(root)
    mapping, rule_f, rule_t, miss_why = detect_mapping(keys, idx)
    used = set(mapping.values())
    orphans = sorted(str(p.relative_to(root)) for ps in idx.values() for p in ps if p not in used)
    stats["audio"] = {"files": sum(len(v) for v in idx.values()), "rule_final": rule_f, "rule_turn": rule_t,
                      "mapped": len(mapping), "missing": dict(miss_why), "unused_files": len(orphans)}
    print(f"  audio: luật final = {rule_f}, luật turn = {rule_t}; ghép {len(mapping)}/{len(keys)} clip; "
          f"thiếu / lỗi: {dict(miss_why) or 0}; file không dùng: {len(orphans)}")

    # ---- dòng
    seen, recs, why = set(), [], Counter()
    resampled = Counter()
    block = set()
    for r in rows:
        sid, turn = r["id"], r["turn"]
        hints = listener_list(r.get("listener_texts"))
        for t in [r.get("consensus_text") or ""] + hints:
            b = " ".join(basic_tokens(t))
            if b:
                block.add(b)
        if (sid, turn) in seen:
            why["bỏ: (id, turn) trùng"] += 1
            continue
        seen.add((sid, turn))
        st = r["status"].strip().lower()
        if st == "escalate":
            why["bỏ: escalate"] += 1
            continue
        if st not in ("agreed", "disputed", "pending"):
            why[f"bỏ: status lạ {st}"] += 1
            continue
        src = mapping.get((sid, turn))
        if src is None:
            why[f"bỏ: thiếu audio ({st})"] += 1
            continue
        raw = (r.get("consensus_text") or "").strip() or (hints[0] if st != "agreed" and hints else "")
        if not raw:
            why[f"bỏ: không có text ({st})"] += 1
            continue
        uid = f"cons_{sid}_{turn}"
        path, dur, sr0, rs = prepare_audio(src, out, f"{sid}/{uid}")
        resampled[f"{sr0} Hz{' -> 16 kHz mono' if rs else ''}"] += 1
        tagged, at, tcat, extra = tag_text(raw, cat, vw, names, full)
        flags = sorted(set(at["flags"]) | set(extra))
        if (r.get("numbers_majority") or "").strip().lower() == "false":
            flags.append("numbers_disagree")
        if st != "agreed":
            flags += [f"consensus:{st}", f"share:{(r.get('share') or '').strip() or '?'}"]
        has_en = "<en>" in tagged
        conf = at["confidence"] if at["confidence"] != "none" or not tcat else "heuristic"
        rec = {"id": uid, "split": "B2" if has_en else "B5", "split_by_tag": {"en": "B2", "vi": "B5"},
               "audio": os.path.relpath(os.path.abspath(path), os.path.abspath(out)).replace(os.sep, "/"),
               "duration": round(dur, 3), "text_tagged": tagged, "speaker": f"consensus:{sid}", "source": "consensus",
               "region": "unk", "gender": "u", "channel": "wideband", "domain": "product", "dataset": "consensus",
               "license": TEST_ONLY_LICENSE, "auto_tag": {"confidence": conf, "flags": flags},
               "consensus": {"status": st, "turn": turn, "tier": r.get("tier", ""), "hist_pos": r.get("hist_pos", ""),
                             "need": r.get("need", ""), "n_blind": r.get("n_blind", ""), "share": r.get("share", ""),
                             "numbers_majority": r.get("numbers_majority", ""), "label_source": r.get("source", ""),
                             "n_listeners": len(hints)},
               "_session": sid, "_hint": r.get("listener_texts") or "",
               "_review_audio": os.path.relpath(os.path.abspath(src), os.path.abspath(root)).replace(os.sep, "/")}
        if tcat:
            rec["term_category"] = tcat
        if st != "agreed":
            rec["review_required"] = True
        recs.append(rec)
    stats["audio"]["resample"] = dict(resampled)

    # ---- speaker embedding (tuỳ chọn)
    spk_note = "không chạy (--spk-embed off)"
    if args.spk_embed != "off" and recs:
        try:
            clips = [(r["_session"], Path(out) / r["audio"]) for r in recs]
            lab, sst = cluster_sessions(clips, args.spk_model, args.spk_threads, args.spk_threshold, args.spk_pct, args.seed)
            for r in recs:
                r["speaker_cluster"] = f"consensus:spk{lab[r['_session']]:03d}"
            with open(out / "speaker_clusters.tsv", "w", encoding="utf-8") as f:
                f.write("session\tspeaker_cluster\n" + "".join(f"{s}\tconsensus:spk{c:03d}\n" for s, c in sorted(lab.items())))
            stats["speaker_embedding"] = sst
            spk_note = (f"{sst['n_clusters']} cụm từ {sst['n_sessions']} phiên ({sst['n_clips']} clip), model {sst['model']}, "
                        f"ngưỡng cosine {sst['threshold']} ({sst['threshold_rule']})")
        except Exception as e:  # noqa: BLE001
            if args.spk_embed == "on":
                raise
            spk_note = f"BỎ QUA (lỗi: {e.__class__.__name__}: {e})"
            stats["speaker_embedding"] = {"skipped": spk_note}
    print(f"  speaker embedding: {spk_note}")

    # ---- tách nhóm
    agreed = [r for r in recs if r["consensus"]["status"] == "agreed"]
    disputed = [r for r in recs if r["consensus"]["status"] != "agreed"]
    b2 = [r for r in agreed if r["split"] == "B2"]
    b5 = [r for r in agreed if r["split"] == "B5"]
    terms_b2 = [canon_terms(r["text_tagged"], cat) for r in b2]
    before_terms = Counter(t for ts in terms_b2 for t, _ in ts)
    drop, cut = set(), {}
    if args.max_per_term:
        rnd = random.Random(args.seed)
        by_term = defaultdict(list)
        for i, ts in enumerate(terms_b2):
            for t, ctx in ts:
                by_term[t].append((i, b2[i]["speaker"], ctx))
        over = {t: c for t, c in by_term.items() if len(c) > args.max_per_term}
        for t, c in sorted(over.items()):
            chosen = pick_diverse(c, args.max_per_term, rnd)
            drop |= {i for i, _, _ in c if i not in chosen}
        after = Counter(t for i, ts in enumerate(terms_b2) if i not in drop for t, _ in ts)
        cut = {t: (len(c), after[t]) for t, c in over.items()}
    b2_keep = [r for i, r in enumerate(b2) if i not in drop]
    b2_cut = [r for i, r in enumerate(b2) if i in drop]
    for r in b2_cut:
        r["overcap"] = True
    kept_terms = Counter(t for i, ts in enumerate(terms_b2) if i not in drop for t, _ in ts)
    catc = Counter()
    for i, ts in enumerate(terms_b2):
        if i not in drop:
            for t, _ in ts:
                catc[span_category(t, b2[i].get("term_category") or {}, cat)] += 1
    dis_split = Counter(r["split"] for r in disputed)

    # ---- ghi
    files = {"b2_consensus_candidates.jsonl": b2_keep, "b2_consensus_overcap.jsonl": b2_cut,
             "b5_consensus_product.jsonl": b5, "disputed_candidates.jsonl": disputed}
    for name, rs in files.items():
        write_jsonl(str(out / name), rs)
        print(f"  {out / name}: {len(rs)} câu, {hours(rs):.2f} h")
    b5_flag = [r for r in b5 if r["auto_tag"]["flags"]]
    for name, rs in (("b2_consensus_candidates_review.csv", b2_keep), ("b5_consensus_product_review.csv", b5_flag),
                     ("disputed_candidates_review.csv", disputed)):
        write_review(out / name, rs)
        print(f"  {out / name}: phiếu duyệt {len(rs)} câu")
    bp = out / "leak_block.txt"
    with open(bp, "w", encoding="utf-8") as f:
        f.write("# text chuẩn hoá của bộ consensus (CHỈ ĐƯỢC TEST): cấm có trong train / dev. Sinh bởi "
                "converters/conv_consensus.py\n" + "".join(f"{t}\n" for t in sorted(block)))
    print(f"  {bp}: {len(block)} câu chặn rò rỉ")
    if bp.resolve() not in [p.resolve() for p in DEFAULT_BLOCK_FILES]:
        print(f"  LƯU Ý: --out khác mặc định: truyền --block-text {bp} cho curate_train / vss_cs_prefilter / validate")

    stats.update({
        "drop": dict(why), "B2": {"n": len(b2_keep), "h": round(hours(b2_keep), 3)},
        "B2_before_cap": {"n": len(b2), "h": round(hours(b2), 3)},
        "B2_overcap": {"n": len(b2_cut), "h": round(hours(b2_cut), 3), "max_per_term": args.max_per_term},
        "B5_product": {"n": len(b5), "h": round(hours(b5), 3), "flagged_for_review": len(b5_flag)},
        "disputed_pending": {"n": len(disputed), "h": round(hours(disputed), 3), "split_guess": dict(dis_split)},
        "terms_cut": {t: list(v) for t, v in sorted(cut.items(), key=lambda x: -x[1][0])},
        "top_terms": kept_terms.most_common(30), "top_terms_before_cap": before_terms.most_common(30),
        "n_terms": len(kept_terms), "category": dict(catc.most_common()), "leak_block": len(block),
        "speaker_embedding_note": spk_note})
    (out / "consensus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Bộ consensus ({TEST_ONLY_LICENSE}) — {args.csv or root / 'consensus_clips_anon.csv'}", "",
          f"{len(rows)} dòng, {stats['sessions']} phiên. Trạng thái: "
          + ", ".join(f"{k} {v}" for k, v in status.most_common()) + ".", "",
          f"Audio: luật ghép final = `{rule_f}`, turn khác = `{rule_t}`; ghép {len(mapping)}/{len(keys)} clip, "
          f"thiếu / lỗi {sum(miss_why.values())} ({dict(miss_why) or 'không'}), file không dùng {len(orphans)}. "
          f"Định dạng: {dict(resampled)}.", "",
          "| Nhóm | Câu | Giờ |", "| --- | --- | --- |",
          f"| B2 (agreed, có <en>) trước trần | {len(b2)} | {hours(b2):.2f} |",
          f"| B2 sau --max-per-term {args.max_per_term} | {len(b2_keep)} | {hours(b2_keep):.2f} |",
          f"| bị cắt bởi --max-per-term {args.max_per_term} | {len(b2_cut)} | {hours(b2_cut):.2f} |",
          f"| B5 domain product (agreed, Việt thuần) | {len(b5)} | {hours(b5):.2f} |",
          f"| disputed / pending (phiếu duyệt) | {len(disputed)} | {hours(disputed):.2f} |", "",
          "Bỏ: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "không"), "",
          f"B5 có cờ cần duyệt: {len(b5_flag)} / {len(b5)}. Disputed / pending đoán split: {dict(dis_split)}.", "",
          f"## Thuật ngữ B2 ({len(kept_terms)} khác nhau sau trần)", "", "| Thuật ngữ | Câu sau trần | Trước trần | Category |",
          "| --- | --- | --- | --- |"]
    tc_all = {}
    for r in b2:
        tc_all.update(r.get("term_category") or {})
    md += [f"| {t} | {c} | {before_terms[t]} | {span_category(t, tc_all, cat)} |" for t, c in kept_terms.most_common(30)]
    md += ["", f"## Thuật ngữ bị cắt bởi --max-per-term {args.max_per_term}: {len(cut)} thuật ngữ, {len(b2_cut)} câu", "",
           "| Thuật ngữ | Câu trước | Câu sau |", "| --- | --- | --- |"]
    md += [f"| {t} | {a} | {b} |" for t, (a, b) in sorted(cut.items(), key=lambda x: -x[1][0])]
    md += ["", "## Category của span (B2 sau trần)", "", "| Category | Span |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in catc.most_common()]
    md += ["", "## Speaker embedding", "", spk_note]
    if "speaker_embedding" in stats and "within" in stats["speaker_embedding"]:
        s = stats["speaker_embedding"]
        md += ["", f"Độ giống cosine giữa clip cùng phiên: {s['within']}; khác phiên (ngẫu nhiên): {s['between']}.",
               f"Số cụm theo ngưỡng: {s.get('sweep')}. speaker_cluster CHỈ dùng cho bootstrap CI (score.py)."]
    md += ["", f"Danh sách chặn rò rỉ: {bp} ({len(block)} câu chuẩn hoá, mọi trạng thái + listener_texts)."]
    (out / "consensus_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  B2 {len(b2_keep)} câu {hours(b2_keep):.2f} h (cắt {len(b2_cut)} bởi --max-per-term {args.max_per_term}); "
          f"B5-product {len(b5)} câu {hours(b5):.2f} h; disputed/pending {len(disputed)}; category {dict(catc)}")
    print(f"  báo cáo: {out / 'consensus_report.md'}")


if __name__ == "__main__":
    main()
