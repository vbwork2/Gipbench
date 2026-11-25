#!/usr/bin/env python3
"""Bước 2 của chuỗi TTS: tổng hợp audio từ kịch bản (src/gipbench/dataset/synth/make_text.py) -> manifest train_tts_raw.jsonl.

  # thử không cần engine thật (âm thanh giả, chỉ để kiểm đường ống)
  python -m gipbench.dataset.synth.synth --script data/tts/script/tts_script.jsonl --voices configs/dataset/synth/tts_voices.tsv \
      --backend mock --out-dir data/tts/raw --bench data/bench_v0_draft/bench_v0_draft.jsonl
  # lệnh ngoài (text truyền qua file UTF-8): chỗ trống {text_file} {voice} {voice_id} {ref_audio} {out}
  python -m gipbench.dataset.synth.synth ... --backend command --command "mytts --in {text_file} --voice {voice} --out {out}"
  # hàm Python của bạn (module:function hoặc file.py:function; chữ ký fn(text, voice, out_path))
  python -m gipbench.dataset.synth.synth ... --backend python --python-fn my_engine:synth [--backend-opt raw_ext=.mp3]

--voices FILE (TSV có header): voice, backend_voice, ref_audio, gender, region (+ tuỳ chọn ref_speaker, ref_source = người
nói / nguồn của audio tham chiếu). `voice` là id ổn định -> speaker = `tts:<backend>:<voice>`.
Phân giọng (xác định theo --seed): xoay vòng sao cho mỗi thuật ngữ gặp >= --min-voices-per-term giọng và không giọng nào
chiếm quá --max-voice-share số câu (cần ít nhất ceil(1 / max-voice-share) giọng). Không đạt được thì ghi cảnh báo vào report
(--strict-voices: thoát mã 3).
Chống rò rỉ giọng: có --bench và voices có ref_audio -> LỖI (thoát mã 2, chưa tổng hợp gì) nếu file tham chiếu là audio của bench
(trùng đường dẫn, hoặc cùng nội dung md5) hoặc ref_speaker / ref_source trùng người nói / nguồn của bench.
Hậu xử lý mỗi câu: resample 16 kHz mono PCM16, cắt lặng đầu / cuối (chừa --keep-sil giây), chuẩn hoá đỉnh (--peak), đo duration.
Chạy tiếp được: câu đã có manifest + file wav hợp lệ (cùng giọng, cùng tts_text) được bỏ qua. Lỗi từng câu ghi vào
<out-dir>/synth_errors.jsonl và KHÔNG dừng cả mẻ; --abort-after-fails N lần liên tiếp thất bại thì dừng (backend hỏng, thoát mã 3).
Ra: <out-dir>/train_tts_raw.jsonl (đúng SCHEMA.md: dataset "tts", synthetic true, train_only true, en_meta[].viet_level,
tts {backend, voice, tts_text, readings, category, flags, stats}), audio/<id>.wav, train_tts_raw.report.md.
Tiếp theo: src/gipbench/infer/asr_hyps.py trên manifest này -> src/gipbench/dataset/synth/verify.py.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.manifest import read_jsonl
from gipbench.dataset.synth.backends import (BACKENDS, TARGET_SR, coerce, load_voices, make_backend, postprocess, read_audio,
                              span_levels, write_wav16)


def assign_voices(lines, voices, min_voices, max_share, seed):
    """Phân giọng cho từng dòng kịch bản. Trả về ({id: Voice}, [cảnh báo]).
    Tham lam theo thứ tự xáo (seed): mỗi câu chọn giọng còn dưới trần, ưu tiên giọng mà nhiều thuật ngữ của câu CHƯA gặp
    (tới khi đủ min_voices), rồi giọng dùng ít nhất; hoà thì bốc ngẫu nhiên (seed)."""
    n_lines, n_v = len(lines), len(voices)
    need = math.ceil(1.0 / max_share - 1e-9) if max_share > 0 else 1
    if n_v < need:
        raise SystemExit(f"LỖI: --max-voice-share {max_share:g} cần ít nhất {need} giọng, voices.tsv chỉ có {n_v}")
    cap = max(int(math.floor(max_share * n_lines + 1e-9)), math.ceil(n_lines / n_v))
    rng = random.Random(f"voices|{seed}")
    order = list(range(n_lines))
    rng.shuffle(order)
    used, seen, out = Counter(), defaultdict(set), {}
    for i in order:
        ln = lines[i]
        ts = ln.get("terms") or []
        cands = [v for v in voices if used[v.id] < cap]
        best = max(cands, key=lambda v: (sum(1 for t in ts if len(seen[t]) < min_voices and v.id not in seen[t]),
                                         -used[v.id], rng.random()))
        out[ln["id"]] = best
        used[best.id] += 1
        for t in ts:
            seen[t].add(best.id)
    warns = []
    cnt = Counter(t for ln in lines for t in (ln.get("terms") or []))
    low = sorted(t for t, c in cnt.items() if c >= min_voices and len(seen[t]) < min_voices)
    if low:
        warns.append(f"{len(low)} thuật ngữ gặp < {min_voices} giọng (vd {', '.join(low[:5])})")
    short = sorted(t for t, c in cnt.items() if c < min_voices)
    if short:
        warns.append(f"{len(short)} thuật ngữ có ít hơn {min_voices} câu nên không thể đủ {min_voices} giọng")
    return out, warns


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_ref_audio(voices, bench_rows):
    """Lỗi rò rỉ giọng tham chiếu: ref_audio là audio của bench / thuộc người nói hoặc nguồn của bench."""
    errs = []
    paths = {os.path.normcase(os.path.realpath(r["_audio_path"])) for r in bench_rows if r.get("_audio_path")}
    spk = {r.get("speaker") for r in bench_rows if r.get("speaker")}
    src = {r.get("source") for r in bench_rows if r.get("source")}
    by_size = None
    for v in voices:
        if v.ref_speaker and v.ref_speaker in spk:
            errs.append(f"giọng {v.id}: ref_speaker '{v.ref_speaker}' là người nói của bench")
        if v.ref_source and v.ref_source in src:
            errs.append(f"giọng {v.id}: ref_source '{v.ref_source}' là nguồn của bench")
        if not v.ref_audio:
            continue
        if not Path(v.ref_audio).exists():
            errs.append(f"giọng {v.id}: ref_audio không tồn tại: {v.ref_audio}")
            continue
        rp = os.path.normcase(os.path.realpath(v.ref_audio))
        if rp in paths:
            errs.append(f"giọng {v.id}: ref_audio {v.ref_audio} là audio của bench")
            continue
        if by_size is None:
            by_size = defaultdict(list)
            for p in paths:
                try:
                    by_size[os.path.getsize(p)].append(p)
                except OSError:
                    pass
        cand = by_size.get(os.path.getsize(rp), [])
        if cand:
            h = md5(rp)
            hit = next((p for p in cand if md5(p) == h), None)
            if hit:
                errs.append(f"giọng {v.id}: ref_audio có nội dung trùng audio của bench ({hit})")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True, help="tts_script.jsonl")
    ap.add_argument("--voices", required=True)
    ap.add_argument("--backend", required=True, choices=BACKENDS,
                    help="edge GỬI văn bản lên dịch vụ Microsoft, không phải mặc định; mock chỉ để test")
    ap.add_argument("--command", default="", help="backend command: mẫu lệnh có {out} (+ {text_file} {voice} {voice_id} {ref_audio})")
    ap.add_argument("--python-fn", default="", help="backend python: module:function")
    ap.add_argument("--backend-opt", action="append", default=[], help="k=v truyền cho backend (lặp được), vd sec_per_syl=0.25")
    ap.add_argument("--timeout", type=float, default=120.0, help="backend command: giây tối đa mỗi câu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bench", nargs="*", default=[], help="manifest bench: kiểm ref_audio / ref_speaker không thuộc bench")
    ap.add_argument("--min-voices-per-term", type=int, default=3)
    ap.add_argument("--max-voice-share", type=float, default=0.4, help="một giọng tối đa tỉ lệ này số câu")
    ap.add_argument("--strict-voices", action="store_true", help="thoát mã 3 nếu không đạt --min-voices-per-term")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provenance", type=Path, help="JSON object describing the pilot model, voices and settings")
    ap.add_argument("--limit", type=int, default=0, help="chỉ tổng hợp N câu đầu của kịch bản (thử nhanh)")
    ap.add_argument("--keep-sil", type=float, default=0.15, help="giây lặng chừa ở mỗi đầu sau khi cắt")
    ap.add_argument("--peak", type=float, default=0.9, help="đỉnh sau chuẩn hoá (0-1)")
    ap.add_argument("--abort-after-fails", type=int, default=20, help="dừng (mã 3) sau N lỗi liên tiếp; 0 = không dừng")
    args = ap.parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance else None
    if args.provenance and not isinstance(provenance, dict):
        ap.error("--provenance must contain a JSON object")

    out = Path(args.out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    script = [json.loads(l) for l in open(args.script, encoding="utf-8") if l.strip()]
    if not script:
        sys.exit("LỖI: kịch bản rỗng")
    voices = load_voices(args.voices)
    bench_rows = [r for b in args.bench for r in read_jsonl(b)]
    if any(v.ref_audio or v.ref_speaker or v.ref_source for v in voices):
        if not bench_rows:
            print("CẢNH BÁO: voices có ref_audio nhưng không truyền --bench: không kiểm được rò rỉ giọng tham chiếu.")
        else:
            errs = check_ref_audio(voices, bench_rows)
            if errs:
                for e in errs:
                    print("LỖI:", e)
                sys.exit(2)
    opts = {}
    for kv in args.backend_opt:
        k, _, v = kv.partition("=")
        opts[k.strip()] = coerce(v.strip())
    backend = make_backend(args.backend, args.command, args.python_fn, opts, args.timeout)

    assign, vwarn = assign_voices(script, voices, args.min_voices_per_term, args.max_voice_share, args.seed)
    todo_lines = script[:args.limit] if args.limit else script

    man_path = out / "train_tts_raw.jsonl"
    rows = {}
    if man_path.exists():
        for r in read_jsonl(str(man_path)):
            rows[r["id"]] = {k: v for k, v in r.items() if not k.startswith("_")}
    if provenance is not None and any(r.get("tts", {}).get("provenance") != provenance for r in rows.values()):
        ap.error("Use a fresh output directory when the TTS provenance changes")

    def valid(ln, r):
        v = assign[ln["id"]]
        p = out / r["audio"]
        return (r.get("speaker") == f"tts:{backend.name}:{v.id}" and (r.get("tts") or {}).get("tts_text") == ln["tts_text"]
                and p.exists() and p.stat().st_size > 44)

    errors, fails, n_done, n_skip = [], 0, 0, 0
    err_path = out / "synth_errors.jsonl"
    mf = open(man_path, "a", encoding="utf-8", newline="\n")
    with tempfile.TemporaryDirectory(prefix="tts_raw_") as td:
        for ln in todo_lines:
            sid, v = ln["id"], assign[ln["id"]]
            if sid in rows and valid(ln, rows[sid]):
                n_skip += 1
                continue
            raw = os.path.join(td, sid + backend.raw_ext)
            try:
                backend.synth(ln["tts_text"], v, raw)
                x, sr = read_audio(raw)
                y, st = postprocess(x, sr, TARGET_SR, args.keep_sil, args.peak)
                if len(y) < 0.2 * TARGET_SR:
                    raise RuntimeError(f"audio quá ngắn ({len(y) / TARGET_SR:.2f} s)")
                write_wav16(str(out / "audio" / f"{sid}.wav"), y, TARGET_SR)
            except Exception as e:  # noqa: BLE001
                fails += 1
                rec = {"id": sid, "voice": v.id, "error": f"{e.__class__.__name__}: {e}"[:500]}
                errors.append(rec)
                with open(err_path, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  LỖI {sid} ({v.id}): {rec['error'][:150]}")
                if args.abort_after_fails and fails >= args.abort_after_fails:
                    mf.close()
                    sys.exit(f"DỪNG: {fails} lỗi liên tiếp, backend có vẻ hỏng (kiểm lệnh / engine). Xem {err_path}")
                continue
            finally:
                try:
                    os.remove(raw)
                except OSError:
                    pass
            fails = 0
            row = {"id": sid, "split": "train", "audio": f"audio/{sid}.wav", "duration": round(len(y) / TARGET_SR, 3),
                   "text_tagged": ln["text_tagged"], "speaker": f"tts:{backend.name}:{v.id}", "source": f"tts:{sid}",
                   "region": v.region, "gender": v.gender, "channel": "wideband", "domain": ln.get("domain") or "other",
                   "dataset": "tts", "synthetic": True, "train_only": True,
                   "en_meta": span_levels(ln["text_tagged"], ln["readings"]),
                   "tts": {"backend": backend.name, "voice": v.id, "tts_text": ln["tts_text"], "readings": ln["readings"],
                           "category": ln.get("category", ""), "flags": ln.get("flags", []), "stats": st}}
            if provenance is not None:
                from gipbench.core.fileio import sha256_file

                row.update(audio_sha256=sha256_file(out / row["audio"]), license=provenance.get("license"),
                           quality_status=provenance.get("quality_status", "pending_review"),
                           label_source="synthetic:tts_conditioning_text", text_orig=ln["text_tagged"])
                row["tts"].update(provenance=provenance, seed=args.seed)
            rows[sid] = row
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  … {n_done} câu mới")
    mf.close()

    final = [rows[ln["id"]] for ln in script if ln["id"] in rows and valid(ln, rows[ln["id"]])]  # giữ cả câu của lần chạy trước
    with open(man_path, "w", encoding="utf-8", newline="\n") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report
    by_v = Counter(r["tts"]["voice"] for r in final)
    hrs_v = Counter()
    for r in final:
        hrs_v[r["tts"]["voice"]] += r["duration"] / 3600
    tv = defaultdict(set)
    for r in final:
        for rd in r["tts"]["readings"]:
            tv[rd["term"]].add(r["tts"]["voice"])
    low = sorted((len(s), t) for t, s in tv.items() if len(s) < args.min_voices_per_term)
    total_h = sum(r["duration"] for r in final) / 3600
    md = ["# Tổng hợp TTS", "",
          f"Kịch bản `{args.script}` ({len(script)} câu; xử lý {len(todo_lines)}), backend **{backend.name}**, "
          f"{len(voices)} giọng, seed {args.seed}.", "",
          f"- câu có audio hợp lệ: {len(final)} ({total_h:.2f} h); tổng hợp mới {n_done}, bỏ qua (đã có) {n_skip}, "
          f"lỗi {len(errors)} (xem synth_errors.jsonl)",
          f"- trần giọng: --max-voice-share {args.max_voice_share:g}; giọng nhiều nhất chiếm "
          f"{(max(by_v.values()) / len(final) if final else 0):.1%}",
          f"- thuật ngữ gặp < {args.min_voices_per_term} giọng trong số đã tổng hợp: {len(low)}"
          + (" (vd " + ", ".join(f"{t} ({n})" for n, t in low[:10]) + ")" if low else "")]
    md += [f"- CẢNH BÁO phân giọng: {w}" for w in vwarn]
    if args.backend == "mock":
        md += ["- **backend mock: âm thanh giả, KHÔNG dùng để train.**"]
    md += ["", "| Giọng | Câu | Giờ | Tỉ lệ câu |", "| --- | --- | --- | --- |"]
    md += [f"| {v} | {c} | {hrs_v[v]:.3f} | {c / len(final):.1%} |" for v, c in by_v.most_common()] if final else []
    if errors:
        md += ["", "### Lỗi (tối đa 30)", ""] + [f"- {e['id']} ({e['voice']}): {e['error']}" for e in errors[:30]]
    (out / "train_tts_raw.report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Đã ghi {man_path}: {len(final)} câu ({total_h:.2f} h), mới {n_done}, bỏ qua {n_skip}, lỗi {len(errors)}")
    for w in vwarn:
        print("CẢNH BÁO:", w)
    if args.strict_voices and vwarn:
        sys.exit(3)


if __name__ == "__main__":
    main()
