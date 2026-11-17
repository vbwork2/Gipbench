#!/usr/bin/env python3
"""Test chuỗi TTS (tts_make_text -> tts_synth -> tts_verify -> curate_train / validate) bằng backend `mock`, trong thư mục tạm.

  python tests/dataset/test_tts.py            # thoát mã 1 nếu có bất kỳ kiểm tra nào sai

Không cần numpy / soundfile / engine TTS / ASR / mạng. Âm thanh mock là giả, hyp ASR là hyp GIẢ do test dựng.
Kiểm: reserved_b3 và thuật ngữ đối chứng không có trong kịch bản; cùng seed -> kịch bản y hệt; không câu nào trùng bench (kể cả
khi bench cố tình chứa một câu sinh ra); text_tagged hợp lệ và đúng span; tts_text không còn tag và dùng cách đọc của lexicon;
đủ ngữ cảnh / giọng, trần giọng; manifest đúng SCHEMA, qua validate.py --train và src/gipbench/model/recipe/curate.py --max-synthetic-share;
tts_verify giữ / loại đúng trên hyp giả; ref_audio thuộc bench bị báo lỗi; câu của --sentences được kiểm.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import wave
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.leak import norm_text
from gipbench.core.manifest import REQUIRED, read_jsonl
from gipbench.core.textnorm import Catalog, normalize_ref, plain_text, tagged_well_formed
from gipbench.dataset.synth.backends import (MockBackend, Voice, analyze, count_syllables, load_lexicon, read_audio, resample,
                              term_key, write_wav16)

FAILS = []
PY = sys.executable


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  [SAI]", msg)
    return bool(cond)


def run(args, expect_rc=0):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY] + [str(a) for a in args], cwd=str(G), capture_output=True, env=env)
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    if expect_rc is not None and r.returncode != expect_rc:
        FAILS.append(f"lệnh {' '.join(str(a) for a in args[:3])} ... trả mã {r.returncode} (mong {expect_rc}):\n{out[-1500:]}")
        print("  [SAI] mã thoát", r.returncode, "\n", out[-800:])
    return r.returncode, out


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def wjl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tone(sec, sr=16000, amp=0.4, freq=220.0):
    return [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(int(sec * sr))]


# ----------------------------------------------------------------------------- dữ liệu thử
CATALOG = [("meeting", "business"), ("deadline", "business"), ("team", "business"), ("feedback", "business"),
           ("marketing", "business"), ("email", "tech"), ("website", "tech"), ("server", "tech"), ("deploy", "tech"),
           ("bug", "tech"), ("code", "tech"), ("app", "tech"), ("wifi", "tech"), ("online", "tech"), ("livestream", "tech"),
           ("update", "tech"), ("download", "tech"), ("password", "tech"), ("login", "tech"), ("link", "tech"),
           ("click", "tech"), ("share", "tech"), ("comment", "tech"), ("group", "tech"), ("chat", "tech"),
           ("workshop", "edu"), ("ielts", "edu"), ("toefl", "edu"), ("scholarship", "edu"), ("master", "edu"),
           ("bachelor", "edu"), ("sale", "daily"), ("combo", "daily"), ("voucher", "daily"), ("booking", "daily"),
           ("skincare", "daily"), ("api", "acronym"), ("kpi", "acronym"), ("ceo", "acronym"), ("sql", "acronym"),
           ("vpn", "acronym"), ("pdf", "acronym"), ("gdp", "acronym"), ("facebook", "brand"), ("youtube", "brand"),
           ("shopee", "brand"), ("zalo", "brand"), ("google", "brand"), ("iphone", "brand"), ("grab", "brand"),
           ("foobar", "tech"), ("hdd", "tech")]  # foobar, hdd: KHÔNG có trong lexicon (dự phòng)
RESERVED = ["kubernetes", "webhook"]


def build_fixtures(d: Path):
    (d / "bench_audio").mkdir()
    write_wav16(str(d / "bench_audio/b1.wav"), tone(2.0), 16000)
    write_wav16(str(d / "bench_audio/b1_copy.wav"), tone(2.0), 16000)  # cùng nội dung -> md5 trùng
    write_wav16(str(d / "other_ref.wav"), tone(2.0, freq=330.0), 16000)
    (d / "bench_audio/b1_copy.wav").write_bytes((d / "bench_audio/b1.wav").read_bytes())
    with open(d / "catalog.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("term\tcategory\tvariants\tphonetic\treserved_b3\n")
        for t, c in CATALOG:
            f.write(f"{t}\t{c}\t\t\t0\n")
        for t in RESERVED:
            f.write(f"{t}\ttech\t\t\t1\n")
    bench = [{"id": "b1_0001", "split": "B1", "audio": "bench_audio/b1.wav", "duration": 2.0, "speaker": "bench_spk1",
              "source": "bench_src1", "domain": "tech", "text_tagged": "hôm nay mình có <en>meeting</en> lúc ba giờ chiều"},
             {"id": "b1_0002", "split": "B1", "audio": "bench_audio/b1.wav", "duration": 2.0, "speaker": "bench_spk2",
              "source": "bench_src2", "domain": "tech", "text_tagged": "em đang dùng <en>zalo</en> để nhắn tin"}]
    wjl(d / "bench.jsonl", bench)
    (d / "voices.tsv").write_text("voice\tbackend_voice\tref_audio\tgender\tregion\n"
                                  "v_a\t\t\tf\tnorth\nv_b\t\t\tm\tsouth\nv_c\t\t\tf\tcentral\nv_d\t\t\tm\tnorth\n", encoding="utf-8")


# ----------------------------------------------------------------------------- các nhóm kiểm tra
def test_make_text(d: Path):
    print("[1] tts_make_text")
    cat = Catalog.load(str(d / "catalog.tsv"))
    lex, probs = load_lexicon(str(G / "configs/dataset/synth/tts_lexicon.tsv"))
    check(not probs, f"lexicon có cảnh báo: {probs[:3]}")
    check(len(lex) >= 80, f"lexicon chỉ có {len(lex)} thuật ngữ (cần ~80+)")
    check(all(2 <= len(v) for v in lex.values() if v[0].kind != "x"), "mỗi thuật ngữ lexicon nên có >= 2 cách đọc")
    base = ["src/gipbench/dataset/synth/make_text.py", "--catalog", d / "catalog.tsv", "--bench", d / "bench.jsonl"]
    run(base + ["--out-dir", d / "s0", "--seed", 0])
    run(base + ["--out-dir", d / "s0b", "--seed", 0])
    run(base + ["--out-dir", d / "s1", "--seed", 1])
    a, b, c = [(d / x / "tts_script.jsonl").read_bytes() for x in ("s0", "s0b", "s1")]
    check(a == b, "cùng seed nhưng kịch bản khác nhau")
    check(a != c, "khác seed mà kịch bản y hệt")
    check((d / "s0/tts_holdout_terms.txt").read_bytes() == (d / "s1/tts_holdout_terms.txt").read_bytes(),
          "nhóm đối chứng phụ thuộc seed (phải ổn định)")
    script = jl(d / "s0/tts_script.jsonl")
    held = {l.strip() for l in open(d / "s0/tts_holdout_terms.txt", encoding="utf-8") if l.strip()}
    check(len(held) > 0, "không có thuật ngữ đối chứng")
    check(not (held & set(RESERVED)), "thuật ngữ đối chứng lẫn reserved_b3")
    bench_n = {norm_text(r["text_tagged"], cat) for r in read_jsonl(str(d / "bench.jsonl"))}
    used = Counter()
    for ln in script:
        used.update(ln["terms"])
        tt = ln["text_tagged"]
        check(tagged_well_formed(tt) is None, f"tag hỏng: {tt}")
        _, spans = normalize_ref(tt, cat)
        check([" ".join(s) for s in spans] == ln["terms"], f"span != terms: {tt} {ln['terms']}")
        check("<" not in ln["tts_text"] and ">" not in ln["tts_text"], f"tts_text còn tag: {ln['tts_text']}")
        check(norm_text(tt, cat) not in bench_n, f"câu trùng bench: {tt}")
        for rd in ln["readings"]:
            check(rd["reading"] in ln["tts_text"], f"tts_text thiếu cách đọc {rd['reading']}: {ln['tts_text']}")
            check(rd["viet_level"] in ("strong", "mid", "native"), "viet_level sai")
            if rd["term"] in lex:
                check(rd["reading"] in {x.reading for x in lex[rd["term"]]}, f"cách đọc ngoài lexicon: {rd}")
                check(f"reading_fallback:{rd['term']}" not in ln["flags"], "term có lexicon mà gắn cờ fallback")
            else:
                check(f"reading_fallback:{rd['term']}" in ln["flags"], f"thiếu cờ fallback cho {rd['term']}")
        for rd in ln["readings"]:
            if rd["term"] in lex and rd["reading"] != rd["term"]:
                check(rd["term"] not in ln["tts_text"].split(), f"tts_text còn chính tả Anh của {rd['term']}: {ln['tts_text']}")
    allt = set(used)
    check(not (allt & set(RESERVED)), "reserved_b3 có trong kịch bản")
    check(not (allt & held), f"thuật ngữ đối chứng có trong kịch bản: {sorted(allt & held)[:5]}")
    check(not any(r in ln["text_tagged"] for ln in script for r in RESERVED), "chuỗi reserved xuất hiện trong text")
    short = [t for t in allt if used[t] < 5]
    check(not short, f"thuật ngữ có < 5 câu (tính cả khi làm thuật ngữ thứ hai): {short[:5]}")
    check(len({ln["domain"] for ln in script}) >= 3, "kịch bản chỉ có < 3 domain")
    # nhiều cách đọc cho thuật ngữ trong lexicon có >= 2 cách
    by_term = defaultdict(set)
    for ln in script:
        for rd in ln["readings"]:
            by_term[rd["term"]].add(rd["reading"])
    multi = [t for t in allt if t in lex and len(lex[t]) >= 2 and len(by_term[t]) < 2]
    check(not multi, f"thuật ngữ có >= 2 cách đọc nhưng chỉ dùng 1: {multi[:5]}")
    levels = {rd["viet_level"] for ln in script for rd in ln["readings"] if rd["term"] in lex}
    check(len(levels) >= 2, f"mức đọc không đa dạng: {levels}")
    # dự phòng: hdd -> đánh vần chữ cái (toàn phụ âm); foobar -> nguyên chính tả
    if "hdd" in allt:
        r = next(rd for ln in script for rd in ln["readings"] if rd["term"] == "hdd")
        check(any(x in r["reading"] for x in ("ếch", "hát")), f"hdd không được đánh vần: {r}")
    # trùng bench: nhét câu sinh ra vào bench -> câu đó phải biến mất, thuật ngữ vẫn đủ câu
    victim = script[0]
    wjl(d / "bench_dup.jsonl", read_jsonl(str(d / "bench.jsonl")) + [{"id": "b1_dup", "split": "B1", "audio": "bench_audio/b1.wav",
                                                                      "duration": 2.0, "speaker": "bench_spk9", "source": "bench_src9",
                                                                      "text_tagged": victim["text_tagged"]}])
    run(["src/gipbench/dataset/synth/make_text.py", "--catalog", d / "catalog.tsv", "--bench", d / "bench_dup.jsonl", "--out-dir", d / "sd", "--seed", 0])
    sd = jl(d / "sd/tts_script.jsonl")
    check(all(ln["text_tagged"] != victim["text_tagged"] for ln in sd), "câu trùng bench vẫn còn trong kịch bản")
    rep = (d / "sd/tts_script.report.md").read_text(encoding="utf-8")
    check("TRÙNG text bench (đã thay bằng mẫu khác): 0" not in rep, "report không đếm câu trùng bench")
    cnt = Counter(ln["terms"][0] for ln in sd)
    check(cnt[victim["terms"][0]] >= 5, "thuật ngữ mất câu trùng bench không được bù lại")
    # --require-lexicon
    run(base + ["--out-dir", d / "sr", "--require-lexicon"])
    sr = jl(d / "sr/tts_script.jsonl")
    check(all("reading_fallback" not in ln["flags"] for ln in sr) and sr, "--require-lexicon vẫn còn thuật ngữ dự phòng")
    check(all(t in lex for ln in sr for t in ln["terms"]), "--require-lexicon còn thuật ngữ ngoài lexicon")
    return script, held


def test_sentences(d: Path, script, held):
    print("[2] --sentences")
    pool = sorted({t for ln in script for t in ln["terms"] if t in {x[0] for x in CATALOG}})
    A, B = pool[0], pool[1]
    H = sorted(held)[0]
    good = f"sáng nay mình họp về {A} rồi bàn tiếp {B} nhé"
    sents = [{"text": good, "terms": [A, B], "domain": "tech"},
             {"text": f"mình có 3 cuộc họp về {A}", "terms": [A]},
             {"text": f"hôm nay mình bàn về {A}", "terms": [A, B]},
             {"text": "sáng nay mình triển khai kubernetes lên máy", "terms": ["kubernetes"]},
             {"text": f"em đang học {H} trong lớp", "terms": [H]},
             {"text": f"hôm nay mình họp về {A} và webhook", "terms": [A]},
             {"text": f"hôm nay họp về <en>{A}</en> và <en>{B}</en>", "terms": [A]}]
    wjl(d / "sentences.jsonl", sents)
    run(["src/gipbench/dataset/synth/make_text.py", "--catalog", d / "catalog.tsv", "--out-dir", d / "ss", "--sentences", d / "sentences.jsonl",
         "--contexts-per-term", 5])
    ss = jl(d / "ss/tts_script.jsonl")
    got = [ln for ln in ss if "from_sentences" in ln["flags"]]
    check(len(got) == 1, f"phải có đúng 1 câu từ --sentences, có {len(got)}")
    if got:
        exp = f"sáng nay mình họp về <en>{A}</en> rồi bàn tiếp <en>{B}</en> nhé"
        check(got[0]["text_tagged"] == exp, f"tag sai: {got[0]['text_tagged']}")
        check(set(got[0]["terms"]) == {A, B}, "terms của câu --sentences sai")
    rep = (d / "ss/tts_script.report.md").read_text(encoding="utf-8")
    check("đọc 7, loại 6" in rep, "report --sentences không báo đúng 7 đọc / 6 loại")
    check(all(H not in ln["terms"] and "kubernetes" not in ln["text_tagged"] and "webhook" not in ln["text_tagged"] for ln in ss),
          "câu bị loại vẫn lọt vào kịch bản (đối chứng / reserved)")


def test_synth(d: Path):
    print("[3] tts_synth (mock)")
    run(["src/gipbench/dataset/synth/make_text.py", "--catalog", d / "catalog.tsv", "--bench", d / "bench.jsonl", "--out-dir", d / "sm",
         "--max-terms", 14, "--seed", 0])
    script = jl(d / "sm/tts_script.jsonl")
    base = ["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices.tsv", "--backend", "mock",
            "--backend-opt", "sample_rate=16000", "--bench", d / "bench.jsonl", "--max-voice-share", 0.4]
    rc, out = run(base + ["--out-dir", d / "raw"])
    rows = read_jsonl(str(d / "raw/train_tts_raw.jsonl"))
    check(len(rows) == len(script) and rows, f"manifest {len(rows)} != kịch bản {len(script)}")
    spans_ok = True
    for r in rows:
        miss = [k for k in REQUIRED if k not in r]
        check(not miss, f"{r['id']} thiếu trường {miss}")
        check(r["split"] == "train" and r["dataset"] == "tts" and r["synthetic"] is True and r["train_only"] is True
              and r["channel"] == "wideband", f"{r['id']} sai trường cố định")
        check(r["speaker"] == f"tts:mock:{r['tts']['voice']}" and r["source"] == f"tts:{r['id']}", f"{r['id']} speaker/source sai")
        n_span = r["text_tagged"].count("<en>")
        check(len(r["en_meta"]) == n_span and all(m["viet_level"] in ("strong", "mid", "native") for m in r["en_meta"]),
              f"{r['id']} en_meta sai")
        check("<" not in r["tts"]["tts_text"], "tts_text còn tag")
        check({"backend", "voice", "tts_text", "readings"} <= set(r["tts"]), "thiếu trường tts")
        try:
            with wave.open(r["_audio_path"], "rb") as w:
                spans_ok = spans_ok and w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
                dur = w.getnframes() / w.getframerate()
            check(abs(dur - r["duration"]) < 0.01 and 1.0 <= dur <= 20.0, f"{r['id']} duration {dur} vs {r['duration']}")
        except Exception as e:  # noqa: BLE001
            check(False, f"{r['id']} audio lỗi: {e}")
    check(spans_ok, "audio không phải 16 kHz mono PCM16")
    # trần giọng + đủ giọng cho mỗi thuật ngữ
    n = len(rows)
    sh = Counter(r["tts"]["voice"] for r in rows)
    check(max(sh.values()) / n <= 0.4 + 1e-9, f"giọng vượt trần 0.4: {sh}")
    tv = defaultdict(set)
    for r in rows:
        for rd in r["tts"]["readings"]:
            tv[rd["term"]].add(r["tts"]["voice"])
    few = {t: len(v) for t, v in tv.items() if len(v) < 3}
    check(not few, f"thuật ngữ gặp < 3 giọng: {few}")
    # xác định + chạy tiếp
    run(base + ["--out-dir", d / "raw2"])
    m1 = {r["id"]: r["speaker"] for r in rows}
    m2 = {r["id"]: r["speaker"] for r in read_jsonl(str(d / "raw2/train_tts_raw.jsonl"))}
    check(m1 == m2, "cùng seed nhưng phân giọng khác nhau")
    rc, out = run(base + ["--out-dir", d / "raw"])
    check("mới 0" in out and f"bỏ qua {n}" in out, f"chạy lại không bỏ qua câu đã có: {out[-300:]}")
    victim = rows[0]
    os.remove(victim["_audio_path"])
    rc, out = run(base + ["--out-dir", d / "raw"])
    check("mới 1" in out and os.path.exists(victim["_audio_path"]), "mất 1 file wav mà không tổng hợp lại đúng 1 câu")
    # resample thật (mock 22050 Hz -> 16 kHz) trên 3 câu
    run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices.tsv", "--backend", "mock",
         "--out-dir", d / "raw_rs", "--limit", 3])
    for r in read_jsonl(str(d / "raw_rs/train_tts_raw.jsonl")):
        x, sr = read_audio(r["_audio_path"])
        exp = count_syllables(r["tts"]["tts_text"]) * 0.2 + 0.24
        check(sr == 16000 and abs(len(x) / sr - exp) < 0.12, f"resample/cắt lặng: {len(x) / sr:.2f} s, mong ~{exp:.2f} s")
        check(abs(max(abs(v) for v in x) - 0.9) < 0.02, "chuẩn hoá đỉnh sai")
    # backend python: lỗi từng câu không dừng cả mẻ
    (d / "bad_engine.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, %r)\nfrom gipbench.dataset.synth.backends import MockBackend\n"
        "_n = [0]\n_m = MockBackend(sample_rate=16000)\n"
        "def synth(text, voice, out_path):\n    _n[0] += 1\n    if _n[0] %% 4 == 0:\n        raise RuntimeError('engine hỏng giả')\n"
        "    _m.synth(text, voice, out_path)\n" % str(G / "src"), encoding="utf-8")
    rc, out = run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices.tsv", "--backend", "python",
                   "--python-fn", f"{(d / 'bad_engine.py').as_posix()}:synth", "--out-dir", d / "raw_py", "--limit", 12,
                   "--abort-after-fails", 5])
    errs = jl(d / "raw_py/synth_errors.jsonl") if (d / "raw_py/synth_errors.jsonl").exists() else []
    got = read_jsonl(str(d / "raw_py/train_tts_raw.jsonl"))
    check(len(errs) == 3 and len(got) == 9, f"python backend: {len(errs)} lỗi, {len(got)} câu (mong 3 / 9)")
    check({e["id"] for e in errs}.isdisjoint({r["id"] for r in got}), "câu lỗi vẫn có trong manifest")
    # backend command: text qua file
    (d / "cmd_engine.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, %r)\nfrom gipbench.dataset.synth.backends import MockBackend, Voice\n"
        "text = Path(sys.argv[1]).read_text(encoding='utf-8')\nMockBackend(sample_rate=16000).synth(text, Voice(sys.argv[2]), sys.argv[3])\n"
        % str(G / "src"), encoding="utf-8")
    cmd = f'"{Path(PY).as_posix()}" "{(d / "cmd_engine.py").as_posix()}" {{text_file}} {{voice}} {{out}}'
    run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices.tsv", "--backend", "command",
         "--command", cmd, "--out-dir", d / "raw_cmd", "--limit", 3])
    rc3 = read_jsonl(str(d / "raw_cmd/train_tts_raw.jsonl"))
    check(len(rc3) == 3, f"backend command: {len(rc3)} câu (mong 3)")
    for r in rc3:
        x, sr = read_audio(r["_audio_path"])
        check(abs(len(x) / sr - (count_syllables(r["tts"]["tts_text"]) * 0.2 + 0.24)) < 0.12, "backend command không nhận đúng text")
    # ref_audio thuộc bench -> LỖI
    for name, ref in (("path", "bench_audio/b1.wav"), ("md5", "bench_audio/b1_copy.wav")):
        (d / f"voices_{name}.tsv").write_text("voice\tbackend_voice\tref_audio\tgender\tregion\n"
                                              f"v_a\t\t{ref}\tf\tnorth\nv_b\t\t\tm\tsouth\nv_c\t\t\tf\tcentral\n", encoding="utf-8")
        rc, out = run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / f"voices_{name}.tsv",
                       "--backend", "mock", "--out-dir", d / f"raw_leak_{name}", "--bench", d / "bench.jsonl", "--limit", 2],
                      expect_rc=2)
        check("LỖI" in out and "bench" in out, f"ref_audio thuộc bench ({name}) không bị báo lỗi: {out[-200:]}")
        check(not (d / f"raw_leak_{name}/train_tts_raw.jsonl").exists(), "đã tổng hợp dù ref_audio thuộc bench")
    (d / "voices_spk.tsv").write_text("voice\tbackend_voice\tref_audio\tgender\tregion\tref_speaker\n"
                                      "v_a\t\t\tf\tnorth\tbench_spk2\nv_b\t\t\tm\tsouth\t\nv_c\t\t\tf\tcentral\t\n", encoding="utf-8")
    run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices_spk.tsv", "--backend", "mock",
         "--out-dir", d / "raw_leak_spk", "--bench", d / "bench.jsonl", "--limit", 2], expect_rc=2)
    (d / "voices_ok.tsv").write_text("voice\tbackend_voice\tref_audio\tgender\tregion\n"
                                     "v_a\t\tother_ref.wav\tf\tnorth\nv_b\t\t\tm\tsouth\nv_c\t\t\tf\tcentral\n", encoding="utf-8")
    run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices_ok.tsv", "--backend", "mock",
         "--backend-opt", "sample_rate=16000", "--out-dir", d / "raw_ok", "--bench", d / "bench.jsonl", "--limit", 2])
    # quá ít giọng cho trần
    run(["src/gipbench/dataset/synth/synth.py", "--script", d / "sm/tts_script.jsonl", "--voices", d / "voices_ok.tsv", "--backend", "mock",
         "--out-dir", d / "raw_few", "--max-voice-share", 0.2, "--limit", 2], expect_rc=1)
    return rows


def hyp_variants(r, lex):
    """Các hyp giả cho một câu: chính tả / cách đọc đã dùng / cách đọc khác trong lexicon / sai hẳn / lệch tiếng Việt."""
    tagged = r["text_tagged"]
    spell = plain_text(tagged)
    used = plain_text(r["tts"]["tts_text"])
    alt = spell
    for rd in r["tts"]["readings"]:
        others = [x.reading for x in lex.get(rd["term"], []) if x.reading != rd["reading"]]
        if others:
            alt = r["tts"]["tts_text"].replace(rd["reading"], others[0])
            break
    vi_wrong = " ".join("xyz" if i % 2 == 0 else w for i, w in enumerate(spell.split()))
    return dict(spell=spell.upper(), reading=used.upper(), lexalt=alt.upper(), wrong="XIN CHÀO CÁC BẠN", viwrong=vi_wrong.upper())


def test_verify(d: Path, rows):
    print("[4] tts_verify")
    lex, _ = load_lexicon(str(G / "configs/dataset/synth/tts_lexicon.tsv"))
    # chọn câu một thuật ngữ trong lexicon có >= 2 cách đọc để thử "cách đọc khác trong lexicon"
    cand = [r for r in rows if len(r["tts"]["readings"]) == 1 and len(lex.get(r["tts"]["readings"][0]["term"], [])) >= 2]
    check(len(cand) >= 6, f"không đủ câu thử ({len(cand)})")
    plan = ["spell", "reading", "lexalt", "wrong", "viwrong"]
    hyps_a, hyps_b, expect_a, expect_both = [], [], set(), set()
    for i, r in enumerate(cand[:5]):
        v = hyp_variants(r, lex)
        kind = plan[i]
        hyps_a.append({"id": r["id"], "hyp": v[kind]})
        # hệ B: đúng ở câu 0, 1; sai ở các câu còn lại
        hyps_b.append({"id": r["id"], "hyp": v["spell"] if i < 2 else v["wrong"]})
        if kind in ("spell", "reading", "lexalt"):
            expect_a.add(r["id"])
        if i < 2:
            expect_both.add(r["id"])
    missing = cand[5]  # không có hyp ở hệ A
    sub = [r for r in cand[:6]]
    # manifest con nằm ở d/, audio tương đối theo manifest gốc -> trỏ lại bằng đường dẫn tuyệt đối
    subrows = []
    for r in sub:
        x = {k: v for k, v in r.items() if not k.startswith("_")}
        x["audio"] = Path(r["_audio_path"]).as_posix()
        subrows.append(x)
    wjl(d / "raw_sub.jsonl", subrows)
    wjl(d / "hyps_a.jsonl", hyps_a)
    wjl(d / "hyps_b.jsonl", hyps_b)
    run(["src/gipbench/dataset/synth/verify.py", "--manifest", d / "raw_sub.jsonl", "--hyps", f"A={d / 'hyps_a.jsonl'}", "--catalog", d / "catalog.tsv",
         "--out", d / "v1.jsonl"])
    kept = {r["id"] for r in read_jsonl(str(d / "v1.jsonl"))}
    check(kept == expect_a, f"một hệ: giữ {sorted(kept)} khác mong đợi {sorted(expect_a)}")
    rej = {r["id"]: r["reasons"] for r in jl(d / "v1.jsonl.rejected.jsonl")}
    check(any("span" in w for w in rej.get(cand[3]["id"], [])), f"câu 'sai hẳn' phải bị loại vì span: {rej.get(cand[3]['id'])}")
    check(any("WER" in w for w in rej.get(cand[4]["id"], [])), f"câu 'lệch tiếng Việt' phải bị loại vì WER: {rej.get(cand[4]['id'])}")
    check("không có hyp" in " ".join(rej.get(missing["id"], [])), "câu thiếu hyp phải bị loại")
    check((d / "v1.jsonl.report.md").exists(), "thiếu report")
    # hai hệ, đồng ý 2
    run(["src/gipbench/dataset/synth/verify.py", "--manifest", d / "raw_sub.jsonl", "--hyps", f"A={d / 'hyps_a.jsonl'}", f"B={d / 'hyps_b.jsonl'}",
         "--catalog", d / "catalog.tsv", "--min-agree", 2, "--out", d / "v2.jsonl"])
    kept2 = {r["id"] for r in read_jsonl(str(d / "v2.jsonl"))}
    check(kept2 == expect_both, f"min-agree 2: giữ {sorted(kept2)} khác mong đợi {sorted(expect_both)}")
    # không có hyps: chỉ âm học + cảnh báo
    rc, out = run(["src/gipbench/dataset/synth/verify.py", "--manifest", d / "raw_sub.jsonl", "--out", d / "v0.jsonl"])
    check("CẢNH BÁO" in out and "CHỈ kiểm âm học" in out, "không có --hyps mà không cảnh báo")
    check(len(read_jsonl(str(d / "v0.jsonl"))) == len(sub), "mock audio sạch phải qua kiểm âm học")
    # audio xấu dựng tay (không hyps)
    ac = []

    def mk(name, samples, clip_stat=0.0, text="xin chào các bạn hôm nay trời đẹp"):
        write_wav16(str(d / f"{name}.wav"), samples, 16000)
        ac.append({"id": name, "split": "train", "audio": (d / f"{name}.wav").as_posix(), "duration": len(samples) / 16000,
                   "text_tagged": text, "speaker": "tts:mock:v_a", "source": f"tts:{name}", "dataset": "tts", "synthetic": True,
                   "train_only": True, "en_meta": [], "tts": {"backend": "mock", "voice": "v_a", "tts_text": text, "readings": [],
                                                              "stats": {"clip_frac": clip_stat}}})
    sil = lambda s: [0.0] * int(s * 16000)  # noqa: E731
    good = sil(0.15) + tone(1.8) + sil(0.15)  # 7 âm tiết / 1.8 s ~ 3.9 âm tiết/s
    mk("good", good)
    mk("clip", good, clip_stat=0.05)
    sq = [0.9999 if (i // 40) % 2 else -0.9999 for i in range(int(1.8 * 16000))]
    mk("clip_file", sil(0.15) + sq + sil(0.15))
    mk("lead_sil", sil(1.2) + tone(1.8) + sil(0.15))
    mk("trail_sil", sil(0.15) + tone(1.8) + sil(1.2))
    mk("inner_sil", sil(0.15) + tone(0.9) + sil(2.0) + tone(0.9) + sil(0.15))
    mk("too_short", sil(0.1) + tone(0.5) + sil(0.1))
    mk("too_long", sil(0.15) + tone(21.0) + sil(0.15), text=" ".join(["xin chào"] * 60))
    mk("slow", sil(0.15) + tone(12.0) + sil(0.15))  # 7 âm tiết / 12 s = 0.6 âm tiết/s
    mk("fast", sil(0.15) + tone(1.0) + sil(0.15), text="xin chào các bạn hôm nay trời đẹp quá đi thôi nhé mọi người ơi xin chào các bạn")
    wjl(d / "raw_ac.jsonl", ac)
    run(["src/gipbench/dataset/synth/verify.py", "--manifest", d / "raw_ac.jsonl", "--out", d / "vac.jsonl"])
    kept_ac = {r["id"] for r in read_jsonl(str(d / "vac.jsonl"))}
    check(kept_ac == {"good"}, f"kiểm âm học: giữ {sorted(kept_ac)}, mong {{'good'}}")
    rej_ac = {r["id"]: " ".join(r["reasons"]) for r in jl(d / "vac.jsonl.rejected.jsonl")}
    for k, w in (("clip", "clipping"), ("clip_file", "clipping"), ("lead_sil", "lặng đầu"), ("trail_sil", "lặng đầu"),
                 ("inner_sil", "lặng giữa"), ("too_short", "thời lượng"), ("too_long", "thời lượng"), ("slow", "chậm"), ("fast", "nhanh")):
        check(w in rej_ac.get(k, ""), f"{k}: lý do loại '{rej_ac.get(k)}' không chứa '{w}'")


def test_downstream(d: Path, rows):
    print("[5] validate.py / curate_train.py")
    # manifest TTS đã lọc: dùng toàn bộ raw (không hyps) để có đủ dữ liệu
    run(["src/gipbench/dataset/synth/verify.py", "--manifest", d / "raw/train_tts_raw.jsonl", "--out", d / "train_tts.jsonl"])
    tts = read_jsonl(str(d / "train_tts.jsonl"))
    check(len(tts) > 0, "không có câu TTS sau lọc âm học")
    rc, out = run(["-m", "gipbench.dataset.release.validate", "--bench", d / "bench.jsonl", "--catalog", d / "catalog.tsv", "--train", d / "train_tts.jsonl",
                   "--block-text"], expect_rc=0)
    check("RÒ RỈ" not in out, f"validate.py báo rò rỉ trên dữ liệu TTS: {out[-400:]}")
    # curate: trộn với dữ liệu 'thật' giả lập, trần synthetic 20%
    real = []
    for i in range(60):
        real.append({"id": f"real_{i}", "split": "train", "audio": f"x/{i}.wav", "duration": 4.0, "speaker": f"spk{i % 20}",
                     "source": f"src{i}", "domain": "tech", "dataset": "ViMedCSS:train",
                     "text_tagged": f"câu thật số {i} có <en>meeting</en> và <en>server</en>"})
    wjl(d / "real.jsonl", real)
    run(["src/gipbench/model/recipe/curate.py", "--manifest", d / "real.jsonl", d / "train_tts.jsonl", "--catalog", d / "catalog.tsv",
         "--out", d / "curated.jsonl", "--max-synthetic-share", 0.2, "--max-per-term", 0, "--max-speaker-share", 0,
         "--max-domain-share", 0, "--block-text"])
    cur = read_jsonl(str(d / "curated.jsonl"))
    h_syn = sum(r["duration"] for r in cur if r.get("synthetic"))
    h_all = sum(r["duration"] for r in cur)
    h_tts_in = sum(r["duration"] for r in tts)
    check(h_syn / h_all <= 0.2 + 1e-6, f"curate_train không giữ trần synthetic: {h_syn:.1f}/{h_all:.1f} s")
    if h_tts_in / (h_tts_in + 240.0) > 0.2:
        check(h_syn / h_all > 0.15, f"trần đang hoạt động nhưng cắt quá tay: {h_syn / h_all:.3f}")
    else:
        check(abs(h_syn - h_tts_in) < 1e-6, "trần chưa chạm mà vẫn bị cắt")


def test_units():
    print("[0] đơn vị")
    v = Voice("v_x", "backend-x", "", "f", "north")
    check(str(v) == "backend-x" and v.id == "v_x" and isinstance(v, str), "Voice")
    x = tone(0.5, 22050)
    y = resample(x, 22050, 16000)
    check(abs(len(y) - 8000) <= 2, f"resample độ dài {len(y)}")
    check(term_key("Meeting ") == "meeting", "term_key")
    a = analyze(tone(1.0), 16000)
    check(abs(a["duration"] - 1.0) < 1e-6 and a["clip_frac"] == 0 and a["lead"] < 0.02, f"analyze {a}")
    import tempfile as _t
    with _t.TemporaryDirectory() as td:
        p = os.path.join(td, "m.wav")
        MockBackend(sec_per_syl=0.2).synth("xin chào các bạn", Voice("v1"), p)
        q = os.path.join(td, "m2.wav")
        MockBackend(sec_per_syl=0.2).synth("xin chào các bạn", Voice("v1"), q)
        check(open(p, "rb").read() == open(q, "rb").read(), "mock không xác định")
        MockBackend().synth("xin chào các bạn", Voice("v2"), q)
        check(open(p, "rb").read() != open(q, "rb").read(), "mock không phụ thuộc giọng")


def main():
    with tempfile.TemporaryDirectory(prefix="tts_test_") as td:
        d = Path(td)
        stages = []
        try:
            test_units()
            build_fixtures(d)
            script, held = test_make_text(d)
            stages.append(lambda: test_sentences(d, script, held))
            rows_holder = []

            def synth():
                rows_holder.append(test_synth(d))
            stages.append(synth)
            stages.append(lambda: test_verify(d, rows_holder[0]) if rows_holder else None)
            stages.append(lambda: test_downstream(d, rows_holder[0]) if rows_holder else None)
        except Exception:  # noqa: BLE001
            FAILS.append("ngoại lệ ở bước chuẩn bị:\n" + traceback.format_exc())
        for st in stages:
            try:
                st()
            except Exception:  # noqa: BLE001
                FAILS.append("ngoại lệ:\n" + traceback.format_exc())
                print(traceback.format_exc())
    print()
    if FAILS:
        print(f"KHÔNG ĐẠT: {len(FAILS)} kiểm tra sai")
        for f in FAILS[:30]:
            print(" -", f[:600])
        sys.exit(1)
    print("ĐẠT: toàn bộ kiểm tra chuỗi TTS")


if __name__ == "__main__":
    main()
