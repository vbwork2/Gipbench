#!/usr/bin/env python3
"""Dev tạm cắt từ pool VSS theo video (TODO Phần 3, phương án 18/11): chọn video ổn định, giữ lại cả video, dùng lại quyết
định nghe cũ, ghép nhãn đã nghe; make_dev_split --interim đòi nhãn người, nới ngưỡng, gắn cờ giới hạn. Dữ liệu GIẢ, CPU.

  python tests/dataset/test_holdout_dev.py
"""
import contextlib
import csv
import io
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-400:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def jl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def rd(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_review(path, rows):
    from gipbench.dataset.pseudolabel.listen_sample import REVIEW_COLS
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in REVIEW_COLS})


def test_select_apply(d):
    from gipbench.dataset.release import holdout_dev as H
    import random
    import struct
    import wave
    def make_wav(name, seed):  # wav thật, có nhiễu riêng (screen_audio bỏ audio hỏng / im lặng; sha phải khác nhau)
        p = d / "wav" / f"{name}.wav"
        p.parent.mkdir(exist_ok=True)
        rnd = random.Random(seed)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(struct.pack("<8000h", *(rnd.randint(-3000, 3000) for _ in range(8000))))
        return p

    rows = []
    for v in range(20):
        for j in range(4):
            cs = j < 3
            text = f"video {v} câu {j} nói <en>meeting</en> nhé" if cs else f"video {v} câu {j} chỉ tiếng việt"
            rows.append({"id": f"v{v}_{j}", "audio": str(make_wav(f"v{v}_{j}", v * 10 + j)), "duration": 10.0, "text_tagged": text,
                         "source": f"vss:video_{v}.mp4", "speaker": f"vss_video_{v}.mp4", "label_source": "pseudo:x"})
    pool = d / "pool.jsonl"
    jl(pool, rows)
    p1 = H.pick_videos(rows, "cs", 0.3, 0)
    p2 = H.pick_videos(list(reversed(rows)), "cs", 0.3, 0)
    check("chọn video ổn định (không phụ thuộc thứ tự dòng)", p1[0] == p2[0], (p1[0], p2[0]))
    check("đủ tỉ lệ giờ yêu cầu", p1[1] >= 0.3 * p1[2] and len(p1[0]) == 6, p1)
    check("seed khác -> tập video khác", H.pick_videos(rows, "cs", 0.3, 1)[0] != p1[0])
    first = sorted(p1[0])[0]
    vid = first.split("_")[1].split(".")[0]
    old = d / "old_reviewed.csv"
    write_review(old, [{"id": f"v{vid}_0", "text_tagged": f"video {vid} câu 0 nói <en>meeting</en> nhé", "decision": "fix",
                        "text_tagged_fixed": f"video {vid} câu 0 nói <en>meetings</en> nhé"},
                       {"id": f"v{vid}_1", "text_tagged": "nhãn cũ khác", "decision": "ok"}])
    out = d / "dev"
    with contextlib.redirect_stdout(io.StringIO()):
        H.cmd_select(Namespace(pool=str(pool), kind="cs", frac=0.3, seed=0, reuse=[str(old)], out=str(out)))
    held, cand = rd(out / "heldout.jsonl"), rd(out / "candidates.jsonl")
    sheet = list(csv.DictReader(open(out / "listen.csv", encoding="utf-8-sig")))
    check("heldout = MỌI câu của video giữ lại (cả câu không CS)", len(held) == 24 and
          {r["source"] for r in held} == p1[0], len(held))
    check("chỉ câu CS vào ứng viên", len(cand) == 18 and all("<en>" in r["text_tagged"] for r in cand), len(cand))
    check("dùng lại quyết định cũ khi nhãn không đổi (không dùng khi nhãn đổi)",
          len(sheet) == 17 and f"v{vid}_0" not in {r["id"] for r in sheet} and f"v{vid}_1" in {r["id"] for r in sheet},
          len(sheet))
    try:
        H.cmd_apply(Namespace(dir=str(out), review="listen_reviewed.csv"))
        check("apply từ chối khi chưa nghe hết", False)
    except SystemExit as e:
        check("apply từ chối khi chưa nghe hết", "chưa nghe" in str(e), e)
    for i, r in enumerate(sheet):
        r["decision"] = "drop" if i == 0 else ("fix" if i == 1 else "ok")
        if i == 1:
            r["text_tagged_fixed"] = r["text_tagged"].replace("<en>meeting</en>", "<en>meeting room</en>")
    write_review(out / "listen_reviewed.csv", sheet)
    with contextlib.redirect_stdout(io.StringIO()):
        H.cmd_apply(Namespace(dir=str(out), review="listen_reviewed.csv"))
    ver = {r["id"]: r for r in rd(out / "verified.jsonl")}
    check("drop bị bỏ, còn 17 câu", len(ver) == 17 and sheet[0]["id"] not in ver, len(ver))
    fixed = ver[sheet[1]["id"]]
    check("fix: dùng nhãn đã sửa, text bỏ tag, giữ nhãn giả cũ",
          "<en>meeting room</en>" in fixed["text_tagged"] and "<en>" not in fixed["text"] and
          fixed["text_tagged_pseudo"] == sheet[1]["text_tagged"] and fixed["label_review"] == "fix", fixed)
    check("quyết định dùng lại áp vào nhãn", "<en>meetings</en>" in ver[f"v{vid}_0"]["text_tagged"])
    check("label_source là nhãn người", all(r["label_source"].startswith("human:") for r in ver.values()))
    return out, pool


def test_interim_release(d, dev_dir, pool):
    from gipbench.core import release
    from gipbench.dataset.release import make_dev_split as mds
    bench = "bench/locked"
    rest = d / "rest.jsonl"
    held_src = {r["source"] for r in rd(dev_dir / "heldout.jsonl")}
    jl(rest, [r for r in rd(pool) if r["source"] not in held_src])
    with patch.object(release, "root", return_value=d / "releases"), patch.object(mds.R, "locked_bench_refs", return_value=[bench]):
        release.write_release(bench, manifest_rows=[{"id": "b", "audio": "x.wav", "duration": 1.0, "text_tagged": "khác hẳn",
                                                     "speaker": "bs", "source": "bsrc"}])
        base = ["--candidates", str(dev_dir / "verified.jsonl"), "--kind", "cs", "--exclude-from", str(rest),
                "--out-dev", str(d / "out" / "dev.jsonl"), "--no-acoustic", "--near-threshold", "0"]
        q = lambda a: contextlib.redirect_stdout(io.StringIO())  # noqa: E731
        with q(None):
            try:
                mds.run(base + ["--interim"])
                check("--interim đòi --carved-from", False)
            except SystemExit as e:
                check("--interim đòi --carved-from", "carved-from" in str(e), e)
            mds.run(base)
        rep = json.loads((d / "out" / "dev.jsonl.report.json").read_text(encoding="utf-8"))
        check("không interim: video VSS không tính là người nói, không đạt", rep["speakers"] == 0 and not rep["qualified"], rep)
        with q(None):
            mds.run(base + ["--interim", "--carved-from", "pool/x@1"])
        rep = json.loads((d / "out" / "dev.jsonl.report.json").read_text(encoding="utf-8"))
        check("interim: 17 câu < 300 -> chưa đạt", rep["interim"] and not rep["qualified"] and rep["unverified_labels"] == 0, rep)
        with q(None), patch.object(mds, "INTERIM_MIN_CS_ROWS", 10):
            mds.run(base + ["--interim", "--carved-from", "pool/x@1", "--release", "dev/dev_interim_cs@1"])
        rl = release.resolve("dev/dev_interim_cs@1")
        f = rl.meta["flags"]
        check("release gắn cờ interim / representative false / carved_from",
              f.get("independent") and f.get("interim") and f.get("representative") is False and
              f.get("carved_from") == ["pool/x@1"] and f.get("speaker_unit") == "video", f)
        # nhãn giả chưa nghe -> không đạt
        pseudo = d / "pseudo.jsonl"
        jl(pseudo, [dict(r, label_source="pseudo:x") for r in rd(dev_dir / "verified.jsonl")])
        with q(None), patch.object(mds, "INTERIM_MIN_CS_ROWS", 10):
            mds.run(["--candidates", str(pseudo)] + base[2:] + ["--interim", "--carved-from", "pool/x@1"])
        rep = json.loads((d / "out" / "dev.jsonl.report.json").read_text(encoding="utf-8"))
        check("interim: nhãn chưa nghe -> không đạt", not rep["qualified"] and rep["unverified_labels"] == 17, rep)


def test_triage(d):
    from gipbench.dataset.release import holdout_dev as H
    t = d / "tri"
    cand = [{"id": f"u{i}", "audio": "x.wav", "duration": 3.0, "text_tagged": f"câu số {i} hôm nay trời đẹp",
             "source": "vss:v.mp4", "label_source": "distill:gip15"} for i in range(30)]
    jl(t / "candidates.jsonl", cand)
    write_review(t / "listen.csv", [dict(r, split="listen", text_tagged_fixed=r["text_tagged"]) for r in cand])
    # 20 câu đầu: cả hai hyp khớp (khác hoa / dấu câu / chữ số vẫn tính là khớp); 10 câu sau: một hyp lệch
    jl(t / "h1.jsonl", [{"id": r["id"], "hyp": r["text_tagged"].replace("câu số", "Câu số") + "."} for r in cand])
    jl(t / "h2.jsonl", [{"id": r["id"], "hyp": r["text_tagged"] if i < 20 else "khác hẳn"} for i, r in enumerate(cand)])
    with contextlib.redirect_stdout(io.StringIO()):
        H.cmd_triage(Namespace(dir=str(t), hyps=[str(t / "h1.jsonl"), str(t / "h2.jsonl")], audit=5, seed=0))
    meta = json.loads((t / "triage_meta.json").read_text(encoding="utf-8"))
    sheet = list(csv.DictReader(open(t / "listen.csv", encoding="utf-8-sig")))
    check("triage: 20 khớp, nhận luôn 15, phiếu còn 10 lệch + 5 kiểm mù",
          meta["n_agree"] == 20 and meta["n_auto"] == 15 and len(sheet) == 15 and (t / "listen_full.csv").exists(), meta)
    check("triage: câu kiểm mù nằm trong phiếu, không có dấu hiệu nhận diện",
          set(meta["audit_ids"]) <= {r["id"] for r in sheet} and all(not r["note"] for r in sheet))
    for r in sheet:
        r["decision"] = "fix" if r["id"] == meta["audit_ids"][0] else "ok"
        if r["decision"] == "fix":
            r["text_tagged_fixed"] = r["text_tagged"] + " nhé"
    write_review(t / "listen_reviewed.csv", sheet)
    with contextlib.redirect_stdout(io.StringIO()):
        H.cmd_apply(Namespace(dir=str(t), review="listen_reviewed.csv"))
    ver = rd(t / "verified.jsonl")
    n_auto = sum(r["label_source"].startswith("auto:asr_agree") for r in ver)
    aud = json.loads((t / "triage_audit.json").read_text(encoding="utf-8"))
    check("apply: 15 câu nhãn auto, 15 câu nhãn người; kiểm mù 1 / 5 không ok",
          len(ver) == 30 and n_auto == 15 and aud["audit_not_ok"] == 1 and aud["audit_n"] == 5, (n_auto, aud))


def test_hint():
    from gipbench.dataset.release.holdout_dev import hint_for
    h = hint_for("mình làm <en>sale</en> là phải chấp nhận thất bại",
                 {"qwen3": "mình làm sale là phải chấp nhận được thất bại", "whisper": "mình làm sell là phải chấp nhận thất bại"})
    parts = h.split(" || ")
    check("hint: đánh dấu từ thay thế và chỗ chèn trên nhãn", parts[0] == "nghe kỹ (2 từ): mình làm ⟦sale⟧ là phải chấp "
          "nhận ⟦thất⟧ bại", parts[0])
    check("hint: từng ASR, số từ lệch", parts[1].startswith("qwen3 (lệch 1)") and "⟦được⟧" in parts[1] and
          parts[2] == "whisper (lệch 1): mình làm ⟦sell⟧ là phải chấp nhận thất bại", parts[1:])
    check("hint: khớp hết (bỏ qua hoa / dấu câu)", hint_for("hôm nay trời đẹp", {"q": "Hôm nay trời đẹp."}).startswith("mọi ASR khớp"))


def main():
    import shutil
    from gipbench.core.paths import REPO
    (REPO / "data" / "tmp").mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=REPO / "data" / "tmp"))  # cùng ổ với repo (write_release ghi đường dẫn tương đối)
    try:
        dev_dir, pool = test_select_apply(d)
        test_interim_release(d, dev_dir, pool)
        test_triage(d)
        test_hint()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'HỎNG ' + str(len(FAIL)) if FAIL else 'TẤT CẢ ĐẠT'}")
    return 1 if FAIL else 0


def test_all():
    assert main() == 0, FAIL


if __name__ == "__main__":
    sys.exit(main())
