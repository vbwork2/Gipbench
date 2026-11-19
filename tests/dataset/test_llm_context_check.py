#!/usr/bin/env python3
"""LLM đánh dấu từ nghi sai ngữ cảnh trên phiếu nghe (dataset/label/llm_context_check.py): chỉ số từ khớp trang duyệt,
lọc trả lời hỏng của LLM, cache theo nhãn, giữ nguyên cột phiếu; trang review_bench.html chỉ tô từ còn khớp. Dữ liệu GIẢ,
không gọi mạng (thay hàm chat).

  python tests/dataset/test_llm_context_check.py
"""
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

G = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(G / "src"))
from gipbench.dataset.label import llm_context_check as L  # noqa: E402

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-400:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


COLS = ["id", "split", "audio", "duration", "text_tagged", "text_tagged_orig", "confidence", "flags", "decision",
        "text_tagged_fixed", "viet_levels", "reading_note", "region", "gender", "note", "hint"]
ROWS = [
    {"id": "a", "text_tagged": "có một cái <en>connection</en> khá tốt nó còn thô sơ",
     "hint": "nghe kỹ (1 từ): có một cái connection khá tốt nó còn ⟦thô⟧ sơ || whisper (lệch 1): có một cái connection khá "
             "tốt nó còn ⟦thâu⟧ sơ || qwen3 (lệch 1): có một cái connection ⟦∅⟧ tốt nó còn thô sơ"},
    {"id": "b", "text_tagged": "hôm nay trời đẹp", "hint": ""},
    {"id": "c", "text_tagged": "đi họp <en>meeting</en> nha", "decision": "ok", "hint": ""},
]


def write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames), list(rd)


def test_units():
    ts = L.tokens(ROWS[0]["text_tagged"])
    check("tokens: chỉ số + cờ EN", [t["w"] for t in ts][3:5] == ["connection", "khá"] and ts[3]["en"] and not ts[4]["en"], ts)
    alts = L.asr_alts(ROWS[0]["hint"])
    check("asr_alts: bỏ dòng 'nghe kỹ', bỏ ⟦ ⟧ và ⟦∅⟧",
          alts == {"whisper": "có một cái connection khá tốt nó còn thâu sơ", "qwen3": "có một cái connection tốt nó còn thô sơ"},
          alts)
    p = L.user_prompt(ROWS[0])
    check("prompt đánh số, đánh dấu [EN]", "3:[EN]connection" in p and "8:thô" in p and "- whisper:" in p, p)
    raw = {"suspects": [
        {"i": 8, "w": "thô", "kind": "ctx", "maybe": "thâu", "why": "x"},  # đúng
        {"i": 2, "w": "connection", "kind": "en"},                         # lệch chỉ số -> sửa về 3
        {"i": 99, "w": "không có", "kind": "ctx"},                         # từ không có -> bỏ
        {"i": 8, "w": "thô", "kind": "ctx"},                               # trùng -> bỏ
        {"i": "5", "w": "tốt", "kind": "lạ", "maybe": "tốc"},              # chỉ số dạng chuỗi, loại lạ -> ctx
        {"i": 6, "w": "nó", "kind": "ctx", "why": "từ thừa"},              # ctx không đề xuất -> bỏ
        {"i": 7, "w": "còn", "kind": "untag", "maybe": "Còn"},             # đề xuất trùng chính từ -> bỏ
        "rác"]}
    sus = L.clean_suspects(raw, ts)
    check("clean_suspects: sửa chỉ số, bỏ rác / trùng / không đề xuất, chuẩn loại",
          [(s["i"], s["w"], s["kind"]) for s in sus] == [(3, "connection", "en"), (5, "tốt", "ctx"), (8, "thô", "ctx")], sus)
    check("clean_suspects: trả lời không phải dict", L.clean_suspects(["x"], ts) == [] and L.clean_suspects(None, ts) == [])


def test_run(d):
    sheet = d / "listen.csv"
    write(sheet, ROWS)
    calls = []

    def chat(system, user):
        calls.append(user)
        if "thô" in user:
            return {"json": {"suspects": [{"i": 8, "w": "thô", "kind": "ctx", "maybe": "thâu", "why": "thô sơ hợp nghĩa?"}]},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return {"json": {"suspects": []}, "usage": {}}

    args = Namespace(dir=str(d), sheet="listen.csv", model="m", reasoning_effort=None, workers=2, limit=0,
                     skip_decided=[], dry_run=False)
    rc = L.run(args, chat=chat)
    header, rows = read(sheet)
    check("run: thoát 0, hỏi 2 câu (bỏ câu đã ok)", rc == 0 and len(calls) == 2, (rc, len(calls)))
    check("run: giữ cột cũ, thêm llm_suspects ở cuối", header == COLS + [L.COL], header)
    got = {r["id"]: r[L.COL] for r in rows}
    check("run: câu a có từ nghi, câu b rỗng, câu c không đụng",
          json.loads(got["a"])[0]["w"] == "thô" and got["b"] == "" and got["c"] == "", got)
    check("run: hint / text giữ nguyên", rows[0]["hint"] == ROWS[0]["hint"] and rows[0]["text_tagged"] == ROWS[0]["text_tagged"])
    rc = L.run(args, chat=chat)
    check("run lại: dùng cache, không hỏi thêm", rc == 0 and len(calls) == 2, len(calls))
    _, cur = read(sheet)
    cur[1]["text_tagged"] = "hôm nay trời mưa"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + [L.COL])
        w.writeheader()
        w.writerows(cur)
    L.run(args, chat=chat)
    check("đổi nhãn: chỉ hỏi lại câu đổi", len(calls) == 3 and "mưa" in calls[-1], calls[-1:])
    rev = d / "listen_reviewed.csv"
    write(rev, [dict(ROWS[0], decision="fix")])
    calls.clear()
    args.skip_decided = [str(rev)]
    args.model = "m2"
    L.run(args, chat=chat)
    check("--skip-decided: bỏ câu đã nghe ở phiếu xuất", len(calls) == 1 and "mưa" in calls[0], calls)

    def bad(system, user):
        raise RuntimeError("mạng hỏng")
    args.model, args.skip_decided = "m3", []
    check("lỗi API: thoát 1, phiếu vẫn ghi được", L.run(args, chat=bad) == 1 and read(sheet)[0] == COLS + [L.COL])
    args.dry_run = True
    check("--dry-run không cần key", L.run(args) == 0)


def test_page():
    node = shutil.which("node") or shutil.which("node.exe")
    if not node:
        print("BỎ QUA test review_bench.html: không tìm thấy node")
        return
    core = re.search(r'<script id="core">(.*?)</script>', (G / "src/gipbench/dataset/label/review_bench.html").read_text(
        encoding="utf-8"), re.S).group(1)
    raw = json.dumps([{"i": 3, "w": "connection", "kind": "en"}, {"i": 8, "w": "thô", "kind": "ctx"}], ensure_ascii=False)
    drv = (core + f"\nconst raw = {json.dumps(raw, ensure_ascii=False)};\n"
           f"const a = suspectsFor(raw, tokenizeTagged({json.dumps(ROWS[0]['text_tagged'], ensure_ascii=False)}));\n"
           "const b = suspectsFor(raw, tokenizeTagged('có một cái connection khá tốt nó còn thâu sơ'));\n"
           "process.stdout.write(JSON.stringify([a.map(s => s.i), b.map(s => s.i), suspectsFor('{hỏng', []), suspectsFor('', [])]));\n")
    p = subprocess.run([node, "-"], input=drv.encode("utf-8"), capture_output=True)
    out = json.loads(p.stdout.decode("utf-8")) if p.returncode == 0 else p.stderr.decode("utf-8", "replace")
    check("trang: suspectsFor khớp chỉ số Python, bỏ mục lệch sau khi sửa chữ, chịu JSON hỏng",
          out == [[3, 8], [3], [], []], out)
    # sửa một từ: thay / gắn EN / bỏ EN / xoá; gợi ý LLM được làm sạch; mục nghi sau chỗ sửa vẫn bám đúng từ
    js = r"""
const T = 'có một cái <en>connection</en> khá tốt nó còn thô sơ';
const S = (t, i, x, en) => serializeTokens(replaceWord(tokenizeTagged(t), i, x, en));
const raw = JSON.stringify([{i: 8, w: "thô", kind: "ctx"}]);
process.stdout.write(JSON.stringify([
  S(T, 8, 'thâu'),
  S('nó không mát với nhau', 2, 'match', true),
  S(T, 3, 'kết nối', false),
  S(T, 3, 'collection'),
  S(T, 0, ''),
  S('a <en>b c</en> d', 1, 'x'),
  serializeTokens(replaceWord(tokenizeTagged('nhiều hơn luật vi cũng'), 2, 'lượt view', undefined, 2)),
  serializeTokens(replaceWord(tokenizeTagged('a b c'), 1, 'x', undefined, 9)),
  cleanMaybe("match (hoặc 'khớp')"), cleanMaybe('đâu đó / từ'), cleanMaybe('"Hòa"'), cleanMaybe(''),
  suspectsFor(raw, tokenizeTagged(S(T, 0, 'có hai'))).map(s => s.i),
  suspectsFor(raw, tokenizeTagged(S(T, 8, 'thâu'))).length,
]));
"""
    p = subprocess.run([node, "-"], input=(core + js).encode("utf-8"), capture_output=True)
    out = json.loads(p.stdout.decode("utf-8")) if p.returncode == 0 else p.stderr.decode("utf-8", "replace")
    want = ["có một cái <en>connection</en> khá tốt nó còn thâu sơ",
            "nó không <en>match</en> với nhau",
            "có một cái kết nối khá tốt nó còn thô sơ",
            "có một cái <en>collection</en> khá tốt nó còn thô sơ",
            "một cái <en>connection</en> khá tốt nó còn thô sơ",
            "a <en>x c</en> d",
            "nhiều hơn lượt view cũng", "a x",
            "match", "đâu đó", "Hòa", "", [9], 0]
    check("trang: replaceWord / cleanMaybe / suspectsFor sau khi sửa", out == want,
          [(k, a, b) for k, (a, b) in enumerate(zip(out, want)) if a != b] if isinstance(out, list) else out)


def main():
    d = Path(tempfile.mkdtemp())
    try:
        test_units()
        test_run(d)
        test_page()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'HỎNG ' + str(len(FAIL)) if FAIL else 'TẤT CẢ ĐẠT'}")
    return 1 if FAIL else 0


def test_all():
    assert main() == 0, FAIL


if __name__ == "__main__":
    sys.exit(main())
