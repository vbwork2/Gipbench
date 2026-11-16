#!/usr/bin/env python3
"""Kiểm thử chế độ khai thác câu KHÓ (lib.agreement.mine_hard + tools/agreement_filter.py --mode hard), không cần mạng / GPU.

  python tools/test_hard_mining.py

Thoát mã != 0 nếu có ca sai. Gồm: ca tay cho mine_hard (nhận / loại từng điều kiện), chạy thật tools/agreement_filter.py
--mode hard trên dữ liệu giả (nhãn, --easy-keep-frac, bộ chặn reserved_b3), và so chế độ confirm với bản trước khi sửa
(git show HEAD:...; bỏ qua nếu HEAD đã chứa chế độ hard).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.agreement import has_span, mine_hard, pair_wer, tag_text  # noqa: E402
from lib.leak import LeakGuard  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402

FAILS = []
NOVW, NODEC = frozenset(), {}  # tắt vi_abbrev / vi_loanwords / term_decisions: ca tay không phụ thuộc configs/


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  LỖI  ") + name + ("" if cond else f"  -> {detail}"))
    if not cond:
        FAILS.append(name)


def mine(teachers, student, cat, **kw):
    return mine_hard(teachers, student, cat, vi_words=NOVW, term_dec=NODEC, **kw)


def unit(tmp):
    cat_p = Path(tmp) / "catalog.tsv"
    cat_p.write_text("term\tcategory\tvariants\tphonetic\treserved_b3\n"
                     "deadline\ttech\t\t\t0\nblockchain\ttech\t\t\t1\n", encoding="utf-8")
    cat = Catalog.load(str(cat_p))
    T = "Chiều nay mình họp team để chốt deadline nhé."
    T2 = T.lower().rstrip(".")
    print("mine_hard, ca tay")

    r = mine([T, T2], "chiều nay mình họp tim để chốt đét lai nhé", cat)
    check("thầy đồng thuận + trò sai -> hard", r["status"] == "hard", r)
    check("nhãn = text thầy chính kèm <en>", r["text_tagged"] == "Chiều nay mình họp <en>team</en> để chốt <en>deadline</en> nhé.",
          r["text_tagged"])
    check("teacher_spans / student_spans song song", r["teacher_spans"] == ["team", "deadline"]
          and r["student_spans"] == ["tim", "đét lai"], (r["teacher_spans"], r["student_spans"]))
    check("teacher_pair_wer, student_vi_wer", r["teacher_pair_wer"] == 0.0 and r["student_vi_wer"] == 0.0, r)

    r = mine([T, T2], "chiều nay mình họp team để chốt deadline nhé", cat)
    check("trò đúng -> không phải hard (easy)", r["status"] == "easy" and not r["teacher_spans"], r)

    r = mine([T, "chiều nay mình họp tim để chốt deadline nhé"], "chiều nay mình họp tim để chốt đét lai nhé", cat)
    check("hai thầy bất đồng ở span -> loại", r["status"] == "reject" and r["reason"] == "span_confirmed", r)

    r = mine(["Hôm nay tôi đi nộp thuế cho công ty."] * 2, "hôm nay tôi đi nộp thuế cho công ty", cat)
    check("thầy đồng thuận nhưng không có từ Anh -> loại", r["status"] == "reject" and r["reason"] == "has_span", r)

    r = mine_hard(["Hôm nay tôi đi nộp BHXH cho công ty."] * 2, "hôm nay tôi đi nộp bê hát xê hát cho công ty", cat)
    check("viết tắt trong configs/vi_abbrev.txt (bhxh) không thành span (auto_tag mặc định)", r["reason"] == "has_span", r)

    guard = LeakGuard([], cat)
    leak = lambda t: guard.reason({}, t)  # noqa: E731
    Tb = "mình đang tìm hiểu blockchain nhé"
    r = mine([Tb, Tb], "mình đang tìm hiểu bờ lốc chen nhé", cat, leak=leak)
    check("span là thuật ngữ reserved_b3 -> loại", r["status"] == "reject" and r["reason"] == "leak", r)
    r = mine([Tb, Tb], "mình đang tìm hiểu bờ lốc chen nhé", cat)
    check("(đối chứng) bỏ bộ chặn thì cùng câu là hard", r["status"] == "hard", r)

    r = mine(["mình cần setup lại máy"] * 2, "mình cần set up lại máy", cat)
    check("thầy 'setup', trò 'set up' không coi là trò sai", r["status"] == "easy", r)
    r = mine(["mình cần set up lại máy"] * 2, "mình cần setup lại máy", cat)
    check("thầy 'set up', trò 'setup' không coi là trò sai", r["status"] == "easy", r)
    check("has_span: ghép bỏ dấu cách", has_span(["set", "up"], ("setup",)) and has_span(["api"], ("a", "p", "i"))
          and not has_span(["setup"], ("set",)) and not has_span(["tim"], ("team",)))

    r = mine(["họp team ngay bây giờ"] * 1 + ["đi ăn cơm với gia đình nhé các bạn"], "họp team", cat)
    check("hai thầy lệch xa (WER cặp > 0.15) -> loại", r["reason"] == "pair", r)
    r = mine([T, T2], "alo alo alo", cat)
    check("trò lệch phần tiếng Việt quá ngưỡng (audio hỏng) -> loại", r["reason"] == "student_vi", r)
    r = mine(["meeting deadline", "meeting deadline"], "mít tinh", cat, max_en_frac=0.5)
    check("câu gần như toàn tiếng Anh -> loại theo max_en_frac", r["reason"] == "en_frac", r)
    r = mine([T], "x", cat)
    check("chỉ 1 thầy -> loại", r["reason"] == "hyp", r)
    r = mine([T, T2], None, cat)
    check("chưa có hyp trò, đạt mọi điều kiện phía thầy -> need_student", r["status"] == "need_student", r)
    r = mine(["Hôm nay tôi đi nộp thuế cho công ty."] * 2, None, cat)
    check("chưa có hyp trò nhưng thầy không có span -> loại luôn (không cần decode trò)", r["reason"] == "has_span", r)
    check("pair_wer đối xứng", pair_wer(["a", "b", "c"], ["a", "c"]) == pair_wer(["a", "c"], ["a", "b", "c"]))
    check("tag_text tôn trọng vi_words truyền vào", "<en>" in tag_text("họp team", cat, NOVW, NODEC)
          and "<en>" not in tag_text("họp team", cat, frozenset({"team"}), NODEC))


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sh(*a, cwd=G):
    p = subprocess.run([sys.executable, *map(str, a)], capture_output=True, text=True, encoding="utf-8", cwd=str(cwd))
    return p.returncode, p.stdout, p.stderr


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def end_to_end(tmp):
    print("tools/agreement_filter.py --mode hard, dữ liệu giả")
    tmp = Path(tmp)
    cat = tmp / "catalog.tsv"
    base = {"split": "train", "dataset": "vss_cs", "source": "vss:s", "domain": "tech", "duration": 4.0,
            "label_source": "pseudo:Zipformer-30M-RNNT-6000h", "train_only": True}
    cases = {  # id -> (nhãn giả, thầy 1, thầy 2, trò)
        "h1": ("chiều nay mình họp <en>tim</en> nhé", "Chiều nay mình họp team nhé.", "chiều nay mình họp team nhé", "chiều nay mình họp tim nhé"),
        "e1": ("mình đặt <en>deadline</en> nhé", "Mình đặt deadline nhé.", "mình đặt deadline nhé", "mình đặt deadline nhé"),
        "n1": ("hôm nay tôi đi làm", "Hôm nay tôi đi làm.", "hôm nay tôi đi làm", "hôm nay tôi đi làm"),
        "r1": ("mình học <en>blockchain</en>", "Mình học blockchain.", "mình học blockchain", "mình học bờ lốc chen"),
        "d1": ("họp <en>tim</en> nhé", "Họp team nhé.", "họp tim nhé", "họp tim nhé"),
        "m1": ("họp <en>team</en> nhé", "Họp team nhé.", "họp team nhé", None),  # thiếu hyp trò: chưa xét
    }
    rows = [dict(base, id=i, speaker=f"spk_{i}", text=t.replace("<en>", "").replace("</en>", ""), text_tagged=t)
            for i, (t, *_) in cases.items()]
    write_jsonl(tmp / "m.jsonl", rows)
    for k, name in ((1, "fw"), (2, "qwen")):
        write_jsonl(tmp / name / "h.jsonl", [{"id": i, "hyp": c[k]} for i, c in cases.items()])
    write_jsonl(tmp / "stu" / "hyps.jsonl", [{"id": i, "hyp": c[3]} for i, c in cases.items() if c[3] is not None])
    cat.write_text("term\tcategory\tvariants\tphonetic\treserved_b3\nblockchain\ttech\t\t\t1\n", encoding="utf-8")

    def run(out, *extra):
        rc, so, se = sh("tools/agreement_filter.py", "--mode", "hard", "--manifest", tmp / "m.jsonl", "--hyps", tmp / "fw",
                        tmp / "qwen", "--student-hyps", tmp / "stu" / "hyps.jsonl", "--catalog", cat, "--block-text",
                        "--out", tmp / out, *extra)
        assert rc == 0, (rc, so, se)
        return {r["id"]: r for r in read(tmp / out)}, so

    kept, so = run("hard.jsonl")
    check("chỉ câu khó được giữ (mặc định easy-keep-frac 0)", set(kept) == {"h1"}, sorted(kept))
    h = kept.get("h1", {})
    check("nhãn / text / label_source / train_only", h.get("text_tagged") == "Chiều nay mình họp <en>team</en> nhé."
          and h.get("text") == "Chiều nay mình họp team nhé." and h.get("label_source") == "teacher_consensus:fw+qwen"
          and h.get("train_only") is True and h.get("text_orig") == "chiều nay mình họp <en>tim</en> nhé", h)
    check("mining", h.get("mining") == {"mode": "hard", "teacher_spans": ["team"], "student_spans": ["tim"],
                                        "teacher_pair_wer": 0.0, "student_vi_wer": 0.0}, h.get("mining"))
    rep = (tmp / "hard.jsonl.report.md").read_text(encoding="utf-8")
    check("report có phễu, cặp thầy -> trò, domain", "| team | tim | 1 |" in rep and "tech" in rep and "chưa xét" in rep.lower()
          and "reserved_b3" in rep, rep)
    check("câu thiếu hyp trò chỉ đếm chưa xét", "thiếu hyp trò: 1" in rep, rep)
    run("need.jsonl", "--write-need-student", tmp / "need_ids.jsonl")
    check("--write-need-student: chỉ câu thiếu hyp trò mà thầy đã đủ điều kiện (m1)",
          [r["id"] for r in read(tmp / "need_ids.jsonl")] == ["m1"], read(tmp / "need_ids.jsonl"))
    kept, _ = run("hard1.jsonl", "--easy-keep-frac", "1")
    check("--easy-keep-frac 1: câu trò đã đúng được nhận, nhãn mining.mode = easy",
          set(kept) == {"h1", "e1"} and kept["e1"]["mining"]["mode"] == "easy", sorted(kept))
    kept, _ = run("hard0.jsonl", "--easy-keep-frac", "0")
    check("--easy-keep-frac 0: không nhận câu dễ", set(kept) == {"h1"}, sorted(kept))
    a, _ = run("seed_a.jsonl", "--easy-keep-frac", "0.5", "--seed", "3")
    b, _ = run("seed_b.jsonl", "--easy-keep-frac", "0.5", "--seed", "3")
    check("--easy-keep-frac có seed: lặp lại cho cùng kết quả", list(a) == list(b))
    # câu r1 (blockchain, reserved_b3) và d1 (thầy bất đồng) không bao giờ vào, kể cả khi giữ hết câu dễ
    kept, _ = run("all.jsonl", "--easy-keep-frac", "1")
    check("reserved_b3 và thầy bất đồng bị loại kể cả --easy-keep-frac 1", not ({"r1", "d1", "n1"} & set(kept)), sorted(kept))
    rc, _, _ = sh("tools/agreement_filter.py", "--mode", "hard", "--manifest", tmp / "m.jsonl", "--hyps", tmp / "fw",
                  "--out", tmp / "x.jsonl")
    check("--mode hard thiếu --student-hyps / chỉ 1 thầy -> thoát mã != 0", rc != 0)


# ---- chế độ confirm không đổi: so từng byte với bản trước khi sửa (git show HEAD:...) ----
def head_file(rel):
    p = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=str(G))
    return p.stdout.decode("utf-8") if p.returncode == 0 else None


def confirm_unchanged(tmp):
    print("chế độ confirm so với bản HEAD")
    old_agr, old_tool = head_file("lib/agreement.py"), head_file("tools/agreement_filter.py")
    if old_agr is None or old_tool is None:
        print("  bỏ qua: không đọc được HEAD (không có git?)")
        return
    if "mine_hard" in old_agr:
        print("  bỏ qua: HEAD đã chứa chế độ hard (đã commit), không còn bản 'trước khi sửa' để so")
        return
    tmp = Path(tmp)
    old = tmp / "old"
    shutil.copytree(G / "lib", old / "lib", ignore=shutil.ignore_patterns("__pycache__"))
    (old / "tools").mkdir(parents=True)
    blk = G / "data" / "raw" / "consensus" / "leak_block.txt"
    if blk.exists():  # cùng danh sách chặn mặc định ở cả hai cây thư mục
        (old / "data" / "raw" / "consensus").mkdir(parents=True)
        shutil.copy(blk, old / "data" / "raw" / "consensus" / "leak_block.txt")
    (old / "lib" / "agreement.py").write_text(old_agr, encoding="utf-8", newline="")
    (old / "tools" / "agreement_filter.py").write_text(old_tool, encoding="utf-8", newline="")
    base = {"split": "train", "dataset": "vss_cs", "source": "vss:s", "domain": "tech", "duration": 4.0}
    samples = [("a", "họp <en>team</en> để chốt <en>deadline</en>", "họp team để chốt deadline", "họp tim để chốt deadline"),
               ("b", "mình học <en>blockchain</en> nhé", "mình học blockchain nhé", "mình học bờ lốc chen nhé"),
               ("c", "đặt <en>setup</en> lại máy", "đặt set up lại máy", "đặt setup lại máy"),
               ("d", "họp <en>tim</en> nhé", "họp team nhé", "họp team nhé"),
               ("e", "xin chào các bạn", "xin chào các bạn", "xin chào các bạn")]
    write_jsonl(tmp / "c.jsonl", [dict(base, id=i, speaker=f"s{i}", text=t.replace("<en>", "").replace("</en>", ""), text_tagged=t)
                                  for i, t, *_ in samples])
    write_jsonl(tmp / "c1" / "h.jsonl", [{"id": i, "hyp": a} for i, _, a, _ in samples])
    write_jsonl(tmp / "c2" / "h.jsonl", [{"id": i, "hyp": b} for i, _, _, b in samples])
    (tmp / "cat.tsv").write_text("term\tcategory\tvariants\tphonetic\treserved_b3\ndeadline\ttech\t\t\t0\n"
                                 "blockchain\ttech\t\t\t1\n", encoding="utf-8")
    runs = [["--min-agree", "1"], ["--min-agree", "2", "--min-span-frac", "0.5"], ["--relabel-spans", "--relabel-max-vi-wer", "0.5"],
            ["--tiered"], ["--max-vi-wer", "0.5", "--write-unconfirmed-ids", "UNC", "--only-with-hyps"]]
    for k, extra in enumerate(runs):
        outs = {}
        for tag, script in (("old", old / "tools" / "agreement_filter.py"), ("new", G / "tools" / "agreement_filter.py")):
            d = tmp / f"run_{k}_{tag}"
            d.mkdir()
            ex = [str(d / "unc.txt") if x == "UNC" else x for x in extra]
            # lệnh chạy từ chính thư mục kết quả để đường dẫn in ra giống nhau
            rc, so, se = sh(script, "--manifest", tmp / "c.jsonl", "--hyps", tmp / "c1", tmp / "c2", "--catalog", tmp / "cat.tsv",
                            "--seed", "0", *ex, "--out", d / "o.jsonl")
            assert rc == 0, (tag, extra, so, se)
            outs[tag] = ({f.name: f.read_bytes() for f in sorted(d.iterdir())},
                         so.replace(str(d), "<D>").replace(str(old), "<R>").replace(str(G), "<R>"))
        check(f"confirm {' '.join(extra)}: file ra + stdout giống hệt bản cũ",
              outs["old"] == outs["new"], {t: (sorted(v[0]), v[1][:200]) for t, v in outs.items()})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        unit(tmp)
        end_to_end(tmp)
        confirm_unchanged(tmp)
    if FAILS:
        print(f"\nLỖI: {len(FAILS)} ca sai: " + "; ".join(FAILS))
        sys.exit(1)
    print("\nTất cả ca đạt.")


if __name__ == "__main__":
    main()
