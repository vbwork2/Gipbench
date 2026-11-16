#!/usr/bin/env python3
"""Test chỉ số theo span (exact / cer / macro / lát tần suất train) và luật không kém hơn của leaderboard.
Chạy: python tools/test_span_metrics.py   (thoát mã != 0 nếu có ca sai). Chỉ dùng dữ liệu tay, không cần bench thật."""
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
import leaderboard  # noqa: E402
from lib.align import _op_spans, align, attribute, char_edit_distance, span_hyps  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402
from score import agg, bootstrap_many, freq_slice, freq_slices, score_utt, term_weights, train_term_freq  # noqa: E402

CAT = Catalog()
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "LỖI ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        FAILS.append(name)


def close(a, b, eps=1e-9):
    return a is not None and b is not None and abs(a - b) < eps


def utt(tagged, hyp, i=0, split="B1", spk="s1", freq=None):
    r = {"id": f"u{i}", "split": split, "speaker": spk, "text_tagged": tagged}
    return score_utt(r, hyp, CAT, freq=freq)


# ---------------------------------------------------------------- exact / cer
u = utt("mở <en>setup</en> đi", "mở set up đi")
s = u["spans"][0]
check("set up vs setup: exact", s["exact"] and s["cer"] == 0 and s["hyp"] == "set up", s)
check("set up vs setup: CS-WER cũ vẫn tính lỗi", s["err"] == 2 and u["err_en"] == 2, s)

u = utt("bệnh <en>bilirubin</en> cao", "bệnh billy rubin cao")
s = u["spans"][0]
check("bilirubin vs billy rubin: không exact", not s["exact"], s)
check("bilirubin vs billy rubin: cer < 1", 0 < s["cer"] < 1 and close(s["cer"], 2 / 9), s)
check("bilirubin vs billy rubin: CS-WER cũ = 2 lỗi / 1 token", s["err"] == 2 and close(agg([u])["cs_wer"], 200.0), s)

u = utt("hôm nay <en>team</en> họp", "hôm nay team họp")
s = u["spans"][0]
check("span đúng hoàn toàn", s["exact"] and s["err"] == 0 and s["cer"] == 0 and s["hyp"] == "team", s)

u = utt("hôm nay <en>team</en> họp", "hôm nay họp")
s = u["spans"][0]
check("span bị xoá: cer = 1, không exact", s["cer"] == 1 and not s["exact"] and s["hyp"] == "" and s["err"] == 1, s)

u = utt("<en>ai</en> hay", "ây ai ây ai hay")  # chèn nhiều: cer thô > 1, agg cắt ở 1
s = u["spans"][0]
check("cer lưu thô (có thể > 1), span_cer cắt ở 1", s["cer"] > 1 and close(agg([u])["span_cer"], 100.0), s)

check("char_edit_distance", char_edit_distance("kitten", "sitting") == 3 and char_edit_distance("", "ab") == 2
      and char_edit_distance("ab", "") == 2 and char_edit_distance("abc", "abc") == 0)

# ---------------------------------------------------------------- span_hyps nhất quán với attribute
rnd = random.Random(1)
vocab = ["a", "b", "c", "d", "e"]
bad = 0
for _ in range(300):
    ref, sp, nsp = [], [], 0
    for _ in range(rnd.randint(1, 5)):  # xen kẽ đoạn Việt (-1) và span Anh liên tục
        L = rnd.randint(1, 3)
        ref += [rnd.choice(vocab) for _ in range(L)]
        if rnd.random() < 0.5:
            sp += [nsp] * L
            nsp += 1
        else:
            sp += [-1] * L
    hyp = [rnd.choice(vocab) for _ in range(rnd.randint(0, 8))]
    ops = align(ref, hyp)
    errs, _, _, _ = attribute(ops, sp)
    own = _op_spans(ops, sp)
    cnt = {}
    for (t, _, _), o in zip(ops, own):
        if t != "C":
            cnt[o] = cnt.get(o, 0) + 1
    sh = span_hyps(ops, sp, hyp, nsp)
    n_hyp_en = sum(len(x) for x in sh)
    n_hyp_en_ref = sum(1 for (t, _, hj), o in zip(ops, own) if o >= 0 and hj is not None)
    if cnt != errs or n_hyp_en != n_hyp_en_ref:
        bad += 1
check("span_hyps / _op_spans khớp attribute (300 ca ngẫu nhiên)", bad == 0, f"{bad} ca lệch")

# ---------------------------------------------------------------- macro khác micro
recs = [utt("<en>rare</en> x", "sai x", 0)] + [utt("<en>team</en> x", "team x", i) for i in range(1, 4)]
a = agg(recs)
check("micro span_acc = 75", close(a["span_acc"], 75.0), a["span_acc"])
check("macro term_acc = 50 (khác micro khi một thuật ngữ lặp nhiều)", close(a["term_acc_macro"], 50.0), a["term_acc_macro"])
check("n_term / n_span", a["n_term"] == 2 and a["n_span"] == 4, (a["n_term"], a["n_span"]))
check("bootstrap: macro tính lại đúng trên mẫu lặp", close(agg(recs + recs[1:])["term_acc_macro"], 50.0) and
      close(agg(recs[1:] * 2)["term_acc_macro"], 100.0))
# paired_ci dùng agg trên mẫu: base sai hết, other đúng hết -> ΔCI > 0 theo metric macro
base_d, oth_d = {}, {}
for i in range(20):
    spk = f"p{i % 5}"
    base_d[f"u{i}"] = utt("<en>team</en> x", "sai x", i, spk=spk)
    oth_d[f"u{i}"] = utt("<en>team</en> x", "team x", i, spk=spk)
ci = leaderboard.paired_ci(base_d, oth_d, ["B1"], "term_acc_macro", 50)
check("paired_ci metric macro", ci is not None and close(ci[0], 100.0) and close(ci[1], 100.0), ci)

# trọng số cố định w = 1 / n_t trên tập đầy đủ cho đúng term_acc_macro
tw = term_weights(recs)
check("term_weights = 1 / số span của thuật ngữ", tw == {"rare": 1.0, "team": 1 / 3}, tw)
check("trọng số cố định trên tập đầy đủ = term_acc_macro", close(agg(recs, True, tw)["term_acc_macro"], agg(recs)["term_acc_macro"]),
      (agg(recs, True, tw)["term_acc_macro"], agg(recs)["term_acc_macro"]))
# nhiều thuật ngữ chỉ xuất hiện 1 lần: CI95 của term_acc_macro phải chứa ước lượng điểm
rs = random.Random(7)
many, k_id = [], 0
for sp in range(25):
    for k in range(20):   # 20 thuật ngữ lặp mỗi người nói một lần, dễ (~90% đúng)
        many.append(utt(f"<en>common{k}</en> x", f"common{k} x" if rs.random() < 0.9 else "sai x", k_id, spk=f"m{sp}"))
        k_id += 1
    for k in range(8):    # 8 thuật ngữ chỉ gặp 1 lần mỗi người nói, khó (~10% đúng)
        many.append(utt(f"<en>rare{sp}x{k}</en> x", f"rare{sp}x{k} x" if rs.random() < 0.1 else "sai x", k_id, spk=f"m{sp}"))
        k_id += 1
est = agg(many)["term_acc_macro"]
ci = bootstrap_many(many, ["term_acc_macro"], 300)["term_acc_macro"]
check("bootstrap macro (nhiều thuật ngữ 1 lần): CI chứa ước lượng điểm", ci and ci[0] <= est <= ci[1], (est, ci))
# cách cũ (macro tính lại theo mẫu) lệch lên trên dữ liệu này: CI cũ KHÔNG chứa ước lượng điểm -> test (a) có phân biệt được
by_spk = {}
for r in many:
    by_spk.setdefault(r["speaker"], []).append(r)
rnd5 = random.Random(0)
old = sorted(agg([x for q in (rnd5.choice(list(by_spk)) for _ in by_spk) for x in by_spk[q]])["term_acc_macro"] for _ in range(300))
check("(đối chứng) macro tính lại theo mẫu bị lệch lên: CI cũ không chứa ước lượng điểm", old[int(0.025 * 300)] > est, (est, old[7]))

# span_acc / cs_wer cũ không đổi: cs_wer của agg = lỗi span / token span
check("agg cs_wer = lỗi span / token span", close(agg(recs)["cs_wer"], 25.0), agg(recs)["cs_wer"])

# ---------------------------------------------------------------- lát tần suất train
check("freq_slice biên", [freq_slice({"train_freq_real": r, "train_freq_synth": s}) for r, s in
                          [(0, 0), (0, 3), (1, 0), (4, 9), (5, 0), (50, 0)]]
      == ["unseen", "synth_only", "real_1-4", "real_1-4", "real_5+", "real_5+"])
check("freq_slice thiếu trường -> None", freq_slice({"term": "x"}) is None)

tmp = Path(tempfile.mkdtemp(prefix="span_test_"))
tr = tmp / "train.jsonl"
rows = [{"id": "t1", "text_tagged": "mình họp <en>team</en> và <en>team</en> nhé"},         # real: team x2
        {"id": "t2", "text": "hôm nay team họp"},                                           # text (không tag): team x1
        {"id": "t3", "text_tagged": "dùng <en>set up</en> rồi", "synthetic": True},           # synth: set up
        {"id": "t4", "text_tagged": "cài <en>set up</en> hai lần <en>set up</en>", "synthetic": True},
        {"id": "t5", "text": "không liên quan gì"}]
tr.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
terms = {("team",), ("set", "up"), ("bilirubin",), ("thiếu",)}
fq = train_term_freq([str(tr)], CAT, terms)
check("train_term_freq đếm thật / synth", fq == {"team": (3, 0), "set up": (0, 3), "bilirubin": (0, 0), "thiếu": (0, 0)}, fq)

bench = [("<en>team</en> x", "team x"), ("<en>set up</en> x", "set up x"), ("<en>bilirubin</en> x", "sai x"),
         ("<en>team</en> x", "sai x")]
fr = [utt(t, h, i, freq=fq) for i, (t, h) in enumerate(bench)]
sl = freq_slices(fr)
check("lát: unseen", sl["unseen"]["n_span"] == 1 and sl["unseen"]["span_acc"] == 0.0, sl["unseen"])
check("lát: synth_only", sl["synth_only"]["n_span"] == 1 and sl["synth_only"]["span_acc"] == 100.0, sl["synth_only"])
check("lát: real_1-4 (team = 3 lần)", sl["real_1-4"]["n_span"] == 2 and sl["real_1-4"]["span_acc"] == 50.0
      and sl["real_1-4"]["n_term"] == 1, sl["real_1-4"])
check("lát: real_5+ rỗng", sl["real_5+"]["n_span"] == 0 and sl["real_5+"]["span_acc"] is None, sl["real_5+"])
check("không có train_freq -> không có lát", freq_slices(recs) == {})

# ---------------------------------------------------------------- leaderboard: luật không kém hơn
res = tmp / "results"


def make_system(name, b5_bad, b1_good):
    d = res / name
    d.mkdir(parents=True)
    out = []
    for sp in range(10):
        for j in range(4):
            i = f"{sp}_{j}"
            out.append(utt("họp <en>team</en> nhé", "họp team nhé" if b1_good else "họp tim nhé", f"b1_{i}", "B1", f"spk{sp}"))
            hyp5 = "hôm nay trời đẹp" if not b5_bad else "hôm nay trời tệ"  # B5: 1 / 4 token sai khi b5_bad
            out.append(utt("hôm nay trời đẹp", hyp5, f"b5_{i}", "B5", f"spk{sp}"))
    # các câu B5 sai đều nhau -> ΔWER B5 ổn định (+25 điểm) khi b5_bad
    with open(d / "utt_scores.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


make_system("base", False, False)
make_system("good", False, True)        # đúng thuật ngữ, B5 không đổi -> đủ điều kiện
make_system("bad_b5", True, True)       # đúng thuật ngữ nhưng B5 tệ đi -> loại vì không kém hơn
make_system("same", False, False)       # giống baseline -> CI chứa 0, không có ý nghĩa


def run_lb(*extra):
    out = tmp / ("lb_" + "_".join(a.strip("-") for a in extra)) if extra else tmp / "lb"
    sys.argv = ["leaderboard.py", "--results", str(res), "--baseline", "base", "--out", str(out), "--iters", "100", *extra]
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        leaderboard.main()
    with open(out / "leaderboard.csv", encoding="utf-8") as f:
        return {r["system"]: r for r in csv.DictReader(f)}, (out / "leaderboard.md").read_text(encoding="utf-8")


rows_, md = run_lb()
check("leaderboard: hệ tốt đủ điều kiện, hạng 1", rows_["good"]["eligible"] == "True" and rows_["good"]["rank"] == "1", rows_["good"])
check("leaderboard: B5 tệ bị loại (cận trên CI95 ΔWER > margin)", rows_["bad_b5"]["eligible"] == "False"
      and "B5 WER không chứng minh được" in rows_["bad_b5"]["reasons"], rows_["bad_b5"]["reasons"])
check("leaderboard: giống baseline -> chưa có ý nghĩa", rows_["same"]["eligible"] == "False"
      and "chưa có ý nghĩa" in rows_["same"]["reasons"], rows_["same"]["reasons"])
check("leaderboard: metric chính mặc định term_acc_macro", rows_["good"]["main_metric"] == "term_acc_macro"
      and close(float(rows_["good"]["main_value"]), 100.0))
check("leaderboard: Luật nhắc kiểm định không kém hơn", "không kém hơn" in md)
rows2, _ = run_lb("--b5-margin", "100")
check("leaderboard: margin B5 rộng thì bad_b5 qua luật B5", "B5 WER không chứng minh được" not in rows2["bad_b5"]["reasons"],
      rows2["bad_b5"]["reasons"])
rows3, _ = run_lb("--main-metric", "cs_wer")
check("leaderboard: --main-metric cs_wer (thấp hơn là tốt)", rows3["good"]["eligible"] == "True"
      and float(rows3["good"]["d_main"]) < 0, rows3["good"])
rows4, _ = run_lb("--main-metric", "span_acc")
check("leaderboard: --main-metric span_acc (cao hơn là tốt)", rows4["good"]["eligible"] == "True"
      and float(rows4["good"]["d_main"]) > 0, rows4["good"])

# ca ranh giới: ước lượng điểm ΔWER B5 dưới margin nhưng cận trên CI95 vượt margin -> loại
res2 = tmp / "results2"


def make_system2(name, good, b5_one_err):
    d = res2 / name
    d.mkdir(parents=True)
    with open(d / "utt_scores.jsonl", "w", encoding="utf-8") as f:
        for sp in range(20):
            for j in range(5):
                i = f"{sp}_{j}"
                rec1 = utt("họp <en>team</en> nhé", "họp team nhé" if good else "họp tim nhé", f"b1_{i}", "B1", f"spk{sp}")
                hyp5 = "hôm nay trời tệ" if (b5_one_err and sp == 0 and j == 0) else "hôm nay trời đẹp"
                rec5 = utt("hôm nay trời đẹp", hyp5, f"b5_{i}", "B5", f"spk{sp}")
                f.write(json.dumps(rec1, ensure_ascii=False) + "\n" + json.dumps(rec5, ensure_ascii=False) + "\n")


make_system2("base", False, False)
make_system2("edge", True, True)  # đúng thuật ngữ; B5 chỉ 1 token sai trong 400 -> ΔWER = +0.25 < 0.5, nhưng dồn vào 1 người nói
res_main, res = res, res2
rows5, _ = run_lb("--b5-margin", "0.5")
res = res_main
e = rows5["edge"]
check("ca ranh giới: ΔWER B5 điểm < margin", 0 < float(e["d_b5"]) < 0.5, e["d_b5"])
check("ca ranh giới: cận trên CI95 ΔWER B5 > margin", float(e["d_b5_ci_hi"]) > 0.5, e["d_b5_ci_hi"])
check("ca ranh giới: bị loại dù ước lượng điểm dưới margin", e["eligible"] == "False" and "B5 WER không chứng minh được" in e["reasons"],
      e["reasons"])

print(f"\n{len(FAILS)} ca sai" if FAILS else "\nTất cả đạt")
sys.exit(1 if FAILS else 0)
