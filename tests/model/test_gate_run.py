#!/usr/bin/env python3
"""model.run với cổng rò rỉ đóng khi lỗi + chặn catalog from_bench + select_ckpt kiểm dev độc lập (audit IP-02, IP-05).
Chạy `python -m gipbench.model.run ...` thật (subprocess) trên release GIẢ trong thư mục tạm; không cần GPU / icefall / dữ liệu thật.

  python tests/model/test_gate_run.py        # cần pyyaml
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core import release as R
from gipbench.core.paths import REPO as G

FAIL = []
YT1, YT2 = "AAAAAAAAAAA", "BBBBBBBBBBB"


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-400:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def row(i, vid=None, text="một câu bình thường", **kw):
    r = {"id": i, "dataset": "vimedcss", "domain": "medical", "split": "test", "duration": 2.0, "text_tagged": text,
         "speaker": f"spk_{i}", "source": f"src_{i}"}
    if vid:
        r.update(speaker=f"yt_{vid}", source=f"youtube:{vid}")
    return dict(r, **kw)


def world(d: Path):
    """Release giả + releases.yaml. Trả về env cho subprocess."""
    os.environ["GIPBENCH_DATA"] = str(d / "data")
    os.environ["GIPBENCH_RELEASES_YAML"] = str(d / "releases.yaml")
    paths._cache = None
    R.write_release("bench/fb1", manifest_rows=[row("c0_1", YT1, "uống <en>vitamin</en> mỗi ngày"), row("c0_2", YT2, "bị <en>stress</en> kéo dài")])
    R.write_release("bench/fb2", manifest_rows=[row("g0_1", None, "xin chào các bạn hôm nay trời đẹp", dataset="ggroup_edu", domain="edu")])
    R.write_release("pool/clean@v1", manifest_rows=[row("p_1", None, "câu hoàn toàn khác", split="train"), row("p_2", "ZZZZZZZZZZZ", "thêm một câu", split="train")])
    R.write_release("pool/leaky@v1", manifest_rows=[row("l_1", YT1, "chào các bạn quay lại kênh", split="train")])
    R.write_release("dev/dv@v1", manifest_rows=[row("d_1", None, "dev câu một", split="validation")])
    R.write_release("dev/dvlegacy@v1", manifest_rows=[row("dl_1", None, "dev câu hai", split="validation")])
    R.write_release("dev/dvind@v1", manifest_rows=[row("di_1", None, "dev câu ba", split="validation")], flags={"independent": True})
    (d / "cat_bench.tsv").write_text("term\tcategory\tvariants\tphonetic\treserved_b3\nstress\tmedical\t\t\t0\n", encoding="utf-8")
    (d / "cat_train.tsv").write_text("term\tcategory\tvariants\tphonetic\treserved_b3\nvaccine\tmedical\t\t\t0\n", encoding="utf-8")
    R.write_release("catalog/benchcat", catalog_src=d / "cat_bench.tsv", flags={"from_bench": True})
    R.write_release("catalog/traincat", catalog_src=d / "cat_train.tsv", flags={"from_bench": False})
    cfg = {"releases": {"bench/fb1": {"flags": {"locked": True}}, "bench/fb2": {"flags": {"locked": True}}}}
    (d / "releases.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return dict(os.environ, PYTHONIOENCODING="utf-8")


def exp(d, name, **over):
    e = {"name": name, "base_model": "models/none", "catalog": "catalog/traincat", "leak_guard": "bench/fb1",
         "dev": {"cs": "dev/dv@v1", "vi": "dev/dv@v1"},
         "pools": {"replay": {"weight": 1.0, "datasets": ["vimedcss"], "refs": ["pool/clean@v1"]}},
         "output": {"work_dir": str(d / "work" / name), "exp_dir": str(d / "exp" / name), "dev_results": str(d / "dev" / name),
                    "artifact": str(d / "art" / name)}}
    e.update(over)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(e, allow_unicode=True), encoding="utf-8")
    return str(p)


def run_exp(env, path, *args):
    p = subprocess.run([sys.executable, "-m", "gipbench.model.run", path, *args], capture_output=True, text=True, encoding="utf-8",
                       cwd=G, env=env)
    return p.returncode, p.stdout + p.stderr


def run_mod(env, mod, *args):
    p = subprocess.run([sys.executable, "-m", mod, *map(str, args)], capture_output=True, text=True, encoding="utf-8", cwd=G, env=env)
    return p.returncode, p.stdout + p.stderr


def main():
    d = Path(tempfile.mkdtemp())
    env = world(d)

    # ---- cổng rò rỉ ở stage 0
    rc, out = run_exp(env, exp(d, "ok"), "--stages", "0")
    rep = json.loads((d / "work/ok/leak_report.json").read_text(encoding="utf-8")) if (d / "work/ok/leak_report.json").exists() else {}
    check("stage 0 sạch: thoát 0, có leak_report.json (passed)", rc == 0 and rep.get("passed") is True, out)
    check("kiểm cả bench khoá ngoài leak_guard (fb2)", "bench/fb2" in rep.get("benches", {}) and rep.get("extra_locked") == ["bench/fb2"], rep.get("extra_locked"))
    check("in danh sách bench sẽ kiểm", "sẽ kiểm 2 bench" in out and "vẫn được kiểm" in out, out)
    md = (d / "work/ok/leak_report.md").read_text(encoding="utf-8") if (d / "work/ok/leak_report.md").exists() else ""
    check("leak_report.md có cả bộ lọc câu cũ (LeakGuard)", "Cổng rò rỉ" in md and "LeakGuard" in md, md[:200])

    reserved = d / "reserved.txt"
    reserved.write_text("meeting\n", encoding="utf-8")
    R.write_release("dev/heldout@1", manifest_rows=[row("heldout", "ZZZZZZZZZZZ", "đây là <en>meeting</en>")],
                    flags={"independent": True})
    rc, out = run_exp(env, exp(d, "dev_overlap", dev={"cs": "dev/heldout@1", "vi": "dev/dvind@v1"},
                              leak_gate={"check_dev": True, "reserved_terms": str(reserved)}), "--stages", "0")
    report = json.loads((d / "work/dev_overlap/leak_report.json").read_text(encoding="utf-8"))
    check("Dev source overlap blocks while reserved terms in dev remain measurable",
          rc == 3 and report["dev"]["dev/heldout@1"]["critical"] > 0 and report["reserved"]["critical"] == 0, out)

    rc, out = run_exp(env, exp(d, "leak", pools={"replay": {"weight": 1.0, "datasets": ["vimedcss"], "refs": ["pool/clean@v1", "pool/leaky@v1"]}}), "--stages", "0")
    rep = json.loads((d / "work/leak/leak_report.json").read_text(encoding="utf-8"))
    check("trùng video với bench khoá: thoát mã 3 trước curate / train", rc == 3 and rep["blocking"] and "DỪNG" in out, (rc, out))
    check("báo cáo nêu bench + loại rò rỉ", "bench/fb1: source_norm_overlap" in out, out)
    check("không có train_curated (stage 1 chưa chạy)", not (d / "work/leak/train_curated.jsonl").exists())

    leak_pools = {"replay": {"weight": 1.0, "datasets": ["vimedcss"], "refs": ["pool/leaky@v1"]}}
    rc, out = run_exp(env, exp(d, "report", pools=leak_pools, leak_gate={"mode": "report"}), "--stages", "0")
    check("leak_gate.mode: report: vẫn ghi báo cáo nhưng thoát 0", rc == 0 and json.loads((d / "work/report/leak_report.json").read_text(encoding="utf-8"))["critical"] > 0, out)

    ex = d / "ex.yaml"
    meta = {"reason": "thử", "approved_by": "u", "date": "2026-11-17"}
    ex.write_text(yaml.safe_dump([dict(kind="source_norm_overlap", value=f"yt:{YT1}", **meta), dict(kind="speaker_overlap", value=f"yt_{YT1}", **meta),
                                  dict(kind="source_overlap", value=f"youtube:{YT1}", **meta)]), encoding="utf-8")
    rc, out = run_exp(env, exp(d, "exempt", pools=leak_pools, leak_gate={"exemptions": str(ex)}), "--stages", "0")
    check("miễn trừ có ghi chép (từng loại phải nêu rõ): thoát 0, vẫn nằm trong báo cáo",
          rc == 0 and "source_norm_overlap:exempted" in (d / "work/exempt/leak_report.json").read_text(encoding="utf-8"), out)
    ex_part = d / "ex_part.yaml"
    ex_part.write_text(yaml.safe_dump([dict(kind="source_norm_overlap", value=f"yt:{YT1}", **meta)]), encoding="utf-8")
    rc, out = run_exp(env, exp(d, "exempt2", pools=leak_pools, leak_gate={"exemptions": str(ex_part)}), "--stages", "0")
    check("miễn trừ một loại không che loại khác (speaker_overlap, source_overlap còn critical)", rc == 3, out)

    cfg = yaml.safe_load((d / "releases.yaml").read_text(encoding="utf-8"))
    cfg["releases"]["bench/ghost"] = {"flags": {"locked": True}}
    (d / "releases.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc, out = run_exp(env, exp(d, "ghost"), "--stages", "0")
    check("bench khoá chưa register: thoát 3, liệt kê 'CHƯA KIỂM'", rc == 3 and "CHƯA KIỂM bench/ghost" in out, out)
    cfg["releases"].pop("bench/ghost")
    (d / "releases.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rc, out = run_exp(env, exp(d, "list", leak_guard=["bench/fb1", "bench/fb2"]), "--stages", "0")
    check("leak_guard là danh sách: chạy được, không còn bench khoá nào ngoài leak_guard", rc == 0 and "vẫn được kiểm" not in out, out)
    rc, out = run_exp(env, exp(d, "dry"), "--stages", "0", "--dry-run")
    check("--dry-run: in danh sách bench sẽ kiểm, không ghi báo cáo", rc == 0 and "sẽ kiểm 2 bench" in out and not (d / "work/dry/leak_report.json").exists(), out)
    rc, out = run_exp(env, exp(d, "badgate", leak_gate={"mode": "warn"}), "--stages", "0")
    check("leak_gate.mode sai: dừng với thông báo", rc != 0 and "leak_gate.mode" in out, out)

    # ---- stage 6 cần báo cáo cổng
    rc, out = run_exp(env, exp(d, "s6a"), "--stages", "6")
    check("stage 6 không có báo cáo cổng: thoát 3, không tạo artifact", rc == 3 and "chưa có báo cáo cổng" in out and not (d / "art/s6a").exists(), out)
    (d / "work/s6b").mkdir(parents=True)
    (d / "work/s6b/leak_report.json").write_text((d / "work/leak/leak_report.json").read_text(encoding="utf-8"), encoding="utf-8")
    rc, out = run_exp(env, exp(d, "s6b"), "--stages", "6")
    check("stage 6 với báo cáo còn critical chưa miễn trừ: thoát 3", rc == 3 and "không tạo artifact" in out and not (d / "art/s6b").exists(), out)

    # ---- chạy riêng stage 1 (curate) cũng phải qua báo cáo cổng của stage 0
    rc, out = run_exp(env, exp(d, "s1a"), "--stages", "1")
    check("stage 1 không có báo cáo cổng: thoát 3, không curate", rc == 3 and "chưa có báo cáo cổng" in out
          and not (d / "work/s1a/train_curated.jsonl").exists(), out)
    (d / "work/s1b").mkdir(parents=True)
    (d / "work/s1b/leak_report.json").write_text((d / "work/leak/leak_report.json").read_text(encoding="utf-8"), encoding="utf-8")
    rc, out = run_exp(env, exp(d, "s1b"), "--stages", "1")
    check("stage 1 với báo cáo còn critical chưa miễn trừ: thoát 3", rc == 3 and not (d / "work/s1b/train_curated.jsonl").exists(), out)

    # ---- catalog from_bench
    rc, out = run_exp(env, exp(d, "cb", catalog="catalog/benchcat"), "--stages", "0", "--dry-run")
    check("catalog from_bench: thoát 3 ngay cả khi --dry-run", rc == 3 and "from_bench" in out and "build_train_catalog" in out, out)
    rc, out = run_exp(env, exp(d, "cbo", catalog="catalog/benchcat", catalog_use="oracle_diagnostic"), "--stages", "0", "--dry-run")
    check("catalog_use: oracle_diagnostic cho qua", rc == 0, out)
    rc, out = run_exp(env, exp(d, "cbu", catalog_use="khac"), "--stages", "0", "--dry-run")
    check("catalog_use sai: từ chối", rc != 0 and "catalog_use" in out, out)
    sysy = [{"system": "x", "hotwords": {"catalog": "catalog/benchcat", "exclude_b3": True}}]
    rc, out = run_exp(env, exp(d, "hw", systems=sysy), "--stages", "0", "--dry-run")
    check("hotwords từ catalog from_bench trong system: thoát 3", rc == 3 and "hotwords của system x" in out, out)
    rc, out = run_exp(env, exp(d, "hwo", systems=[dict(sysy[0], inference_mode="oracle_bias")]), "--stages", "0", "--dry-run")
    check("system khai báo inference_mode: oracle_bias thì cho qua", rc == 0, out)

    # ---- curate / select_ckpt
    m = d / "m.jsonl"
    m.write_text(json.dumps(row("m_1", None, "câu <en>vaccine</en> thật", split="train", audio="a.wav")) + "\n", encoding="utf-8")
    off = ["--max-per-term", "0", "--max-speaker-share", "0", "--max-domain-share", "0", "--max-synthetic-share", "0", "--block-text"]
    rc, out = run_mod(env, "gipbench.model.recipe.curate", "--manifest", m, "--catalog", d / "cat_bench.tsv", "--out", d / "c1.jsonl", *off)
    check("curate từ chối catalog from_bench (exit 3)", rc == 3 and "from_bench" in out, out)
    rc, out = run_mod(env, "gipbench.model.recipe.curate", "--manifest", m, "--catalog", d / "cat_bench.tsv", "--out", d / "c2.jsonl",
                      "--allow-bench-catalog", *off)
    check("curate --allow-bench-catalog (chẩn đoán oracle) chạy được", rc == 0 and (d / "c2.jsonl").exists(), out)
    rc, out = run_mod(env, "gipbench.model.recipe.curate", "--manifest", m, "--catalog", d / "cat_train.tsv", "--out", d / "c3.jsonl", *off)
    check("curate với catalog train chạy được", rc == 0, out)

    def dirs(name, ok):
        for k, txt in (("dev_cs", "tôi uống <en>vaccine</en>"), ("dev_vi", "xin chào")):
            p = d / name / k
            p.mkdir(parents=True, exist_ok=True)
            (p / "hyps.jsonl").write_text(json.dumps({"id": "s1" if k == "dev_cs" else "v1", "hyp": txt.replace("<en>", "").replace("</en>", "") if ok else "sai"},
                                                       ensure_ascii=False) + "\n", encoding="utf-8")
        return d / name
    dcs = d / "dev_cs.jsonl"
    dcs.write_text(json.dumps({"id": "s1", "split": "validation", "speaker": "a", "dataset": "ViMedCSS:validation",
                               "text_tagged": "tôi uống <en>vaccine</en>"}, ensure_ascii=False) + "\n", encoding="utf-8")
    dvi = d / "dev_vi.jsonl"
    dvi.write_text(json.dumps({"id": "v1", "split": "validation", "speaker": "b", "dataset": "vivos", "text": "xin chào"}, ensure_ascii=False) + "\n", encoding="utf-8")
    base, cand = dirs("base", False), dirs("cand", True)
    sel = ["--catalog", d / "cat_train.tsv", "--dev-cs", dcs, "--dev-vi", dvi, "--base", base, "--cand", cand]
    rc, out = run_mod(env, "gipbench.model.select.select_ckpt", *sel, "--out", d / "s0")
    j = json.loads((d / "s0.json").read_text(encoding="utf-8")) if (d / "s0.json").exists() else {}
    check("select_ckpt không truyền dev-ref: chạy như cũ, JSON không có khoá nguồn gốc (golden không đổi)", rc == 0 and j.get("pick") == "cand" and "rule" not in j, out)
    rc, out = run_mod(env, "gipbench.model.select.select_ckpt", *sel, "--out", d / "s1", "--dev-ref-cs", "dev/dvlegacy@v1", "--dev-ref-vi", "dev/dvlegacy@v1")
    check("dev không có flags.independent: từ chối (exit 3)", rc == 3 and "independent" in out, out)
    rc, out = run_mod(env, "gipbench.model.select.select_ckpt", *sel, "--out", d / "s2", "--dev-ref-cs", "dev/dvlegacy@v1", "--dev-ref-vi", "dev/dvlegacy@v1", "--allow-legacy-dev")
    j = json.loads((d / "s2.json").read_text(encoding="utf-8"))
    check("--allow-legacy-dev: cảnh báo, chạy, ghi dev_flags + rule + catalog_flags", rc == 0 and "CẢNH BÁO" in out and j["rule"]["name"] == "term_acc_macro"
          and j["dev_flags"]["cs"] == {} and j["catalog_flags"] == {"from_bench": False} and j["dev_refs"]["cs"] == "dev/dvlegacy@v1", (out, j))
    rc, out = run_mod(env, "gipbench.model.select.select_ckpt", *sel, "--out", d / "s3", "--dev-ref-cs", "dev/dvind@v1", "--dev-ref-vi", "dev/dvind@v1", "--rule", "cs_wer")
    j = json.loads((d / "s3.json").read_text(encoding="utf-8"))
    check("dev độc lập + --rule cs_wer: không cần cờ, ghi luật đã dùng", rc == 0 and j["rule"]["name"] == "cs_wer" and j["pick"] == "cand"
          and "CS-WER dev CS thấp nhất" in (d / "s3.md").read_text(encoding="utf-8"), (out, j))
    rc, out = run_mod(env, "gipbench.model.select.select_ckpt", "--catalog", d / "cat_bench.tsv", *sel[2:], "--out", d / "s4")
    check("select_ckpt từ chối catalog from_bench", rc == 3 and "from_bench" in out, out)
    rc, out = run_mod(env, "gipbench.model.select.select_ckpt", *sel, "--out", d / "s5", "--dev-ref-cs", "dev/khong_co@v1")
    check("dev-ref không tồn tại: dừng (không bỏ qua im lặng)", rc == 3, out)

    test_e4_selector(env, d)
    test_vi_selector(env, d)

    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


def test_e4_selector(env, d):
    from gipbench.core.manifest import write_jsonl

    cs_labels = ["mình dùng <en>cloud</en>", "mình dùng <en>stocks</en>"]
    vi_label = " ".join(["chào"] * 115)
    cs, vi = d / "e4_cs.jsonl", d / "e4_vi.jsonl"
    write_jsonl(cs, [row(f"cs{i}", text=label, domain=domain, speaker=f"cs-spk{i}")
                     for i, (label, domain) in enumerate(zip(cs_labels, ("tech", "finance")))])
    write_jsonl(vi, [row(f"vi{i}", text=vi_label, speaker=f"vi-spk{i}") for i in range(3)])
    vi_base = [vi_label.replace("chào", "sáng", 1), vi_label, vi_label]
    cs_good = ["mình dùng cloud", "mình dùng stocks"]
    cs_base = ["mình dùng mây", "mình dùng stocks"]

    def save_hyps(name, cs_hyps, vi_hyps):
        for label, texts in (("cs", cs_hyps), ("vi", vi_hyps)):
            write_jsonl(d / name / f"dev_{label}" / "hyps.jsonl",
                        [{"id": f"{label}{i}", "hyp": text} for i, text in enumerate(texts)])
        return d / name

    base = save_hyps("e4_base", cs_base, vi_base)
    late = save_hyps("ck20", cs_good, vi_base)
    early = save_hyps("ck10", cs_good, vi_base)
    domain = save_hyps("domain", ["mình dùng cloud", "mình dùng cổ phiếu"], vi_base)
    hallucination = save_hyps("hallucination", cs_good, [vi_label.replace("chào", "meeting", 1), vi_label, vi_label])
    ci = save_hyps("ci", cs_good, [*vi_base[:2], vi_label.replace("chào", "sáng", 1)])
    limits = ["--rule", "cs_wer", "--max-vi-ci-upper", "0.5", "--max-vi-hallucination-delta", "0.0",
              "--max-cs-domain-delta", "1.0"]
    command = ["--catalog", d / "cat_train.tsv", "--dev-cs", cs, "--dev-vi", vi, "--base", base, *limits]
    rc, output = run_mod(env, "gipbench.model.select.select_ckpt", *command, "--cand", late, early, domain, hallucination, ci,
                         "--out", d / "e4_sel")
    check("E4 selector runs with retention thresholds", rc == 0, output)
    if rc == 0:
        report = json.loads((d / "e4_sel.json").read_text(encoding="utf-8"))
        results = report["results"]
        check("E4 breaks ties by earliest checkpoint", report["pick"] == "ck10")
        check("E4 rejects a regressed domain", not results["domain"]["eligible"]
              and results["domain"]["retention"]["cs_domain_deltas"]["finance"] > 1)
        check("E4 rejects increased English insertions at the same WER", not results["hallucination"]["eligible"]
              and results["hallucination"]["wer_vi"] == results["e4_base"]["wer_vi"])
        check("E4 rejects the upper CI at an acceptable point estimate", not results["ci"]["eligible"]
              and results["ci"]["retention"]["wer_vi_ci95"][1] > 0.5
              and results["ci"]["wer_vi"] - results["e4_base"]["wer_vi"] < 0.3)
    partial = save_hyps("partial", cs_good[:1], vi_base)
    rc, output = run_mod(env, "gipbench.model.select.select_ckpt", *command, "--cand", partial, "--out", d / "e4_partial")
    check("E4 rejects incomplete dev hypotheses", rc != 0 and "one hypothesis per dev utterance" in output, output)


def test_vi_selector(env, d):
    from gipbench.core.manifest import write_jsonl

    text = "chào bạn hôm nay"
    vi = d / "vi_only.jsonl"
    write_jsonl(vi, [row(f"v{i}", text=text, speaker=f"speaker{i}") for i in range(3)])

    def hypotheses(name, texts):
        path = d / "vi_only" / name
        write_jsonl(path / "dev_vi/hyps.jsonl", [{"id": f"v{i}", "hyp": hyp} for i, hyp in enumerate(texts)])
        return path

    base = hypotheses("base", ["chào bạn", "chào bạn hôm", text])
    late = hypotheses("ck2400", [text] * 3)
    early = hypotheses("ck1200", [text] * 3)
    uncertain = hypotheses("uncertain", [text, text, "chào bạn hôm"])
    incomplete = hypotheses("incomplete", [text])
    command = ["--catalog", d / "cat_train.tsv", "--dev-vi", vi, "--dev-ref-vi", "dev/dvind@v1", "--base", base,
               "--rule", "vi_wer", "--max-vi-delta", "0.0", "--max-vi-ci-upper", "0.0"]
    for name, candidates, expected in (("vi_pick", [late, early], "ck1200"), ("vi_base", [uncertain], "base")):
        rc, output = run_mod(env, "gipbench.model.select.select_ckpt", *command, "--cand", *candidates, "--out", d / name)
        report = json.loads((d / f"{name}.json").read_text(encoding="utf-8")) if rc == 0 else {}
        check(f"VI-only selection chooses {expected}", rc == 0 and report.get("pick") == expected, output)
        if name == "vi_base" and rc == 0:
            metrics = report["results"]["uncertain"]
            check("VI-only selection rejects an improvement with a positive upper CI",
                  metrics["wer_vi_delta"] < 0 < metrics["wer_vi_ci95"][1] and not metrics["eligible"], metrics)
    rc, output = run_mod(env, "gipbench.model.select.select_ckpt", *command, "--cand", incomplete, "--out", d / "vi_partial")
    check("VI-only selection rejects incomplete hypotheses", rc != 0 and "one hypothesis per dev utterance" in output, output)


if __name__ == "__main__":
    main()
