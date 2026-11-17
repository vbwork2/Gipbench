#!/usr/bin/env python3
"""Cổng rò rỉ train -> bench khoá (core.leakgate) và các hàm release liên quan (locked_bench_refs, catalog_flags,
forbid_bench_catalog). Dữ liệu GIẢ trong thư mục tạm, CPU.  python tests/core/test_leakgate.py   (cần pyyaml)

Ca bám audit 17/11/2026 mục D.3: bench khoá phải được kiểm kể cả khi không có trong leak_guard (ca 6), bench khoá không resolve
được thì dừng (đóng khi lỗi), trùng video khác định dạng id bị bắt, miễn trừ có ghi chép, chế độ report không dừng.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

from gipbench.core import leakgate as LG
from gipbench.core import paths
from gipbench.core import release as R

FAIL = []
YT1, YT2, YT3 = "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"  # YouTube ID 11 ký tự


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def jl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def yt(i, vid, text, **kw):
    return dict({"id": i, "dataset": "vimedcss", "domain": "medical", "speaker": f"yt_{vid}", "source": f"youtube:{vid}",
                 "text_tagged": text, "duration": 2.0}, **kw)


def main():
    d = Path(tempfile.mkdtemp())
    os.environ["GIPBENCH_DATA"] = str(d / "data")
    os.environ["GIPBENCH_RELEASES_YAML"] = str(d / "releases.yaml")
    paths._cache = None
    # ---- bench giả: fb1 (C0-like, có video + tag), fb2 (G0-like, không id người nói), fdraft (chưa khoá)
    (d / "a.wav").write_bytes(b"same-bytes")
    (d / "b.wav").write_bytes(b"same-bytes")
    R.write_release("bench/fb1", manifest_rows=[yt("c0_1", YT1, "uống <en>vitamin</en> mỗi ngày"),
                                                yt("c0_2", YT2, "bệnh nhân bị <en>stress</en> kéo dài")])
    R.write_release("bench/fb2", manifest_rows=[{"id": "g0_1", "dataset": "ggroup_edu", "domain": "edu", "speaker": "g0_1",
                                                 "source": "ggroup:edu:1", "text_tagged": "xin chào các bạn hôm nay trời đẹp"}])
    R.write_release("bench/fdraft", manifest_rows=[yt("d_1", YT3, "câu của bench nháp")])
    R.write_release("bench/fba", manifest_rows=[{"id": "a_1", "dataset": "x", "speaker": "a_1", "source": "a_1", "text_tagged": "một",
                                                 "audio": "x", "_audio_path": str(d / "a.wav")}])
    cfg = {"releases": {"bench/fb1": {"flags": {"locked": True}}, "bench/fb2": {"flags": {"locked": True}},
                        "bench/fdraft": {"flags": {"locked": False}}, "bench/fba": {"flags": {"locked": True}},
                        "pool/p": {"flags": {"locked": True}}}}  # locked chỉ tính cho bench/*
    (d / "releases.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    check("locked_bench_refs: chỉ bench/* có flags.locked", R.locked_bench_refs() == ["bench/fb1", "bench/fb2", "bench/fba"], R.locked_bench_refs())
    check("plan_refs: leak_guard trước, bench khoá thiếu được thêm",
          LG.plan_refs(["bench/fdraft", "bench/fb1"], R.locked_bench_refs()) ==
          (["bench/fdraft", "bench/fb1", "bench/fb2", "bench/fba"], ["bench/fb2", "bench/fba"]))
    check("guard_refs: chuỗi hoặc danh sách", LG.guard_refs({"leak_guard": "bench/x"}) == ["bench/x"]
          and LG.guard_refs({"leak_guard": ["a", "b"]}) == ["a", "b"])
    clean = jl(d / "in/clean.jsonl", [yt("t_1", "ZZZZZZZZZZZ", "câu hoàn toàn khác một", speaker="spkZ"),
                                      {"id": "t_2", "dataset": "vivos", "domain": "read", "speaker": "vivos_VIVOSSPK01", "text": "tôi đi học"}])
    leaky = jl(d / "in/leaky.jsonl", [yt("vmc_1", YT1, "chào các bạn quay lại kênh", speaker=f"https://www.youtube.com/watch?v={YT1}",
                                          source="x")])
    text_leak = d / "in/tts.txt"
    text_leak.write_text("xin chào các bạn hôm nay trời đẹp\ncâu khác hẳn\n", encoding="utf-8")
    kw = dict(locked=None, audio_sha="metadata")

    # ---- 1. sạch: kiểm cả bench khoá không có trong leak_guard
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], out_dir=d / "w1", **kw)
    check("sạch: không blocking, kiểm đủ 4 bench (fdraft + 3 bench khoá)", not r["blocking"] and r["passed"]
          and sorted(r["benches"]) == ["bench/fb1", "bench/fb2", "bench/fba", "bench/fdraft"], (r["critical"], sorted(r["benches"])))
    check("sạch: bench khoá ngoài leak_guard ghi nhận", r["extra_locked"] == ["bench/fb1", "bench/fb2", "bench/fba"], r["extra_locked"])
    check("ghi leak_report.json + .md", (d / "w1/leak_report.json").exists() and "Cổng rò rỉ" in (d / "w1/leak_report.md").read_text(encoding="utf-8"))
    check("báo cáo ghi rõ chế độ audio (không đọc file)", "KHÔNG đọc file" in (d / "w1/leak_report.md").read_text(encoding="utf-8"))

    # ---- 2. trùng video khác định dạng id (URL vs youtube:ID) ở bench KHOÁ không có trong leak_guard
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean), ("pool/leaky", leaky)], out_dir=d / "w2", **kw)
    kinds = {f["kind"] for f in r["benches"]["bench/fb1"]["findings"] if f["severity"] == "critical"}
    check("trùng video (URL vs youtube:ID) -> source_norm_overlap critical, blocking", "source_norm_overlap" in kinds and r["blocking"], kinds)
    by = {x["input"]: x for x in r["benches"]["bench/fb1"]["by_input"]}
    check("báo cáo quy đúng pool gây rò rỉ", by["pool/leaky"]["source_norm"] == 1 and by["pool/clean"]["source_norm"] == 0, by)
    check("bench draft (không khoá, không trùng) sạch; bench khoá fb2 sạch",
          r["benches"]["bench/fdraft"]["critical"] == 0 and r["benches"]["bench/fb2"]["critical"] == 0)
    md = (d / "w2/leak_report.md").read_text(encoding="utf-8")
    check("markdown nêu bench và loại rò rỉ", "bench/fb1" in md and "source_norm_overlap" in md and "DỪNG" in md)

    # ---- 3. mode report: vẫn báo critical nhưng không dừng
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/leaky", leaky)], mode="report", **kw)
    check("mode report: critical > 0 nhưng không blocking", r["critical"] > 0 and not r["blocking"] and not r["passed"])

    # ---- 4. bench khoá không resolve được -> critical, không bỏ qua im lặng (D.3 ca 6)
    cfg["releases"]["bench/ghost"] = {"flags": {"locked": True}}
    (d / "releases.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], **kw)
    check("bench khoá chưa register -> unresolved critical + blocking", "bench/ghost" in r["unresolved"] and r["blocking"] and r["critical"] >= 1, r["unresolved"])
    check("unresolved nêu lệnh register", "register" in r["unresolved"]["bench/ghost"])
    cfg["releases"].pop("bench/ghost")
    (d / "releases.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # ---- 5. miễn trừ có ghi chép
    ex = d / "ex.yaml"
    ex.write_text(yaml.safe_dump([{"kind": "source_norm_overlap", "value": f"yt:{YT1}", "reason": "video công khai, đã gỡ câu bench",
                                   "approved_by": "người dùng", "date": "2026-11-17"}]), encoding="utf-8")
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/leaky", leaky)], exemptions=str(ex), **kw)
    check("miễn trừ theo value: không blocking, đếm exempted", not r["blocking"] and r["exempted"] >= 1 and r["critical"] == 0, (r["critical"], r["exempted"]))
    check("miễn trừ vẫn nằm trong báo cáo (mức exempted)", "source_norm_overlap:exempted" in json.dumps(r))
    check("exemptions_sha256 ghi lại", len(r["exemptions_sha256"]) == 64)
    bad = d / "bad.yaml"
    bad.write_text(yaml.safe_dump([{"kind": "source_norm_overlap", "value": "x"}]), encoding="utf-8")
    try:
        LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], exemptions=str(bad), **kw)
        check("miễn trừ thiếu lý do / người duyệt / ngày bị từ chối", False)
    except ValueError as e:
        check("miễn trừ thiếu lý do / người duyệt / ngày bị từ chối", "thiếu" in str(e), str(e))

    # ---- 6. text TTS / LM khai báo (leak_gate.text_files) trùng câu bench
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], text_files=[str(text_leak)], **kw)
    kinds = {f["kind"] for f in r["benches"]["bench/fb2"]["findings"] if f["severity"] == "critical"}
    check("câu TTS trùng câu bench -> text_exact_overlap critical", "text_exact_overlap" in kinds and r["blocking"], kinds)
    check("nhãn đầu vào text:tts.txt", any(i["label"] == "text:tts.txt" for i in r["inputs"]))

    # ---- 7. thuật ngữ giữ riêng
    res_t = d / "reserved.txt"
    res_t.write_text("# giữ riêng\nstress\n", encoding="utf-8")
    withterm = jl(d / "in/withterm.jsonl", [{"id": "w_1", "dataset": "x", "speaker": "sw", "source": "sw", "text_tagged": "anh ấy bị <en>stress</en> nặng"}])
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/withterm", withterm)], reserved_terms=str(res_t), **kw)
    check("thuật ngữ giữ riêng có trong text train -> critical + blocking", r["reserved"]["critical"] == 1 and r["blocking"], r["reserved"])
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], reserved_terms=str(res_t), **kw)
    check("thuật ngữ giữ riêng không có trong train -> sạch", r["reserved"]["critical"] == 0 and not r["blocking"])
    ex2 = d / "ex2.yaml"
    ex2.write_text(yaml.safe_dump([{"kind": "reserved_term_in_train", "value": "stress", "reason": "thử", "approved_by": "u", "date": "2026-11-17"}]), encoding="utf-8")
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/withterm", withterm)], reserved_terms=str(res_t), exemptions=str(ex2), **kw)
    check("miễn trừ thuật ngữ giữ riêng theo value", r["reserved"]["critical"] == 0 and r["reserved"]["exempted"] == 1 and not r["blocking"], r["reserved"])
    empty = d / "empty.txt"
    empty.write_text("# chưa chốt\n", encoding="utf-8")
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], reserved_terms=str(empty), **kw)
    check("danh sách thuật ngữ giữ riêng rỗng -> dừng (không coi là sạch)", "reserved_terms" in r["unresolved"] and r["blocking"])

    # ---- 8. chế độ kiểm audio
    wa = jl(d / "in/wa.jsonl", [{"id": "tr_1", "dataset": "x", "speaker": "q1", "source": "q1", "text": "khác", "audio": str(d / "b.wav")}])
    base = dict(locked=["bench/fba"], mode="fail", out_dir=None)
    r = LG.run_gate("t", [], [("pool/wa", wa)], audio_sha="hash", **base)
    check("audio_sha=hash: cùng bytes khác tên file -> audio_sha256_overlap critical",
          any(f["kind"] == "audio_sha256_overlap" for f in r["benches"]["bench/fba"]["findings"]) and r["blocking"])
    r = LG.run_gate("t", [], [("pool/wa", wa)], audio_sha="metadata", **base)
    kinds = {f["kind"]: f["severity"] for f in r["benches"]["bench/fba"]["findings"]}
    check("audio_sha=metadata: không đọc file, không thấy trùng nhưng ghi audio_unverifiable (review)",
          "audio_sha256_overlap" not in kinds and kinds.get("audio_unverifiable") == "review" and not r["blocking"], kinds)
    r = LG.run_gate("t", [], [("pool/wa", wa)], audio_sha="metadata", require_audio_sha=True, **base)
    check("require_audio_sha: thiếu hash = critical (đóng khi lỗi)", r["blocking"] and any(
        f["kind"] == "audio_unverifiable" and f["severity"] == "critical" for f in r["benches"]["bench/fba"]["findings"]))
    r = LG.run_gate("t", [], [("pool/wa", wa)], audio_sha="skip", **base)
    check("audio_sha=skip: không có finding audio, báo cáo ghi rõ KHÔNG kiểm",
          not any(f["kind"].startswith("audio") for f in r["benches"]["bench/fba"]["findings"]) and "KHÔNG kiểm trùng audio" in LG.render_md(r))

    # ---- 9. cấu hình + model card
    for bad_cfg in ({"mode": "warn"}, {"audio_sha": "all"}, {"audio_sha": "skip", "require_audio_sha": True}):
        try:
            LG.gate_config({"leak_gate": bad_cfg})
            check(f"gate_config từ chối {bad_cfg}", False)
        except ValueError:
            check(f"gate_config từ chối {bad_cfg}", True)
    g = LG.gate_config({})
    check("gate_config mặc định: fail, audio metadata, không near", g["mode"] == "fail" and g["audio_sha"] == "metadata" and not g["near"], g)
    r = LG.run_gate("t", ["bench/fdraft"], [("pool/clean", clean)], out_dir=d / "w9", **kw)
    s = LG.card_summary(r, d / "w9/leak_report.json")
    check("card_summary: sha256 của leak_report.json", s["report_sha256"] == hashlib.sha256((d / "w9/leak_report.json").read_bytes()).hexdigest()
          and s["passed"] and "bench/fb1" in s["benches_checked"])

    # ---- 10. catalog from_bench
    cb, ct = d / "cat_bench.tsv", d / "cat_train.tsv"
    for p, body in ((cb, "term\tcategory\tvariants\tphonetic\treserved_b3\nstress\tmedical\t\t\t0\n"),
                    (ct, "term\tcategory\tvariants\tphonetic\treserved_b3\nvaccine\tmedical\t\t\t0\n")):
        p.write_text(body, encoding="utf-8")
    R.write_release("catalog/benchcat", catalog_src=cb, flags={"from_bench": True})
    R.write_release("catalog/traincat", catalog_src=ct, flags={"from_bench": False})
    check("catalog_flags theo ref", R.catalog_flags("catalog/benchcat") == {"from_bench": True} and R.catalog_flags("catalog/traincat") == {"from_bench": False})
    check("catalog_flags theo đường dẫn file nguồn (khớp sha256)", R.catalog_flags(str(cb)) == {"from_bench": True})
    unknown = d / "other.tsv"
    unknown.write_text("term\tcategory\tvariants\tphonetic\treserved_b3\nxyz\t\t\t\t0\n", encoding="utf-8")
    check("catalog không thuộc release nào -> {} (không biết)", R.catalog_flags(str(unknown)) == {})
    try:
        R.forbid_bench_catalog(str(cb), "thử")
        check("forbid_bench_catalog từ chối catalog from_bench", False)
    except R.ReleaseError as e:
        check("forbid_bench_catalog từ chối catalog from_bench", "from_bench" in str(e) and "build_train_catalog" in str(e))
    check("allow=True (oracle) cho qua, catalog train luôn qua",
          R.forbid_bench_catalog(str(cb), "thử", allow=True)["from_bench"] is True and R.forbid_bench_catalog("catalog/traincat", "thử") == {"from_bench": False})
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
