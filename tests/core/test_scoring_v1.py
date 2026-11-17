#!/usr/bin/env python3
"""Ca biên của chấm điểm (core.scoring + core.normprofiles + core.evalstats + bench.evaluate). CPU, không cần dữ liệu.

  python tests/core/test_scoring_v1.py      # thoát mã 1 nếu có ca hỏng
"""
import json
import sys
import tempfile
from pathlib import Path

from gipbench.core.evalstats import check_hyps, cluster_bootstrap, paired_bootstrap, utt_counts
from gipbench.core.normprofiles import get_profile
from gipbench.core.scoring import score_utt
from gipbench.core.textnorm import Catalog, normalize_hyp, normalize_ref

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def cat_from(rows):
    p = Path(tempfile.mkdtemp()) / "cat.tsv"
    p.write_text("term\tcategory\tvariants\tphonetic\treserved_b3\n" + "".join("\t".join(r) + "\n" for r in rows),
                 encoding="utf-8")
    return Catalog.load(str(p))


CAT = cat_from([("code", "tech", "", "cốt", "0"), ("pull request", "tech", "pull-request", "", "0"),
                ("api", "tech", "", "", "0"), ("kpi", "business", "", "", "0"), ("zalo", "brand", "", "", "0")])


def utt(ref, hyp, prof="legacy_v0", cat=CAT):
    p = get_profile(prof)
    r = {"id": "u", "split": "B1", "speaker": "s", "text_tagged": p.pre(ref)}
    rec = score_utt(r, p.pre(hyp), cat)
    if p.lenient:
        rec["err_en"] = rec["err_en_len"]
    return rec


def main():
    # 1. một từ Anh -> hai token Việt: S + I quy về span Anh, CS-WER 200% (> 100% là hợp lệ)
    r = utt("họp <en>meeting</en> nhé", "họp mít tinh nhé")
    check("1 từ EN -> 2 token: err_en = 2 (CS-WER 200%)", r["err_en"] == 2 and r["n_en"] == 1 and r["err_vi"] == 0, r)
    # 2. chèn ở ranh giới: token chèn kề span Anh (bên trái) tính cho span Anh
    r = utt("tôi <en>deploy</en> nhé", "tôi đi ploy nhé")
    check("chèn ở ranh giới Việt/Anh quy về span Anh", r["err_en"] == 2 and r["err_vi"] == 0, r)
    # 3. viết tắt: hoa / thường không tính lỗi
    check("API viết hoa = api", utt("<en>API</en> mới", "api mới")["err"] == 0)
    # 4. KPI đọc rời từng chữ: 1 S + 2 I
    r = utt("chỉ số <en>KPI</en>", "chỉ số k p i")
    check("KPI -> 'k p i' = 3 lỗi span Anh", r["err_en"] == 3, r)
    # 5. viết tắt có dấu chấm: legacy phạt, strict_v1 không phạt
    check("A.P.I: legacy_v0 phạt", utt("<en>API</en> mới", "A.P.I mới")["err"] > 0)
    check("A.P.I: strict_v1 không phạt", utt("<en>API</en> mới", "A.P.I mới", "gipbench_strict_v1")["err"] == 0)
    # 6. từ mượn không gắn tag (vitamin) là token tiếng Việt, đúng thì 0 lỗi, không có span
    r = utt("uống vitamin c mỗi ngày", "uống vitamin c mỗi ngày")
    check("từ mượn vitamin: 0 lỗi, 0 token EN", r["err"] == 0 and r["n_en"] == 0, r)
    # 7. brand Zalo nghe thành tiếng Việt
    r = utt("nhắn <en>Zalo</en> nha", "nhắn da lô nha")
    check("Zalo -> 'da lô' = 2 lỗi span Anh", r["err_en"] == 2, r)
    # 8. cụm nhiều từ: variants của catalog đưa 'pull-request' về dạng chuẩn
    check("pull-request (variants) = pull request", utt("tạo <en>pull request</en>", "tạo pull-request")["err"] == 0)
    # 9. viết liền / viết rời: CS-WER phạt nhưng span exact = True
    r = utt("mở <en>pull request</en>", "mở pullrequest")
    check("'pullrequest' vs 'pull request': CS-WER phạt, span exact", r["err_en"] == 2 and r["spans"][0]["exact"], r)
    # 10. đồng âm code / cốt: strict phạt, lenient diagnostic (cột phonetic) không phạt
    check("code/cốt: strict = 1 lỗi", utt("sửa <en>code</en>", "sửa cốt")["err_en"] == 1)
    check("code/cốt: lenient_diagnostic = 0 lỗi",
          utt("sửa <en>code</en>", "sửa cốt", "gipbench_lenient_diagnostic_v1")["err_en"] == 0)
    # 11. bảng chuẩn KHÔNG biến từ Việt thành từ Anh ở strict: 'cốt lõi' giữ nguyên
    check("strict: 'cốt lõi' không bị đổi thành 'code'", normalize_hyp("cốt lõi", CAT, "strict") == ["cốt", "lõi"])
    # 12. số -> chữ đối xứng
    check("số: '2024' = 'hai nghìn không trăm hai mươi bốn'",
          utt("năm 2024", "năm hai nghìn không trăm hai mươi bốn")["err"] == 0)
    # 13. dấu câu không tính
    check("dấu câu bị bỏ", utt("Xin chào, bạn!", "xin chào bạn")["err"] == 0)
    # 14. vị trí dấu thanh hoà / hòa: legacy phạt, strict_v1 không
    check("hoà/hòa: legacy phạt", utt("hoà bình", "hòa bình")["err"] == 1)
    check("hoà/hòa: strict_v1 không phạt", utt("hoà bình", "hòa bình", "gipbench_strict_v1")["err"] == 0)
    # 15. strict_v1 giữ nguyên ranh giới span (và chuẩn colour -> color trong span)
    t, spans = normalize_ref(get_profile("gipbench_strict_v1").pre("Hoà <en>Colour</en> ổn"), CAT)
    check("strict_v1 giữ span <en>", spans == [("color",)] and [x.span for x in t] == [-1, 0, -1], (t, spans))
    # 16. CER: một ký tự sai trên chuỗi có dấu cách
    c = utt_counts(["ba", "con"], ["ba", "cơn"])
    check("CER = 1 / 6 ký tự", c["cerr"] == 1 and c["nc"] == 6 and c["err"] == 1, c)
    # 17. hyps thiếu / thừa / trùng id
    k = check_hyps(["a", "b", "c"], ["a", "a", "d"])
    check("check_hyps: thiếu b,c; thừa d; trùng a", k["missing"] == ["b", "c"] and k["extra"] == ["d"]
          and k["duplicate_hyp_ids"] == ["a"] and not k["ok"], k)
    check("check_hyps: đủ thì ok", check_hyps(["a", "b"], ["b", "a"])["ok"])
    # 18. bootstrap: 1 đơn vị -> không có CI; hai hệ giống hệt -> Δ = 0, CI [0, 0]; hệ tốt hơn -> CI < 0
    recs = [{"id": f"u{i}", "unit": f"s{i % 4}", "err": i % 3, "n": 5, "cerr": 1, "nc": 10} for i in range(20)]
    check("bootstrap 1 đơn vị -> None", cluster_bootstrap([dict(r, unit="x") for r in recs], ["wer"], "unit")["wer"] is None)
    p = paired_bootstrap(recs, recs, "wer", "unit", iters=200)
    check("paired: hệ giống hệt -> Δ 0, CI [0,0]", p["delta"] == 0 and tuple(p["ci95"]) == (0.0, 0.0), p)
    p = paired_bootstrap(recs, [dict(r, err=0) for r in recs], "wer", "unit", iters=200)
    check("paired: hệ tốt hơn -> CI Δ < 0", p["delta"] < 0 and p["ci95"][1] < 0, p)
    # 19. bench.evaluate: final mode thiếu hyp -> thoát mã 2; diagnostic -> chạy, hyp thiếu = rỗng
    from gipbench.bench import evaluate as E
    d = Path(tempfile.mkdtemp())
    man = d / "m.jsonl"
    man.write_text("".join(json.dumps({"id": f"u{i}", "dataset": "toy", "speaker": f"s{i % 3}", "duration": 1.0,
                                       "text_tagged": "xin chào <en>code</en>"}, ensure_ascii=False) + "\n"
                           for i in range(6)), encoding="utf-8")
    hy = d / "sys" / "hyps.jsonl"
    hy.parent.mkdir()
    hy.write_text("".join(json.dumps({"id": f"u{i}", "hyp": "xin chào code"}) + "\n" for i in range(5)), encoding="utf-8")
    try:
        E.main(["--manifest", str(man), "--hyps", str(hy), "--mode", "final", "--inference-mode", "open_vocab",
                "--allow-draft", "--iters", "50"])
        check("evaluate final: thiếu hyp -> thoát mã 2", False, "không thoát")
    except SystemExit as e:
        check("evaluate final: thiếu hyp -> thoát mã 2", e.code == 2, e.code)
    s = E.main(["--manifest", str(man), "--hyps", str(hy), "--iters", "50"])
    ov = s["profiles"]["gipbench_strict_v1"]["overall"]
    check("evaluate diagnostic: hyp thiếu = rỗng (WER 4/24)", abs(ov["wer"] - 100 * 4 / 24) < 1e-9 and s["warnings"], ov["wer"])
    hy.write_text(hy.read_text(encoding="utf-8") + json.dumps({"id": "u5", "hyp": "xin chào code"}) + "\n", encoding="utf-8")
    try:
        E.main(["--manifest", str(man), "--hyps", str(hy), "--mode", "final", "--allow-draft", "--profile",
                "gipbench_lenient_diagnostic_v1", "--inference-mode", "open_vocab"])
        check("evaluate final: hồ sơ lenient bị từ chối", False)
    except SystemExit as e:
        check("evaluate final: hồ sơ lenient bị từ chối", e.code == 2)
    try:
        E.main(["--manifest", str(man), "--hyps", str(hy), "--mode", "final", "--inference-mode", "open_vocab"])
        check("evaluate final: benchmark draft bị từ chối nếu thiếu --allow-draft", False)
    except SystemExit as e:
        check("evaluate final: benchmark draft bị từ chối nếu thiếu --allow-draft", e.code == 2)
    s = E.main(["--manifest", str(man), "--hyps", str(hy), "--mode", "final", "--allow-draft", "--inference-mode",
                "open_vocab", "--iters", "50"])
    check("evaluate final: đủ hyps -> chạy, WER 0", s["profiles"]["gipbench_strict_v1"]["overall"]["wer"] == 0)
    provenance_and_baseline(E, d, man, hy)
    cs_breakdown_cases()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


def bd(ref, hyp, prof="gipbench_strict_v1"):
    """Phân rã CS-WER của một câu (core.scoring.cs_breakdown) kèm bản ghi score_utt cùng hồ sơ."""
    from gipbench.core.align import align
    from gipbench.core.scoring import cs_breakdown
    p = get_profile(prof)
    rt = normalize_ref(p.pre(ref), Catalog(), "strict")[0]
    ht = normalize_hyp(p.pre(hyp), Catalog(), "strict")
    ref_t, sp = [t.tok for t in rt], [t.span for t in rt]
    return cs_breakdown(align(ref_t, ht), sp, ref_t, ht), utt(ref, hyp, prof, Catalog())


def cs_breakdown_cases():
    # audit 2.2 / IP-04: tách S / D / I của lỗi span Anh, lỗi ở ranh giới, giữ tiếng Việt, chèn từ Anh vào câu Việt
    # căn chỉnh đặt 'mít' là chèn TRƯỚC 'meeting'->'tinh' (khi bằng điểm): chèn chỉ kề phải, ở ranh giới Việt | Anh
    b, r = bd("họp <en>meeting</en> nhé", "họp mít tinh nhé")
    check("meeting -> mít tinh: S_en 1 + I_en_right_only 1 (ranh giới), cs_wer_sd chỉ tính S",
          (b["S_en"], b["D_en"], b["I_en_left"], b["I_en_right_only"], b["boundary_ins"], b["err_en_sd"])
          == (1, 0, 0, 1, 1, 1), b)
    b, r = bd("<en>meeting</en> nhé", "mít tinh nhé")
    check("span Anh đầu câu, chèn sau phép thay: I_en_left", b["I_en_left"] + b["I_en_right_only"] == 1 and b["S_en"] == 1, b)
    b, r = bd("tôi <en>deploy</en>", "tôi à deploy")
    check("chèn giữa token Việt (trái) và token Anh (phải): I_en_right_only + boundary_ins",
          b["I_en_right_only"] == 1 and b["boundary_ins"] == 1 and b["err_en_sd"] == 0, b)
    b, r = bd("<en>deploy</en> <en>server</en>", "deploy à server")
    check("chèn giữa hai span Anh: không phải ranh giới chuyển ngữ", b["boundary_ins"] == 0 and b["I_en_left"] == 1, b)
    b, r = bd("xin chào bạn", "ok xin chào bạn")
    check("câu thuần Việt, hyp có 'ok' thừa: hyp_latin_extra 1, vi_only err 1 / 3",
          (b["utt_vi_only"], b["hyp_latin_extra"], b["err_vi_only"], b["n_vi_only"]) == (1, 1, 1, 3), b)
    b, r = bd("xin chào bạn", "xin chào cho bạn")
    check("âm tiết Việt không dấu ('cho') không tính là từ Anh chèn", b["hyp_latin_extra"] == 0 and b["err_vi_only"] == 1, b)
    b, r = bd("anh ấy họp", "anh ấy meeting team")
    check("'meeting', 'team' (không phải âm tiết Việt) tính là từ Anh chèn", b["hyp_latin_extra"] == 2, b)
    b, r = bd("gửi file cho anh", "gửi file ok cho anh")
    check("ref không tag nhưng có từ Anh ('file'): không phải câu thuần Việt", b["utt_vi_only"] == 0 and b["hyp_latin_extra"] == 0, b)
    b, r = bd("họp <en>meeting</en> nhé", "họp meeting ok nhé")
    check("câu có span Anh: không tính vi_only / hyp_latin_extra", b["utt_vi_only"] == 0 and b["hyp_latin_extra"] == 0
          and b["n_vi_only"] == 0, b)
    for ref, hyp in [("họp <en>meeting</en> nhé", "họp mít tinh nhé"), ("tôi <en>deploy</en> nhé", "tôi đi ploy nhé"),
                     ("<en>pull request</en> này của <en>team</en> mình", "bun ri quét này team"),
                     ("a <en>b c</en> d", ""), ("", "x y"), ("xin <en>chào</en>", "xin chào chào chào")]:
        b, r = bd(ref, hyp)
        check(f"bất biến err_en = S_en + D_en + I_en_left + I_en_right_only ({ref!r} / {hyp!r})",
              r["err_en"] == b["S_en"] + b["D_en"] + b["I_en_left"] + b["I_en_right_only"], (r, b))


def provenance_and_baseline(E, d, man, hy):
    # IP-03: baseline hyps phải được kiểm như hyps chính (audit: cắt baseline còn 1/2 -> Δ đảo dấu mà vẫn thoát 0)
    base = d / "base" / "hyps.jsonl"
    base.parent.mkdir()
    base.write_text("".join(json.dumps({"id": f"u{i}", "hyp": "xin chào"}) + "\n" for i in range(3)), encoding="utf-8")
    try:
        E.main(["--manifest", str(man), "--hyps", str(hy), "--mode", "final", "--allow-draft", "--inference-mode",
                "open_vocab", "--iters", "20", "--baseline-hyps", str(base), "--out", str(d / "o1")])
        check("evaluate final: baseline thiếu id -> thoát mã 2", False, "không thoát")
    except SystemExit as e:
        check("evaluate final: baseline thiếu id -> thoát mã 2", e.code == 2, e.code)
    s = E.main(["--manifest", str(man), "--hyps", str(hy), "--iters", "20", "--baseline-hyps", str(base), "--out", str(d / "o2")])
    check("evaluate diagnostic: baseline thiếu id -> cảnh báo + baseline_hyp_check",
          s["baseline_hyp_check"]["missing"] == 3 and any("baseline" in w for w in s["warnings"]), s.get("baseline_hyp_check"))
    check("summary: baseline_hyps_sha256", len(s.get("baseline_hyps_sha256") or "") == 64)
    # provenance: label_field, overrides, scorer_code_sha256 ổn định
    s2 = E.main(["--manifest", str(man), "--hyps", str(hy), "--iters", "20", "--out", str(d / "o3")])
    check("summary: label_field", s2.get("label_field") == "text_tagged", s2.get("label_field"))
    check("summary: scorer_code_sha256 (64 hex) ổn định giữa hai lần chạy",
          len(s2.get("scorer_code_sha256") or "") == 64 and s2["scorer_code_sha256"] == s["scorer_code_sha256"])
    check("summary: profile.code_sha256", len(s2["profile"].get("code_sha256") or "") == 64)
    check("summary: overrides ghi manifest khi chấm ngoài registry", "manifest" in (s2.get("overrides") or {}), s2.get("overrides"))
    ov = s2["profiles"]["gipbench_strict_v1"]["overall"]
    check("summary: có metric mới cs_wer_sd / boundary_ins_rate / vi_only_wer / en_hallucination_per100 / cs_errors",
          all(k in ov for k in ("cs_wer_sd", "boundary_ins_rate", "vi_only_wer", "en_hallucination_per100", "cs_errors")), list(ov))
    s3 = E.main(["--manifest", str(man), "--hyps", str(hy), "--iters", "20", "--label-field", "text", "--out", str(d / "o4")])
    check("summary: --label-field khác mặc định ghi vào overrides",
          s3["label_field"] == "text" and s3["overrides"].get("label_field") == "text", s3.get("overrides"))
    try:
        E.main(["--manifest", str(man), "--hyps", str(hy), "--mode", "final", "--allow-draft", "--inference-mode",
                "open_vocab", "--iters", "20", "--label-field", "text_tagged_orig", "--out", str(d / "o5")])
        check("evaluate final: label_field không khai báo trong benchmark -> thoát mã 2", False, "không thoát")
    except SystemExit as e:
        check("evaluate final: label_field không khai báo trong benchmark -> thoát mã 2", e.code == 2, e.code)


if __name__ == "__main__":
    main()
