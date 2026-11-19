#!/usr/bin/env python3
"""Kiểm nhiễm bẩn test -> train / hotwords (core.decontam).  python tests/core/test_leakage.py"""
import sys
import tempfile
from pathlib import Path

from gipbench.core.decontam import audit, dup_key, find_terms_in_texts, near_duplicates, test_terms_from_rows

FAIL = []
YT = "AbCdEfGh-Jk"  # YouTube ID 11 ký tự (có '-')


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def kinds_of(rep):
    return {f["kind"]: f for f in rep["findings"]}


def vamped(d: Path):
    """Ca của audit 17/11 mục D.3: bộ kiểm cũ bị vượt im lặng (lệch định dạng id, thiếu audio, clip resample...)."""
    from gipbench.core import decontam as D
    # D.3 ca 1: test speaker là URL, train source là 'youtube:ID' -> cùng video
    t = [{"id": "e0_a", "speaker": f"https://www.youtube.com/watch?v={YT}", "source": "e0_a", "text": "câu một ở đây"}]
    tr = [{"id": "vmc_1", "speaker": f"yt_{YT}", "source": f"youtube:{YT}", "text": "một câu hoàn toàn khác"}]
    k = kinds_of(audit(t, tr, check_audio_sha=False))
    check("D.3-1 URL youtube vs youtube:ID -> source_norm_overlap critical",
          k.get("source_norm_overlap", {}).get("severity") == "critical" and k["source_norm_overlap"]["test_rows"] == 1, k)
    check("D.3-1 trường cũ speaker/source_overlap giữ nguyên (không thấy trùng chuỗi)",
          "speaker_overlap" not in k and "source_overlap" not in k, k)
    # D.3 ca 2: test không có audio_sha256 / file -> audio_unverifiable, critical khi require_audio_sha
    t2 = [{"id": "x", "text": "a"}, {"id": "y", "text": "b", "audio_sha256": "f" * 64}]
    tr2 = [{"id": "z", "text": "c", "_audio_path": str(d / "khong_co.wav")}]
    k = kinds_of(audit(t2, tr2, require_audio_sha=True))
    au = k.get("audio_unverifiable", {})
    check("D.3-2 thiếu hash audio -> audio_unverifiable critical (require_audio_sha)",
          au.get("severity") == "critical" and au.get("test_rows") == 1 and au.get("train_rows") == 1, au)
    k = kinds_of(audit(t2, tr2))
    check("D.3-2 không require -> audio_unverifiable mức review", k.get("audio_unverifiable", {}).get("severity") == "review", k)
    # D.3 ca 3: cùng clip gốc nhưng resample (sha khác, id khác) -> clip_overlap qua mã clip gốc
    t3 = [{"id": "e0_cv17_vi_common_voice_vi_123.mp3", "upstream_id": "common_voice_vi_123.mp3", "audio_sha256": "a" * 64,
           "speaker": "cv_1111222233334444", "text": "một"}]
    tr3 = [{"id": "cv_common_voice_vi_123", "audio_sha256": "b" * 64, "speaker": "cv_9999888877776666", "text": "hai"}]
    k = kinds_of(audit(t3, tr3))
    check("D.3-3 cùng clip CV đã resample -> clip_overlap critical", k.get("clip_overlap", {}).get("severity") == "critical", k)
    # D.3 ca 4: train là đoạn 0-8 s, test là đoạn 4-12 s cùng bản ghi (id khác, text khác) -> critical theo nguồn
    t4 = [{"id": "c0_vmc_Med_CS-5-9", "speaker": f"yt_{YT}", "source": f"youtube:{YT}",
           "text_tagged": "uống <en>vitamin</en> mỗi ngày nhé các bạn"}]
    tr4 = [{"id": "vmc_Med_CS-5-2", "source": f"https://youtu.be/{YT}", "text": "chào mừng các bạn quay lại kênh"}]
    k = kinds_of(audit(t4, tr4, check_audio_sha=False))
    check("D.3-4 đoạn khác của cùng bản ghi (youtu.be) -> source_norm_overlap", "source_norm_overlap" in k, k)
    # D.3 ca 5: hotwords lấy từ text test không có <en> -> test_ngram_in_runtime_vocab (review)
    t5 = [{"id": "g1", "text_tagged": "thủ tục cấp căn cước công dân gắn chip"}]
    k = kinds_of(audit(t5, [], check_audio_sha=False, runtime_vocab=["căn cước công dân", "zalo"]))
    ng = k.get("test_ngram_in_runtime_vocab", {})
    check("D.3-5 hotwords chứa cụm lấy từ text test (không tag) -> review",
          ng.get("severity") == "review" and ng.get("examples", [{}])[0].get("phrase") == "căn cước công dân", ng)
    # ca 7: source_key không bao giờ trả id câu
    sk = D.source_key
    check("source_key: G-Group (source = id câu) -> None",
          sk({"id": "g0_gg_edu_1", "speaker": "g0_gg_edu_1", "source": "ggroup:edu:1"}) is None)
    check("source_key: speaker = id -> None", sk({"id": "e0_lsvsc_x.wav", "speaker": "e0_lsvsc_x.wav"}) is None)
    check("source_key: yt_ / youtube: / watch?v= / youtu.be cho cùng khoá",
          len({sk({"id": "1", "speaker": f"yt_{YT}"}), sk({"id": "2", "source": f"youtube:{YT}"}),
               sk({"id": "3", "speaker": f"https://www.youtube.com/watch?v={YT}&t=3"}),
               sk({"id": "4", "source": f"https://youtu.be/{YT}"})}) == 1)
    check("source_key: CV cv_<hash> và cv:<hash> cùng khoá",
          sk({"id": "a", "speaker": "cv_018d3b0534bf0c4c"}) == sk({"id": "b", "source": "cv:018d3b0534bf0c4c"}) is not None)
    check("source_key: VIVOS", sk({"id": "a", "speaker": "vivos_VIVOSDEV02"}) == sk({"id": "b", "source": "vivos:VIVOSDEV02"}) == "vivos:VIVOSDEV02")
    check("source_key: id câu dạng yt_ không bị lấy (chỉ xét speaker/source/upstream_id/audio)",
          sk({"id": f"yt_{YT}"}) is None)
    # identity_unknown: báo tỉ lệ câu test không xác định được nguồn
    k = kinds_of(audit([{"id": "a", "text": "x"}, {"id": "b", "speaker": f"yt_{YT}", "text": "y"}], [], check_audio_sha=False))
    iu = k.get("identity_unknown", {})
    check("identity_unknown: 1/2 câu test", iu.get("severity") == "review" and iu.get("test_rows") == 1
          and abs(iu.get("share", 0) - 0.5) < 1e-9, iu)
    # trùng text trong bộ đọc có chủ ý (VIVOS) -> review, không critical; ngoài bộ đọc vẫn critical
    tv = [{"id": "g0_vivos_VIVOSDEV02_R1", "dataset": "vivos", "domain": "read", "text": "hôm nay trời đẹp quá"}]
    trv = [{"id": "vivos_VIVOSSPK01_R9", "dataset": "vivos:train", "domain": "read", "text": "Hôm nay trời đẹp quá."}]
    k = kinds_of(audit(tv, trv, check_audio_sha=False))
    check("trùng text bộ đọc cùng họ -> lexical_overlap_read_corpus (review)",
          "text_exact_overlap" not in k and k.get("lexical_overlap_read_corpus", {}).get("severity") == "review", k)
    k = kinds_of(audit(tv, [dict(trv[0], dataset="vss_cs", domain="other")], check_audio_sha=False))
    check("trùng text khác họ -> text_exact_overlap critical", "text_exact_overlap" in k, k)
    # miễn trừ: finding được miễn vẫn in, không tính critical
    ex = [{"kind": "source_norm_overlap", "value": f"yt:{YT}", "reason": "test", "approved_by": "u", "date": "2026-11-17"}]
    rep = audit(t, tr, check_audio_sha=False, exemptions=ex)
    k = kinds_of(rep)
    check("miễn trừ theo value: không còn critical source_norm_overlap, có bản exempted",
          "source_norm_overlap" not in k and k.get("source_norm_overlap:exempted", {}).get("severity") == "exempted"
          and rep["critical"] == 0, rep)
    # CLI: --require-audio-sha mặc định bật khi --fail-on critical; stats in vào markdown
    import contextlib
    import io
    import json
    from gipbench.dataset.release import leak_audit as LA
    tf, trf = d / "t.jsonl", d / "tr.jsonl"
    tf.write_text(json.dumps({"id": "q1", "speaker": f"yt_{YT}", "text": "một hai ba bốn"}, ensure_ascii=False) + "\n", encoding="utf-8")
    trf.write_text(json.dumps({"id": "q2", "speaker": "khac", "text": "năm sáu bảy"}, ensure_ascii=False) + "\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            LA.main(["--test", str(tf), "--train", str(trf), "--out", str(d / "rep")])
            code = 0
        except SystemExit as e:
            code = e.code
    md = (d / "rep" / "leak_report.md").read_text(encoding="utf-8")
    check("CLI: thiếu hash audio + mặc định -> thoát 1 (đóng khi lỗi)", code == 1, code)
    check("CLI: markdown in số câu không đọc được audio", "không xác định được sha256 audio" in md, md)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            LA.main(["--test", str(tf), "--train", str(trf), "--no-audio-sha"])
            code = 0
        except SystemExit as e:
            code = e.code
    check("CLI: --no-audio-sha (bỏ kiểm có chủ ý) và không trùng -> thoát 0", code == 0, code)


def main():
    d = Path(tempfile.mkdtemp())
    (d / "a.wav").write_bytes(b"same-bytes")
    (d / "b.wav").write_bytes(b"same-bytes")
    (d / "c.wav").write_bytes(b"other")
    test = [
        {"id": "t1", "speaker": "spkA", "source": "vidA", "text_tagged": "hôm nay tôi <en>deploy</en> cái <en>backend</en>",
         "_audio_path": str(d / "a.wav")},
        {"id": "t2", "speaker": "spkB", "source": "vidB", "text_tagged": "Bệnh nhân cần xét nghiệm máu định kỳ mỗi tháng một lần",
         "_audio_path": str(d / "c.wav")},
        {"id": "t3", "speaker": "spkC", "source": "vidC", "text_tagged": "xin chào", "_audio_path": str(d / "c.wav")},
    ]
    train = [
        {"id": "x1", "speaker": "spkA", "source": "vidZ", "text": "khác hẳn", "_audio_path": str(d / "b.wav")},
        {"id": "x2", "speaker": "spkQ", "source": "vidB", "text": "bệnh nhân cần xét nghiệm máu định kì mỗi tháng 1 lần nhé"},
        {"id": "x3", "speaker": "spkR", "source": "vidR", "text": "XIN CHÀO!"},
        {"id": "x4", "speaker": "spkS", "source": "vidS", "text": "mình sẽ deploy lên server"},
    ]
    rep = audit(test, train, reserved_terms=["deploy", "kubernetes"], runtime_vocab=["BACKEND", "zalo"],
                test_terms=test_terms_from_rows(test))
    kinds = {f["kind"]: f for f in rep["findings"]}
    check("trùng speaker", "speaker_overlap" in kinds)
    check("trùng source", "source_overlap" in kinds)
    check("trùng sha256 audio (khác tên file)", "audio_sha256_overlap" in kinds)
    check("trùng text sau chuẩn hoá (hoa/thường, dấu câu)", kinds.get("text_exact_overlap", {}).get("count") == 1,
          kinds.get("text_exact_overlap"))
    check("thuật ngữ FT-unseen có trong train",
          [e["term"] for e in kinds.get("reserved_term_in_train", {}).get("examples", [])] == ["deploy"])
    check("hotwords chứa thuật ngữ lấy từ test", kinds.get("test_term_in_runtime_vocab", {}).get("examples") == ["backend"])
    check("không có id trùng", "id_overlap" not in kinds)
    check("đếm critical >= 6", rep["critical"] >= 6, rep["critical"])
    nd = near_duplicates(test, train, threshold=0.6)
    check("gần trùng (kỳ/kì, số 1, thêm 'nhé')", [x["test_id"] for x in nd] == ["t2"], nd)
    check("câu ngắn không xét gần trùng", all(x["test_id"] != "t3" for x in nd))
    check("dup_key chuẩn chính tả + số", dup_key("Định kỳ 1 lần!") == dup_key("định kì một lần"))
    check("find_terms khớp cụm nhiều từ",
          find_terms_in_texts(["pull request"], ["tạo Pull Request mới", "pull nhé"]) == {"pull request": 1})
    clean = audit([test[2]], [{"id": "z", "speaker": "q", "source": "q", "text": "tạm biệt"}], check_audio_sha=False)
    check("tập sạch -> 0 critical", clean["critical"] == 0, clean)
    check("tập sạch: chỉ còn finding identity_unknown (không xác định được nguồn)",
          [x["kind"] for x in clean["findings"]] == ["identity_unknown"], clean)
    vamped(d)
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
