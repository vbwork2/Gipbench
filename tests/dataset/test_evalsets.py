#!/usr/bin/env python3
"""gipbench.dataset.ingest.evalsets + fetch kind "evalset", bằng dữ liệu giả. Không cần mạng.

  python tests/dataset/test_evalsets.py
"""
import csv
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from gipbench.dataset.ingest import evalsets as EV
from gipbench.dataset.release import fetch as F

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def wav(sec, sr, ch=1, freq=440.0):
    t = np.arange(int(sec * sr)) / sr
    x = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if ch > 1:
        x = np.stack([x] * ch, axis=1)
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def main():
    d = Path(tempfile.mkdtemp())
    # parquet giả: shard 0 có 2 câu 16 kHz mono (giữ byte), shard 1 có 1 câu 44.1 kHz stereo (resample) trùng tên file
    w16, w44 = wav(1.0, 16000), wav(2.0, 44100, ch=2)
    p0, p1 = d / "test-00000-of-00002.parquet", d / "test-00001-of-00002.parquet"
    pq.write_table(pa.table({"audio": [{"bytes": w16, "path": "001.wav"}, {"bytes": w16, "path": "002.wav"}],
                             "transcription": ["Xin chào", "hai"], "gender": ["female", "male"]}), p0)
    pq.write_table(pa.table({"audio": [{"bytes": w44, "path": "001.wav"}], "transcription": ["ba"], "gender": ["male"]}), p1)
    EV.SPECS["fake_pq"] = {"repo": "x/y", "revision": "r", "license": "CC0", "split": "test", "reader": "parquet",
                           "files": [("a", EV.sha256_file(p0)), ("b", EV.sha256_file(p1))], "id": None,
                           "id_with_shard": True, "text": "transcription", "speaker": None, "meta": {"gender": "gender"}}
    rows = []
    for f in (p0, p1):
        rows += [r for r, why in EV.materialize("fake_pq", [str(f)], d / "out" / "audio") if not why]
    ids = [r["id"] for r in rows]
    check("id kèm tên shard: cùng tên file ở hai shard không trùng", len(set(ids)) == 3, ids)
    check("lời thoại giữ nguyên chữ gốc", rows[0]["text_tagged"] == "Xin chào")
    check("16 kHz mono -> giữ nguyên byte", rows[0]["audio_derivation"] == "upstream_bytes"
          and Path(rows[0]["_path"]).read_bytes() == w16)
    i = sf.info(rows[2]["_path"])
    check("44.1 kHz stereo -> resample 16 kHz mono, độ dài giữ nguyên",
          rows[2]["audio_derivation"] == EV.DERIVED_TAG and i.samplerate == 16000 and i.channels == 1
          and abs(i.frames / 16000 - 2.0) < 0.002, i)
    check("audio_sha256 = sha256 file đã ghi", all(r["audio_sha256"] == EV.sha256_file(r["_path"]) for r in rows))
    check("trường meta được chép (gender)", rows[1].get("gender") == "male")
    # FLEURS giả: tsv không header (7 cột) + tar.gz
    tsv = d / "test.tsv"
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(["11", "a1.wav", "Năm 2020, mưa.", "năm 2020 mưa", "x", "16000", "FEMALE"])
    tgz = d / "test.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        ti = tarfile.TarInfo("test/a1.wav")
        ti.size = len(w16)
        tf.addfile(ti, io.BytesIO(w16))
    EV.SPECS["fake_fleurs"] = {"repo": "x/f", "revision": "r", "license": "CC-BY-4.0", "split": "test", "reader": "fleurs",
                               "files": [("t", EV.sha256_file(tsv)), ("a", EV.sha256_file(tgz))]}
    fr = [r for r, why in EV.materialize("fake_fleurs", [str(tsv), str(tgz)], d / "outf") if not why]
    check("FLEURS: lấy raw_transcription, giữ bản chuẩn hoá của dataset riêng",
          len(fr) == 1 and fr[0]["text_tagged"] == "Năm 2020, mưa." and fr[0]["text_norm_upstream"] == "năm 2020 mưa", fr)
    # CV17 giả: tsv có header + tar; mp3 giả không đọc được -> phải báo lỗi rõ, không ghi bừa
    ctsv = d / "cv.tsv"
    with open(ctsv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["client_id", "path", "sentence_id", "sentence", "gender", "age", "accents"])
        wr.writerow(["c" * 40, "common_voice_vi_1.mp3", "s1", "Hương thu", "", "", ""])
    ctar = d / "cv.tar"
    with tarfile.open(ctar, "w") as tf:
        ti = tarfile.TarInfo("vi_test_0/common_voice_vi_1.mp3")
        ti.size = 9
        tf.addfile(ti, io.BytesIO(b"ID3fakemp"))
    EV.SPECS["fake_cv"] = {"repo": "x/c", "revision": "r", "license": "CC0-1.0", "split": "test", "reader": "cv17",
                           "files": [("t", EV.sha256_file(ctsv)), ("a", EV.sha256_file(ctar))]}
    try:
        list(EV.materialize("fake_cv", [str(ctsv), str(ctar)], d / "outc"))
        check("CV: audio hỏng -> báo lỗi", False)
    except Exception:
        check("CV: audio hỏng -> báo lỗi", True)
    # fetch kind evalset: người nhận dựng lại đúng audio theo upstream_id, đổi tên theo id manifest
    F.SOURCES["fake_pq"] = {"kind": "evalset", "repo": "x/y", "revision": "r", "files": EV.SPECS["fake_pq"]["files"]}
    man = d / "m.jsonl"
    mrows = [{k: v for k, v in r.items() if k != "_path"} for r in rows]
    man.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mrows), encoding="utf-8")
    try:
        F.main(["--manifest", str(man), "--out", str(d / "recv"), "--local-source", f"fake_pq={p0}", f"fake_pq={p1}"])
        rc = 0
    except SystemExit as e:
        rc = e.code
    rep = json.loads((d / "recv" / "fetch_report.json").read_text(encoding="utf-8"))
    check("fetch evalset: đủ 3 / 3 câu, sha256 khớp, thoát mã 0", rc == 0 and rep["rows_ok"] == 3 and not rep["problems"], rep)
    bad = d / "bad.parquet"
    pq.write_table(pa.table({"audio": [{"bytes": w16, "path": "x.wav"}], "transcription": ["x"], "gender": ["m"]}), bad)
    try:
        F.main(["--manifest", str(man), "--out", str(d / "recv2"), "--local-source", f"fake_pq={bad}", f"fake_pq={p1}"])
        check("fetch evalset: file nguồn khác bản khoá -> dừng", False)
    except SystemExit as e:
        check("fetch evalset: file nguồn khác bản khoá -> dừng", e.code not in (0, None))
    # bộ không cho phân phối lại lời thoại: package bỏ text, fetch lấy lại từ nguồn, manifest_sha256 trùng bản khoá
    from gipbench.dataset.release.build_general import manifest_sha256
    from gipbench.dataset.release.package import share_rows, strip_text
    EV.SPECS["fake_nd"] = dict(EV.SPECS["fake_pq"], redistribute_text=False)
    F.SOURCES["fake_nd"] = dict(F.SOURCES["fake_pq"])
    nd = []
    for f in (p0, p1):
        nd += [r for r, why in EV.materialize("fake_nd", [str(f)], d / "nd" / "audio") if not why]
    check("bộ không cho phân phối lại: dòng mang text_redistributable = false",
          all(r.get("text_redistributable") is False for r in nd))
    full = share_rows([dict(r, audio="x") for r in nd])
    shared = strip_text(full)
    check("package bỏ lời thoại khỏi bản share", all("text_tagged" not in r for r in shared))
    man3 = d / "m3.jsonl"
    man3.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in shared), encoding="utf-8")
    try:
        F.main(["--manifest", str(man3), "--out", str(d / "recv3"), "--local-source", f"fake_nd={p0}", f"fake_nd={p1}"])
    except SystemExit:
        pass
    got3 = [json.loads(l) for l in open(d / "recv3" / "manifest.jsonl", encoding="utf-8")]
    check("fetch lấy lại lời thoại từ nguồn", [r.get("text_tagged") for r in got3] == [r["text_tagged"] for r in full])
    check("manifest sau fetch trùng manifest_sha256 bản khoá", manifest_sha256(got3) == manifest_sha256(full))
    # bộ cho phân phối lại nhưng nằm trong share_policy.text_not_redistributable: package bỏ text (không thêm cờ),
    # fetch vẫn lấy lại, manifest trùng bản khoá
    pol_rows = share_rows([dict(r) for r in mrows])
    shared5 = strip_text(pol_rows, {"fake_pq"})
    check("strip_text theo policy: bỏ text, không thêm trường cờ",
          all("text_tagged" not in r and "text_redistributable" not in r for r in shared5))
    man5 = d / "m5.jsonl"
    man5.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in shared5), encoding="utf-8")
    try:
        F.main(["--manifest", str(man5), "--out", str(d / "recv5"), "--local-source", f"fake_pq={p0}", f"fake_pq={p1}"])
    except SystemExit:
        pass
    got5 = [json.loads(l) for l in open(d / "recv5" / "manifest.jsonl", encoding="utf-8")]
    check("fetch lấy lại lời thoại bộ thuộc policy, manifest trùng bản khoá",
          manifest_sha256(got5) == manifest_sha256(pol_rows) and all("text_tagged" in r for r in got5))
    # dòng metadata-only (không audio_sha256 / duration): fetch dựng audio, kiểm bằng sha256 file nguồn
    meta_rows = [{k: v for k, v in r.items() if k not in ("audio_sha256", "duration", "_path")} for r in rows]
    for r in meta_rows:
        r["audio_derivation"] = EV.METADATA_DERIVATION
    check("base_row giống nhau giữa nhập đầy đủ và metadata-only",
          all(EV.base_row("fake_pq", r["id"], r["upstream_id"], {"text": r["text_tagged"], "speaker": None,
                                                                 "session": None, "gender": r.get("gender")})
              == {k: v for k, v in r.items() if k != "audio_derivation"} for r in meta_rows))
    man4 = d / "m4.jsonl"
    man4.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in share_rows(meta_rows)), encoding="utf-8")
    try:
        F.main(["--manifest", str(man4), "--out", str(d / "recv4"), "--local-source", f"fake_pq={p0}", f"fake_pq={p1}"])
        rc4 = 0
    except SystemExit as e:
        rc4 = e.code
    rep4 = json.loads((d / "recv4" / "fetch_report.json").read_text(encoding="utf-8"))
    check("fetch dòng metadata-only: đủ câu, kiểm bằng sha256 file nguồn", rc4 == 0 and rep4["rows_ok"] == 3
          and rep4["rows_checked_by_source_sha_only"] == 3, rep4)
    check("mọi bộ E0 thật đều có nguồn trong fetch.SOURCES",
          all(k in F.SOURCES for k in ("fleurs_vi", "cv17_vi", "vietmed", "lsvsc", "vimd", "speech_massive_vi", "vimedcss_test")))
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
