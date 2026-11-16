#!/usr/bin/env python3
"""fetch (tar / parquet giữ byte / parquet resample / Common Voice cục bộ) + package, bằng dữ liệu giả. Không cần mạng.

  python tests/dataset/test_fetch.py
"""
import hashlib
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

from gipbench.dataset.release import fetch as F

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def wav_bytes(sec, sr, freq=440.0):
    t = np.arange(int(sec * sr)) / sr
    b = io.BytesIO()
    sf.write(b, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run(args):
    try:
        F.main(args)
        return 0
    except SystemExit as e:
        return e.code


def main():
    d = Path(tempfile.mkdtemp())
    # VIVOS giả: tar.gz có vivos/test/waves/<spk>/<id>.wav
    v1, v2 = wav_bytes(1.0, 16000, 300), wav_bytes(1.2, 16000, 500)
    tgz = d / "vivos.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        for name, b in [("vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R001.wav", v1),
                        ("vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R002.wav", v2),
                        ("vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav", v1)]:
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    # G-Group giả: parquet {id, audio{bytes,path}, text}
    g1 = b"ID3fake-mp3-bytes-1"
    gpq = d / "tech.parquet"
    pq.write_table(pa.table({"id": ["113015", "999"], "audio": [{"bytes": g1, "path": "113015.mp3"},
                                                                {"bytes": b"x", "path": "999.mp3"}],
                             "text": ["a", "b"]}), gpq)
    # ViMedCSS giả: 44.1 kHz -> resample 16 kHz
    m44 = wav_bytes(2.0, 44100)
    mpq = [d / "hard0.parquet", d / "hard1.parquet"]
    pq.write_table(pa.table({"segment_id": ["Med_CS-1-1"], "audio": [{"bytes": m44, "path": "a.wav"}]}), mpq[0])
    pq.write_table(pa.table({"segment_id": ["Med_CS-2-1"], "audio": [{"bytes": m44, "path": "b.wav"}]}), mpq[1])
    # Common Voice giả
    cv = d / "cv" / "clips"
    cv.mkdir(parents=True)
    (cv / "common_voice_vi_1.mp3").write_bytes(b"cvbytes")
    rows = [
        {"id": "g0_vivos_VIVOSDEV01_R001", "dataset": "vivos", "upstream_id": "VIVOSDEV01_R001", "audio_sha256": sha(v1), "duration": 1.0},
        {"id": "g0_vivos_VIVOSDEV02_R002", "dataset": "vivos", "upstream_id": "VIVOSDEV02_R002", "audio_sha256": sha(v2), "duration": 1.2},
        {"id": "g0_gg_tech_113015", "dataset": "ggroup_tech", "upstream_id": "113015", "audio_sha256": sha(g1), "duration": 1.0},
        {"id": "c0_vmc_Med_CS-1-1", "dataset": "vimedcss", "upstream_id": "Med_CS-1-1", "audio_sha256": "khac", "duration": 2.0},
        {"id": "c0_vmc_Med_CS-2-1", "dataset": "vimedcss", "upstream_id": "Med_CS-2-1", "audio_sha256": "khac", "duration": 2.0},
        {"id": "g0_cv1", "dataset": "commonvoice_vi_27", "upstream_id": "common_voice_vi_1.mp3",
         "audio_sha256": sha(b"cvbytes"), "duration": 1.0},
    ]
    man = d / "m.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # bản khoá giả cho nguồn (thay SOURCES thật trong lúc test)
    F.SOURCES["vivos"]["files"] = [("data/vivos.tar.gz", F.sha256_file(tgz))]
    F.SOURCES["ggroup_tech"]["files"] = [("data/x.parquet", F.sha256_file(gpq))]
    F.SOURCES["vimedcss"]["files"] = [("h0", F.sha256_file(mpq[0])), ("h1", F.sha256_file(mpq[1]))]
    out = d / "out"
    rc = run(["--manifest", str(man), "--out", str(out), "--cv-root", str(d / "cv"),
              "--local-source", f"vivos={tgz}", f"ggroup_tech={gpq}", f"vimedcss={mpq[0]}", f"vimedcss={mpq[1]}"])
    rep = json.loads((out / "fetch_report.json").read_text(encoding="utf-8"))
    got = [json.loads(l) for l in open(out / "manifest.jsonl", encoding="utf-8")]
    check("đủ 6 / 6 dòng, thoát mã 0", rc == 0 and rep["rows_ok"] == 6, rep)
    check("VIVOS tách đúng file test, giữ nguyên byte", F.sha256_file(out / got[0]["audio"]) == sha(v1))
    check("G-Group giữ nguyên byte mp3", (out / got[2]["audio"]).read_bytes() == g1)
    info = sf.info(str(out / got[3]["audio"]))
    check("ViMedCSS resample 16 kHz mono, độ dài giữ nguyên", info.samplerate == 16000 and info.channels == 1
          and abs(info.frames / 16000 - 2.0) < 0.002, info)
    check("ViMedCSS lệch sha nhưng độ dài khớp -> chỉ cảnh báo", rep["derived_sha_mismatch_duration_ok"] == 2)
    check("manifest đầu ra có đường dẫn audio tương đối", all(not Path(r["audio"]).is_absolute() for r in got))
    # chạy lại riêng một bộ vào cùng --out: dòng của các bộ khác phải còn
    run(["--manifest", str(man), "--out", str(out), "--datasets", "ggroup_tech", "--local-source", f"ggroup_tech={gpq}"])
    again = [json.loads(l) for l in open(out / "manifest.jsonl", encoding="utf-8")]
    check("chạy từng bộ vào cùng thư mục không làm mất dòng bộ khác", len(again) == 6 and [r["id"] for r in again] == [r["id"] for r in rows])
    # audio giữ byte bị sửa -> lỗi
    man2 = d / "m2.jsonl"
    man2.write_text(json.dumps(dict(rows[2], audio_sha256="0" * 64)) + "\n", encoding="utf-8")
    check("sha256 lệch ở nguồn giữ byte -> thoát mã 1",
          run(["--manifest", str(man2), "--out", str(d / "o2"), "--local-source", f"ggroup_tech={gpq}"]) == 1)
    # file nguồn khác bản khoá -> dừng
    other = d / "other.parquet"
    pq.write_table(pa.table({"id": ["1"], "audio": [{"bytes": b"z", "path": "z.mp3"}]}), other)
    check("file nguồn khác sha256 bản khoá -> dừng",
          run(["--manifest", str(man2), "--out", str(d / "o3"), "--local-source", f"ggroup_tech={other}"]) not in (0, None))
    # thiếu --cv-root: bỏ qua bộ đó, có báo, không lỗi
    man3 = d / "m3.jsonl"
    man3.write_text(json.dumps(rows[5]) + "\n", encoding="utf-8")
    rc = run(["--manifest", str(man3), "--out", str(d / "o4")])
    rep = json.loads((d / "o4" / "fetch_report.json").read_text(encoding="utf-8"))
    check("Common Voice không có --cv-root -> bỏ qua có báo", rc == 0 and "cv-root" in rep["datasets"]["commonvoice_vi_27"])
    # package: từ chối dòng test_only; bỏ audio / trường _
    from gipbench.dataset.release.package import share_rows
    try:
        share_rows([dict(rows[2], license="test_only")])
        check("package từ chối dòng test_only", False)
    except SystemExit:
        check("package từ chối dòng test_only", True)
    s = share_rows([dict(rows[3], audio="x/y.wav", _line=3)])
    check("package bỏ audio / trường _, ghi audio_derivation",
          "audio" not in s[0] and "_line" not in s[0] and s[0]["audio_derivation"] == "soxr_hq_16k_mono_pcm16")
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
