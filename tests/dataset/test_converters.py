#!/usr/bin/env python3
"""Kiểm thử các converter bằng dataset GIẢ có đúng schema của bản thật (không cần mạng).

  python tests/dataset/test_converters.py [thư_mục_tạm]
"""
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from gipbench.core.paths import REPO as G
T = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gipbench_convtest")
rng = np.random.default_rng(0)


def wav_bytes(dur, fmt="WAV", sr=16000, channels=1):
    b = io.BytesIO()
    x = (0.05 * rng.standard_normal((int(sr * dur), channels))).astype("float32")
    sf.write(b, x if channels > 1 else x[:, 0], sr, format=fmt)
    return b.getvalue()


def run(*a, ok=(0,)):
    print("$", " ".join(str(x) for x in a))
    p = subprocess.run([sys.executable, *map(str, a)], capture_output=True, text=True)
    print(p.stdout[-1800:], p.stderr[-600:] if p.returncode not in ok else "")
    assert p.returncode in ok, f"exit {p.returncode}"
    return p.stdout


def main():
    from datasets import Audio, Dataset
    dl = T / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    # --- G-Group: id, audio, text, duration
    gg = {"tech": ["đây là google search google search rất tận dụng", "bấm connect metamask để móc các tài khoản",
                   "hôm nay mình hướng dẫn cài đặt máy in"],
          "finance": ["khi mua ipo luôn nên nhà đầu tư mới mới", "cả tuần bitcoin sideway em đánh margin toàn win"],
          "pubadmin": ["anh chị mang theo căn cước công dân bản gốc", "nộp hồ sơ online trên cổng dịch vụ công"]}
    for name, texts in gg.items():
        ds = Dataset.from_dict({"id": [f"{100000 + i}" for i in range(len(texts))],
                                "audio": [{"bytes": wav_bytes(3 + i), "path": f"{i}.wav"} for i in range(len(texts))],
                                "text": texts, "duration": [3.0 + i for i in range(len(texts))]}).cast_column("audio", Audio())
        ds.to_parquet(str(dl / f"ggroup_{name}.parquet"))
    # --- ViMedCSS
    vm = {"train": [("Med_CS-0-1", "5 alpha reductase là một enzyme chuyển đổi testosterone.", "reductase; testosteron", "GUEt9JwUd-4"),
                    ("Med_CS-1-1", "Bổ sung vitamin và lipid cho cơ thể.", "lipid", "y578jWB56QE")],
          "validation": [("Med_CS-2-1", "methionin là một axit amin thiết yếu.", "methionin; amin", "PKaw6828hUM")],
          "test": [("Med_CS-3-1", "Khi mà bị viêm tuyến Bartholin thì chị em thấy sưng đau.", "Bartholin", "yrtgA1ZH14Q")],
          "hard": [("Med_CS-4-1", "Thuốc này ức chế leukotriene trong cơ thể.", "leukotriene", "hardvid0001"),
                   ("Med_CS-4-2", "Hoạt chất quercetin giúp giảm viêm, không thấy thuật ngữ.", "prostan", "hardvid0002")]}
    for split, rows in vm.items():
        ds = Dataset.from_dict({"segment_id": [r[0] for r in rows], "audio": [{"bytes": wav_bytes(4, sr=44100, channels=2), "path": "audio.wav"} for _ in rows],  # như bản thật: 44.1 kHz
                                "duration_seconds": [4] * len(rows), "segment_text": [r[1] for r in rows],
                                "cs_terms_list": [r[2] for r in rows], "cs_terms_count": [len(r[2].split(";")) for r in rows],
                                "topic": ["Medical Sciences"] * len(rows),
                                "original_video_link": [f"https://www.youtube.com/watch?v={r[3]}" for r in rows],
                                "original_video_title": ["t"] * len(rows), "start_time": ["00:12"] * len(rows),
                                "end_time": ["00:21"] * len(rows)}).cast_column("audio", Audio())
        ds.to_parquet(str(dl / f"vimedcss_{split}.parquet"))
    # --- SEA: nhiều cặp ngôn ngữ, chỉ lấy en-vi
    sea = [("en-vi", "vidAAAAAAA1", 0.0, "tuần này team mình chạy deadline cho kịp release"),
           ("en-vi", "vidAAAAAAA1", 9.5, "anh review giúp em cái pull request nhé"),
           ("en-vi", "vidBBBBBBB2", 3.0, "mình order trên shopee có voucher freeship"),
           ("en-vi", "vidCCCCCCC3", 1.0, "bạn nào có feedback thì comment bên dưới"),
           ("en-vi", "y578jWB56QE", 5.0, "bổ sung lipid và protein mỗi ngày"),
           ("en-zh", "vidZZZZZZZ9", 0.0, "今天我们 meeting 一下"),
           ("en-vi", "vidDDDDDDD4", 2.0, "hôm nay trời đẹp quá")]
    ds = Dataset.from_dict({"audio": [{"bytes": wav_bytes(5, "FLAC"), "path": None} for _ in sea], "text": [s[3] for s in sea],
                            "language_pair": [s[0] for s in sea], "switch_level": ["insertion"] * len(sea),
                            "n_switches": [2] * len(sea), "duration": [5.0] * len(sea),
                            "spans": [[{"lang": "vi", "text": "x", "start": 0.0, "end": 1.0}, {"lang": "en", "text": "deadline", "start": 1.0, "end": 2.0}]] * len(sea),
                            "youtube_id": [s[1] for s in sea], "start_sec": [s[2] for s in sea],
                            "end_sec": [s[2] + 5 for s in sea]}).cast_column("audio", Audio())
    ds.to_parquet(str(dl / "sea.parquet"))
    # --- VIVOS, Common Voice, VietSuperSpeech (thư mục)
    for part, utts in (("test", [("VIVOSSPK01_R001", "KHÁCH SẠN"), ("VIVOSSPK02_R003", "TÔI ĐI CHỢ")]), ("train", [("VIVOSSPK11_R001", "XIN CHÀO CÁC BẠN")])):
        d = dl / "vivos" / part
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompts.txt").write_text("\n".join(f"{u} {t}" for u, t in utts) + "\n", encoding="utf-8")
        for u, _ in utts:
            p = d / "waves" / u.split("_")[0]
            p.mkdir(parents=True, exist_ok=True)
            (p / f"{u}.wav").write_bytes(wav_bytes(2))
    cv = dl / "cv"
    (cv / "clips").mkdir(parents=True, exist_ok=True)
    for part, rows in (("test", [("spkaaa", "c1.wav", "Hôm nay tôi đi học."), ("spkbbb", "c2.wav", "Mở Facebook lên xem nào.")]), ("train", [("spkccc", "c3.wav", "Trời mưa rồi.")])):
        with open(cv / f"{part}.tsv", "w", encoding="utf-8") as f:
            f.write("client_id\tpath\tsentence\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in rows))
        for _, b, _ in rows:
            (cv / "clips" / b).write_bytes(wav_bytes(2))
    vss = dl / "vss"
    (vss / "audio").mkdir(parents=True, exist_ok=True)
    items = [{"audio": "audio/a.wav", "text": "À CHO TÔI MẤY NGÀN LIKE ĐÓ", "duration": 11.0, "source": "ep1.mp4"},
             {"audio": "audio/b.wav", "text": "HÔM NAY CHÚNG TA NÓI VỀ CHUYỆN HỌC HÀNH", "duration": 12.0, "source": "ep2.mp4"}]
    for it in items:
        (vss / it["audio"]).write_bytes(wav_bytes(3))
    (vss / "train.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    cat = G / "configs/dataset/labeling/catalog.example.tsv"
    raw = T / "raw"
    run(G / "src/gipbench/dataset/ingest/ggroup.py", "--names", "tech", "finance", "pubadmin", "--out", raw / "ggroup", "--catalog", cat, "--parquet", dl / "ggroup_{name}.parquet", "--vi-words")
    loan = T / "vi_loanwords_test.txt"  # danh sách cố định cho test, không phụ thuộc configs/
    loan.write_text("word\tkind\tstatus\tsource\naxit\tvi_spelling\táp dụng\ttest\nlipid\tsame_surface\tchờ duyệt\ttest\n"
                    "vitamin\tsame_surface\tchờ duyệt\ttest\n", encoding="utf-8")
    run(G / "src/gipbench/dataset/ingest/vimedcss.py", "--out", raw / "vimedcss", "--catalog", cat, "--parquet", dl / "vimedcss_{split}.parquet",
        "--vi-words", loan)
    run(G / "src/gipbench/dataset/ingest/sea.py", "--out", raw / "sea", "--catalog", cat, "--parquet", dl / "sea.parquet", "--bench-hours", "0.002", "--vi-words")
    run(G / "src/gipbench/dataset/ingest/local.py", "vivos", "--root", dl / "vivos", "--out", raw / "vivos", "--vi-words")
    run(G / "src/gipbench/dataset/ingest/local.py", "commonvoice", "--root", cv, "--out", raw / "cv", "--vi-words")
    run(G / "src/gipbench/dataset/ingest/local.py", "vietsuperspeech", "--root", vss, "--out", raw / "vss", "--vi-words")

    def L(p):
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    b1 = L(raw / "ggroup/b1_candidates.jsonl")
    assert len(b1) == 5 and any("<en>connect metamask</en>" in r["text_tagged"] for r in b1), b1
    assert all(r["trusted_text"] is True for r in b1 + L(raw / "ggroup/b5_ggroup.jsonl"))  # G-Group: text do người gõ
    assert len(L(raw / "ggroup/b5_ggroup.jsonl")) == 1
    b3 = L(raw / "vimedcss/b3_candidates.jsonl")
    assert "<en>leukotriene</en>" in b3[0]["text_tagged"] and "term_not_found" in b3[1]["auto_tag"]["flags"]
    catv = list(csv.DictReader(open(raw / "vimedcss/catalog_vimedcss.tsv", encoding="utf-8"), delimiter="\t"))
    assert {r["term"]: r["reserved_b3"] for r in catv}["leukotriene"] == "1"
    # Phase 0b T3: hai nhãn — Spec (bỏ tag từ mượn) và theo dataset
    trv = {r["id"]: r for r in L(raw / "vimedcss/train_vimedcss.jsonl")}["vmc_Med_CS-1-1"]
    assert "<en>lipid</en>" in trv["text_tagged_orig"] and "<en>" not in trv["text_tagged"], trv
    assert "loanword_untagged" in trv["auto_tag"]["flags"]
    dv = L(raw / "vimedcss/dev_vimedcss.jsonl")[0]
    assert dv["text_tagged_orig"] == "<en>methionin</en> là một axit <en>amin</en> thiết yếu.", dv["text_tagged_orig"]
    assert "<en>axit" not in dv["text_tagged"] and "<en>methionin</en>" in dv["text_tagged"], dv["text_tagged"]
    assert "lipid" not in {r["term"] for r in catv}  # từ mượn không vào catalog EN
    # Phase 0b T5: audio 44.1 kHz (stereo) -> WAV 16 kHz mono, duration lấy từ audio thật
    ai = sf.info(str(raw / "vimedcss" / trv["audio"]))
    assert (ai.samplerate, ai.channels, ai.format) == (16000, 1, "WAV") and abs(trv["duration"] - 4.0) < 0.01, ai
    assert "text_tagged_orig" in next(csv.reader(open(raw / "vimedcss/b3_candidates_review.csv", encoding="utf-8-sig")))
    seab, seat = L(raw / "sea/b1_candidates_sea.jsonl"), L(raw / "sea/train_sea.jsonl")
    assert not ({r["source"] for r in seab} & {r["source"] for r in seat}) and len(seab) + len(seat) == 5
    assert len(L(raw / "vivos/b5_vivos.jsonl")) == 2 and len(L(raw / "cv/b5_cv.jsonl")) == 1
    assert len(L(raw / "vss/train_vss.jsonl")) == 1

    # --- người duyệt: 1 ok, 1 fix, 1 drop trong phiếu G-Group; ok hết B3 trừ câu không tìm thấy thuật ngữ
    rp = raw / "ggroup/b1_candidates_review.csv"
    rows = list(csv.DictReader(open(rp, encoding="utf-8-sig")))
    assert "reading_note" in rows[0], list(rows[0])
    rows[0]["decision"], rows[0]["viet_levels"], rows[0]["region"] = "fix", "mid", "south"
    rows[0]["reading_note"] = "google: gu gồ -> EN"
    rows[0]["text_tagged_fixed"] = "đây là <en>google search</en> <en>google search</en> rất tận dụng"
    rows[1]["decision"] = "ok"
    rows[2]["decision"] = "drop"
    with open(rp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rp3 = raw / "vimedcss/b3_candidates_review.csv"
    rows = list(csv.DictReader(open(rp3, encoding="utf-8-sig")))
    rows[0]["decision"] = "ok"
    rows[1]["decision"] = "drop"
    with open(rp3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", raw / "ggroup/b1_candidates.jsonl", raw / "ggroup/b5_ggroup.jsonl",
              raw / "vimedcss/b3_candidates.jsonl", raw / "sea/b1_candidates_sea.jsonl", raw / "vivos/b5_vivos.jsonl", raw / "cv/b5_cv.jsonl",
              "--reviews", rp, rp3, "--catalogs", cat, raw / "vimedcss/catalog_vimedcss.tsv",
              "--out", T / "data/bench_v0.jsonl", "--catalog-out", T / "configs/catalog_v0.tsv")
    bench = L(T / "data/bench_v0.jsonl")
    assert sorted(r["split"] for r in bench) == ["B1", "B1", "B3", "B5", "B5", "B5", "B5"], [r["split"] for r in bench]
    fixed = next(r for r in bench if r["review"] == "fix")
    assert fixed["en_meta"] == [{"viet_level": "mid"}] and fixed["region"] == "south"
    assert fixed["reading_note"] == "google: gu gồ -> EN"  # Phase 0b: cột ghi chú cách đọc đi vào bench
    # --- Phase 0d T8: src/gipbench/dataset/label/review_bench.html — lõi JS (parseCSV / applyReview / toCSV) chạy bằng Node (chương trình
    #     đưa qua stdin: `node -`); CSV xuất ra phải được build_bench_v0.py đọc đúng
    node = shutil.which("node") or shutil.which("node.exe")
    if not node:
        print("BỎ QUA test review_bench.html: không tìm thấy node")
    else:
        core = re.search(r'<script id="core">(.*?)</script>', (G / "src/gipbench/dataset/label/review_bench.html").read_text(encoding="utf-8"),
                         re.S).group(1)
        src_rows = list(csv.DictReader(open(rp, encoding="utf-8-sig")))
        ids = [r["id"] for r in src_rows[:3]]
        fix_text = 'đây là, "thử" <en>google</en> nhé'  # có dấu phẩy và ngoặc kép: kiểm tra quote CSV
        revs = {r["id"]: {"decision": ""} for r in src_rows[3:]}
        revs.update({ids[0]: {"decision": "ok", "region": "north"}, ids[1]: {"decision": "drop"},
                     ids[2]: {"decision": "fix", "fixed": fix_text, "viet_levels": "strong", "gender": "f"}})
        drv = (core + "\nconst {header, rows} = parseCSV(" + json.dumps(open(rp, encoding="utf-8").read()) + ");\n"
               "const R = " + json.dumps(revs, ensure_ascii=False) + ";\n"
               "process.stdout.write(toCSV(ensureCols(header), rows.map(r => applyReview(r, R[r.id]))));\n"
               "console.error(tagError('<en>a</en> <en>b') + '|' + tagError('<en>a</en>') + '|' + tagHtml('x <en>a&b</en>'));\n")
        p = subprocess.run([node, "-"], input=drv.encode("utf-8"), capture_output=True)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        assert p.stderr.decode("utf-8").strip() == 'số <en> (2) khác số </en> (1)||x <b class="en">a&amp;b</b>', p.stderr
        out_csv = T / "review_html_out.csv"
        out_csv.write_bytes(p.stdout)
        out_rows = list(csv.DictReader(open(out_csv, encoding="utf-8-sig")))
        assert list(out_rows[0]) == list(src_rows[0]) and len(out_rows) == len(src_rows)  # giữ nguyên cột, số dòng
        assert out_rows[2]["text_tagged_fixed"] == fix_text and out_rows[1]["decision"] == "drop"
        run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", raw / "ggroup/b1_candidates.jsonl", "--reviews", out_csv,
            "--catalogs", cat, "--out", T / "data/bench_html.jsonl", "--catalog-out", T / "configs/catalog_html.tsv")
        bh = {r["id"]: r for r in L(T / "data/bench_html.jsonl")}
        assert set(bh) == {ids[0], ids[2]}, sorted(bh)
        assert bh[ids[2]]["text_tagged"] == fix_text and bh[ids[2]]["en_meta"] == [{"viet_level": "strong"}]
        assert bh[ids[2]]["gender"] == "f" and bh[ids[0]]["region"] == "north" and bh[ids[0]]["review"] == "ok"
    # --- bench đi tiếp được vào validate (phải bắt video y578jWB56QE có ở cả train ViMedCSS và SEA nếu rơi vào bench) và eda
    run(G / "src/gipbench/dataset/release/validate.py", "--bench", T / "data/bench_v0.jsonl", "--catalog", T / "configs/catalog_v0.tsv", "--check-audio",
        "--train", raw / "vimedcss/train_vimedcss.jsonl", raw / "sea/train_sea.jsonl", raw / "vss/train_vss.jsonl", ok=(0, 1))
    run(G / "src/gipbench/bench/report/eda.py", "--bench", T / "data/bench_v0.jsonl", "--catalog", T / "configs/catalog_v0.tsv", "--out", T / "eda")
    run(G / "src/gipbench/model/features/to_lhotse.py", "--manifest", raw / "vimedcss/train_vimedcss.jsonl", raw / "sea/train_sea.jsonl", "--name", "cs_train",
        "--bench", T / "data/bench_v0.jsonl", "--catalog", T / "configs/catalog_v0.tsv", "--out", T / "lhotse", "--min-dur", "1")
    # --- Phase 0: đọc N dòng đầu từ parquet nhiều shard (đường --limit khi tải từ HF), filesystem cục bộ thay HF
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gipbench.dataset.ingest.common import iter_hf_parquet, load_vi_words
    shard_dir = T / "hf/org/ds/data"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for k in range(2):
        pq.write_table(pa.table({"id": [f"{k}-{j}" for j in range(5)], "text": ["x"] * 5}),
                       str(shard_dir / f"test-0000{k}-of-00002.parquet"), row_group_size=2)
    lfs = fsspec.filesystem("file")
    got = [r["id"] for r in iter_hf_parquet("org/ds", "test", 7, batch_size=2, fs=lfs, root=str(T / "hf") + "/")]
    assert got == ["0-0", "0-1", "0-2", "0-3", "0-4", "1-0", "1-1"], got
    assert iter_hf_parquet("org/ds", "hard", 3, fs=lfs, root=str(T / "hf") + "/") is None
    # --- VietSuperSpeech: --limit áp cho từng phần train/dev
    (vss / "dev.json").write_text(json.dumps([{"audio": "audio/b.wav", "text": "HỌC HÀNH CHĂM CHỈ", "duration": 3.0,
                                               "source": "ep3.mp4"}], ensure_ascii=False), encoding="utf-8")
    run(G / "src/gipbench/dataset/ingest/local.py", "vietsuperspeech", "--root", vss, "--out", raw / "vss_l1", "--limit", "1")
    assert len(L(raw / "vss_l1/dev_vss.jsonl")) == 1  # trước đây dev không được đọc khi có --limit
    # --- src/gipbench/dataset/report/converter_check.py
    run(G / "src/gipbench/dataset/report/converter_check.py", "--manifests", raw / "ggroup/b1_candidates.jsonl", raw / "vss/train_vss.jsonl",
        "--samples", "2", "--out", T / "reports/history/converter_check.md")
    ck = (T / "reports/history/converter_check.md").read_text(encoding="utf-8")
    assert "| 5 |" in ck and "heuristic" in ck and "2 câu mẫu" in ck, ck
    # các lần chạy converter ở trên dùng --vi-words rỗng để không phụ thuộc danh sách đang chờ duyệt;
    # danh sách mặc định trong configs/ vẫn phải đọc được
    assert {"bidv", "axit"} <= load_vi_words(), "configs/dataset/labeling/vi_abbrev.txt / vi_loanwords.txt thiếu mục cơ bản"
    # --- src/gipbench/dataset/label/eval_autotag.py: số liệu tính tay trên 3 câu vàng
    gold = T / "gold.jsonl"
    gold.write_text("\n".join(json.dumps({"id": i, "gold": g}, ensure_ascii=False) for i, g in [
        ("a", "mình [[deploy]] lên [[server]] rồi"),          # auto_tag: deploy + server (catalog) -> 2 TP
        ("b", "chấm [[gov]] chấm {{vn}} nhé"),                 # auto_tag gắn cả vn -> 1 FP (nằm trong viết tắt VN)
        ("c", "học ((ai ten)) với cái [[kit]] này")]) + "\n", encoding="utf-8")  # kit: -t không dấu -> không phải tiếng Việt
    # --vi-words rỗng: số liệu tính tay giả định KHÔNG có danh sách vi_abbrev (configs/dataset/labeling/vi_abbrev.txt có vn từ Phase 0d)
    run(G / "src/gipbench/dataset/label/eval_autotag.py", "--gold", gold, "--catalog", cat, "--out", T / "reports/eval", "--spotcheck", "2",
        "--vi-words", "--term-decisions", "")  # không phụ thuộc quyết định thuật ngữ thật (vd người dùng chốt gov = vi)
    ev = json.loads((T / "reports/eval.json").read_text(encoding="utf-8"))
    s = ev["subset_en"]
    assert (s["gold_spans"], s["tp"], ev["phonetic_spans"], ev["vi_abbr_spans"]) == (4, 4, 1, 1), s
    assert abs(s["precision_neutral"] - 1.0) < 1e-9 and abs(s["precision"] - 0.8) < 1e-9, s
    # --- T0.2: luật chính tả âm tiết trong is_vietnamese_token
    from gipbench.dataset.ingest.common import auto_tag, is_vietnamese_token
    from gipbench.core.textnorm import Catalog
    cat_obj = Catalog.load(str(cat))
    not_vi = ["chip", "kit", "mac", "net", "hot", "tech", "op", "chat", "top", "chain", "main", "sing", "boat", "tuan"]
    vi = ["con", "cho", "an", "hay", "nhanh", "sinh", "quan", "huynh", "keng", "toan", "khuya", "ngoai", "giu", "nghe",
          "mình", "tiếp", "ăn", "ba"]
    assert not [w for w in not_vi if is_vietnamese_token(w)], [w for w in not_vi if is_vietnamese_token(w)]
    assert all(is_vietnamese_token(w) for w in vi), [w for w in vi if not is_vietnamese_token(w)]
    assert auto_tag("căn cước công dân gắn chip", term_dec={})["spans"] == ["chip"]  # trước đây lọt vào B5
    # --- Phase 0b, quy ước theo cách đọc: viết tắt ngắn đoán bằng heuristic -> cờ cho người duyệt
    assert "abbr_reading:crm" in auto_tag("anh mở cái crm lên nhé", term_dec={})["flags"]
    assert not any(f.startswith("abbr_reading") for f in auto_tag("anh mở cái kpi lên", cat_obj, term_dec={})["flags"])  # có trong catalog
    # --- Phase 0b T2: vi_abbrev — nguồn giả -> src/gipbench/dataset/label/wordlists.py -> auto_tag không gắn <en>
    import gzip
    lex = T / "lex"
    lex.mkdir(parents=True, exist_ok=True)
    tick = [{"symbol": s, "board": b, "type": "STOCK", "organShortName": n} for s, b, n in
            [("FPT", "HSX", "FPT Corp"), ("PLX", "HSX", "Petrolimex"), ("GAS", "HSX", "PV Gas"), ("SHS", "HNX", "CK SHS"),
             ("QNS", "UPCOM", "Đường Quảng Ngãi"), ("API", "HNX", "Đầu tư Châu Á")]] + \
        [{"symbol": "E1VFVN30", "board": "HSX", "type": "ETF"}]
    (lex / "vietcap_symbols.json").write_bytes(gzip.compress(json.dumps(tick).encode()))
    (lex / "wiki_Danh_sách_ngân_hàng_tại_Việt_Nam.txt").write_text(
        "== Ngân hàng ==\n{| class=\"wikitable\"\n|-\n|1\n|[[Ngân hàng X|Đầu tư và Phát triển Việt Nam]]\n|BIDV||1 ||x\n"
        "|-\n|2\n|[[Bắc Á]]\n|Bac A Bank, BAB||2 ||x\n|-\n|3\n|[[Kỹ Thương]]\n|Techcombank||3||x\n|}\n"
        "==Ngân hàng 100% vốn nước ngoài==\n{|\n|-\n|1\n|[[HSBC]]\n|HSBC||9||x\n|}\n", encoding="utf-8")
    (lex / "wiki_tinh.txt").write_text("{|\n|-\n|1\n|'''[[Hà Nội]]'''\n|-\n|2\n|[[TP. Hồ Chí Minh]]\n|-\n|3\n|[[Phú Thọ]]\n"
                                       "|-\n|4\n|[[Đà Nẵng]]\n|}\n", encoding="utf-8")
    (lex / "wikt_viettat_pages.json").write_text(json.dumps({
        "THPT": "# {{abbreviation of|vi|trung học phổ thông}}.", "ROM": "# {{abbr of|vi|Read Only Memory}}",
        "AIDS": "# [[hội chứng]] [[suy giảm]]\n# {{abbreviation of|en|acquired immune deficiency syndrome}}",
        "STT": "#[[số|Số]] [[thứ tự]]."}, ensure_ascii=False), encoding="utf-8")
    (lex / "google-10000-english-no-swears.txt").write_text("the\ngas\npt\n", encoding="utf-8")
    run(G / "src/gipbench/dataset/label/wordlists.py", "--src", lex, "--catalog", cat, "--out-abbrev", T / "vi_abbrev.txt",
        "--project-abbrev", "")
    ab_lines = (T / "vi_abbrev.txt").read_text(encoding="utf-8").splitlines()
    from gipbench.dataset.ingest.common import load_vi_words
    ab = load_vi_words([T / "vi_abbrev.txt"])
    assert ab == {"fpt", "plx", "shs", "bidv", "bab", "hn", "hcm", "thpt", "stt"}, sorted(ab)
    assert any(l.startswith("#-gas\t") and "google-10000" in l for l in ab_lines)  # trùng từ Anh phổ biến -> loại
    assert any(l.startswith("#-api\t") and "catalog" in l for l in ab_lines)       # trùng catalog -> loại
    assert any(l.startswith("#-pt\t") for l in ab_lines)                           # chữ cái đầu Phú Thọ, trùng 'pt'
    tg = auto_tag("mua fpt với bidv rồi gửi api", cat_obj, vi_words=ab, term_dec={})
    assert tg["spans"] == ["api"] and "vi_word:fpt" in tg["flags"], tg               # catalog vẫn thắng danh sách
    assert auto_tag("mua fpt", cat_obj, vi_words=set(), term_dec={})["spans"] == ["fpt"]
    # Phase 0d T7: quyết định tay (--project-abbrev): add thắng google-10000 (pt), exclude (shs); catalog vẫn thắng (api)
    proj = T / "vi_abbrev_project.tsv"
    proj.write_text("# chú thích\nabbr\taction\tgroup\tfull\tsource\n"
                    "vn\tadd\tviết tắt (quy ước dự án)\tViệt Nam\tquy ước dự án, đọc kiểu Việt\n"
                    "pt\tadd\tviết tắt (quy ước dự án)\tphú thọ\tquy ước dự án, đọc kiểu Việt; bật lại\n"
                    "api\tadd\tviết tắt (quy ước dự án)\tx\tquy ước dự án, đọc kiểu Việt\n"
                    "shs\texclude\tmã chứng khoán\t\tbỏ theo quyết định dự án\n", encoding="utf-8")
    out_ab = run(G / "src/gipbench/dataset/label/wordlists.py", "--src", lex, "--catalog", cat, "--out-abbrev", T / "vi_abbrev_p.txt",
                 "--project-abbrev", proj)
    abp = load_vi_words([T / "vi_abbrev_p.txt"])
    assert abp == (ab | {"vn", "pt"}) - {"shs"}, sorted(abp)
    abp_lines = (T / "vi_abbrev_p.txt").read_text(encoding="utf-8").splitlines()
    assert any(l.startswith("vn\t") and l.endswith("quy ước dự án, đọc kiểu Việt") for l in abp_lines)
    assert any(l.startswith("#-shs\t") and "loại tay: bỏ theo quyết định dự án" in l for l in abp_lines)
    assert any(l.startswith("#-api\t") and "catalog" in l for l in abp_lines) and "luật catalog: 1 mục" in out_ab, out_ab
    # §12.3: mã CHỈ thuộc nhóm mã chứng khoán, gặp >= --nonfin-min lần trong text không phải tài chính -> loại
    (lex / "nonfin.txt").write_text("".join(f"S{i} GAS PLX VN FPT\n" for i in range(3)) + "S9 PLX\n", encoding="utf-8")
    (lex / "nonfin.json").write_text(json.dumps([{"text": "plx shs"}]), encoding="utf-8")
    out_nf = run(G / "src/gipbench/dataset/label/wordlists.py", "--src", lex, "--catalog", cat, "--out-abbrev", T / "vi_abbrev_nf.txt",
                 "--project-abbrev", proj, "--nonfin-text", lex / "nonfin.txt", lex / "nonfin.json", "--nonfin-min", "3")
    abn = load_vi_words([T / "vi_abbrev_nf.txt"])
    # plx 5 lần, fpt 3 lần -> loại; vn (thêm tay) và shs (đã exclude) không xét; gas đã bị loại trước đó
    assert abn == abp - {"plx", "fpt"}, sorted(abn)
    assert "2 mã bị loại: plx (5), fpt (3)" in out_nf, out_nf
    assert any(l.startswith("#-plx\t") and "gặp 5 lần trong text không phải tài chính" in l
               for l in (T / "vi_abbrev_nf.txt").read_text(encoding="utf-8").splitlines())
    # --- Phase 0b T3: vi_loanwords — vi_spelling + same_surface (mọi bộ, §12) / --exclude / google-10000
    (lex / "Viet74K.txt").write_text("axit\ncanxi\nvitamin\niot\nmega\ncon\nHà Nội\na-xít\n", encoding="utf-8")
    (lex / "words_alpha.txt").write_text("vitamin\nlipid\n", encoding="utf-8")
    (lex / "google-10000-english-no-swears.txt").write_text("the\ngas\npt\nmega\n", encoding="utf-8")
    vm_meta = lex / "vmc"
    vm_meta.mkdir(exist_ok=True)
    with open(vm_meta / "train_set.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["segment_id", "segment_text", "cs_terms_list"])
        cw.writerow(["a", "uống vitamin và lipid mỗi ngày", "vitamin; lipid"])
    run(G / "src/gipbench/dataset/label/wordlists.py", "--src", lex, "--vimedcss-meta", vm_meta, "--out-loan", T / "vi_loan.txt",
        "--exclude", "iot:IoT trong tech")
    lo_lines = (T / "vi_loan.txt").read_text(encoding="utf-8").splitlines()
    # same_surface = không EN ở MỌI bộ (decisions §12): mặc định nạp cả nhóm này
    assert load_vi_words([T / "vi_loan.txt"]) == {"axit", "canxi", "vitamin"}, load_vi_words([T / "vi_loan.txt"])
    assert load_vi_words([T / "vi_loan.txt"], same_surface=False) == {"axit", "canxi"}  # lipid không có trong Viet74K
    assert any(l.startswith("#-iot\t") and "IoT trong tech" in l for l in lo_lines)
    assert any(l.startswith("#-mega\t") and "google-10000" in l for l in lo_lines)  # trùng từ Anh phổ biến -> loại
    # --- Phase 0b T4: VSS -> vss_cs_candidates (chỉ phần train, chặn rò rỉ với bench B1–B3 và reserved_b3)
    vss2 = dl / "vss2"
    (vss2 / "audio").mkdir(parents=True, exist_ok=True)
    tr = [("t0", "HÔM NAY TRỜI ĐẸP QUÁ", "ep1.mp4"),                       # không có từ Anh -> train_vss
          ("t1", "MÌNH MỞ FACEBOOK LÊN XEM NHÉ", "ep1.mp4"),               # -> vss_cs
          ("t2", "ANH REVIEW GIÚP EM CÁI PULL REQUEST NHÉ", "ep2.mp4"),    # trùng text câu bench B1 -> loại
          ("t3", "BÊN ĐÓ DÙNG KUBERNETES CHO CẢ HỆ THỐNG", "ep2.mp4"),     # reserved_b3 -> loại
          ("t4", "CHẠY DEADLINE XONG THÌ ĐI ĂN", "ep3.mp4")]               # -> vss_cs
    dv = [("d0", "CÁI MEETING HÔM QUA HƠI DÀI", "ep9.mp4")]                # có từ Anh nhưng ở phần dev -> bỏ
    for part, rows_ in (("train", tr), ("dev", dv)):
        js = []
        for uid, text, src in rows_:
            (vss2 / "audio" / f"{uid}.wav").write_bytes(wav_bytes(2))
            js.append({"audio": f"audio/{uid}.wav", "text": text, "duration": 2.0, "source": src})
        (vss2 / f"{part}.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    bench_proxy = T / "bench_proxy.jsonl"
    bench_proxy.write_text(json.dumps({"id": "b1x", "split": "B1", "audio": "x.wav", "duration": 2, "speaker": "yt_q",
                                       "source": "youtube:q", "text_tagged": "anh <en>review</en> giúp em cái "
                                       "<en>pull request</en> nhé"}, ensure_ascii=False) + "\n", encoding="utf-8")
    out_v = run(G / "src/gipbench/dataset/ingest/local.py", "vietsuperspeech", "--root", vss2, "--out", raw / "vss2", "--catalog", cat,
                "--bench", bench_proxy, "--vi-words")
    cs = L(raw / "vss2/vss_cs_candidates.jsonl")
    assert sorted(r["id"] for r in cs) == ["vss_0000001", "vss_0000004"], [r["id"] for r in cs]
    assert all(r["dataset"] == "vss_cs" and r["train_only"] and r["split"] == "train" and "<en>" in r["text_tagged"] for r in cs)
    assert len(L(raw / "vss2/train_vss.jsonl")) == 1 and "vss_cs bỏ: text trùng câu bench: 1" in out_v
    assert "vss_cs bỏ: có thuật ngữ reserved_b3: 1" in out_v and "vss_cs bỏ: câu có từ Anh ở phần dev: 1" in out_v
    assert (raw / "vss2/vss_cs_terms.tsv").read_text(encoding="utf-8").startswith("term\tcount\n")
    # --- Phase 0b T4: lọc đồng thuận (src/gipbench/dataset/pseudolabel/agreement.py + src/gipbench/dataset/pseudolabel/agreement_filter.py)
    from gipbench.dataset.pseudolabel.agreement import agree
    lab = "mở <en>facebook</en> rồi <en>check</en> mail"
    hy = ["MỞ FB RỒI CHECK MAIL", "mở phây búc rồi check mail"]  # fb = biến thể catalog của facebook
    assert agree(lab, hy, cat_obj, 1) == {"keep": True, "spans": ["facebook", "check"], "confirmed": [1, 2]}
    assert not agree(lab, hy, cat_obj, 2)["keep"] and agree(lab, hy, cat_obj, 2, min_span_frac=0.5)["keep"]
    assert not agree("hôm nay trời đẹp", hy, cat_obj)["keep"]  # không có span -> không có gì để xác nhận
    ha, hb = T / "hyp_a.jsonl", T / "hyp_b.jsonl"
    ha.write_text('{"id": "vss_0000001", "hyp": "MÌNH MỞ FACEBOOK LÊN XEM NHÉ"}\n{"id": "vss_0000004", "hyp": "CHẠY ĐÉT LAI XONG"}\n',
                  encoding="utf-8")
    hb.write_text('{"id": "vss_0000001", "hyp": "mình mở phây búc lên"}\n', encoding="utf-8")
    run(G / "src/gipbench/dataset/pseudolabel/agreement_filter.py", "--manifest", raw / "vss2/vss_cs_candidates.jsonl", "--hyps", ha, hb,
        "--catalog", cat, "--out", T / "vss_cs_agreed.jsonl")
    ag = L(T / "vss_cs_agreed.jsonl")
    assert [r["id"] for r in ag] == ["vss_0000001"] and ag[0]["agreement"]["confirmed"] == [1], ag
    # --- Phase 0c: gán lại TỪNG span (lib.agreement.relabel_spans, agreement_filter --relabel-spans)
    from gipbench.dataset.pseudolabel.agreement import relabel_spans
    lab = "với cái <en>netword</en> của dương thì mọi người sẽ đi tìm dương cho nên là dương cũng sẽ mau được tìm thấy"
    h1 = "Với cái network của Dương, thì mọi người sẽ đi tìm Dương. Cho nên là Dương cũng sẽ mau được tìm thấy."
    h2 = "với cái network của dương thì mọi người sẽ đi tìm dương cho nên là dương cũng sẽ mau được tìm thấy"
    rl = relabel_spans(lab, [h1, h2], cat_obj)
    assert rl["changed"] == [(0, "netword", "network")] and "<en>network</en> của dương" in rl["text_tagged"], rl
    assert rl["vi_wer"] == [0.0, 0.0]
    assert relabel_spans(lab, [h1, h2.replace("network", "nét")], cat_obj)["reason"] == "các ASR không cho cùng chữ"
    noisy = "với cái network của dương thì ai cũng đi kiếm dương nên là dương mau được thấy"  # lệch nhiều từ tiếng Việt
    assert relabel_spans(lab, [h1, noisy], cat_obj)["reason"] == "WER tiếng Việt vượt ngưỡng"
    assert relabel_spans(lab, [h1], cat_obj)["reason"] == "thiếu hyp" and relabel_spans(lab, [h1, ""], cat_obj)["reason"] == "thiếu hyp"
    rl2 = relabel_spans("mở <en>facebook</en> rồi <en>chếch</en> mail", ["mở facebook rồi check mail"] * 2, cat_obj)
    assert rl2["changed"] == [(1, "chếch", "check")] and rl2["text_tagged"] == "mở <en>facebook</en> rồi <en>check</en> mail", rl2
    # Phase 0d: span mới không có chữ có dấu tiếng Việt, không dài hơn span cũ quá 1 token
    from gipbench.dataset.pseudolabel.agreement import span_rule_violation
    assert span_rule_violation(("rocket",), ("rap", "việt")) == "span mới có chữ có dấu tiếng Việt"
    assert span_rule_violation(("ad",), ("a", "b", "c")).startswith("span mới dài hơn span cũ")
    assert span_rule_violation(("concep",), ("concept",)) == "" and span_rule_violation(("post",), ("host", "talk")) == ""
    rl3 = relabel_spans("nghe <en>rocket</en> hay lắm", ["nghe rap việt hay lắm"] * 2, cat_obj)
    assert rl3["changed"] == [] and rl3["rejected"] == [(0, "rocket", "rap việt", "span mới có chữ có dấu tiếng Việt")], rl3
    rl4 = relabel_spans("cái <en>sho</en> hay", ["cái host talk show hay"] * 2, cat_obj)
    assert rl4["changed"] == [] and rl4["rejected"][0][3].startswith("span mới dài hơn"), rl4
    rel_m = T / "relabel_in.jsonl"
    base = {"split": "train", "audio": "audio/x.wav", "duration": 3.0, "speaker": "vss_ep7", "source": "vss:ep7", "dataset": "vss_cs"}
    rel_rows = [dict(base, id="r1", text_tagged=lab),
                dict(base, id="r2", text_tagged="bên đó dùng <en>cu bơ nét</en> cho cả hệ thống của công ty mình")]
    rel_m.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rel_rows), encoding="utf-8")
    k8s = "bên đó dùng kubernetes cho cả hệ thống của công ty mình"  # kubernetes = reserved_b3 trong catalog mẫu
    for name, hy_ in (("rq", [h1, k8s]), ("rw", [h2, k8s])):
        (T / f"{name}.jsonl").write_text("".join(json.dumps({"id": i, "hyp": h}, ensure_ascii=False) + "\n"
                                                 for i, h in zip(["r1", "r2"], hy_)), encoding="utf-8")
    run(G / "src/gipbench/dataset/pseudolabel/agreement_filter.py", "--manifest", rel_m, "--hyps", T / "rq.jsonl", T / "rw.jsonl", "--catalog", cat,
        "--out", T / "rel_filter.jsonl", "--write-unconfirmed-ids", T / "rel_unconf.txt")
    assert L(T / "rel_filter.jsonl") == [] and (T / "rel_unconf.txt").read_text(encoding="utf-8").split() == ["r1", "r2"]
    out_rl = run(G / "src/gipbench/dataset/pseudolabel/agreement_filter.py", "--manifest", rel_m, "--hyps", T / "rq.jsonl", T / "rw.jsonl", "--catalog", cat,
                 "--relabel-spans", "--listen-csv", T / "listen.csv", "--out", T / "rel_relabel.jsonl")
    rr = L(T / "rel_relabel.jsonl")
    assert [r["id"] for r in rr] == ["r1"] and rr[0]["relabeled"] and rr[0]["text_orig"] == lab, rr  # r2: rò rỉ sau khi gán lại
    assert rr[0]["text"].startswith("với cái network của dương") and "<en>" not in rr[0]["text"]
    assert "rò rỉ sau khi gán lại: có thuật ngữ reserved_b3 | 1" in (T / "rel_relabel.jsonl.report.md").read_text(encoding="utf-8")
    lis = list(csv.DictReader(open(T / "listen.csv", encoding="utf-8-sig")))
    assert len(lis) == 1 and lis[0]["span_cu_moi"] == "netword -> network" and lis[0]["audio_url"].endswith("audio/x.wav"), lis
    assert "gán lại thêm 1 câu" in out_rl
    # --- Phase 0d T3: ngưỡng WER tiếng Việt cho nhánh chỉ lọc (min theo các ASR có hyp, mặc định 0.15)
    from gipbench.dataset.pseudolabel.agreement import vi_wer
    lab_ok = lab.replace("netword", "network")
    h3 = "với cái network của dương thì ai sẽ đi tìm dương cho nên là dương cũng sẽ mau được thấy"  # 3 lỗi / 21 từ Việt
    assert vi_wer(lab_ok, h2, cat_obj) == 0.0 and abs(vi_wer(lab_ok, h3, cat_obj) - 3 / 21) < 1e-9
    assert vi_wer(lab_ok, noisy, cat_obj) > 0.15 and vi_wer("<en>hello world</en>", "hello world", cat_obj) == float("inf")
    vm = T / "viwer_in.jsonl"
    vm.write_text(json.dumps(dict(base, id="v1", text_tagged=lab_ok), ensure_ascii=False) + "\n"
                  + json.dumps(dict(base, id="v2", text_tagged=lab_ok), ensure_ascii=False) + "\n", encoding="utf-8")
    (T / "vq.jsonl").write_text(json.dumps({"id": "v1", "hyp": h3}, ensure_ascii=False) + "\n"
                                + json.dumps({"id": "v2", "hyp": noisy}, ensure_ascii=False) + "\n", encoding="utf-8")
    (T / "vw.jsonl").write_text(json.dumps({"id": "v2", "hyp": noisy}, ensure_ascii=False) + "\n", encoding="utf-8")
    run(G / "src/gipbench/dataset/pseudolabel/agreement_filter.py", "--manifest", vm, "--hyps", T / "vq.jsonl", T / "vw.jsonl", "--catalog", cat,
        "--out", T / "viwer_015.jsonl")
    v15 = L(T / "viwer_015.jsonl")
    assert [r["id"] for r in v15] == ["v1"] and abs(v15[0]["agreement"]["vi_wer"] - 0.1429) < 1e-4, v15  # v2: 0.38 > 0.15
    assert "1 câu trong khoảng 0.10–0.15" in (T / "viwer_015.jsonl.report.md").read_text(encoding="utf-8")
    run(G / "src/gipbench/dataset/pseudolabel/agreement_filter.py", "--manifest", vm, "--hyps", T / "vq.jsonl", T / "vw.jsonl", "--catalog", cat,
        "--max-vi-wer", "0.10", "--out", T / "viwer_010.jsonl")
    assert L(T / "viwer_010.jsonl") == []
    # --- Phase 0d T4: cache hyps theo id + hash audio, backfill hyps cũ, rebuild_vss_cs.sh chỉ từ cache
    from gipbench.infer.cache import HashIndex, HypLookup, file_sha256
    c4 = T / "cache4"
    (c4 / "root/audio/p0").mkdir(parents=True, exist_ok=True)
    for k, d in (("a1", 2), ("a2", 3), ("a3", 4), ("a4", 5), ("d9", 6)):
        (c4 / f"root/audio/p0/{k}.wav").write_bytes(wav_bytes(d))
    c_rows = [("a1", "mở <en>zoom</en> lên đi em nhé"), ("a2", "cái <en>zoom</en> hôm nay lag quá"),
              ("a3", "vào <en>zoom</en> họp nha mọi người"), ("a4", "bật <en>zoom</en> lên giúp anh nha")]
    c_m = c4 / "cand.jsonl"
    c_m.write_text("".join(json.dumps(dict(base, id=i, text_tagged=t, audio=str(c4 / f"root/audio/p0/{i}.wav")),
                                      ensure_ascii=False) + "\n" for i, t in c_rows), encoding="utf-8")
    (c4 / "hq").mkdir(exist_ok=True)
    (c4 / "hw").mkdir(exist_ok=True)
    (c4 / "hq/x.jsonl").write_text("".join(json.dumps({"id": i, "hyp": h}, ensure_ascii=False) + "\n" for i, h in [
        ("a1", "mở zoom lên đi em nhé"), ("a2", "cái dum hôm nay lag quá"), ("a4", "bật dum lên giúp anh nha")]),
        encoding="utf-8")  # a3: chưa chạy tầng 1; a2: chưa chạy tầng 2
    (c4 / "hw/x.jsonl").write_text(json.dumps({"id": "a4", "hyp": "bật zoom lên giúp anh nha"}, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    idx = c4 / "sha.tsv"
    out_bf = run(G / "src/gipbench/infer/cache_cli.py", "backfill", "--manifest", c_m, "--hyps", c4 / "hq", c4 / "hw",
                 "--hash-index", idx, "--audio-root", c4 / "root")
    assert "thêm hash: 4" in out_bf and (c4 / "hq/x.jsonl.bak").exists(), out_bf
    hq_rows = L(c4 / "hq/x.jsonl")
    assert all(h["audio_sha256"] == file_sha256(c4 / f"root/audio/p0/{h['id']}.wav") for h in hq_rows)
    assert hq_rows[0]["audio_rel"] == "audio/p0/a1.wav"
    hi = HashIndex(str(idx))
    look = HypLookup([str(c4 / "hq"), str(c4 / "hw")], hi, str(c4 / "root"))
    assert look.get({"id": "a1", "audio": str(c4 / "root/audio/p0/a1.wav")}) == ["mở zoom lên đi em nhé", None]
    hi.put("audio/p0/d9.wav", file_sha256(c4 / "root/audio/p0/d9.wav"))
    # cùng id a1 nhưng audio khác (vd converter đánh lại id): hash không khớp -> không dùng lại hyp cũ
    assert look.get({"id": "a1", "audio": str(c4 / "root/audio/p0/d9.wav")}) == [None, None]
    out_rb = subprocess.run(["bash", str(G / "scripts/dataset/rebuild_vss_cs.sh"), "--bench", str(bench_proxy), "--catalog", str(cat),
                             "--candidates", str(c_m), "--out-dir", str(c4 / "rb"), "--hyps", f"{c4 / 'hq'} {c4 / 'hw'}",
                             "--hash-index", str(idx), "--audio-root", str(c4 / "root")],
                            capture_output=True, text=True, env={**os.environ, "PY": sys.executable})
    assert out_rb.returncode == 0, out_rb.stdout + out_rb.stderr
    assert sorted(r["id"] for r in L(c4 / "rb/vss_cs_clean.jsonl")) == ["a1", "a4"], out_rb.stdout
    rb_md = (c4 / "rb/vss_cs_clean.jsonl.report.md").read_text(encoding="utf-8")
    assert "chưa có hyp tầng 1: 1" in rb_md and "chờ tầng 2 (còn span chưa xác nhận, thiếu hyp tầng sau): 1" in rb_md, rb_md
    assert "Giữ **2 / 2**" in rb_md
    # --- Phase 0c: lọc rẻ trước ASR (src/gipbench/dataset/pseudolabel/vss_cs_prefilter.py)
    pre_rows = [dict(base, id="p0", text_tagged="<en>you build a factory not only one year</en> nha"),  # en_frac 8/9 -> bỏ
                dict(base, id="p1", duration=25.0, text_tagged="dài quá <en>zoom</en> nhé"),           # > 20 s -> bỏ
                dict(base, id="p2", text_tagged="mở <en>zoom</en> lên đi em"),
                dict(base, id="p3", text_tagged="cái <en>zoom</en> hôm nay lag quá"),
                dict(base, id="p4", text_tagged="vào <en>zoom</en> họp nha mọi người"),
                dict(base, id="p5", text_tagged="bảng <en>kanban</en> trên <en>zoom</en> đó"),            # thuật ngữ hiếm -> ưu tiên
                dict(base, id="p6", text_tagged="cụm <en>kubernetes</en> bên mình mới dựng"),             # reserved_b3 -> bỏ
                dict(base, id="p7", text_tagged="cái <en>blorptech</en> này lạ ghê")]                    # thuật ngữ 1 lần -> bỏ (giữ 10% của 1 = 0)
    pre_m = T / "pre_in.jsonl"
    pre_m.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pre_rows), encoding="utf-8")
    out_p = run(G / "src/gipbench/dataset/pseudolabel/vss_cs_prefilter.py", "--manifest", pre_m, "--catalog", cat, "--term-cap", "2",
                "--out", T / "pre_out.jsonl", "--sample", "1", "--sample-out", T / "pre_pilot.jsonl")
    po = L(T / "pre_out.jsonl")
    assert len(po) == 2 and "p5" in [r["id"] for r in po] and all(r["id"] in ("p2", "p3", "p4", "p5") for r in po), po
    assert "--max-en-frac 0.5: 6 câu" in out_p and "rò rỉ B1–B3 + reserved_b3: 5 câu" in out_p, out_p
    assert "bỏ câu chỉ có thuật ngữ 1 lần (giữ 10%): 4 câu" in out_p, out_p
    run(G / "src/gipbench/dataset/pseudolabel/vss_cs_prefilter.py", "--manifest", pre_m, "--catalog", cat, "--term-cap", "0", "--hapax-keep-frac", "1",
        "--out", T / "pre_out_nohapax.jsonl")
    assert "p7" in [r["id"] for r in L(T / "pre_out_nohapax.jsonl")]  # --hapax-keep-frac 1: không lọc
    assert len(L(T / "pre_pilot.jsonl")) == 1 and po[0]["prefilter"]["en_frac"] <= 0.5
    # --- Phase 1 T1.1: src/gipbench/model/recipe/curate.py — mỗi luật hoạt động độc lập (các luật khác tắt)
    cur = T / "curate"
    cur.mkdir(parents=True, exist_ok=True)

    def crow(i, text, spk, dom, **kw):
        return dict(id=i, split="train", audio=f"a/{i}.wav", duration=10.0, text_tagged=text, speaker=spk,
                    source=f"src:{spk}", domain=dom, dataset="toy", **kw)
    ctx = ["chạy", "kịp", "trễ", "dời", "lo", "gấp"]
    c_rows = [crow(f"t{k}", f"{ctx[k]} <en>deadline</en> {ctx[(k + 1) % 6]} nha", s, "medical")
              for k, s in enumerate(["sA", "sA", "sA", "sB", "sB", "sC"])]
    c_rows += [crow("f1", "họp <en>meeting</en> tài chính", "sF", "finance"), crow("f2", "lãi suất tăng", "sF", "finance"),
               crow("m1", "mở <en>meeting</en> đi", "sM", "tech", label_source="pseudo:x"),
               crow("m2", "dời <en>meeting</en> lại", "sN", "tech", label_source="pseudo:x"),
               crow("y1", "bật <en>meeting</en> lên", "tts1", "tech", synthetic=True),
               crow("y2", "tắt <en>meeting</en> đi", "tts2", "tech", synthetic=True)]
    cm_in = cur / "in.jsonl"
    cm_in.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in c_rows), encoding="utf-8")
    OFF = ["--max-per-term", "0", "--max-speaker-share", "0", "--max-domain-share", "0", "--max-synthetic-share", "0"]

    def curate(name, *flags):
        run(G / "src/gipbench/model/recipe/curate.py", "--manifest", cm_in, "--catalog", cat, "--out", cur / f"{name}.jsonl", *OFF, *flags)
        return {r["id"]: r for r in L(cur / f"{name}.jsonl")}
    allids = {r["id"] for r in c_rows}
    assert set(curate("none")) == allids  # tắt hết luật: giữ nguyên
    assert set(curate("holdout", "--holdout-domain", "finance")) == allids - {"f1", "f2"}  # cả câu không CS
    (cur / "hyp.jsonl").write_text(json.dumps({"id": "m1", "hyp": "mở meeting đi"}, ensure_ascii=False) + "\n"
                                   + json.dumps({"id": "m2", "hyp": "dời mít tinh lại"}, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    assert set(curate("agree", "--agreement-hyps", cur / "hyp.jsonl")) == allids - {"m2"}  # chỉ câu nhãn máy bị xét
    pt = curate("perterm", "--max-per-term", "2")
    dl = [r for r in pt.values() if "deadline" in r["text_tagged"]]
    assert len(dl) == 2 and len({r["speaker"] for r in dl}) == 2, dl  # xoay vòng người nói
    assert sum("meeting" in r["text_tagged"] for r in pt.values()) == 2 and "f2" in pt
    assert "| deadline | 6 | 2 |" in (cur / "perterm.jsonl.report.md").read_text(encoding="utf-8")
    sp = curate("spk", "--max-speaker-share", "0.2")  # sA 30/110 s > 20% -> còn 20 s (20% của tổng mới 100 s)
    assert sum(r["speaker"] == "sA" for r in sp.values()) == 2 and len(sp) == len(allids) - 1
    dm = curate("dom", "--max-domain-share", "0.5")  # medical 60/110 s > 50% -> còn 50 s
    assert sum(r["domain"] == "medical" for r in dm.values()) == 5 and len(dm) == len(allids) - 1
    sy = curate("syn", "--max-synthetic-share", "0.1")  # synthetic 20 s / 110 s > 10% -> còn 10 s
    assert sum(bool(r.get("synthetic")) for r in sy.values()) == 1 and len(sy) == len(allids) - 1
    curate("few", "--min-speakers-per-term", "4")
    few_md = (cur / "few.jsonl.report.md").read_text(encoding="utf-8")
    assert "deadline (3)" in few_md and "meeting (" not in few_md, few_md  # chỉ báo cáo, không loại
    # --- Phase 1 T1.3: src/gipbench/dataset/derive/confusables.py — dạng gần âm từ --pairs + phonetic catalog; bench / train tách
    #     theo người nói và nguồn (union-find: s1, s2 cùng nguồn srcX phải đi cùng một phía)
    cf = T / "confusable"
    cf.mkdir(parents=True, exist_ok=True)
    cf_rows = [dict(id=f"v{k}", split="train", audio=f"a/v{k}.wav", duration=3.0, text_tagged=t, speaker=f"s{k}",
                    source=src, dataset="vivos:train") for k, (t, src) in enumerate([
                        ("anh đi tìm chìa khoá", "srcX"), ("tim em đập nhanh", "srcX"), ("ra nhà ga đón mẹ", "src2"),
                        ("cuộc mít tinh rất đông", "src3"), ("tìm mãi không thấy", "src4"), ("đạp ga cho xe chạy", "src5"),
                        ("hôm nay trời đẹp", "src6"), ("mở <en>team</en> chat đi", "src7"), ("tìm đường về nhà", "src8")], 1)]
    cf_in = cf / "in.jsonl"
    cf_in.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cf_rows), encoding="utf-8")
    out_cf = run(G / "src/gipbench/dataset/derive/confusables.py", "--manifest", cf_in, "--catalog", cat, "--out-dir", cf / "out",
                 "--bench-n", "3", "--max-per-form-bench", "2")
    cb, ct = L(cf / "out/b5_confusable.jsonl"), L(cf / "out/train_confusable.jsonl")
    assert 1 <= len(cb) <= 3 and ct and len(cb) + len(ct) == 7, (cb, ct)  # 9 câu - 1 không có dạng - 1 có <en>
    assert all(r["split"] == "B5" and r["domain"] == "confusable" for r in cb)
    assert Counter(r["confusable"]["form"] for r in cb).most_common(1)[0][1] <= 2
    assert not ({r["speaker"] for r in cb} & {r["speaker"] for r in ct}) and not ({r["source"] for r in cb} & {r["source"] for r in ct})
    sides = {r["id"]: "b" for r in cb} | {r["id"]: "t" for r in ct}
    assert sides["cf_v1"] == sides["cf_v2"]  # cùng nguồn srcX -> cùng phía
    mt = next(r for r in cb + ct if r["id"] == "cf_v4")
    assert mt["confusable"] == {"form": "mít tinh", "term": "meeting", "from": "catalog phonetic"}, mt
    assert next(r for r in cb + ct if r["id"] == "cf_v3")["confusable"]["term"] == "gas"  # từ configs/dataset/labeling/confusables.tsv
    rv = list(csv.DictReader(open(cf / "out/b5_confusable_review.csv", encoding="utf-8-sig")))
    assert len(rv) == len(cb) and "nghĩa Việt? y/n" in rv[0] and rv[0]["text"] and "phải = 0" in out_cf  # chỉ đọc text
    # §12.6: --train-from: train CHỈ từ nguồn cho phép; nguồn khác (vd consensus, chung source) chỉ vào bench, duyệt trước
    cons = [dict(id=f"c{k}", split="B5", audio=f"a/c{k}.wav", duration=2.0, text_tagged=t, speaker=f"consensus:p{k}",
                 speaker_cluster=f"consensus:spk{k % 2}", source="consensus", dataset="consensus", license="test_only")
            for k, t in enumerate(["tìm quán ăn gần đây", "tim tôi đau quá", "ra ga lúc mấy giờ", "tìm lại ví"], 1)]
    cf_c = cf / "cons.jsonl"
    cf_c.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cons), encoding="utf-8")
    out_cf2 = run(G / "src/gipbench/dataset/derive/confusables.py", "--manifest", cf_in, cf_c, "--train-from", cf_in, "--catalog", cat,
                  "--out-dir", cf / "out2", "--bench-n", "3", "--max-per-group-bench", "2")
    cb2, ct2 = L(cf / "out2/b5_confusable.jsonl"), L(cf / "out2/train_confusable.jsonl")
    assert {r["dataset"] for r in ct2} == {"confusable:vivos:train"}, ct2  # consensus không bao giờ vào train
    assert [r["dataset"] for r in cb2] == ["confusable:consensus"] * 3, cb2  # bench lấy nguồn không-train trước
    assert max(Counter(r["speaker_cluster"] for r in cb2).values()) <= 2  # nhóm theo speaker_cluster, không theo source
    assert len(ct2) == 7 and "phải = 0" in out_cf2, (ct2, out_cf2)
    out_cf3 = run(G / "src/gipbench/dataset/derive/confusables.py", "--manifest", cf_in, cf_c, "--train-from", cf_in, "--catalog", cat,
                  "--out-dir", cf / "out3", "--bench-n", "10", "--bench-not-from-train")
    assert {r["dataset"] for r in L(cf / "out3/b5_confusable.jsonl")} == {"confusable:consensus"}  # nguồn train giữ cho train
    assert len(L(cf / "out3/train_confusable.jsonl")) == 7, out_cf3
    # build_bench_v0: cột "nghĩa Việt? y/n" (y = ok, n = drop); clip trùng audio với câu đã nhận bị bỏ
    rv2 = list(csv.DictReader(open(cf / "out2/b5_confusable_review.csv", encoding="utf-8-sig")))
    for k, r in enumerate(rv2):
        r["nghĩa Việt? y/n"] = "y" if k else "n"
    with open(cf / "out2/rv.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rv2[0]))
        w.writeheader()
        w.writerows(rv2)
    out_b = run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", cf / "out2/b5_confusable.jsonl", cf_c,
                "--reviews", cf / "out2/rv.csv", "--out", cf / "out2/bench.jsonl")
    bb = L(cf / "out2/bench.jsonl")
    assert sum(r["id"].startswith("cf_") for r in bb) == 2 and "người duyệt loại" in out_b, out_b
    assert "audio trùng câu đã nhận" in out_b and len(bb) == 4, (out_b, [r["id"] for r in bb])
    # --- Phase 1 T1.2: src/gipbench/dataset/pseudolabel/distill.py — chỉ giữ câu greedy == beam, lọc độ dài / từ mỗi giây / logprob
    ds = T / "distill"
    ds.mkdir(parents=True, exist_ok=True)
    d_rows = [("d1", 3.0, "HÔM NAY TRỜI ĐẸP", "HÔM NAY TRỜI ĐẸP", -0.1),      # giữ
              ("d2", 3.0, "ANH ĐI LÀM", "ANH ĐI CHƠI", -0.1),                # greedy khác beam
              ("d3", 3.0, "", "", None),                                     # hyp rỗng
              ("d4", 0.5, "ỪM", "ỪM", -0.1),                                 # quá ngắn
              ("d5", 1.5, " ".join(["MỘT"] * 20), " ".join(["MỘT"] * 20), -0.1),  # 13 từ/giây: ảo
              ("d6", 3.0, "CHÀO CÁC BẠN NHÉ", "CHÀO CÁC BẠN NHÉ", -3.0)]      # logprob thấp
    (ds / "m.jsonl").write_text("".join(json.dumps({"id": i, "split": "train", "audio": f"a/{i}.wav", "duration": d,
                                                     "text": "nhãn cũ", "speaker": "s", "source": "x", "dataset": "vivos:train"},
                                                    ensure_ascii=False) + "\n" for i, d, *_ in d_rows), encoding="utf-8")
    for name, col in (("g", 2), ("b", 3)):
        (ds / f"{name}.jsonl").write_text("".join(json.dumps({"id": r[0], "hyp": r[col], "logprob_mean": r[4]},
                                                             ensure_ascii=False) + "\n" for r in d_rows), encoding="utf-8")
    out_d = run(G / "src/gipbench/dataset/pseudolabel/distill.py", "--manifest", ds / "m.jsonl", "--greedy-hyps", ds / "g.jsonl",
                "--beam-hyps", ds / "b.jsonl", "--min-logprob-mean", "-1.0", "--out", ds / "out.jsonl")
    dk = L(ds / "out.jsonl")
    assert [r["id"] for r in dk] == ["d1"] and dk[0]["text"] == "hôm nay trời đẹp" and dk[0]["distill"], dk
    assert dk[0]["dataset"] == "distill:vivos" and dk[0]["label_source"].startswith("distill:")
    for k in ("greedy khác beam: 1", "hyp rỗng: 1", "độ dài ngoài 1–20 s: 1", "từ/giây ngoài 1–6: 1", "logprob_mean < -1: 1"):
        assert k in out_d, (k, out_d)
    # --- Phase 0d T5: cấu hình trộn: vss_cs thuộc nhóm riêng cs_pseudo (audio thật, nhãn giả), không thuộc cs_real
    from gipbench.model.recipe.mix_plan import load_mix
    mix = {g["group"]: g for g in load_mix(G / "configs/model/experiments/e3_v2.yaml")}
    old_mix = {g["group"]: g for g in load_mix(G / "docs/history/train_mix.tsv")}
    assert {k: (g["weight"], g["env"], g["datasets"]) for k, g in mix.items()} == \
        {k: (g["weight"], g["env"], g["datasets"]) for k, g in old_mix.items()}, "e3_v2.yaml phải khớp train_mix.tsv cuối"
    assert mix["cs_pseudo"]["datasets"] == ["vss_cs"] and "vss_cs" not in mix["cs_real"]["datasets"]
    assert "audio thật, nhãn giả" in mix["cs_pseudo"]["note"]
    assert {k: float(v["weight"]) for k, v in mix.items()} == {"cs_real": 0.20, "cs_pseudo": 0.10, "synth": 0.20,
                                                               "replay_human": 0.25, "distill": 0.20, "confusable": 0.05}
    assert abs(sum(float(r["weight"]) for r in mix.values()) - 1.0) < 1e-9
    out_mp = run(G / "src/gipbench/model/recipe/mix_plan.py", "--manifest", raw / "vss2/vss_cs_candidates.jsonl", "--assume", "ViMedCSS:medical:24.3")
    assert "**CẢNH BÁO:** domain medical (ViMedCSS) chiếm 66.7% phần CS (cs_real + cs_pseudo) > trần 35%" in out_mp, out_mp
    assert "| cs_pseudo | 0.10 | vss_cs |" in out_mp and "| conversation | 33.3% |" in out_mp, out_mp
    bad_mix = subprocess.run([sys.executable, str(G / "src/gipbench/model/recipe/mix_plan.py")], capture_output=True, text=True,
                             env={**os.environ, "MIX_CS_REAL": "0.5"})
    assert bad_mix.returncode == 1 and "tổng tỉ trọng = 1.3000" in bad_mix.stderr, bad_mix.stderr
    # --- Phase 0b T3: score.py --label-field (B3: số chính theo Spec, số phụ theo nhãn gốc)
    sb = T / "score_lf"
    sb.mkdir(parents=True, exist_ok=True)
    (sb / "bench.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"id": "x1", "split": "B3", "audio": "a.wav", "duration": 2, "speaker": "s1", "source": "youtube:a",
         "text_tagged": "uống vitamin và <en>leukotriene</en>", "text_tagged_orig": "uống <en>vitamin</en> và <en>leukotriene</en>"},
        {"id": "x2", "split": "B1", "audio": "b.wav", "duration": 2, "speaker": "s2", "source": "youtube:b",
         "text_tagged": "mở <en>facebook</en> lên"}]) + "\n", encoding="utf-8")
    (sb / "hyps.jsonl").write_text('{"id": "x1", "hyp": "UỐNG VI TA MIN VÀ LEUKOTRIENE"}\n{"id": "x2", "hyp": "MỞ FACEBOOK LÊN"}\n',
                                   encoding="utf-8")
    for lf in ("text_tagged", "text_tagged_orig"):
        run(G / "src/gipbench/bench/score.py", "--bench", sb / "bench.jsonl", "--catalog", cat, "--hyps", sb / "hyps.jsonl",
            "--out", sb / lf, "--label-field", lf, "--iters", "10")
    s_spec = json.loads((sb / "text_tagged/summary.json").read_text(encoding="utf-8"))["groups"]["B3"]
    s_orig = json.loads((sb / "text_tagged_orig/summary.json").read_text(encoding="utf-8"))["groups"]["B3"]
    assert (s_spec["n_en"], s_spec["cs_wer"]) == (1, 0.0), s_spec          # vitamin là tiếng Việt: lỗi tính vào N-WER
    assert s_orig["n_en"] == 2 and s_orig["cs_wer"] > 0, s_orig            # theo dataset: vitamin là span EN bị sai
    assert json.loads((sb / "text_tagged_orig/summary.json").read_text(encoding="utf-8"))["groups"]["B1"]["cs_wer"] == 0.0
    assert len(list(csv.reader(open(T / "reports/eval_spotcheck.csv", encoding="utf-8-sig")))) == 3
    test_consensus(cat, L)
    test_term_decisions(cat, L)
    test_domain(cat, L)
    test_backup()
    print("TẤT CẢ KIỂM THỬ CONVERTER ĐẠT")


def test_consensus(cat, L):
    """src/gipbench/dataset/ingest/consensus.py (bộ CHỈ ĐƯỢC TEST) + chặn rò rỉ test_only ở curate / prefilter / validate,
    review_required ở build_bench_v0, category name + bootstrap theo speaker_cluster ở score.py. CSV giả cùng schema."""
    from gipbench.dataset.ingest.common import REVIEW_COLS as REVIEW_COLS_
    zr = T / "dl/zac_fake"
    shutil.rmtree(zr, ignore_errors=True)
    cols = ["id", "turn", "hist_pos", "tier", "need", "n_blind", "status", "share", "numbers_majority", "source",
            "consensus_text", "reviewers", "confirmed_after_prompt", "listener_texts"]
    Z = [("public_t01", "final", "agreed", "Nhắn cho Hugo Lambert lúc 9 giờ nhé.", "True", "", True),
         ("public_t10", "final", "agreed", "Zalo hết pin rồi.", "True", "", True),
         ("public_t01", "u1", "agreed", "Mai mà mưa thì huỷ đá bóng, gọi báo Lan nhé.", "True", "", True),
         ("public_t02", "final", "agreed", "nhắn zalo cho mẹ là con về muộn", "True", "", True),
         ("public_t03", "final", "agreed", "mở zalo gọi cho anh tuấn", "True", "", True),
         ("public_t04", "final", "agreed", "zalo của chị lan bị khoá rồi", "True", "", True),
         ("public_t05", "final", "disputed", "", "True", "A: Gọi cho Sarah nhé || B: gọi cho sa ra nhé", True),
         ("public_t05", "u1", "pending", "tắt đèn phòng khách", "True", "E: Tắt đèn phòng khách. || Q: tắt đèn phòng khách", True),
         ("public_t06", "final", "escalate", "", "", "P: câu bị đẩy lên cấp trên", True),
         ("public_t07", "final", "agreed", "câu này không có audio", "True", "", False),
         ("public_t08", "final", "agreed", "bật quạt phòng ngủ giúp mình", "True", "", "44k"),
         ("public_t09", "u1", "agreed", "chuyển ba trăm nghìn cho bố", "False", "", True)]
    rows = []
    for sid, turn, st, text, nm, lt, audio in Z:
        rows.append(dict(zip(cols, [sid, turn, "", "confirm", "1", "1", st, "1.0" if st == "agreed" else "0.5", nm,
                                    "blind", text, "A; B", "", lt])))
        if audio:
            d = zr / "audio" / sid
            d.mkdir(parents=True, exist_ok=True)
            stem = sid if turn == "final" else f"{sid}_{turn}"
            (d / f"{stem}.wav").write_bytes(wav_bytes(2.0, sr=44100, channels=2) if audio == "44k" else wav_bytes(2.0))
    (zr / "audio/public_t99").mkdir(parents=True, exist_ok=True)
    (zr / "audio/public_t99/public_t99.wav").write_bytes(wav_bytes(1.5))  # file không có trong CSV
    with open(zr / "consensus_clips_anon.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    co = T / "raw/consensus"
    shutil.rmtree(co, ignore_errors=True)
    out = run(G / "src/gipbench/dataset/ingest/consensus.py", "--root", zr, "--out", co, "--catalog", cat, "--max-per-term", "2",
              "--spk-model", T / "khong_co_model.onnx")
    assert "luật final = <id>, luật turn = <id>_<turn>; ghép 11/12 clip" in out and "file không dùng: 1" in out, out
    assert "speaker embedding: BỎ QUA" in out, out  # tuỳ chọn: thiếu model thì bỏ qua, không dừng
    b2 = {r["id"]: r for r in L(co / "b2_consensus_candidates.jsonl")}
    cut = L(co / "b2_consensus_overcap.jsonl")
    b5 = {r["id"]: r for r in L(co / "b5_consensus_product.jsonl")}
    dis = {r["id"]: r for r in L(co / "disputed_candidates.jsonl")}
    assert len(b2) == 3 and len(cut) == 2 and all(r["overcap"] for r in cut), (b2, cut)  # zalo ×4, trần 2
    h = b2["cons_public_t01_final"]
    assert h["text_tagged"] == "nhắn cho <en>hugo lambert</en> lúc chín giờ nhé", h
    assert h["term_category"] == {"hugo lambert": "name"} and "name:hugo lambert" in h["auto_tag"]["flags"]
    assert b5["cons_public_t01_u1"]["text_tagged"] == "mai mà mưa thì huỷ đá bóng gọi báo lan nhé"
    assert "numbers_disagree" in b5["cons_public_t09_u1"]["auto_tag"]["flags"]
    allr = list(b2.values()) + cut + list(b5.values()) + list(dis.values())
    assert all(r["speaker"] == f"consensus:{r['id'][5:15]}" and r["source"] == "consensus" and r["domain"] == "product"
               and r["license"] == "test_only" for r in allr), allr
    assert not any("t06" in r["id"] or "t07" in r["id"] for r in allr)  # escalate bỏ, thiếu audio bỏ
    assert not list(co.glob("dev*")) and not any(r["split"] not in ("B2", "B5") for r in allr)  # KHÔNG tách dev
    r8 = b5["cons_public_t08_final"]
    i8 = sf.info(str(co / r8["audio"]))
    assert r8["audio"].startswith("audio/") and (i8.samplerate, i8.channels) == (16000, 1)  # 44.1 kHz stereo -> 16 kHz mono
    assert b5["cons_public_t01_u1"]["audio"].startswith("../")  # đã đúng định dạng: trỏ thẳng file gốc
    d5 = dis["cons_public_t05_final"]
    assert d5["split"] == "B2" and d5["review_required"] and "<en>sarah</en>" in d5["text_tagged"], d5
    assert dis["cons_public_t05_u1"]["split"] == "B5" and "consensus:pending" in dis["cons_public_t05_u1"]["auto_tag"]["flags"]
    rv = list(csv.DictReader(open(co / "disputed_candidates_review.csv", encoding="utf-8-sig")))
    assert list(rv[0])[:len(REVIEW_COLS_)] == REVIEW_COLS_ and rv[0]["hint"].startswith("A: Gọi cho Sarah nhé"), rv[0]
    assert rv[0]["audio"] == "audio/public_t05/public_t05.wav" and rv[0]["status"] == "disputed"
    b5rv = list(csv.DictReader(open(co / "b5_consensus_product_review.csv", encoding="utf-8-sig")))
    assert [r["id"] for r in b5rv] == ["cons_public_t09_u1"]  # chỉ câu B5 có cờ
    block = (co / "leak_block.txt").read_text(encoding="utf-8")
    for t in ("nhắn cho hugo lambert lúc chín giờ nhé", "câu bị đẩy lên cấp trên", "gọi cho sa ra nhé"):
        assert f"\n{t}\n" in block, t  # mọi trạng thái, cả listener_texts
    rep = (co / "consensus_report.md").read_text(encoding="utf-8")
    assert "| bị cắt bởi --max-per-term 2 | 2 |" in rep and "| zalo | 4 | 2 |" in rep and "| name | 1 |" in rep, rep
    # build_bench_v0: disputed (review_required) không vào bench khi chưa duyệt, kể cả --accept all; fix bỏ <en> -> B5
    rvp = T / "cons_review.csv"
    with open(rvp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rv[0]))
        w.writeheader()
        for r in rv:
            if r["id"] == "cons_public_t05_final":
                r.update(decision="fix", text_tagged_fixed="gọi cho sa ra nhé")
            w.writerow(r)
    for name, reviews in (("noreview", []), ("reviewed", [rvp])):
        run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", co / "b2_consensus_candidates.jsonl", co / "disputed_candidates.jsonl",
            co / "b5_consensus_product.jsonl", "--reviews", *reviews, "--accept", "all", "--out", T / f"data/bench_cons_{name}.jsonl")
    bn = {r["id"]: r for r in L(T / "data/bench_cons_noreview.jsonl")}
    br = {r["id"]: r for r in L(T / "data/bench_cons_reviewed.jsonl")}
    assert not any(k in bn for k in dis) and "cons_public_t05_u1" not in br, sorted(bn)
    assert br["cons_public_t05_final"]["split"] == "B5" and br["cons_public_t05_final"]["text_tagged"] == "gọi cho sa ra nhé"
    # chặn rò rỉ: curate_train (license / source test_only + text trùng danh sách chặn), prefilter, validate
    tr = T / "cons_train.jsonl"
    trows = [dict(id="x1", split="train", audio="a.wav", duration=3.0, text_tagged="nhắn <en>zalo</en> cho mẹ là con về muộn",
                  speaker="vss_a", source="vss:a", dataset="vss_cs", domain="conversation"),   # text trùng consensus
             dict(id="x2", split="train", audio="b.wav", duration=3.0, text_tagged="mở <en>meeting</en> đi",
                  speaker="consensus:public_t02", source="consensus", dataset="consensus", license="test_only"),
             dict(id="x3", split="train", audio="c.wav", duration=3.0, text_tagged="bật <en>wifi</en> lên giúp anh",
                  speaker="vss_b", source="vss:b", dataset="vss_cs", domain="conversation", license="cc-by")]
    tr.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in trows), encoding="utf-8")
    OFF = ["--max-per-term", "0", "--max-speaker-share", "0", "--max-domain-share", "0", "--max-synthetic-share", "0"]
    run(G / "src/gipbench/model/recipe/curate.py", "--manifest", tr, "--catalog", cat, "--out", T / "cons_cur.jsonl", *OFF,
        "--block-text", co / "leak_block.txt")
    assert [r["id"] for r in L(T / "cons_cur.jsonl")] == ["x3"]
    cur_md = (T / "cons_cur.jsonl.report.md").read_text(encoding="utf-8")
    assert "license test_only: 1 câu" in cur_md and "text trùng danh sách chặn: 1 câu" in cur_md, cur_md
    run(G / "src/gipbench/model/recipe/curate.py", "--manifest", tr, "--catalog", cat, "--out", T / "cons_cur2.jsonl", *OFF, "--block-text")
    assert [r["id"] for r in L(T / "cons_cur2.jsonl")] == ["x1", "x3"]  # tắt danh sách text; test_only vẫn bị chặn
    out_pf = run(G / "src/gipbench/dataset/pseudolabel/vss_cs_prefilter.py", "--manifest", tr, "--catalog", cat, "--term-cap", "0", "--hapax-keep-frac", "1",
                 "--block-text", co / "leak_block.txt", "--out", T / "cons_pf.jsonl")
    assert [r["id"] for r in L(T / "cons_pf.jsonl")] == ["x3"], out_pf
    assert "danh sách chặn test_only" in (T / "cons_pf.jsonl.report.md").read_text(encoding="utf-8")
    dev = T / "cons_dev.jsonl"
    dev.write_text(json.dumps(dict(trows[2], id="d1", source="consensus"), ensure_ascii=False) + "\n", encoding="utf-8")
    ok_train = T / "cons_train_ok.jsonl"
    ok_train.write_text(json.dumps(trows[2], ensure_ascii=False) + "\n", encoding="utf-8")
    vb = T / "data/bench_cons_reviewed.jsonl"
    v_bad = run(G / "src/gipbench/dataset/release/validate.py", "--bench", vb, "--catalog", cat, "--train", tr, "--block-text", co / "leak_block.txt", ok=(1,))
    assert "RÒ RỈ: dữ liệu test_only có trong train / dev" in v_bad and "cons_train.jsonl:x2 (license test_only)" in v_bad, v_bad
    assert "trùng text danh sách chặn" in v_bad, v_bad
    v_dev = run(G / "src/gipbench/dataset/release/validate.py", "--bench", vb, "--catalog", cat, "--train", ok_train, "--dev", dev, ok=(1,))
    assert "cons_dev.jsonl:d1 (source consensus (test_only))" in v_dev, v_dev
    run(G / "src/gipbench/dataset/release/validate.py", "--bench", vb, "--catalog", cat, "--train", ok_train, ok=(0,))
    # score.py: category từ term_category (name), bootstrap theo speaker_cluster
    sc = T / "score_cons"
    sc.mkdir(parents=True, exist_ok=True)
    sb = [dict(r, speaker_cluster=f"consensus:spk{k % 2:03d}") for k, r in enumerate(L(T / "data/bench_cons_noreview.jsonl"))]
    (sc / "bench.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sb), encoding="utf-8")
    (sc / "hyps.jsonl").write_text("".join(json.dumps({"id": r["id"], "hyp": re.sub(r"</?en>", "", r["text_tagged"])},
                                                      ensure_ascii=False) + "\n" for r in sb), encoding="utf-8")
    run(G / "src/gipbench/bench/score.py", "--bench", sc / "bench.jsonl", "--catalog", cat, "--hyps", sc / "hyps.jsonl", "--out", sc, "--iters", "20")
    sl = (sc / "slices.md").read_text(encoding="utf-8")
    assert "## Category thuật ngữ (B2, theo span)" in sl and "| name | 1 | 0.00 |" in sl and "| brand |" in sl, sl
    sm = json.loads((sc / "summary.json").read_text(encoding="utf-8"))["groups"]
    assert sm["B2"]["boot_units"] == 2 and sm["B5"]["boot_units"] == 2, sm  # 2 cụm, không phải số phiên
    assert {json.loads(l)["boot_unit"] for l in open(sc / "utt_scores.jsonl", encoding="utf-8")} == {"consensus:spk000", "consensus:spk001"}



def test_term_decisions(cat, L):
    """§12.4: chốt tag theo thuật ngữ — auto_tag theo configs/dataset/labeling/term_decisions.tsv, lib.term_decisions (chờ / chặn / cũ),
    build_bench_v0 --accept trusted-text (+ --pending-out, --draft-splits), src/gipbench/dataset/label/term_decisions_cli.py ask / import."""
    from gipbench.dataset.ingest.common import auto_tag
    from gipbench.dataset.ingest.consensus import tag_text
    from gipbench.core.term_decisions import load_term_decisions, term_status, write_term_rows
    from gipbench.core.textnorm import Catalog
    cat_obj = Catalog.load(str(cat))
    # auto_tag: en thắng vi_words, vi thắng catalog; cụm nhiều từ vi không sinh cờ vi_word từng từ
    td = {"sjc": "vi", "test": "en", "zalo": "vi", "mr tanaka": "drop_tag"}
    t = auto_tag("giá vàng sjc và chạy test trên zalo", cat_obj, vi_words={"sjc", "test"}, term_dec=td)
    assert t["spans"] == ["test"] and "vi_word:sjc" in t["flags"] and "vi_word:zalo" in t["flags"], t
    assert auto_tag("giá vàng sjc", cat_obj, vi_words=set(), term_dec={})["spans"] == ["sjc"]  # {} = tắt
    # quyết định en không tách cụm: ghép với từ Anh liền kề, không sinh cờ abbr_reading cho phần còn lại
    t3 = auto_tag("đặt stop loss ở new york", None, vi_words=set(), term_dec={"stop": "en", "new": "en"})
    assert t3["spans"] == ["stop loss", "new york"] and not any(f.startswith("abbr_reading") for f in t3["flags"]), t3
    t4 = auto_tag("mở crm nhé", None, vi_words=set(), term_dec={"crm": "en"})
    assert t4["spans"] == ["crm"] and t4["span_kinds"] == ["D"] and t4["flags"] == [], t4
    t2 = auto_tag("gặp mr tanaka nhé", None, vi_words=set(), term_dec=td)
    assert t2["spans"] == [] and not any(f.startswith("vi_word") for f in t2["flags"]), t2
    # consensus tag_text: cụm tên có decision vi / en
    names = {"sam": "given", "kim": "surname", "lucas": "given"}
    assert "<en>" not in tag_text("gọi lucas nhé", cat_obj, set(), names, set(), {"lucas": "vi"})[0]
    tg, _, tcat, fl = tag_text("gọi sam kim nhé", cat_obj, set(), names, set(), {"sam kim": "en"})
    assert tg == "gọi <en>sam kim</en> nhé" and tcat == {"sam kim": "name"}, (tg, tcat, fl)
    assert "maybe_name:sam kim" in tag_text("gọi sam kim nhé", cat_obj, set(), names, set(), {})[3]
    # term_status
    r_ok = {"text_tagged": "mở <en>crm</en> nhé", "auto_tag": {"flags": ["abbr_reading:crm", "heuristic_span"]}}
    assert term_status(r_ok, {})["pending"] == ["crm"]
    assert term_status(r_ok, {"crm": "en"}) == {"pending": [], "blocking": [], "stale": []}
    assert term_status(r_ok, {"crm": "vi"})["stale"] == ["crm"]  # quyết định vi mà nhãn còn <en>: phải chạy lại converter
    r_vw = {"text_tagged": "co giãn tốt", "auto_tag": {"flags": ["vi_word:co", "vi_word:fpt"]}}
    assert term_status(r_vw, {})["pending"] == ["fpt"]  # 'co' là âm tiết Việt hợp lệ: không cần chốt
    assert term_status(r_vw, {"fpt": "en"})["stale"] == ["fpt"]
    r_amb = {"text_tagged": "bật <en>snap to grid</en>", "auto_tag": {"flags": ["ambiguous_joined", "heuristic_span"]}}
    assert term_status(r_amb, {})["pending"] == ["snap to grid"]
    r_blk = {"text_tagged": "chuyển ba trăm", "auto_tag": {"flags": ["numbers_disagree"]}}
    assert term_status(r_blk, {})["blocking"] == ["numbers_disagree"]
    # file quyết định: ghi / đọc / decision lạ
    D = T / "term_dec"
    shutil.rmtree(D, ignore_errors=True)
    D.mkdir(parents=True)
    write_term_rows(D / "td.tsv", [{"term": "CRM", "decision": "en", "note": "x"}, {"term": "ai", "decision": "", "note": ""}])
    assert load_term_decisions(D / "td.tsv") == {"crm": "en"} and load_term_decisions("") == {}
    (D / "bad.tsv").write_text("term\tdecision\tnote\nabc\tyes\t\n", encoding="utf-8")
    try:
        load_term_decisions(D / "bad.tsv")
        raise AssertionError("decision lạ phải báo lỗi")
    except ValueError:
        pass
    # build_bench_v0 --accept trusted-text
    (D / "a.wav").write_bytes(wav_bytes(1.0))

    def row(i, split, text, flags, trusted=True, **kw):
        return dict({"id": i, "split": split, "audio": f"{i}.wav", "duration": 1.0, "speaker": i, "source": i,
                     "domain": "tech", "dataset": "x", "text_tagged": text, "_audio_path": str(D / f"{i}.wav"),
                     "auto_tag": {"confidence": "heuristic" if "<en>" in text else "none", "flags": flags}},
                    **({"trusted_text": True} if trusted else {}), **kw)
    rows = [row("t_ok", "B1", "mở <en>crm</en> nhé", ["abbr_reading:crm", "heuristic_span"]),
            row("t_clear", "B1", "học <en>deep learning</en>", ["heuristic_span"]),
            row("t_pend", "B1", "mở <en>xyz</en> nhé", ["abbr_reading:xyz", "heuristic_span"]),
            row("t_stale", "B2", "giá <en>sjc</en>", ["abbr_reading:sjc", "heuristic_span"]),
            row("t_blk", "B5", "chuyển ba trăm", ["numbers_disagree"]),
            row("t_b5", "B5", "co giãn tốt", ["vi_word:co"]),
            row("u_b1", "B1", "mở <en>crm</en>", ["abbr_reading:crm", "heuristic_span"], trusted=False),
            row("u_b3", "B3", "uống <en>aspirin</en>", ["heuristic_span"], trusted=False)]
    (D / "cand.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_term_rows(D / "dec.tsv", [{"term": "crm", "decision": "en"}, {"term": "sjc", "decision": "vi"}])
    out = run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", D / "cand.jsonl", "--accept", "trusted-text",
              "--term-decisions", D / "dec.tsv", "--out", D / "bench.jsonl", "--pending-out", D / "pend.tsv")
    got = {r["id"]: r["review"] for r in L(D / "bench.jsonl")}
    assert got == {"t_ok": "term_decisions", "t_clear": "term_decisions", "t_b5": "term_decisions"}, (got, out)
    assert "chờ: nhãn cũ, chạy lại converter (B2)" in out and "chờ duyệt câu: cờ không theo thuật ngữ (B5)" in out, out
    assert (D / "pend.tsv").read_text(encoding="utf-8").splitlines()[1] == "xyz\t1\t0.000"
    out = run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", D / "cand.jsonl", "--accept", "trusted-text",
              "--term-decisions", D / "dec.tsv", "--out", D / "bench2.jsonl", "--draft-splits", "B3")
    got2 = {r["id"]: r["review"] for r in L(D / "bench2.jsonl")}
    assert got2.get("u_b3") == "draft_unreviewed" and "u_b1" not in got2 and "bản nháp" in out, (got2, out)
    run(G / "src/gipbench/dataset/release/build_bench.py", "--candidates", D / "cand.jsonl", "--accept", "trusted-text",
        "--term-decisions", D / "dec.tsv", "--out", D / "bench3.jsonl", "--catalogs", cat,
        "--catalog-out", D / "cat3.tsv", "--catalog-from-bench")
    c3 = {l.split("\t")[0]: l.split("\t") for l in (D / "cat3.tsv").read_text(encoding="utf-8").splitlines()[1:]}
    assert c3["crm"][1] == "tech" and c3["deep learning"][4] == "0" and "facebook" in c3, sorted(c3)  # + catalog gốc
    # src/gipbench/dataset/label/term_decisions_cli.py ask (gộp top_review) / import
    top = D / "top.csv"
    with open(top, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "rank", "entry", "group_or_kind", "count", "count_ggroup", "count_vimedcss", "count_vss",
                    "example_source", "example_id", "example_text", "list_source"])
        w.writerow(["vi_loanwords.txt", 1, "virus", "same_surface", 50, 0, 40, 10, "vimedcss", "m1", "virus cúm", "x"])
        w.writerow(["vi_abbrev.txt", 1, "crm", "x", 3, 3, 0, 0, "ggroup", "g1", "mở crm", "x"])
    out = run(G / "src/gipbench/dataset/label/term_decisions_cli.py", "ask", "--manifests", D / "cand.jsonl", "--decisions", D / "td.tsv",
              "--top-review", top, "--catalog", cat, "--out", D / "ask.csv", "--auto-out", D / "auto.csv")
    ask = list(csv.DictReader(open(D / "ask.csv", encoding="utf-8-sig")))
    terms = [r["term"] for r in ask]
    assert "crm" not in terms and terms[:2] == ["sjc", "xyz"] and terms[-1] == "virus", (terms, out)  # crm đã chốt
    assert all(r["n_sentences"] == "1" and r["example_1"] for r in ask[:2]) and ask[-1]["flags"] == "top_review"
    assert ask[-1]["decision"] == "vi" and "same_surface" in ask[-1]["note"], ask[-1]
    auto = list(csv.DictReader(open(D / "auto.csv", encoding="utf-8-sig")))
    assert [r["term"] for r in auto] == ["deep learning"], auto
    rows_ask = list(csv.DictReader(open(D / "ask.csv", encoding="utf-8-sig")))
    for r in rows_ask:
        r["decision"] = {"sjc": "vi", "xyz": "drop_tag"}.get(r["term"], "")
    with open(D / "ask.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_ask[0]))
        w.writeheader()
        w.writerows(rows_ask)
    run(G / "src/gipbench/dataset/label/term_decisions_cli.py", "import", "--ask", D / "ask.csv", "--decisions", D / "td.tsv", "--out", D / "td2.tsv")
    assert load_term_decisions(D / "td2.tsv") == {"crm": "en", "sjc": "vi", "xyz": "drop_tag"}
    assert load_term_decisions(D / "td.tsv") == {"crm": "en"}  # --out khác: không đụng file gốc



def test_domain(cat, L):
    """§10 / §12.6: suy domain từ text (src/gipbench/core/domain.py, src/gipbench/dataset/label/infer_domain.py), báo cáo domain, đề xuất reserved_b3
    ngoài y khoa (src/gipbench/dataset/label/b3_nonmed_terms.py), mix_plan --fit-cap (trần 35% / 50%)."""
    from gipbench.core.domain import DomainInferer, load_keywords
    from gipbench.core.textnorm import Catalog
    inf = DomainInferer(load_keywords(), Catalog.load(str(cat)))
    assert inf.infer("nhà đầu tư mua cổ phiếu khi lãi suất giảm")[0] == "finance"
    assert inf.infer("hôm nay trời đẹp quá")[0] == "other"                       # không đủ điểm
    assert inf.infer("bác sĩ nói học sinh nên uống thuốc ở trường")[0] == "other"  # hoà điểm -> other
    assert inf.infer("mở <en>app</en> rồi sửa <en>code</en> trên máy tính")[0] == "tech"
    D = T / "domain"
    shutil.rmtree(D, ignore_errors=True)
    D.mkdir(parents=True)
    rows = [dict(id=f"v{i}", split="train", audio=f"v{i}.wav", duration=3600.0, text_tagged=t, speaker=f"s{i % 3}",
                 source=f"vss:s{i % 3}", domain=d, dataset="vss_cs") for i, (t, d) in enumerate([
                     ("nhà đầu tư mua cổ phiếu và trái phiếu", "conversation"),
                     ("mình viết <en>code</en> cho phần mềm trên máy tính", "conversation"),
                     ("đi <en>blockchain</en> học <en>blockchain</en> với <en>docker</en>", "conversation"),
                     ("hôm nay trời đẹp", "conversation"), ("chuyện ngân hàng tài chính", "finance")])]
    (D / "vss.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = run(G / "src/gipbench/dataset/label/infer_domain.py", "--manifest", D / "vss.jsonl", "--out", D / "vss_dom.jsonl",
              "--catalog", cat, "--report", D / "dom.md")
    dom = {r["id"]: r for r in L(D / "vss_dom.jsonl")}
    assert dom["v0"]["domain"] == "finance" and dom["v0"]["domain_source"] == "inferred" and dom["v0"]["domain_orig"] == "conversation"
    assert dom["v1"]["domain"] == "tech" and dom["v3"]["domain"] == "other"
    assert dom["v4"]["domain"] == "finance" and "domain_source" not in dom["v4"]  # domain thật: giữ nguyên
    assert "câu mẫu được suy ra là finance" in (D / "dom.md").read_text(encoding="utf-8"), out
    # báo cáo domain (manifest + meta CSV kiểu ViMedCSS)
    with open(D / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "duration_seconds", "segment_text", "cs_terms_list"])
        w.writerow(["m1", "7200", "uống aspirin mỗi ngày", "aspirin"])
    run(G / "src/gipbench/dataset/label/domain_report.py", "--catalog", cat, "--out", D / "rep.md",
        "--group", f"cs_pseudo={D / 'vss_dom.jsonl'}", f"B3={D / 'meta.csv'}")
    rep = (D / "rep.md").read_text(encoding="utf-8")
    assert "| cs_pseudo | 5 | 5.00 | 3 |" in rep and "| B3 | 1 | 2.00 | 1 |" in rep and "blockchain 2" in rep, rep
    # đề xuất reserved_b3 ngoài y khoa: nguồn G-Group (không finance) + vss_cs, bỏ consensus
    gg = [dict(id=f"g{i}", split="B1", audio="x.wav", duration=2.0, text_tagged=t, speaker=f"g{i}", source=f"g{i}", domain=d,
               dataset=f"vi-asr-{d}-test") for i, (t, d) in enumerate([
                   ("cài <en>docker</en> lên máy", "tech"), ("chạy <en>docker</en> nhé", "tech"),
                   ("mở <en>zalo</en> lên", "tech"), ("mở <en>zalo</en> đi", "tech"),
                   ("mua <en>etf</en> nhé", "finance"), ("bán <en>etf</en> đi", "finance"), ("<en>etf</en> tăng", "finance")])]
    (D / "gg.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gg), encoding="utf-8")
    cons = [dict(id="c1", split="B2", audio="c.wav", duration=2.0, text_tagged="nhắn <en>zalo</en> nhé", speaker="c",
                 source="consensus", domain="product", dataset="consensus", license="test_only")]
    (D / "cons.jsonl").write_text(json.dumps(cons[0], ensure_ascii=False) + "\n", encoding="utf-8")
    out = run(G / "src/gipbench/dataset/label/b3_nonmed_terms.py", "--ggroup", D / "gg.jsonl", "--vss-cs", D / "vss_dom.jsonl",
              "--exclude-manifest", D / "cons.jsonl", "--catalog", cat, "--min-speakers", "2", "--common-top", "0",
              "--out", D / "b3.csv", "--report", D / "b3.md")
    b3 = {r["term"]: r for r in csv.DictReader(open(D / "b3.csv", encoding="utf-8-sig"))}
    assert set(b3) == {"docker"} and b3["docker"]["selected"] == "1" and b3["docker"]["n_vss_cs"] == "1", (b3, out)
    b3md = (D / "b3.md").read_text(encoding="utf-8")
    assert "mất **1 câu, 1.00 h**" in b3md and "2 câu B1 chứa" in b3md, b3md  # zalo (consensus), etf (finance) bị bỏ
    # mix_plan --fit-cap: medical (ViMedCSS) bị cắt về đúng trần, 'other' miễn trần
    out = run(G / "src/gipbench/model/recipe/mix_plan.py", "--manifest", D / "vss_dom.jsonl", "--assume", "ViMedCSS:medical:20",
              "--domain-cap", "0.5", "--fit-cap", "--out-json", D / "mix.json")
    mj = json.loads((D / "mix.json").read_text(encoding="utf-8"))
    assert mj["feasible"] and abs(mj["target_share"]["medical"] - 0.5) < 1e-6, mj
    assert abs(sum(mj["target_share"].values()) - 1) < 1e-6 and "Domain bị cắt: medical" in out, out



def test_backup():
    """scripts/backup.sh: nén manifest / phiếu duyệt / configs / cache hyps, KHÔNG nén audio."""
    import tarfile
    R = T / "bk_root"
    shutil.rmtree(R, ignore_errors=True)
    for f in ["data/raw/x/a.jsonl", "data/raw/x/a_review.csv", "data/raw/consensus/leak_block.txt",
              "data/raw/consensus/speaker_clusters.tsv", "data/raw/x/audio/a.jsonl", "data/raw/x/a.wav",
              "configs/c.tsv", "results/asr_hyps/q/h.jsonl", "results/other/z.jsonl", "data/dl/big.wav"]:
        (R / f).parent.mkdir(parents=True, exist_ok=True)
        (R / f).write_text("x", encoding="utf-8")
    p = subprocess.run(["bash", str(G / "scripts/backup.sh"), "--root", str(R), "--dest", str(T / "bk_dest")],
                       capture_output=True, text=True)
    out = p.stdout
    assert p.returncode == 0, (p.stdout, p.stderr)
    tars = sorted((T / "bk_dest").glob("*.tar.gz"))
    names = set(tarfile.open(tars[-1]).getnames())
    assert names == {"data/raw/x/a.jsonl", "data/raw/x/a_review.csv", "data/raw/consensus/leak_block.txt",
                     "data/raw/consensus/speaker_clusters.tsv", "configs/c.tsv", "results/asr_hyps/q/h.jsonl"}, names
    assert "Đã sao lưu 6 file" in out, out


if __name__ == "__main__":
    main()
