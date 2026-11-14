#!/usr/bin/env python3
"""Kiểm thử các converter bằng dataset GIẢ có đúng schema của bản thật (không cần mạng).

  python tools/test_converters.py [thư_mục_tạm]
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

G = Path(__file__).resolve().parents[1]
T = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gipbench_convtest")
rng = np.random.default_rng(0)


def wav_bytes(dur, fmt="WAV"):
    b = io.BytesIO()
    sf.write(b, (0.05 * rng.standard_normal(int(16000 * dur))).astype("float32"), 16000, format=fmt)
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
        ds = Dataset.from_dict({"segment_id": [r[0] for r in rows], "audio": [{"bytes": wav_bytes(4), "path": "audio.wav"} for _ in rows],
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

    cat = G / "configs/catalog.example.tsv"
    raw = T / "raw"
    run(G / "converters/conv_ggroup.py", "--names", "tech", "finance", "pubadmin", "--out", raw / "ggroup", "--catalog", cat, "--parquet", dl / "ggroup_{name}.parquet")
    run(G / "converters/conv_vimedcss.py", "--out", raw / "vimedcss", "--catalog", cat, "--parquet", dl / "vimedcss_{split}.parquet")
    run(G / "converters/conv_sea.py", "--out", raw / "sea", "--catalog", cat, "--parquet", dl / "sea.parquet", "--bench-hours", "0.002")
    run(G / "converters/conv_local.py", "vivos", "--root", dl / "vivos", "--out", raw / "vivos")
    run(G / "converters/conv_local.py", "commonvoice", "--root", cv, "--out", raw / "cv")
    run(G / "converters/conv_local.py", "vietsuperspeech", "--root", vss, "--out", raw / "vss")

    def L(p):
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    b1 = L(raw / "ggroup/b1_candidates.jsonl")
    assert len(b1) == 5 and any("<en>connect metamask</en>" in r["text_tagged"] for r in b1), b1
    assert len(L(raw / "ggroup/b5_ggroup.jsonl")) == 1
    b3 = L(raw / "vimedcss/b3_candidates.jsonl")
    assert "<en>leukotriene</en>" in b3[0]["text_tagged"] and "term_not_found" in b3[1]["auto_tag"]["flags"]
    catv = list(csv.DictReader(open(raw / "vimedcss/catalog_vimedcss.tsv", encoding="utf-8"), delimiter="\t"))
    assert {r["term"]: r["reserved_b3"] for r in catv}["leukotriene"] == "1"
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
    out = run(G / "converters/build_bench_v0.py", "--candidates", raw / "ggroup/b1_candidates.jsonl", raw / "ggroup/b5_ggroup.jsonl",
              raw / "vimedcss/b3_candidates.jsonl", raw / "sea/b1_candidates_sea.jsonl", raw / "vivos/b5_vivos.jsonl", raw / "cv/b5_cv.jsonl",
              "--reviews", rp, rp3, "--catalogs", cat, raw / "vimedcss/catalog_vimedcss.tsv",
              "--out", T / "data/bench_v0.jsonl", "--catalog-out", T / "configs/catalog_v0.tsv")
    bench = L(T / "data/bench_v0.jsonl")
    assert sorted(r["split"] for r in bench) == ["B1", "B1", "B3", "B5", "B5", "B5", "B5"], [r["split"] for r in bench]
    fixed = next(r for r in bench if r["review"] == "fix")
    assert fixed["en_meta"] == [{"viet_level": "mid"}] and fixed["region"] == "south"
    assert fixed["reading_note"] == "google: gu gồ -> EN"  # Phase 0b: cột ghi chú cách đọc đi vào bench
    # --- bench đi tiếp được vào validate (phải bắt video y578jWB56QE có ở cả train ViMedCSS và SEA nếu rơi vào bench) và eda
    run(G / "validate.py", "--bench", T / "data/bench_v0.jsonl", "--catalog", T / "configs/catalog_v0.tsv", "--check-audio",
        "--train", raw / "vimedcss/train_vimedcss.jsonl", raw / "sea/train_sea.jsonl", raw / "vss/train_vss.jsonl", ok=(0, 1))
    run(G / "eda.py", "--bench", T / "data/bench_v0.jsonl", "--catalog", T / "configs/catalog_v0.tsv", "--out", T / "eda")
    run(G / "ft/to_lhotse.py", "--manifest", raw / "vimedcss/train_vimedcss.jsonl", raw / "sea/train_sea.jsonl", "--name", "cs_train",
        "--bench", T / "data/bench_v0.jsonl", "--catalog", T / "configs/catalog_v0.tsv", "--out", T / "lhotse", "--min-dur", "1")
    # --- Phase 0: đọc N dòng đầu từ parquet nhiều shard (đường --limit khi tải từ HF), filesystem cục bộ thay HF
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq
    sys.path.insert(0, str(G))
    from converters.common import iter_hf_parquet
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
    run(G / "converters/conv_local.py", "vietsuperspeech", "--root", vss, "--out", raw / "vss_l1", "--limit", "1")
    assert len(L(raw / "vss_l1/dev_vss.jsonl")) == 1  # trước đây dev không được đọc khi có --limit
    # --- tools/converter_check.py
    run(G / "tools/converter_check.py", "--manifests", raw / "ggroup/b1_candidates.jsonl", raw / "vss/train_vss.jsonl",
        "--samples", "2", "--out", T / "reports/converter_check.md")
    ck = (T / "reports/converter_check.md").read_text(encoding="utf-8")
    assert "| 5 |" in ck and "heuristic" in ck and "2 câu mẫu" in ck, ck
    # --- tools/eval_autotag.py: số liệu tính tay trên 3 câu vàng
    gold = T / "gold.jsonl"
    gold.write_text("\n".join(json.dumps({"id": i, "gold": g}, ensure_ascii=False) for i, g in [
        ("a", "mình [[deploy]] lên [[server]] rồi"),          # auto_tag: deploy + server (catalog) -> 2 TP
        ("b", "chấm [[gov]] chấm {{vn}} nhé"),                 # auto_tag gắn cả vn -> 1 FP (nằm trong viết tắt VN)
        ("c", "học ((ai ten)) với cái [[kit]] này")]) + "\n", encoding="utf-8")  # kit: -t không dấu -> không phải tiếng Việt
    run(G / "tools/eval_autotag.py", "--gold", gold, "--catalog", cat, "--out", T / "reports/eval", "--spotcheck", "2")
    ev = json.loads((T / "reports/eval.json").read_text(encoding="utf-8"))
    s = ev["subset_en"]
    assert (s["gold_spans"], s["tp"], ev["phonetic_spans"], ev["vi_abbr_spans"]) == (4, 4, 1, 1), s
    assert abs(s["precision_neutral"] - 1.0) < 1e-9 and abs(s["precision"] - 0.8) < 1e-9, s
    # --- T0.2: luật chính tả âm tiết trong is_vietnamese_token
    from converters.common import auto_tag, is_vietnamese_token
    from lib.textnorm import Catalog
    cat_obj = Catalog.load(str(cat))
    not_vi = ["chip", "kit", "mac", "net", "hot", "tech", "op", "chat", "top", "chain", "main", "sing", "boat", "tuan"]
    vi = ["con", "cho", "an", "hay", "nhanh", "sinh", "quan", "huynh", "keng", "toan", "khuya", "ngoai", "giu", "nghe",
          "mình", "tiếp", "ăn", "ba"]
    assert not [w for w in not_vi if is_vietnamese_token(w)], [w for w in not_vi if is_vietnamese_token(w)]
    assert all(is_vietnamese_token(w) for w in vi), [w for w in vi if not is_vietnamese_token(w)]
    assert auto_tag("căn cước công dân gắn chip")["spans"] == ["chip"]  # trước đây lọt vào B5
    # --- Phase 0b, quy ước theo cách đọc: viết tắt ngắn đoán bằng heuristic -> cờ cho người duyệt
    assert "abbr_reading:crm" in auto_tag("anh mở cái crm lên nhé")["flags"]
    assert not any(f.startswith("abbr_reading") for f in auto_tag("anh mở cái kpi lên", cat_obj)["flags"])  # có trong catalog
    # --- Phase 0b T2: vi_abbrev — nguồn giả -> tools/build_wordlists.py -> auto_tag không gắn <en>
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
    run(G / "tools/build_wordlists.py", "--src", lex, "--catalog", cat, "--out-abbrev", T / "vi_abbrev.txt")
    ab_lines = (T / "vi_abbrev.txt").read_text(encoding="utf-8").splitlines()
    from converters.common import load_vi_words
    ab = load_vi_words([T / "vi_abbrev.txt"])
    assert ab == {"fpt", "plx", "shs", "bidv", "bab", "hn", "hcm", "thpt", "stt"}, sorted(ab)
    assert any(l.startswith("#-gas\t") and "google-10000" in l for l in ab_lines)  # trùng từ Anh phổ biến -> loại
    assert any(l.startswith("#-api\t") and "catalog" in l for l in ab_lines)       # trùng catalog -> loại
    assert any(l.startswith("#-pt\t") for l in ab_lines)                           # chữ cái đầu Phú Thọ, trùng 'pt'
    tg = auto_tag("mua fpt với bidv rồi gửi api", cat_obj, vi_words=ab)
    assert tg["spans"] == ["api"] and "vi_word:fpt" in tg["flags"], tg               # catalog vẫn thắng danh sách
    assert auto_tag("mua fpt", cat_obj, vi_words=set())["spans"] == ["fpt"]
    assert len(list(csv.reader(open(T / "reports/eval_spotcheck.csv", encoding="utf-8-sig")))) == 3
    print("TẤT CẢ KIỂM THỬ CONVERTER ĐẠT")


if __name__ == "__main__":
    main()
