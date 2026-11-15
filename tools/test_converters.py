#!/usr/bin/env python3
"""Kiểm thử các converter bằng dataset GIẢ có đúng schema của bản thật (không cần mạng).

  python tools/test_converters.py [thư_mục_tạm]
"""
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

G = Path(__file__).resolve().parents[1]
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

    cat = G / "configs/catalog.example.tsv"
    raw = T / "raw"
    run(G / "converters/conv_ggroup.py", "--names", "tech", "finance", "pubadmin", "--out", raw / "ggroup", "--catalog", cat, "--parquet", dl / "ggroup_{name}.parquet", "--vi-words")
    loan = T / "vi_loanwords_test.txt"  # danh sách cố định cho test, không phụ thuộc configs/
    loan.write_text("word\tkind\tstatus\tsource\naxit\tvi_spelling\táp dụng\ttest\nlipid\tsame_surface\tchờ duyệt\ttest\n"
                    "vitamin\tsame_surface\tchờ duyệt\ttest\n", encoding="utf-8")
    run(G / "converters/conv_vimedcss.py", "--out", raw / "vimedcss", "--catalog", cat, "--parquet", dl / "vimedcss_{split}.parquet",
        "--vi-words", loan)
    run(G / "converters/conv_sea.py", "--out", raw / "sea", "--catalog", cat, "--parquet", dl / "sea.parquet", "--bench-hours", "0.002", "--vi-words")
    run(G / "converters/conv_local.py", "vivos", "--root", dl / "vivos", "--out", raw / "vivos", "--vi-words")
    run(G / "converters/conv_local.py", "commonvoice", "--root", cv, "--out", raw / "cv", "--vi-words")
    run(G / "converters/conv_local.py", "vietsuperspeech", "--root", vss, "--out", raw / "vss", "--vi-words")

    def L(p):
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    b1 = L(raw / "ggroup/b1_candidates.jsonl")
    assert len(b1) == 5 and any("<en>connect metamask</en>" in r["text_tagged"] for r in b1), b1
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
    from converters.common import iter_hf_parquet, load_vi_words
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
    # các lần chạy converter ở trên dùng --vi-words rỗng để không phụ thuộc danh sách đang chờ duyệt;
    # danh sách mặc định trong configs/ vẫn phải đọc được
    assert {"bidv", "axit"} <= load_vi_words(), "configs/vi_abbrev.txt / vi_loanwords.txt thiếu mục cơ bản"
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
    # --- Phase 0b T3: vi_loanwords — vi_spelling (mọi bộ) / same_surface (chỉ ViMedCSS) / --exclude / google-10000
    (lex / "Viet74K.txt").write_text("axit\ncanxi\nvitamin\niot\nmega\ncon\nHà Nội\na-xít\n", encoding="utf-8")
    (lex / "words_alpha.txt").write_text("vitamin\nlipid\n", encoding="utf-8")
    (lex / "google-10000-english-no-swears.txt").write_text("the\ngas\npt\nmega\n", encoding="utf-8")
    vm_meta = lex / "vmc"
    vm_meta.mkdir(exist_ok=True)
    with open(vm_meta / "train_set.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["segment_id", "segment_text", "cs_terms_list"])
        cw.writerow(["a", "uống vitamin và lipid mỗi ngày", "vitamin; lipid"])
    run(G / "tools/build_wordlists.py", "--src", lex, "--vimedcss-meta", vm_meta, "--out-loan", T / "vi_loan.txt",
        "--exclude", "iot:IoT trong tech")
    lo_lines = (T / "vi_loan.txt").read_text(encoding="utf-8").splitlines()
    assert load_vi_words([T / "vi_loan.txt"]) == {"axit", "canxi"}, load_vi_words([T / "vi_loan.txt"])
    assert load_vi_words([T / "vi_loan.txt"], same_surface=True) == {"axit", "canxi", "vitamin"}  # lipid không có trong Viet74K
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
    out_v = run(G / "converters/conv_local.py", "vietsuperspeech", "--root", vss2, "--out", raw / "vss2", "--catalog", cat,
                "--bench", bench_proxy, "--vi-words")
    cs = L(raw / "vss2/vss_cs_candidates.jsonl")
    assert sorted(r["id"] for r in cs) == ["vss_0000001", "vss_0000004"], [r["id"] for r in cs]
    assert all(r["dataset"] == "vss_cs" and r["train_only"] and r["split"] == "train" and "<en>" in r["text_tagged"] for r in cs)
    assert len(L(raw / "vss2/train_vss.jsonl")) == 1 and "vss_cs bỏ: text trùng câu bench: 1" in out_v
    assert "vss_cs bỏ: có thuật ngữ reserved_b3: 1" in out_v and "vss_cs bỏ: câu có từ Anh ở phần dev: 1" in out_v
    assert (raw / "vss2/vss_cs_terms.tsv").read_text(encoding="utf-8").startswith("term\tcount\n")
    # --- Phase 0b T4: lọc đồng thuận (lib/agreement.py + tools/agreement_filter.py)
    from lib.agreement import agree
    lab = "mở <en>facebook</en> rồi <en>check</en> mail"
    hy = ["MỞ FB RỒI CHECK MAIL", "mở phây búc rồi check mail"]  # fb = biến thể catalog của facebook
    assert agree(lab, hy, cat_obj, 1) == {"keep": True, "spans": ["facebook", "check"], "confirmed": [1, 2]}
    assert not agree(lab, hy, cat_obj, 2)["keep"] and agree(lab, hy, cat_obj, 2, min_span_frac=0.5)["keep"]
    assert not agree("hôm nay trời đẹp", hy, cat_obj)["keep"]  # không có span -> không có gì để xác nhận
    ha, hb = T / "hyp_a.jsonl", T / "hyp_b.jsonl"
    ha.write_text('{"id": "vss_0000001", "hyp": "MÌNH MỞ FACEBOOK LÊN XEM NHÉ"}\n{"id": "vss_0000004", "hyp": "CHẠY ĐÉT LAI XONG"}\n',
                  encoding="utf-8")
    hb.write_text('{"id": "vss_0000001", "hyp": "mình mở phây búc lên"}\n', encoding="utf-8")
    run(G / "tools/agreement_filter.py", "--manifest", raw / "vss2/vss_cs_candidates.jsonl", "--hyps", ha, hb,
        "--catalog", cat, "--out", T / "vss_cs_agreed.jsonl")
    ag = L(T / "vss_cs_agreed.jsonl")
    assert [r["id"] for r in ag] == ["vss_0000001"] and ag[0]["agreement"]["confirmed"] == [1], ag
    # --- Phase 0c: gán lại TỪNG span (lib.agreement.relabel_spans, agreement_filter --relabel-spans)
    from lib.agreement import relabel_spans
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
    from lib.agreement import span_rule_violation
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
    run(G / "tools/agreement_filter.py", "--manifest", rel_m, "--hyps", T / "rq.jsonl", T / "rw.jsonl", "--catalog", cat,
        "--out", T / "rel_filter.jsonl", "--write-unconfirmed-ids", T / "rel_unconf.txt")
    assert L(T / "rel_filter.jsonl") == [] and (T / "rel_unconf.txt").read_text(encoding="utf-8").split() == ["r1", "r2"]
    out_rl = run(G / "tools/agreement_filter.py", "--manifest", rel_m, "--hyps", T / "rq.jsonl", T / "rw.jsonl", "--catalog", cat,
                 "--relabel-spans", "--listen-csv", T / "listen.csv", "--out", T / "rel_relabel.jsonl")
    rr = L(T / "rel_relabel.jsonl")
    assert [r["id"] for r in rr] == ["r1"] and rr[0]["relabeled"] and rr[0]["text_orig"] == lab, rr  # r2: rò rỉ sau khi gán lại
    assert rr[0]["text"].startswith("với cái network của dương") and "<en>" not in rr[0]["text"]
    assert "rò rỉ sau khi gán lại: có thuật ngữ reserved_b3 | 1" in (T / "rel_relabel.jsonl.report.md").read_text(encoding="utf-8")
    lis = list(csv.DictReader(open(T / "listen.csv", encoding="utf-8-sig")))
    assert len(lis) == 1 and lis[0]["span_cu_moi"] == "netword -> network" and lis[0]["audio_url"].endswith("audio/x.wav"), lis
    assert "gán lại thêm 1 câu" in out_rl
    # --- Phase 0d T3: ngưỡng WER tiếng Việt cho nhánh chỉ lọc (min theo các ASR có hyp, mặc định 0.15)
    from lib.agreement import vi_wer
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
    run(G / "tools/agreement_filter.py", "--manifest", vm, "--hyps", T / "vq.jsonl", T / "vw.jsonl", "--catalog", cat,
        "--out", T / "viwer_015.jsonl")
    v15 = L(T / "viwer_015.jsonl")
    assert [r["id"] for r in v15] == ["v1"] and abs(v15[0]["agreement"]["vi_wer"] - 0.1429) < 1e-4, v15  # v2: 0.38 > 0.15
    assert "1 câu trong khoảng 0.10–0.15" in (T / "viwer_015.jsonl.report.md").read_text(encoding="utf-8")
    run(G / "tools/agreement_filter.py", "--manifest", vm, "--hyps", T / "vq.jsonl", T / "vw.jsonl", "--catalog", cat,
        "--max-vi-wer", "0.10", "--out", T / "viwer_010.jsonl")
    assert L(T / "viwer_010.jsonl") == []
    # --- Phase 0d T4: cache hyps theo id + hash audio, backfill hyps cũ, rebuild_vss_cs.sh chỉ từ cache
    from lib.hyp_cache import HashIndex, HypLookup, file_sha256
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
    out_bf = run(G / "tools/hyp_cache.py", "backfill", "--manifest", c_m, "--hyps", c4 / "hq", c4 / "hw",
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
    out_rb = subprocess.run(["bash", str(G / "tools/rebuild_vss_cs.sh"), "--bench", str(bench_proxy), "--catalog", str(cat),
                             "--candidates", str(c_m), "--out-dir", str(c4 / "rb"), "--hyps", f"{c4 / 'hq'} {c4 / 'hw'}",
                             "--hash-index", str(idx), "--audio-root", str(c4 / "root")],
                            capture_output=True, text=True, env={**os.environ, "PY": sys.executable})
    assert out_rb.returncode == 0, out_rb.stdout + out_rb.stderr
    assert sorted(r["id"] for r in L(c4 / "rb/vss_cs_clean.jsonl")) == ["a1", "a4"], out_rb.stdout
    rb_md = (c4 / "rb/vss_cs_clean.jsonl.report.md").read_text(encoding="utf-8")
    assert "chưa có hyp tầng 1: 1" in rb_md and "chờ tầng 2 (còn span chưa xác nhận, thiếu hyp tầng sau): 1" in rb_md, rb_md
    assert "Giữ **2 / 2**" in rb_md
    # --- Phase 0c: lọc rẻ trước ASR (tools/vss_cs_prefilter.py)
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
    out_p = run(G / "tools/vss_cs_prefilter.py", "--manifest", pre_m, "--catalog", cat, "--term-cap", "2",
                "--out", T / "pre_out.jsonl", "--sample", "1", "--sample-out", T / "pre_pilot.jsonl")
    po = L(T / "pre_out.jsonl")
    assert len(po) == 2 and "p5" in [r["id"] for r in po] and all(r["id"] in ("p2", "p3", "p4", "p5") for r in po), po
    assert "--max-en-frac 0.5: 6 câu" in out_p and "rò rỉ B1–B3 + reserved_b3: 5 câu" in out_p, out_p
    assert "bỏ câu chỉ có thuật ngữ 1 lần (giữ 10%): 4 câu" in out_p, out_p
    run(G / "tools/vss_cs_prefilter.py", "--manifest", pre_m, "--catalog", cat, "--term-cap", "0", "--hapax-keep-frac", "1",
        "--out", T / "pre_out_nohapax.jsonl")
    assert "p7" in [r["id"] for r in L(T / "pre_out_nohapax.jsonl")]  # --hapax-keep-frac 1: không lọc
    assert len(L(T / "pre_pilot.jsonl")) == 1 and po[0]["prefilter"]["en_frac"] <= 0.5
    # --- Phase 0d T5: cấu hình trộn: vss_cs thuộc nhóm riêng cs_pseudo (audio thật, nhãn giả), không thuộc cs_real
    mix = {r["group"]: r for r in csv.DictReader((l for l in open(G / "configs/train_mix.tsv", encoding="utf-8")
                                                   if not l.startswith("#")), delimiter="\t")}
    assert mix["cs_pseudo"]["datasets"].split(",") == ["vss_cs"] and "vss_cs" not in mix["cs_real"]["datasets"]
    assert "audio thật, nhãn giả" in mix["cs_pseudo"]["note"]
    assert {k: float(v["weight"]) for k, v in mix.items()} == {"cs_real": 0.35, "cs_pseudo": 0.10, "synth": 0.05,
                                                               "replay_human": 0.25, "distill": 0.20, "confusable": 0.05}
    assert abs(sum(float(r["weight"]) for r in mix.values()) - 1.0) < 1e-9
    out_mp = run(G / "tools/mix_plan.py", "--manifest", raw / "vss2/vss_cs_candidates.jsonl", "--assume", "ViMedCSS:medical:24.3")
    assert "**CẢNH BÁO:** domain medical (ViMedCSS) chiếm 77.8% phần CS (cs_real + cs_pseudo) > trần 35%" in out_mp, out_mp
    assert "| cs_pseudo | 0.10 | vss_cs |" in out_mp and "| conversation | 22.2% |" in out_mp, out_mp
    bad_mix = subprocess.run([sys.executable, str(G / "tools/mix_plan.py")], capture_output=True, text=True,
                             env={**os.environ, "MIX_CS_REAL": "0.5"})
    assert bad_mix.returncode == 1 and "tổng tỉ trọng = 1.1500" in bad_mix.stderr, bad_mix.stderr
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
        run(G / "score.py", "--bench", sb / "bench.jsonl", "--catalog", cat, "--hyps", sb / "hyps.jsonl",
            "--out", sb / lf, "--label-field", lf, "--iters", "10")
    s_spec = json.loads((sb / "text_tagged/summary.json").read_text(encoding="utf-8"))["groups"]["B3"]
    s_orig = json.loads((sb / "text_tagged_orig/summary.json").read_text(encoding="utf-8"))["groups"]["B3"]
    assert (s_spec["n_en"], s_spec["cs_wer"]) == (1, 0.0), s_spec          # vitamin là tiếng Việt: lỗi tính vào N-WER
    assert s_orig["n_en"] == 2 and s_orig["cs_wer"] > 0, s_orig            # theo dataset: vitamin là span EN bị sai
    assert json.loads((sb / "text_tagged_orig/summary.json").read_text(encoding="utf-8"))["groups"]["B1"]["cs_wer"] == 0.0
    assert len(list(csv.reader(open(T / "reports/eval_spotcheck.csv", encoding="utf-8-sig")))) == 3
    print("TẤT CẢ KIỂM THỬ CONVERTER ĐẠT")


if __name__ == "__main__":
    main()
