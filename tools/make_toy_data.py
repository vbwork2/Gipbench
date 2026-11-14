#!/usr/bin/env python3
"""Sinh dữ liệu GIẢ LẬP để chạy thử toàn bộ bộ công cụ (không dùng cho kết quả thật).

  python tools/make_toy_data.py --out toy

Tạo: toy/bench.jsonl + audio giả (nhiễu + âm), toy/train_cs.jsonl (cài sẵn 3 lỗi rò rỉ để
validate.py bắt), và toy/results/{e0_baseline,e1_hotwords,e3_ft}/hyps.jsonl mô phỏng lỗi.
"""
import argparse
import json
import random
import zlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.textnorm import Catalog, basic_tokens, normalize_ref  # noqa: E402

FRAMES = [
    "hôm nay bên mình có {T} lúc ba giờ chiều nha",
    "anh gửi lại cái {T} cho em kiểm tra nhé",
    "em vừa cập nhật {T} xong rồi chị ơi",
    "cái {T} này dùng ổn không bạn",
    "tuần sau phải chốt {T} với khách hàng",
    "mình nghĩ là nên làm {T} trước rồi tính tiếp",
    "bạn đã đăng ký {T} chưa vậy",
    "{T} bên đó đang bị lỗi từ sáng tới giờ",
    "chị ơi cho em hỏi về {T} với {T2} một chút",
    "sau buổi {T} thì team gửi {T2} cho sếp",
]
VI_ONLY = [
    "hôm nay trời mưa to quá không đi đâu được",
    "anh ơi cho em hỏi thủ tục làm hộ chiếu",
    "chiều nay mẹ đi chợ mua rau với cá",
    "khách hàng muốn đổi địa chỉ giao hàng sang quận ba",
    "em đã chuyển khoản rồi nhưng chưa thấy xác nhận",
    "tổng đài viên sẽ gọi lại cho quý khách trong ngày",
    "năm nay con tôi thi vào lớp mười",
    "nhà mình cách trường khoảng hai cây số",
]
REGIONS = ["north", "central", "south"]
SPLIT_PLAN = {  # split: (số utt, số speaker, kênh chính, domain)
    "B1": (150, 45, "wideband", ["tech", "finance", "edu", "daily"]),
    "B2": (80, 18, "phone8k", ["product"]),
    "B3": (50, 22, "wideband", ["tech", "finance"]),
    "B4": (120, 14, "wideband", ["tech", "business", "edu"]),
    "B5": (80, 25, "phone8k", ["pubadmin", "daily"]),
}


def synth(path, dur, sr=16000, rng=None):
    import soundfile as sf
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = 0.02 * rng.standard_normal(n)
    for _ in range(int(dur * 3)):
        f0, a, s = rng.uniform(120, 260), rng.uniform(0.05, 0.2), rng.integers(0, max(1, n - sr // 4))
        e = min(n, s + sr // 4)
        x[s:e] += a * np.sin(2 * np.pi * f0 * t[s:e]) * np.hanning(e - s)
    if rng.random() < 0.05:
        x *= 6  # vài file bị clipping
    sf.write(path, np.clip(x, -1, 1).astype("float32"), sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy")
    ap.add_argument("--catalog", default=str(Path(__file__).resolve().parents[1] / "configs/catalog.example.tsv"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rnd, rng = random.Random(args.seed), np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cat = Catalog.load(args.catalog)
    normal = [" ".join(t) for t, v in cat.terms.items() if not v.reserved_b3]
    reserved = [" ".join(t) for t, v in cat.terms.items() if v.reserved_b3]
    weights = [1 / (i + 1) ** 0.9 for i in range(len(normal))]  # đuôi dài kiểu Zipf

    rows = []
    for split, (n, nspk, ch, doms) in SPLIT_PLAN.items():
        spks = [f"{split.lower()}_spk{i:03d}" for i in range(nspk)]
        spk_region = {s: rnd.choice(REGIONS) for s in spks}
        spk_gender = {s: rnd.choice("fm") for s in spks}
        for i in range(n):
            spk = rnd.choice(spks) if rnd.random() < 0.7 else spks[0]  # 1 speaker chiếm nhiều
            if split == "B5":
                tt, meta = rnd.choice(VI_ONLY), []
            else:
                pool = reserved if split == "B3" else normal
                t1 = rnd.choice(pool) if split == "B3" else rnd.choices(normal, weights)[0]
                t2 = rnd.choices(normal, weights)[0]
                fr = rnd.choice(FRAMES)
                tt = fr.replace("{T}", f"<en>{t1}</en>", 1).replace("{T2}", f"<en>{t2}</en>").replace("{T}", f"<en>{t1}</en>")
                meta = [{"viet_level": rnd.choices(["strong", "mid", "native"], [5, 3, 2])[0]} for _ in range(tt.count("<en>"))]
            ntok = len(basic_tokens(tt.replace("<en>", " ").replace("</en>", " ")))
            dur = round(min(25.0, max(1.2, ntok * rnd.uniform(0.22, 0.38))), 2)
            uid = f"{split.lower()}_{i:05d}"
            ap_ = out / "audio" / f"{uid}.wav"
            synth(ap_, dur, rng=rng)
            rows.append({"id": uid, "split": split, "audio": f"audio/{uid}.wav", "duration": dur,
                         "text_tagged": tt, "speaker": spk,
                         "source": f"{'call' if ch == 'phone8k' else 'yt'}:{split}_{i // 4:04d}",
                         "region": spk_region[spk], "gender": spk_gender[spk],
                         "channel": ch if rnd.random() < 0.85 else ("wideband" if ch == "phone8k" else "phone8k"),
                         "domain": rnd.choice(doms), "en_meta": meta})
    with open(out / "bench.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # train có cài 3 lỗi rò rỉ
    train = [
        {"id": "tr_1", "split": "train", "audio": "audio/b1_00000.wav", "duration": rows[0]["duration"],
         "text": "bên mình mới deploy kubernetes cho khách", "speaker": "tr_spk1", "source": "yt:other_1"},
        {"id": "tr_2", "split": "train", "audio": "audio/b1_00001.wav", "duration": 3.0,
         "text": "chiều nay họp team", "speaker": rows[5]["speaker"], "source": "yt:other_2"},
        {"id": "tr_3", "split": "train", "audio": "audio/b1_00002.wav", "duration": 3.0,
         "text": "gửi email cho chị nhé", "speaker": "tr_spk3", "source": rows[10]["source"]},
    ]
    with open(out / "train_cs.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # hypothesis mô phỏng
    phon = {}
    for line in open(args.catalog, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4 and p[0] != "term" and p[3]:
            phon[" ".join(basic_tokens(p[0]))] = p[3].split("|")[0]
    systems = {  # tên: (xác suất từ Anh sai, lỗi tiếng Việt, tỉ lệ chèn sai thuật ngữ ở B5, rtf)
        "e0_baseline": (0.62, 0.050, 0.00, 0.050),
        "e1_hotwords": (0.45, 0.051, 0.06, 0.056),
        "e3_ft": (0.24, 0.050, 0.01, 0.050),
    }
    vi_vocab = sorted({t for r in rows for t in basic_tokens(r["text_tagged"].replace("<en>", " ").replace("</en>", " "))})
    for sysname, (p_en, p_vi, p_ins, rtf) in systems.items():
        r2 = random.Random(zlib.crc32(sysname.encode()))
        d = out / "results" / sysname
        d.mkdir(parents=True, exist_ok=True)
        tot = 0.0
        with open(d / "hyps.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                toks, spans = normalize_ref(r["text_tagged"], cat)
                hard = 1.0 + (0.5 if r["split"] == "B3" else 0) + (0.3 if r["channel"] == "phone8k" else 0)
                lv = [m.get("viet_level") for m in r.get("en_meta", [])]
                out_toks, sp_done = [], set()
                for k, tk in enumerate(toks):
                    if tk.span >= 0:
                        if tk.span in sp_done:
                            continue
                        sp_done.add(tk.span)
                        term = " ".join(spans[tk.span])
                        boost = 1.25 if (tk.span < len(lv) and lv[tk.span] == "strong") else 0.85
                        if r2.random() < min(0.95, p_en * hard * boost):
                            if term in phon and r2.random() < 0.5:
                                out_toks += phon[term].split()[:1] if r2.random() < 0.5 else phon[term].split()
                            else:
                                out_toks.append(term[:-1] + "ơ")
                        else:
                            out_toks += term.split()
                    else:
                        out_toks.append(r2.choice(vi_vocab) if r2.random() < p_vi * hard else tk.tok)
                if r["split"] == "B5" and r2.random() < p_ins:
                    out_toks.insert(r2.randrange(len(out_toks) + 1), r2.choice(normal))
                f.write(json.dumps({"id": r["id"], "hyp": " ".join(out_toks).upper()}, ensure_ascii=False) + "\n")
                tot += r["duration"]
        (d / "decode_summary.json").write_text(json.dumps(
            {"system": sysname, "audio_sec": tot, "proc_sec": tot * rtf, "rtf": rtf, "note": "GIẢ LẬP"}, indent=1))
    print(f"Đã tạo {len(rows)} utterance giả lập trong {out}/")


if __name__ == "__main__":
    main()
