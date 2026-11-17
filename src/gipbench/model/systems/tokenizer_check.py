#!/usr/bin/env python3
"""
Kiểm tra tokenizer (BPE) của Gipformer 1.5 với từ tiếng Anh chen trong tiếng Việt.

Trả lời 3 câu hỏi:
  1. Vocab có biểu diễn được các chữ cái không có trong tiếng Việt (F, J, W, Z) không?
  2. Từ tiếng Anh bị xé thành bao nhiêu token so với từ tiếng Việt?
  3. Có cần mở rộng vocab (thí nghiệm E6) hay chỉ FT là đủ?

Tuỳ chọn: giải mã vài file audio có/không có hotwords để xem baseline E0 vs E1.

Cài đặt:
  pip install sentencepiece huggingface_hub            # phần tokenizer
  pip install sherpa-onnx sherpa-onnx-core soundfile    # thêm nếu dùng --audio

Ví dụ:
  python check_gipformer_tokenizer.py                       # tự tải bpe.model + tokens.txt từ HF
  python check_gipformer_tokenizer.py --model-dir ./gip15   # dùng thư mục model có sẵn
  python check_gipformer_tokenizer.py --terms my_terms.txt  # thêm danh mục thuật ngữ của bạn (1 dòng/thuật ngữ)
  python check_gipformer_tokenizer.py --audio a.wav b.wav   # thêm so sánh có/không hotwords
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO_ID = "g-group-ai-lab/gipformer1.5-68M-rnnt"
NON_VI_LETTERS = ["F", "J", "W", "Z"]

# --- Thuật ngữ tiếng Anh hay gặp trong lời nói tiếng Việt (mặc định) -------------------
EN_TERMS = {
    "tech": """facebook zalo youtube tiktok google iphone samsung laptop wifi bluetooth email
password online offline app website server cloud data database backend frontend deploy bug
fix code coding python javascript java api ui ux update upgrade download upload file folder
link click share like comment post livestream streamer game gaming chatgpt ai chatbot
smartphone tablet macbook windows android ios chip ram ssd camera zoom teams template
dashboard login logout account admin user test demo version feature framework docker
kubernetes github git commit merge sprint scrum agile jira notion figma canva shopee lazada
grab momo vnpay otp""".split(),
    "business_finance": """meeting deadline kpi okr ceo cto manager leader team teamwork
marketing sales brand branding startup freelancer intern hr ok okay report plan project
budget invoice voucher sale freeship cod order ship shipper crypto bitcoin ethereum blockchain
nft stock trading trader forex chart portfolio fintech visa mastercard cashback feedback
review slide excel powerpoint""".split(),
    "edu": """ielts toefl toeic gpa scholarship major workshop seminar mentor coaching homework
quiz essay speaking listening writing reading vocabulary grammar master phd bachelor""".split(),
    "daily": """menu combo size booking hotel resort taxi gym yoga spa makeup skincare outfit
style fashion shopping mall party sorry wow cool crush fan idol kpop show concert vlog vlogger
influencer content creator trend viral hashtag selfie photo video playlist spotify netflix
podcast setup unbox""".split(),
    "phrases": ["pull request", "internet banking", "mobile banking", "qr code", "flash sale",
                "part time", "full time", "credit card", "check in", "check out", "thank you",
                "online course", "smart home", "big data", "machine learning"],
}

# --- Từ tiếng Việt thông dụng để so sánh --------------------------------------------------
VI_TERMS = """tôi bạn chúng ta không được người việc làm công ty hôm nay ngày mai khách hàng
sản phẩm điện thoại tiền ngân hàng tài khoản thông tin hỗ trợ cảm ơn xin chào vâng dạ nhưng
khuyến mãi thanh toán giao hàng đăng ký nhanh chóng chuyện trường giáo dục học sinh đặc biệt
quyết định chứng khoán thị trường phát triển nguyễn hà nội sài gòn đà nẵng thuyền khuyết điểm
gì ở đâu nước thương nghĩa nghiêng chuyển khoản hợp đồng bảo hiểm lãi suất hạn mức mật khẩu
ứng dụng tin nhắn cuộc gọi tổng đài nhân viên trưởng phòng báo cáo kế hoạch dự án ngân sách
hoá đơn đơn hàng vận chuyển giảm giá miễn phí đổi trả bảo hành sửa chữa cài đặt tải về đăng nhập
""".split()


def load_files(model_dir):
    if model_dir:
        d = Path(model_dir)
        return d / "bpe.model", d / "tokens.txt", d
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Cần: pip install huggingface_hub  (hoặc dùng --model-dir)")
    bpe = Path(hf_hub_download(REPO_ID, "bpe.model"))
    tok = Path(hf_hub_download(REPO_ID, "tokens.txt"))
    return bpe, tok, bpe.parent


def detect_case(sp):
    up = low = 0
    for i in range(sp.get_piece_size()):
        p = sp.id_to_piece(i)
        up += sum(c.isupper() for c in p)
        low += sum(c.islower() for c in p)
    return "upper" if up >= low else "lower"


def apply_case(s, case):
    return s.upper() if case == "upper" else s.lower()


def analyse(sp, terms, case):
    rows = []
    unk = sp.unk_id()
    for t in terms:
        tc = apply_case(t, case)
        ids = sp.encode(tc)
        pieces = [sp.id_to_piece(i) for i in ids]
        words = tc.split()
        rows.append({
            "term": tc,
            "pieces": " ".join(pieces),
            "n_pieces": len(ids),
            "pieces_per_word": len(ids) / max(1, len(words)),
            "has_unk": unk in ids,
            "roundtrip_ok": sp.decode(ids) == tc,
            "chars_per_piece": len(tc.replace(" ", "")) / max(1, len(ids)),
        })
    return rows


def summary(rows):
    ppw = [r["pieces_per_word"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "mean_pieces_per_word": statistics.mean(ppw),
        "median_pieces_per_word": statistics.median(ppw),
        "pct_single_piece": 100 * sum(r["n_pieces"] == 1 for r in rows) / n,
        "pct_ge4_per_word": 100 * sum(x >= 4 for x in ppw) / n,
        "pct_unk": 100 * sum(r["has_unk"] for r in rows) / n,
        "pct_roundtrip_fail": 100 * sum(not r["roundtrip_ok"] for r in rows) / n,
        "mean_chars_per_piece": statistics.mean(r["chars_per_piece"] for r in rows),
    }


def export_bpe_vocab(sp, path):
    # Định dạng sherpa-onnx cần cho --modeling-unit bpe: "piece<TAB>score"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")


def decode_audio(model_dir, tokens, bpe_vocab, hotwords_file, audios, quant, score):
    import sherpa_onnx
    import soundfile as sf
    sfx = ".int8.onnx" if quant == "int8" else ".onnx"
    d = Path(model_dir)
    files = {k: d / f"{k}{sfx}" for k in ("encoder", "decoder", "joiner")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        sys.exit("Thiếu file ONNX: " + ", ".join(missing) + "  (dùng --model-dir trỏ tới thư mục model đầy đủ)")

    def make(hot):
        kw = dict(encoder=str(files["encoder"]), decoder=str(files["decoder"]),
                  joiner=str(files["joiner"]), tokens=str(tokens), num_threads=2,
                  decoding_method="modified_beam_search", max_active_paths=4)
        if hot:
            kw.update(hotwords_file=str(hotwords_file), hotwords_score=score,
                      modeling_unit="bpe", bpe_vocab=str(bpe_vocab))
        return sherpa_onnx.OfflineRecognizer.from_transducer(**kw)

    base, hot = make(False), make(True)
    print("\n=== Giải mã audio: không hotwords (E0) vs có hotwords (E1), score =", score)
    for a in audios:
        samples, sr = sf.read(a, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        out = []
        for rec in (base, hot):
            s = rec.create_stream()
            s.accept_waveform(sr, samples)
            rec.decode_stream(s)
            out.append(s.result.text)
        print(f"\n[{a}]\n  E0: {out[0]}\n  E1: {out[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="thư mục chứa bpe.model, tokens.txt (và *.onnx nếu dùng --audio)")
    ap.add_argument("--terms", help="file thuật ngữ tiếng Anh của bạn, 1 dòng/thuật ngữ")
    ap.add_argument("--out", default="tokenizer_report", help="thư mục ghi kết quả")
    ap.add_argument("--audio", nargs="*", help="(tuỳ chọn) file audio để so E0 vs E1")
    ap.add_argument("--quantize", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--hotwords-score", type=float, default=1.5)
    args = ap.parse_args()

    import sentencepiece as spm
    bpe_path, tokens_path, model_dir = load_files(args.model_dir)
    sp = spm.SentencePieceProcessor(model_file=str(bpe_path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case = detect_case(sp)
    n_tokens_txt = sum(1 for _ in open(tokens_path, encoding="utf-8"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    byte_fallback = any(p.startswith("<0x") for p in pieces)

    print("=== 1. Tổng quan vocab")
    print(f"  bpe.model: {sp.get_piece_size()} piece | tokens.txt: {n_tokens_txt} dòng | chữ: {case} | byte_fallback: {byte_fallback}")

    print("\n=== 2. Chữ cái không có trong tiếng Việt")
    letters_bad = []
    for L in NON_VI_LETTERS:
        Lc = apply_case(L, case)
        n_with = sum(Lc in p for p in pieces)
        ids = sp.encode(apply_case(L + "A" + L, case))
        is_unk = sp.unk_id() in ids
        if is_unk:
            letters_bad.append(L)
        print(f"  {L}: có trong {n_with} piece | encode ra <unk>: {'CÓ' if is_unk else 'không'}")

    en_terms = [t for cat in EN_TERMS.values() for t in cat]
    if args.terms:
        en_terms += [l.strip() for l in open(args.terms, encoding="utf-8") if l.strip()]
    en_terms = list(dict.fromkeys(en_terms))
    en_rows, vi_rows = analyse(sp, en_terms, case), analyse(sp, VI_TERMS, case)
    en_s, vi_s = summary(en_rows), summary(vi_rows)

    print("\n=== 3. Mức độ xé vụn: tiếng Anh vs tiếng Việt")
    labels = {
        "n": "số thuật ngữ", "mean_pieces_per_word": "token/từ (TB)", "median_pieces_per_word": "token/từ (trung vị)",
        "pct_single_piece": "% thuật ngữ = 1 token", "pct_ge4_per_word": "% từ ≥ 4 token",
        "pct_unk": "% có <unk>", "pct_roundtrip_fail": "% decode lại sai", "mean_chars_per_piece": "ký tự/token (TB)",
    }
    print(f"  {'':24s}{'tiếng Anh':>12s}{'tiếng Việt':>12s}")
    for k, lab in labels.items():
        print(f"  {lab:24s}{en_s[k]:12.2f}{vi_s[k]:12.2f}")

    worst = sorted(en_rows, key=lambda r: (-r["has_unk"], -r["pieces_per_word"]))[:25]
    print("\n  25 thuật ngữ bị xé nặng nhất:")
    for r in worst:
        flag = " <unk>" if r["has_unk"] else ""
        print(f"    {r['term']:18s} {r['n_pieces']:2d} token  {r['pieces']}{flag}")

    # Ghi file
    for name, rows in (("en_terms.csv", en_rows), ("vi_terms.csv", vi_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    ok_terms = [r["term"] for r in en_rows if not r["has_unk"] and r["roundtrip_ok"]]
    (out / "hotwords.txt").write_text("\n".join(ok_terms) + "\n", encoding="utf-8")
    export_bpe_vocab(sp, out / "bpe.vocab")

    # Kết luận
    print("\n=== 4. Kết luận")
    if letters_bad or en_s["pct_unk"] > 5:
        print(f"  ĐỎ: vocab không biểu diễn được {', '.join(letters_bad) or 'một phần'} "
              f"({en_s['pct_unk']:.0f}% thuật ngữ ra <unk>). Model KHÔNG THỂ viết đúng các từ này dù FT bao nhiêu:")
        print("  → E6 (mở rộng BPE + khởi tạo lại embedding/joiner output) là bắt buộc trước E3,")
        print("    hoặc tạm thời chấp nhận phiên âm và dựa vào hậu xử lý (E8) để đổi về chính tả Anh.")
    elif en_s["mean_pieces_per_word"] >= 3.0 or en_s["pct_ge4_per_word"] >= 30:
        print(f"  VÀNG: biểu diễn được nhưng xé vụn ({en_s['mean_pieces_per_word']:.1f} token/từ, "
              f"tiếng Việt {vi_s['mean_pieces_per_word']:.1f}).")
        print("  → Chạy E3 trước; nếu CS-WER chững lại thì E6 là ứng viên mạnh. Hotwords vẫn dùng được.")
    else:
        print(f"  XANH: tiếng Anh {en_s['mean_pieces_per_word']:.1f} token/từ, không có <unk>. Vocab không phải nút thắt;")
        print("  → bỏ E6, tập trung E1 (hotwords) và E3 (FT trên CS thật).")
    print(f"\n  Đã ghi: {out}/en_terms.csv, vi_terms.csv, hotwords.txt ({len(ok_terms)} thuật ngữ), bpe.vocab")

    if args.audio:
        decode_audio(model_dir, tokens_path, out / "bpe.vocab", out / "hotwords.txt",
                     args.audio, args.quantize, args.hotwords_score)


if __name__ == "__main__":
    main()
