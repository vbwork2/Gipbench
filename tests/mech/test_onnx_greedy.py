#!/usr/bin/env python3
"""Decoder greedy có blank penalty theo điều kiện (gipformer_mech.onnx_greedy) và fbank Kaldi bằng NumPy
(gipformer_mech.fbank). CPU, session ONNX giả, không cần model.

  python tests/mech/test_onnx_greedy.py      # thoát mã 1 nếu có ca hỏng
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from gipformer_mech.fbank import fbank, mel_banks
from gipformer_mech.onnx_greedy import OnnxTransducer, condition, continuation_mask, pieces_to_text, read_pieces

FAIL = []
PIECES = ["<blk>", "<unk>", "▁SMI", "T", "H", "▁A"]   # 0 blank, 1 unk, 2 đầu từ, 3-4 nối từ, 5 đầu từ
BLANK, SMI, T, H, A = 0, 2, 3, 4, 5


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def logit(blank, **others):
    row = np.full(len(PIECES), -10.0, dtype=np.float32)
    row[BLANK] = blank
    for name, value in others.items():
        row[{"smi": SMI, "t": T, "h": H, "a": A, "unk": 1}[name]] = value
    return row


class Session:
    def __init__(self, fn):
        self.fn = fn

    def run(self, _outputs, feeds):
        return [self.fn(feeds)]


def fake_model(frames):
    """Joiner trả logit của frame theo thứ tự; decoder mã hoá ngữ cảnh để kiểm cập nhật trạng thái."""
    model = object.__new__(OnnxTransducer)
    model.pieces, model.blank, model.unk, model.context_size = PIECES, BLANK, 1, 2
    model.continuation = continuation_mask(PIECES, BLANK)
    model._decoder_cache = {}
    model.contexts = []
    model.decoder = Session(lambda feeds: feeds["y"].astype(np.float32))
    position = iter(range(len(frames)))

    def joiner(feeds):
        model.contexts.append(tuple(int(v) for v in feeds["decoder_out"][0]))
        return frames[next(position)][None]

    model.joiner = Session(joiner)
    return model


def encoder(n):
    return np.zeros((n, 4), dtype=np.float32)


def test_helpers():
    d = Path(tempfile.mkdtemp())
    (d / "tokens.txt").write_text("".join(f"{p} {i}\n" for i, p in enumerate(PIECES)), encoding="utf-8")
    check("read_pieces: đọc tokens.txt theo ID", read_pieces(d / "tokens.txt") == PIECES)
    (d / "bad.txt").write_text("<blk> 0\nA 2\n", encoding="utf-8")
    try:
        read_pieces(d / "bad.txt")
        check("read_pieces: ID không liên tục -> ValueError", False)
    except ValueError:
        check("read_pieces: ID không liên tục -> ValueError", True)
    mask = continuation_mask(PIECES, BLANK)
    check("continuation_mask: chỉ piece không có ▁ và không phải <...>", mask.tolist() == [False, False, False, True, True, False])
    check("pieces_to_text", pieces_to_text([SMI, T, H, A], PIECES) == "SMITH A")
    c = continuation_mask(PIECES, BLANK)
    check("condition continuation: ứng viên nối từ + đã có token", condition("continuation", logit(1, h=0.5), 1, c, BLANK))
    check("condition continuation: chưa có token -> tắt", not condition("continuation", logit(1, h=0.5), 0, c, BLANK))
    check("condition continuation: ứng viên đầu từ -> tắt", not condition("continuation", logit(1, a=0.5, h=0.1), 1, c, BLANK))
    check("condition word_start: ứng viên đầu từ", condition("word_start", logit(1, a=0.5, h=0.1), 0, c, BLANK))
    check("condition uniform: luôn bật", condition("uniform", logit(1), 0, c, BLANK))


def test_greedy():
    frames = [logit(0, smi=1), logit(1, t=0.8), logit(1, h=-1.0), logit(1, a=0.9)]
    base = fake_model(frames).greedy(encoder(4))
    check("greedy λ=0: chỉ phát token thắng blank", base["ids"] == [SMI], base["ids"])
    cont = fake_model(frames).greedy(encoder(4), penalty=0.5, gate="continuation")
    check("continuation λ=0.5: cứu T (thua 0.2), bỏ H (thua 2.0), không cứu ▁A",
          cont["ids"] == [SMI, T], cont["ids"])
    check("continuation: đếm frame bị phạt", cont["penalized_frames"] == 2 and cont["condition_frames"] == 2, cont)
    uni = fake_model(frames).greedy(encoder(4), penalty=0.5, gate="uniform")
    check("uniform λ=0.5: cứu cả ▁A", uni["ids"] == [SMI, T, A], uni["ids"])
    ws = fake_model(frames).greedy(encoder(4), penalty=0.5, gate="word_start")
    check("word_start λ=0.5: chỉ cứu ▁A", ws["ids"] == [SMI, A], ws["ids"])
    model = fake_model(frames)
    model.greedy(encoder(4), penalty=0.5, gate="continuation")
    check("ngữ cảnh khởi đầu (-1, blank), cập nhật sau mỗi token",
          model.contexts[:3] == [(-1, BLANK), (BLANK, SMI), (SMI, T)], model.contexts)
    unk = fake_model([logit(0, unk=2), logit(0, smi=1)]).greedy(encoder(2))
    check("unk không được phát", unk["ids"] == [SMI], unk["ids"])
    r1 = fake_model(frames).greedy(encoder(4), 0.5, "random", np.random.default_rng([1, 7]), 0.5)
    r2 = fake_model(frames).greedy(encoder(4), 0.5, "random", np.random.default_rng([1, 7]), 0.5)
    check("random: cùng seed -> cùng kết quả", r1["ids"] == r2["ids"] and r1["penalized_frames"] == r2["penalized_frames"])
    r0 = fake_model(frames).greedy(encoder(4), 0.5, "random", np.random.default_rng(3), 0.0)
    check("random rate 0 = baseline", r0["ids"] == base["ids"] and r0["penalized_frames"] == 0)
    try:
        fake_model(frames).greedy(encoder(4), 0.5, "random")
        check("random thiếu rng -> ValueError", False)
    except ValueError:
        check("random thiếu rng -> ValueError", True)


def test_fbank():
    rate = 16000
    t = np.arange(rate) / rate
    feats = fbank(0.3 * np.sin(2 * np.pi * 1000 * t), rate, 80)
    check("fbank: số frame snip_edges=False = (N + 80) // 160", feats.shape == (100, 80), feats.shape)
    check("fbank: float32, hữu hạn", feats.dtype == np.float32 and np.isfinite(feats).all())
    banks = mel_banks(80, 512, rate, 20.0, -400.0)
    peak_bin = int(np.argmax(banks[:, int(1000 * 512 / rate)]))
    check("fbank: đỉnh của sin 1 kHz nằm ở bin mel phủ 1 kHz", abs(int(np.median(feats.argmax(1))) - peak_bin) <= 1,
          (int(np.median(feats.argmax(1))), peak_bin))
    check("mel_banks: không có trọng số trên 7600 Hz", not banks[:, int(7600 * 512 / rate) + 1:].any())
    check("fbank: audio rỗng -> 0 frame", fbank(np.zeros(10), rate).shape == (0, 80))
    loud = fbank(0.6 * np.sin(2 * np.pi * 1000 * t), rate, 80)
    above = feats > np.log(np.finfo(np.float32).eps) + 1
    check("fbank: gấp đôi biên độ -> log energy +ln 4 (bin trên ngưỡng sàn)",
          above.any() and np.allclose((loud - feats)[above], np.log(4), atol=1e-3),
          float(np.abs((loud - feats)[above] - np.log(4)).max()))
    floor = float(np.log(np.finfo(np.float32).eps))
    check("fbank: bin năng lượng 0 kẹp ở log(FLT_EPSILON)", np.allclose(fbank(np.zeros(rate), rate), floor))


def main():
    test_helpers()
    test_greedy()
    test_fbank()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
