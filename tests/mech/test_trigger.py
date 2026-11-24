#!/usr/bin/env python3
"""Gate và replay có gate (gipformer_mech.trigger). CPU, model giả 4 chiều, không cần ONNX.

  python tests/mech/test_trigger.py      # thoát mã 1 nếu có ca hỏng
"""
import sys

import numpy as np

from gipformer_mech import joiner_probe as jp
from gipformer_mech import trigger as tg

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


class Toy:
    """blank = 0, piece = 1, unk = 2; decoder_out luôn bằng 0 nên h = tanh(enc)."""
    blank, unk, context_size = 0, 2, 2

    def decoder_out(self, context):
        return np.zeros((1, 4), dtype=np.float32)


WEIGHT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -5]], dtype=np.float32)
BIAS = np.zeros(3, dtype=np.float32)
# frame 0: blank thắng rõ; frame 1: blank thắng sát (margin ~0.08); frame 2: blank thắng rõ
ENC = np.array([[2.0, 0.0, 0, 0], [0.5, 0.4, 0, 0], [2.0, 0.0, 0, 0]], dtype=np.float32)


def test_replay():
    model = Toy()
    base = tg.gated_replay(model, ENC, WEIGHT, BIAS, None, 0.0)
    check("không gate -> chỉ blank", base["ids"] == [] and base["fired"] == 0, base)
    plain = jp.replay(model, ENC, WEIGHT, BIAS)
    check("penalty 0 khớp joiner_probe.replay", [int(t) for t in plain["tokens"] if t not in (0, 2)] == base["ids"])
    margin = tg.Gate("margin", -0.5)
    out = tg.gated_replay(model, ENC, WEIGHT, BIAS, margin, 1.0)
    check("gate margin bật ở frame sát -> phát piece", out["ids"] == [1] and out["fired"] == 1, out)
    out = tg.gated_replay(model, ENC, WEIGHT, BIAS, margin, 0.0)
    check("penalty 0 -> gate không có tác dụng", out["ids"] == [] and out["fired"] == 0, out)
    never = tg.Gate("probe", 1e9, np.ones(4), 0.0, np.zeros(4), np.ones(4))
    check("tau rất lớn -> không bao giờ bật", tg.gated_replay(model, ENC, WEIGHT, BIAS, never, 8.0)["ids"] == [])
    h = np.tanh(ENC)
    scores = margin.scores(h, h @ WEIGHT.T + BIAS, 0)
    check("điểm margin = piece tốt nhất − blank", np.allclose(scores, h[:, 1] - h[:, 0]), str(scores))


def test_threshold_and_probe():
    scores = np.arange(1000, dtype=float)
    tau = tg.threshold_for_rate(scores, 0.01)
    check("rate 1% -> đúng 10 điểm trên tau", int((scores > tau).sum()) == 10, tau)
    for bad in (0.0, 1.0):
        try:
            tg.threshold_for_rate(scores, bad)
            check(f"rate {bad} -> ValueError", False)
        except ValueError:
            check(f"rate {bad} -> ValueError", True)
    rng = np.random.default_rng(0)
    pos = rng.normal(1.5, 1.0, (200, 8))
    neg = rng.normal(-1.5, 1.0, (2000, 8))
    w, c, mean, std = tg.fit_probe(pos, neg)
    gate = tg.Gate("probe", 0.0, w, c, mean, std)
    acc = ((gate.scores(pos, None, 0) > 0).mean() + (gate.scores(neg, None, 0) <= 0).mean()) / 2
    check("probe tách hai cụm (độ chính xác cân bằng > 0,95)", acc > 0.95, acc)
    r = tg.random_direction(8, 1)
    check("hướng ngẫu nhiên: chuẩn 1, cùng seed cùng kết quả",
          abs(np.linalg.norm(r) - 1) < 1e-9 and np.array_equal(r, tg.random_direction(8, 1)))
    check("fold theo văn bản: tất định, trong {0, 1}",
          tg.text_fold("XIN CHAO") == tg.text_fold("XIN CHAO") and {tg.text_fold(str(i)) for i in range(50)} == {0, 1})


def main():
    test_replay()
    test_threshold_and_probe()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
