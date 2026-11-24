#!/usr/bin/env python3
"""Phân rã Joiner theo đơn vị ẩn (gipformer_mech.joiner_probe). CPU, trọng số giả, không cần model.

  python tests/mech/test_joiner_probe.py      # thoát mã 1 nếu có ca hỏng
"""
import sys

import numpy as np

from gipformer_mech import joiner_probe as jp

FAIL = []
V, U, BLANK = 12, 16, 0


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def toy(seed=0, n=40):
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(V, U)).astype(np.float32)
    bias = rng.normal(size=V).astype(np.float32)
    h = np.tanh(rng.normal(size=(n, U))).astype(np.float32)
    return weight, bias, h


def test_decomposition():
    weight, bias, h = toy()
    logits = h @ weight.T + bias
    targets = np.arange(len(h)) % (V - 1) + 1
    c = jp.contributions(h, weight, BLANK, targets)
    m = jp.margins(c, bias, BLANK, targets)
    true = logits[:, BLANK] - logits[np.arange(len(h)), targets]
    check("phân rã: tổng đóng góp + chênh bias = margin thật", np.allclose(m, true, atol=1e-5), float(np.abs(m - true).max()))
    check("phân rã: đóng góp của một đơn vị = (W_blank - W_t) * h",
          np.isclose(c[3, 5], (weight[BLANK, 5] - weight[targets[3], 5]) * h[3, 5]))
    units = [2, 7, 11]
    after = jp.ablated_logits(h, weight, bias, units)
    removed = (h[:, units] @ weight[:, units].T)
    check("zero đơn vị = bỏ đúng số hạng của nó khỏi mọi logit", np.allclose(after, logits - removed, atol=1e-5))
    ablated_margin = after[:, BLANK] - after[np.arange(len(h)), targets]
    check("margin sau zero = margin - tổng đóng góp của đơn vị bị zero",
          np.allclose(ablated_margin, m - c[:, units].sum(axis=1), atol=1e-5))
    share = jp.share_explained(c, m, units)
    check("share = đóng góp của tập / margin", np.allclose(share, c[:, units].sum(axis=1) / m))
    check("không đơn vị nào bị zero thì không đổi", np.allclose(jp.ablated_logits(h, weight, bias, []), logits, atol=1e-6))


def test_selection():
    event = np.array([5.0, 1.0, 0.0, 3.0, 2.0])
    ordinary = np.array([5.0, 0.0, 0.0, 0.0, 2.5])
    check("select_units: xếp theo phần dư so với frame thường, không theo đóng góp thô",
          jp.select_units(event, ordinary, 2).tolist() == [3, 1], jp.select_units(event, ordinary, 2).tolist())
    try:
        jp.select_units(event, ordinary[:3], 2)
        check("select_units: vector lệch cỡ -> ValueError", False)
    except ValueError:
        check("select_units: vector lệch cỡ -> ValueError", True)
    folds = {jp.fold_of(f"fpt_{i:06d}") for i in range(200)}
    check("fold_of: cả hai fold đều xuất hiện", folds == {0, 1})
    check("fold_of: cố định theo utt_id", jp.fold_of("fpt_000123") == jp.fold_of("fpt_000123"))


def test_random_sets():
    weight, bias, h = toy(1, 60)
    targets = np.arange(len(h)) % (V - 1) + 1
    c = jp.contributions(h, weight, BLANK, targets)
    m = jp.margins(c, bias, BLANK, targets)
    m = np.where(m > 0.1, m, 0.1)
    a = jp.random_unit_shares(c, m, 4, 50, seed=3)
    b = jp.random_unit_shares(c, m, 4, 50, seed=3)
    check("random_unit_shares: cùng seed -> cùng kết quả", np.array_equal(a, b) and a.shape == (50,))
    full = jp.random_unit_shares(c, m, U, 5, seed=0)
    check("random_unit_shares: chọn đủ mọi đơn vị -> share = tổng đóng góp / margin",
          np.allclose(full, np.median(c.sum(axis=1) / m)))


def test_harm_and_curve():
    weight, bias, h = toy(2, 200)
    base = (h @ weight.T + bias).argmax(axis=1)
    harm = jp.argmax_change(h, weight, bias, [], BLANK)
    check("argmax_change: không zero gì -> không đổi", harm["change_rate"] == 0.0)
    harm = jp.argmax_change(h, weight, bias, list(range(U)), BLANK, chunk=64)
    expected = float((bias.argmax() != base).mean())
    check("argmax_change: zero hết đơn vị -> chỉ còn bias, đếm đúng qua nhiều chunk",
          abs(harm["change_rate"] - expected) < 1e-9 and harm["blank_frames"] + harm["piece_frames"] == len(h))
    curve = jp.concentration_curve(np.array([4.0, 3.0, 2.0, 1.0, -5.0]), (1, 2, 4, 5))
    check("concentration_curve: chỉ tính phần dư dương, tăng dần tới 1",
          curve == {1: 0.4, 2: 0.7, 4: 1.0, 5: 1.0}, curve)
    check("participation_ratio: một đơn vị = 1, đều = n",
          abs(jp.participation_ratio(np.array([3.0, 0, 0, 0])) - 1) < 1e-9
          and abs(jp.participation_ratio(np.ones(8)) - 8) < 1e-9)


def test_mean_contribution():
    weight, bias, h = toy(4, 30)
    mean = jp.mean_contribution(h, weight, bias, BLANK, chunk=7)
    competitor = jp.best_nonblank(h @ weight.T + bias, BLANK)
    check("mean_contribution: khớp tính một lần, độc lập cỡ chunk",
          np.allclose(mean, jp.contributions(h, weight, BLANK, competitor).mean(axis=0), atol=1e-6))
    check("best_nonblank: không bao giờ chọn blank", not (competitor == BLANK).any())


def main():
    test_decomposition()
    test_selection()
    test_random_sets()
    test_harm_and_curve()
    test_mean_contribution()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
