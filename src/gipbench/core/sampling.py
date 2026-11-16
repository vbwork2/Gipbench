"""Lấy mẫu đa dạng dùng chung (dataset: consensus; model: curate trần per-term). Chuyển nguyên từ ft/curate_train.py."""
from collections import defaultdict


def pick_diverse(cands, cap, rnd):
    """cands: [(idx, speaker, ctx)]. Xoay vòng người nói, ưu tiên ngữ cảnh chưa gặp. Trả về tập idx được chọn."""
    by_spk = defaultdict(list)
    shuffled = cands[:]
    rnd.shuffle(shuffled)
    for c in shuffled:
        by_spk[c[1]].append(c)
    order = list(by_spk)
    rnd.shuffle(order)
    picked, seen_ctx = set(), set()
    while len(picked) < cap and any(by_spk[s] for s in order):
        for s in order:
            lst = by_spk[s]
            if not lst:
                continue
            k = next((j for j, c in enumerate(lst) if c[2] not in seen_ctx), 0)
            c = lst.pop(k)
            picked.add(c[0])
            seen_ctx.add(c[2])
            if len(picked) >= cap:
                break
    return picked
