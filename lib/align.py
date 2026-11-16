"""Căn chỉnh Levenshtein theo từ, có truy vết, để quy lỗi về span tiếng Anh / tiếng Việt."""
from typing import List, Optional, Tuple

Op = Tuple[str, Optional[int], Optional[int]]  # ("C"|"S"|"D"|"I", ref_idx, hyp_idx)


def align(ref: List[str], hyp: List[str]) -> List[Op]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            c = prev[j - 1] + (0 if ri == hyp[j - 1] else 1)
            de = prev[j] + 1
            ins = row[j - 1] + 1
            row[j] = c if c <= de and c <= ins else (de if de <= ins else ins)
    ops: List[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def attribute(ops: List[Op], ref_span: List[int]):
    """Quy mỗi lỗi về một span.

    S/D: thuộc span của token reference đó.
    I (chèn): thuộc span tiếng Anh nếu một trong hai token reference kề bên là tiếng Anh
    (ưu tiên bên trái), ngược lại thuộc tiếng Việt. Quy ước này gom lỗi kiểu
    "meeting" -> "mít tinh" (1 S + 1 I) vào đúng span tiếng Anh.

    Trả về: errs_by_span (dict span -> số lỗi, -1 = tiếng Việt), S, D, I.
    """
    errs = {}
    S = D = I = 0
    last_ref = None
    # chỉ số token ref kế tiếp cho từng vị trí op
    next_ref = [None] * len(ops)
    nxt = None
    for k in range(len(ops) - 1, -1, -1):
        next_ref[k] = nxt
        if ops[k][1] is not None:
            nxt = ops[k][1]
    for k, (t, ri, _) in enumerate(ops):
        if t == "C":
            last_ref = ri
            continue
        if t in ("S", "D"):
            sp = ref_span[ri]
            last_ref = ri
            S += t == "S"
            D += t == "D"
        else:
            I += 1
            left = ref_span[last_ref] if last_ref is not None else -1
            right = ref_span[next_ref[k]] if next_ref[k] is not None else -1
            sp = left if left >= 0 else (right if right >= 0 else -1)
        errs[sp] = errs.get(sp, 0) + 1
    return errs, S, D, I


def _op_spans(ops: List[Op], ref_span: List[int]) -> List[int]:
    """Span sở hữu từng op: C/S/D theo token reference; I theo đúng luật kề bên của `attribute`
    (trái nếu là tiếng Anh, không thì phải, không thì -1 = tiếng Việt)."""
    next_ref = [None] * len(ops)
    nxt = None
    for k in range(len(ops) - 1, -1, -1):
        next_ref[k] = nxt
        if ops[k][1] is not None:
            nxt = ops[k][1]
    owner: List[int] = []
    last_ref = None
    for k, (t, ri, _) in enumerate(ops):
        if t != "I":
            owner.append(ref_span[ri])
            last_ref = ri
            continue
        left = ref_span[last_ref] if last_ref is not None else -1
        right = ref_span[next_ref[k]] if next_ref[k] is not None else -1
        owner.append(left if left >= 0 else (right if right >= 0 else -1))
    return owner


def span_hyps(ops: List[Op], ref_span: List[int], hyp: List[str], n_spans: int) -> List[List[str]]:
    """Token hyp căn với từng span reference: token ở op C/S của các token ref thuộc span, cộng token chèn
    được `attribute` quy về span đó (cùng luật kề bên). Span bị xoá hết cho danh sách rỗng.
    Trả về list độ dài n_spans, mỗi phần tử là token hyp theo thứ tự trong hyp."""
    out: List[List[str]] = [[] for _ in range(n_spans)]
    for (t, _, hj), sp in zip(ops, _op_spans(ops, ref_span)):
        if sp >= 0 and hj is not None:
            out[sp].append(hyp[hj])
    return out


def char_edit_distance(a: str, b: str) -> int:
    """Khoảng cách Levenshtein theo ký tự."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (ca != cb), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]
