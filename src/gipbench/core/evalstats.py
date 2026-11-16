"""Thống kê đánh giá dùng chung (bench chấm điểm, model chọn checkpoint): kiểm hyps, đếm lỗi theo câu, bootstrap theo
cụm và bootstrap cặp.

Mọi metric ở đây là TỈ LỆ CỦA TỔNG (micro): wer = Σ lỗi / Σ token ref. Bootstrap lấy mẫu lại theo ĐƠN VỊ (speaker,
session, source hoặc utt; có hoàn lại), mỗi mẫu tính lại tỉ lệ từ tổng của các đơn vị được chọn. CI 95% = phân vị
2.5 / 97.5 (cùng quy ước với core.scoring.bootstrap_many).
"""
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gipbench.core.align import align, char_edit_distance

# metric -> (tử số, mẫu số) trong bộ đếm của một câu
RATIOS = {
    "wer": ("err", "n"),
    "cer": ("cerr", "nc"),
    "cs_wer": ("err_en", "n_en"),
    "n_wer": ("err_vi", "n_vi"),
    "term_recall": ("matched_terms", "ref_terms"),
}
COUNT_KEYS = sorted({k for pair in RATIOS.values() for k in pair})


def check_hyps(manifest_ids: Sequence[str], hyp_ids: Sequence[str], hyp_texts: Optional[Sequence[str]] = None) -> Dict:
    """So danh sách id của manifest với hyps. Trả về số liệu + danh sách lỗi; `ok` = không thiếu, không thừa, không trùng."""
    mc, hc = Counter(manifest_ids), Counter(hyp_ids)
    dup_m = sorted(i for i, c in mc.items() if c > 1)
    dup_h = sorted(i for i, c in hc.items() if c > 1)
    missing = sorted(set(mc) - set(hc))
    extra = sorted(set(hc) - set(mc))
    empty = sum(1 for t in (hyp_texts or []) if not (t or "").strip())
    return {"n_manifest": len(manifest_ids), "n_hyps": len(hyp_ids), "duplicate_manifest_ids": dup_m,
            "duplicate_hyp_ids": dup_h, "missing": missing, "extra": extra, "empty_hyps": empty,
            "ok": not (dup_m or dup_h or missing or extra)}


def utt_counts(ref: List[str], hyp: List[str]) -> Dict[str, int]:
    """Đếm lỗi token (theo khoảng trắng) và lỗi ký tự cho một câu. CER tính trên chuỗi token nối bằng 1 dấu cách
    (dấu cách là một ký tự), Levenshtein theo ký tự."""
    ops = align(ref, hyp)
    S = sum(o[0] == "S" for o in ops)
    D = sum(o[0] == "D" for o in ops)
    I = sum(o[0] == "I" for o in ops)
    r, h = " ".join(ref), " ".join(hyp)
    return {"n": len(ref), "S": S, "D": D, "I": I, "err": S + D + I,
            "cerr": char_edit_distance(r, h), "nc": len(r)}


def ratio(sums: Dict[str, float], metric: str) -> Optional[float]:
    num, den = RATIOS[metric]
    return 100.0 * sums.get(num, 0) / sums[den] if sums.get(den) else None


def sum_counts(recs: Iterable[Dict]) -> Dict[str, float]:
    s: Dict[str, float] = Counter()
    for r in recs:
        for k in COUNT_KEYS:
            s[k] += r.get(k) or 0
    return dict(s)


def _unit_sums(recs: Iterable[Dict], unit_key: str) -> Dict[str, Dict[str, float]]:
    by = defaultdict(list)
    for r in recs:
        by[str(r.get(unit_key) or r["id"])].append(r)
    return {u: sum_counts(rs) for u, rs in by.items()}


def _pct(v: List[float]) -> Optional[Tuple[float, float]]:
    if not v:
        return None
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def cluster_bootstrap(recs: Sequence[Dict], metrics: Sequence[str], unit_key: str, iters: int = 1000,
                      seed: int = 0) -> Dict[str, Optional[Tuple[float, float]]]:
    """CI 95% của các metric RATIOS, lấy mẫu lại theo `unit_key` của từng câu (thiếu thì dùng id câu)."""
    us = _unit_sums(recs, unit_key)
    units = sorted(us)
    if len(units) < 2:
        return {m: None for m in metrics}
    rnd = random.Random(seed)
    vals = {m: [] for m in metrics}
    for _ in range(iters):
        tot: Dict[str, float] = Counter()
        for _k in units:
            for k, v in us[rnd.choice(units)].items():
                tot[k] += v
        for m in metrics:
            x = ratio(tot, m)
            if x is not None:
                vals[m].append(x)
    return {m: _pct(v) for m, v in vals.items()}


def paired_bootstrap(recs_a: Sequence[Dict], recs_b: Sequence[Dict], metric: str, unit_key: str, iters: int = 1000,
                     seed: int = 0) -> Dict:
    """Δ = metric(B) - metric(A) trên các câu CHUNG id, CI 95% bằng bootstrap cặp theo đơn vị (cùng mẫu đơn vị cho
    hai hệ). Đơn vị lấy từ recs_a."""
    a = {r["id"]: r for r in recs_a}
    b = {r["id"]: r for r in recs_b}
    ids = sorted(set(a) & set(b))
    ra, rb = [a[i] for i in ids], [b[i] for i in ids]
    ua = _unit_sums(ra, unit_key)
    ub = _unit_sums([dict(y, **{unit_key: x.get(unit_key)}) for x, y in zip(ra, rb)], unit_key)
    units = sorted(ua)
    pa, pb = ratio(sum_counts(ra), metric), ratio(sum_counts(rb), metric)
    out = {"metric": metric, "n_common": len(ids), "n_only_a": len(set(a) - set(b)), "n_only_b": len(set(b) - set(a)),
           "a": pa, "b": pb, "delta": None if pa is None or pb is None else pb - pa, "ci95": None,
           "units": len(units), "unit_key": unit_key}
    if len(units) < 2 or out["delta"] is None:
        return out
    rnd = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ta: Dict[str, float] = Counter()
        tb: Dict[str, float] = Counter()
        for _k in units:
            u = rnd.choice(units)
            for k, v in ua[u].items():
                ta[k] += v
            for k, v in ub[u].items():
                tb[k] += v
        xa, xb = ratio(ta, metric), ratio(tb, metric)
        if xa is not None and xb is not None:
            diffs.append(xb - xa)
    out["ci95"] = _pct(diffs)
    return out
