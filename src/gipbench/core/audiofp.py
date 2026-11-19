"""Dấu vân tay âm thanh (Haitsma & Kalker 2002) để bắt trùng audio mà sha256 không thấy: cùng lời nói nhưng đã
resample / mã hoá lại / cắt ranh giới khác (vd video đăng lại dưới YouTube ID khác), hoặc G-Group không có id nguồn.

  fingerprint(path)                 -> mảng uint32, một sub-fingerprint 32 bit mỗi khung (0.37 s, bước 11.6 ms)
  FPIndex(items).match(fp, ...)     -> các clip train có đoạn khớp với fp (bỏ phiếu theo độ lệch, rồi kiểm BER)

Sub-fingerprint: năng lượng 33 băng log 300–2000 Hz; bit m = dấu của hiệu năng lượng giữa hai băng kề nhau, lấy hiệu
theo thời gian. Bền với đổi tần số lấy mẫu, nén mp3 / ogg, đổi âm lượng, nhiễu nhẹ. Hai đoạn coi là cùng audio khi
tỉ lệ bit khác (BER) trên phần chồng nhau < `max_ber` (mặc định 0.30; audio khác nhau cho BER ~0.5) và phần chồng nhau
>= `min_overlap_s`. Đoạn im lặng / phẳng sinh sub-fingerprint lặp lại: khung có năng lượng thấp bị bỏ khỏi chỉ mục, và
sub-fingerprint quá phổ biến (> `stop_df` lần) không dùng để bỏ phiếu.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FRAME_S = 0.37
HOP_S = 0.0116
N_BANDS = 33
F_LO, F_HI = 300.0, 2000.0
FP_SR = 5512
_POW = (1 << np.arange(32, dtype=np.uint64)).astype(np.uint64)


def load_mono(path) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def _band_matrix(sr: int, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    """(bin đầu, bin cuối + 1) của 33 băng log-spaced."""
    edges = np.geomspace(F_LO, F_HI, N_BANDS + 1)
    b = np.clip(np.round(edges * nfft / sr).astype(int), 1, nfft // 2)
    return b[:-1], np.maximum(b[1:], b[:-1] + 1)


def _resample(x: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Hạ mẫu về FP_SR (5 512 Hz như bài gốc; băng cao nhất 2 kHz) để FFT nhỏ hơn ~4 lần."""
    if sr == FP_SR:
        return x, sr
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(FP_SR, sr)
    return resample_poly(x, FP_SR // g, sr // g).astype(np.float32), FP_SR


def fingerprint_signal(x: np.ndarray, sr: int, chunk: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """(sub-fingerprint uint32 [n], cờ khung có tín hiệu bool [n]). Clip ngắn hơn một khung -> mảng rỗng."""
    x, sr = _resample(x, sr)
    L, hop = int(round(FRAME_S * sr)), max(1, int(round(HOP_S * sr)))
    if len(x) < L + hop:
        return np.zeros(0, np.uint32), np.zeros(0, bool)
    nfft = 1 << (L - 1).bit_length()
    lo, hi = _band_matrix(sr, nfft)
    win = np.hanning(L).astype(np.float32)
    n = 1 + (len(x) - L) // hop
    E = np.empty((n, N_BANDS), np.float64)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))[:, None] * hop + np.arange(L)[None, :]
        P = np.abs(np.fft.rfft(x[idx] * win, nfft, axis=1)) ** 2
        cs = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)
        E[s:s + len(idx)] = cs[:, hi] - cs[:, lo]
    d = E[:, :-1] - E[:, 1:]                      # [n, 32]
    bits = (d[1:] - d[:-1]) > 0                   # [n-1, 32]
    fp = (bits.astype(np.uint64) * _POW).sum(axis=1).astype(np.uint32)
    tot = E.sum(axis=1)
    thr = max(1e-10, 1e-3 * float(np.median(tot[tot > 0])) if (tot > 0).any() else 1e-10)
    voiced = (tot[1:] > thr) & (tot[:-1] > thr)
    return fp, voiced


def fingerprint(path) -> Tuple[np.ndarray, np.ndarray]:
    x, sr = load_mono(path)
    return fingerprint_signal(x, sr)


def ber(a: np.ndarray, b: np.ndarray) -> float:
    """Tỉ lệ bit khác giữa hai dãy sub-fingerprint cùng độ dài."""
    if len(a) == 0:
        return 1.0
    x = np.bitwise_xor(a, b)
    return float(np.unpackbits(x.view(np.uint8)).sum()) / (32 * len(a))


class FPIndex:
    """Chỉ mục sub-fingerprint -> (clip, khung) của tập train, tìm clip có đoạn khớp với một clip test."""

    def __init__(self, items: Iterable[Tuple[str, np.ndarray, np.ndarray]], stop_df: int = 200):
        self.ids: List[str] = []
        self.fps: List[np.ndarray] = []
        hs, cs, fs = [], [], []
        for cid, fp, voiced in items:
            k = len(self.ids)
            self.ids.append(cid)
            self.fps.append(fp)
            sel = np.nonzero(voiced)[0]
            hs.append(fp[sel])
            cs.append(np.full(len(sel), k, np.int32))
            fs.append(sel.astype(np.int32))
        h = np.concatenate(hs) if hs else np.zeros(0, np.uint32)
        c = np.concatenate(cs) if cs else np.zeros(0, np.int32)
        f = np.concatenate(fs) if fs else np.zeros(0, np.int32)
        o = np.argsort(h, kind="stable")
        self.h, self.c, self.f = h[o], c[o], f[o]
        u, cnt = np.unique(self.h, return_counts=True)
        self.stop = set(u[cnt > stop_df].tolist())
        self.stop_df = stop_df

    def match(self, fp: np.ndarray, voiced: np.ndarray, *, min_votes: int = 3, max_ber: float = 0.30,
              min_overlap_s: float = 1.0, max_candidates: int = 20) -> List[Dict]:
        """Các clip train khớp: [{train, offset_s, overlap_s, ber}], BER thấp nhất trước."""
        q = np.nonzero(voiced)[0]
        if len(q) == 0 or len(self.h) == 0:
            return []
        qh = fp[q]
        left = np.searchsorted(self.h, qh, "left")
        right = np.searchsorted(self.h, qh, "right")
        n = right - left
        ok = (n > 0) & (n <= self.stop_df)
        if not ok.any():
            return []
        left, n, qf = left[ok], n[ok], q[ok]
        # mọi cặp (khung test, mục chỉ mục) có cùng sub-fingerprint, không vòng lặp Python
        rep = np.repeat(np.arange(len(n)), n)
        j = left[rep] + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
        c = self.c[j].astype(np.int64)
        off = self.f[j].astype(np.int64) - qf[rep].astype(np.int64)
        SH = 1 << 21
        key, v = np.unique(c * (1 << 22) + (off + SH), return_counts=True)
        votes = dict(zip(key.tolist(), v.tolist()))
        # gộp độ lệch ±1 khung
        best: Dict[int, Tuple[int, int]] = {}
        for k, cnt in votes.items():
            tot = cnt + votes.get(k - 1, 0) + votes.get(k + 1, 0)
            ci, oi = k >> 22, (k & ((1 << 22) - 1)) - SH
            if tot >= min_votes and (ci not in best or tot > best[ci][0]):
                best[ci] = (tot, oi)
        out = []
        # clip ngắn hơn min_overlap_s: yêu cầu chồng >= 90% độ dài clip (sàn 0.5 s)
        min_ov = int(max(0.5, min(min_overlap_s, 0.9 * len(fp) * HOP_S)) / HOP_S)
        for c, (v, off) in sorted(best.items(), key=lambda kv: -kv[1][0])[:max_candidates]:
            tr = self.fps[c]
            lo_q, hi_q = max(0, -off), min(len(fp), len(tr) - off)
            if hi_q - lo_q < min_ov:
                continue
            e = ber(fp[lo_q:hi_q], tr[lo_q + off:hi_q + off])
            if e <= max_ber:
                out.append({"train": self.ids[c], "offset_s": round(off * HOP_S, 2),
                            "overlap_s": round((hi_q - lo_q) * HOP_S, 2), "ber": round(e, 3), "votes": v})
        return sorted(out, key=lambda m: m["ber"])


def _fp_one(p: Optional[str]):
    try:
        return fingerprint(p)
    except Exception:  # noqa: BLE001 - file hỏng / định dạng lạ: không có fingerprint
        return None


def fingerprint_rows(rows: Sequence[Dict], log_every: int = 0, workers: int = 1,
                     cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
                     ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """[(id, fp, voiced)] cho các dòng có file audio đọc được (dòng lỗi đọc bị bỏ, đếm qua độ dài kết quả).
    cache: {đường dẫn audio: (fp, voiced)} dùng lại và được bổ sung (cùng file ở nhiều release chỉ tính một lần)."""
    cache = {} if cache is None else cache
    paths = [str(r.get("_audio_path")) for r in rows]
    todo = sorted({p for p in paths if p not in cache})
    if workers > 1 and len(todo) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            for k, (p, res) in enumerate(zip(todo, pool.imap(_fp_one, todo, chunksize=16)), 1):
                if res is not None:
                    cache[p] = res
                if log_every and k % log_every == 0:
                    print(f"  fingerprint {k}/{len(todo)}", flush=True)
    else:
        for k, p in enumerate(todo, 1):
            res = _fp_one(p)
            if res is not None:
                cache[p] = res
            if log_every and k % log_every == 0:
                print(f"  fingerprint {k}/{len(todo)}", flush=True)
    return [(str(r.get("id")), *cache[p]) for r, p in zip(rows, paths) if p in cache]


def save_cache(path, cache: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Lưu {đường dẫn: (fp, voiced)} vào một file .npz (nối mảng + chỉ số đầu)."""
    keys = sorted(cache)
    lens = np.array([len(cache[k][0]) for k in keys], np.int64)
    fp = np.concatenate([cache[k][0] for k in keys]) if keys else np.zeros(0, np.uint32)
    vo = np.concatenate([cache[k][1] for k in keys]) if keys else np.zeros(0, bool)
    np.savez_compressed(str(path), keys=np.array(keys, dtype=object), lens=lens, fp=fp, voiced=vo,
                        version=np.array([FP_SR, N_BANDS]))


def load_cache(path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    from pathlib import Path
    if not Path(path).exists():
        return {}
    z = np.load(str(path), allow_pickle=True)
    if list(z["version"]) != [FP_SR, N_BANDS]:
        return {}
    ends = np.cumsum(z["lens"])
    starts = ends - z["lens"]
    fp, vo = z["fp"], z["voiced"]
    return {k: (fp[s:e], vo[s:e]) for k, s, e in zip(z["keys"].tolist(), starts, ends)}
