"""Phần dùng chung cho bộ công cụ TTS (src/gipbench/dataset/synth/make_text.py, tts_synth.py, tts_verify.py): backend TTS, tiện ích
âm thanh, bảng cách đọc (configs/dataset/synth/tts_lexicon.tsv) và cách đọc dự phòng.

Backend (giao diện duy nhất): `synth(text, voice, out_path) -> None`, ghi file âm thanh vào out_path.
  mock     âm thanh giả, xác định theo hash, độ dài tỉ lệ số âm tiết; CHỈ để test, không chứa lời nói.
  command  lệnh ngoài: --command "tts-cli --in {text_file} --voice {voice} --ref {ref_audio} --out {out}"
           (text truyền qua file UTF-8 để tránh lỗi quote; tách lệnh bằng shlex TRƯỚC khi thay chỗ trống;
           chỗ trống: {text_file} {voice} {voice_id} {ref_audio} {out}).
  python   nạp hàm theo "module:function" hoặc "đường/dẫn/file.py:function"; hàm có chữ ký fn(text, voice, out_path).
  edge     gói `edge-tts`. LƯU Ý QUYỀN RIÊNG TƯ: backend này GỬI VĂN BẢN LÊN DỊCH VỤ ONLINE CỦA MICROSOFT, không phải mặc
           định; chỉ dùng khi văn bản kịch bản được phép rời máy. Adapter viết theo API công khai của edge-tts
           (Communicate(text, voice).save(path), ra mp3) nhưng CHƯA được chạy thử trong repo này (không có mạng / gói).
`voice` truyền cho backend là `Voice` (lớp con của str: giá trị chuỗi = backend_voice; thuộc tính .id, .ref_audio,
.gender, .region), nên hàm của người dùng coi nó như chuỗi cũng được.

Âm thanh: đọc wav bằng module chuẩn `wave` (PCM), các định dạng khác (mp3 / flac / float wav) cần `soundfile` hoặc
`ffmpeg` trên PATH. Mọi phép xử lý (resample, cắt lặng, chuẩn hoá) viết bằng Python thuần, có dùng `soxr` + numpy
(đã có trong requirements.txt) để resample nhanh nếu cài. Không cần numpy để chạy.
"""
import array
import csv
import hashlib
import importlib
import importlib.util
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gipbench.core.textnorm import basic_tokens, parse_tagged

TARGET_SR = 16000
VIET_LEVELS = ("strong", "mid", "native")
KINDS = ("word", "acronym_en", "acronym_vi", "brand")


# ----------------------------------------------------------------------------- giọng
class Voice(str):
    """Giọng TTS. Giá trị chuỗi = backend_voice (tên giọng của engine); .id = id trong voices.tsv (vào speaker)."""

    def __new__(cls, id: str, backend_voice: str = "", ref_audio: str = "", gender: str = "u", region: str = "unk"):
        o = super().__new__(cls, backend_voice or id)
        o.id = id
        o.ref_audio = ref_audio or ""
        o.gender = gender or "u"
        o.region = region or "unk"
        o.ref_speaker = ""
        o.ref_source = ""
        return o


def load_voices(path: str) -> List[Voice]:
    """TSV (có header, bỏ dòng '#'): voice, backend_voice, ref_audio, gender, region (+ ref_speaker, ref_source tuỳ chọn:
    người nói / nguồn của audio tham chiếu, để kiểm rò rỉ). Đường dẫn ref_audio tương đối theo thư mục chứa voices.tsv."""
    base = Path(path).resolve().parent
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader((l for l in f if not l.startswith("#") and l.strip()), delimiter="\t"):
            vid = (r.get("voice") or "").strip()
            if not vid:
                continue
            if vid in seen:
                raise SystemExit(f"{path}: giọng '{vid}' bị khai báo hai lần")
            seen.add(vid)
            ref = (r.get("ref_audio") or "").strip()
            if ref and not Path(ref).is_absolute():
                ref = str(base / ref)
            v = Voice(vid, (r.get("backend_voice") or "").strip(), ref, (r.get("gender") or "u").strip(),
                      (r.get("region") or "unk").strip())
            v.ref_speaker = (r.get("ref_speaker") or "").strip()
            v.ref_source = (r.get("ref_source") or "").strip()
            out.append(v)
    if not out:
        raise SystemExit(f"{path}: không có giọng nào")
    return out


# ----------------------------------------------------------------------------- bảng cách đọc
@dataclass(frozen=True)
class Reading:
    reading: str
    viet_level: str
    kind: str = "word"
    note: str = ""
    fallback: bool = False


def term_key(term: str) -> str:
    """Khoá chuẩn của một thuật ngữ: token sau basic_tokens, nối bằng dấu cách."""
    return " ".join(basic_tokens(term))


def fold(s: str) -> str:
    """Bỏ dấu + đ -> d + chữ thường: so cách đọc không phụ thuộc thanh điệu (ASR hay đổi dấu)."""
    s = unicodedata.normalize("NFC", s).lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def load_lexicon(path: Optional[str]) -> Tuple[Dict[str, List[Reading]], List[str]]:
    """Đọc configs/dataset/synth/tts_lexicon.tsv -> ({term_key: [Reading]}, [cảnh báo]). Dòng trùng (term, reading) bị bỏ."""
    lex: Dict[str, List[Reading]] = {}
    problems: List[str] = []
    if not path:
        return lex, problems
    with open(path, encoding="utf-8") as f:
        rows = csv.DictReader((l for l in f if not l.startswith("#") and l.strip()), delimiter="\t")
        for n, r in enumerate(rows, 1):
            term, reading = (r.get("term") or "").strip(), (r.get("reading") or "").strip()
            level, kind = (r.get("viet_level") or "").strip(), (r.get("kind") or "word").strip()
            if not term or not reading:
                problems.append(f"dòng dữ liệu {n}: thiếu term / reading")
                continue
            if level not in VIET_LEVELS:
                problems.append(f"dòng dữ liệu {n} ({term}): viet_level '{level}' không hợp lệ")
                continue
            if kind not in KINDS:
                problems.append(f"dòng dữ liệu {n} ({term}): kind '{kind}' không hợp lệ")
                continue
            k = term_key(term)
            rd = Reading(" ".join(reading.split()), level, kind, (r.get("note") or "").strip())
            if any(x.reading == rd.reading for x in lex.setdefault(k, [])):
                problems.append(f"dòng dữ liệu {n} ({term}): cách đọc '{rd.reading}' bị lặp")
                continue
            lex[k].append(rd)
    return lex, problems


# bảng tên chữ cái: kiểu Anh (ây bi xi ...) và kiểu Việt (a bê xê ...)
EN_LETTER = dict(a="ây", b="bi", c="xi", d="đi", e="i", f="ép", g="gi", h="ếch", i="ai", j="giây", k="kây", l="eo",
                 m="em", n="en", o="âu", p="pi", q="kiu", r="a", s="ét", t="ti", u="diu", v="vi", w="đấp bồ diu",
                 x="ếch xì", y="quai", z="di")
VI_LETTER = dict(a="a", b="bê", c="xê", d="đê", e="e", f="ép", g="giê", h="hát", i="i", j="gi", k="ca", l="e lờ",
                 m="em mờ", n="en nờ", o="o", p="pê", q="quy", r="e rờ", s="ét", t="tê", u="u", v="vê", w="vê kép",
                 x="ích", y="i grếc", z="dét")
_VOWELS = set("aeiouy")


def is_spelled_token(tok: str) -> bool:
    """Viết tắt toàn phụ âm (hr, ssd, sql, vpn...) -> đọc từng chữ cái."""
    return tok.isascii() and tok.isalpha() and 2 <= len(tok) <= 6 and not (set(tok) & _VOWELS)


def fallback_readings(term_tokens: Sequence[str]) -> List[Reading]:
    """Cách đọc dự phòng cho thuật ngữ KHÔNG có trong lexicon (luật đơn giản, đều là phỏng đoán, mang cờ fallback):
    token toàn phụ âm -> đánh vần chữ cái kiểu Anh (native, acronym_en) và kiểu Việt (strong, acronym_vi);
    còn lại -> đưa nguyên chính tả cho backend (viet_level 'mid' chỉ là giá trị giữ chỗ vì schema bắt buộc)."""
    toks = list(term_tokens)
    if any(is_spelled_token(t) for t in toks):
        en = " ".join(" ".join(EN_LETTER[c] for c in t) if is_spelled_token(t) else t for t in toks)
        vi = " ".join(" ".join(VI_LETTER[c] for c in t) if is_spelled_token(t) else t for t in toks)
        return [Reading(en, "native", "acronym_en", "dự phòng: đánh vần Anh", True),
                Reading(vi, "strong", "acronym_vi", "dự phòng: đánh vần Việt", True)]
    return [Reading(" ".join(toks), "mid", "word", "dự phòng: nguyên chính tả", True)]


def build_tts_text(text_tagged: str, readings: Dict[str, str]) -> str:
    """Thay mỗi <en>span</en> bằng cách đọc của thuật ngữ (khoá = term_key); span không có cách đọc giữ nguyên chính tả."""
    out = []
    for seg, is_en in parse_tagged(text_tagged):
        out.append(readings.get(term_key(seg), seg.strip()) if is_en else seg)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def span_levels(text_tagged: str, readings: Iterable[Dict]) -> List[Dict]:
    """en_meta theo thứ tự span <en>: {"viet_level": mức của cách đọc đã dùng cho thuật ngữ đó}."""
    lv = {term_key(r["term"]): r.get("viet_level", "mid") for r in readings}
    return [{"viet_level": lv.get(term_key(seg), "mid")} for seg, is_en in parse_tagged(text_tagged) if is_en]


class TermIndex:
    """Tìm thuật ngữ (n-gram token) trong chuỗi token: khớp dài nhất, không chồng lấn. Giá trị tuỳ ý người dùng gắn."""

    def __init__(self):
        self.d: Dict[Tuple[str, ...], object] = {}
        self.max_n = 1

    def add(self, canon: Tuple[str, ...], value=None):
        if canon:
            self.d[canon] = value
            self.max_n = max(self.max_n, len(canon))

    def find(self, toks: Sequence[str]) -> List[Tuple[int, int, Tuple[str, ...]]]:
        out, i = [], 0
        while i < len(toks):
            for n in range(min(self.max_n, len(toks) - i), 0, -1):
                k = tuple(toks[i:i + n])
                if k in self.d:
                    out.append((i, i + n, k))
                    i += n
                    break
            else:
                i += 1
        return out


# ----------------------------------------------------------------------------- âm tiết, tốc độ nói
_VOWEL_GROUP = re.compile(r"[aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+")


def count_syllables(text: str) -> int:
    """Số âm tiết nói ra: token tiếng Việt = 1; token Anh nguyên chính tả = số cụm nguyên âm (tối thiểu 1)."""
    from gipbench.core.tagging import is_vietnamese_token
    n = 0
    for t in basic_tokens(text):
        if not any(c.isalpha() for c in t):
            continue
        n += 1 if is_vietnamese_token(t) else max(1, len(_VOWEL_GROUP.findall(t)))
    return n


# ----------------------------------------------------------------------------- đọc / ghi âm thanh
def write_wav16(path: str, samples: Sequence[float], sr: int):
    """Ghi mono PCM16 (cắt về [-1, 1])."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = array.array("h", (int(round(max(-1.0, min(1.0, s)) * 32767)) for s in samples))
    if sys.byteorder == "big":
        a.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.tobytes())


def _read_wave(path: str) -> Tuple[List[float], int]:
    with wave.open(str(path), "rb") as w:
        nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = array.array("h")
        a.frombytes(raw)
        if sys.byteorder == "big":
            a.byteswap()
        vals = [x / 32768.0 for x in a]
    elif sw == 1:
        vals = [(x - 128) / 128.0 for x in raw]
    elif sw == 3:
        vals = [int.from_bytes(raw[i:i + 3], "little", signed=True) / 8388608.0 for i in range(0, len(raw) - 2, 3)]
    elif sw == 4:
        a = array.array("i")
        a.frombytes(raw)
        vals = [x / 2147483648.0 for x in a]
    else:
        raise ValueError(f"wav {sw * 8}-bit không hỗ trợ")
    if nch > 1:
        vals = [sum(vals[i:i + nch]) / nch for i in range(0, len(vals) - nch + 1, nch)]
    return vals, sr


def read_audio(path: str) -> Tuple[List[float], int]:
    """Đọc audio -> (mẫu float mono [-1, 1], sample rate). Thử soundfile -> module wave -> ffmpeg; lỗi có hướng dẫn."""
    path = str(path)
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1).tolist(), int(sr)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001  soundfile đọc lỗi (vd định dạng lạ): thử cách khác
        pass
    try:
        return _read_wave(path)
    except Exception as e:  # noqa: BLE001
        err = e
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "x.wav")
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-acodec", "pcm_s16le", tmp],
                               capture_output=True)
            if r.returncode == 0:
                return _read_wave(tmp)
            err = RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
    raise RuntimeError(f"không đọc được audio {path}: {err}. Cài `pip install soundfile` (mp3 / flac / wav float) "
                       f"hoặc đặt ffmpeg vào PATH; wav PCM đọc được bằng module chuẩn.")


def resample(x: Sequence[float], sr_in: int, sr_out: int) -> List[float]:
    """Đổi tần số lấy mẫu. Dùng soxr (HQ) nếu có numpy + soxr, không thì bộ lọc Lanczos (a = 3) bằng Python thuần
    (chậm hơn ~vài trăm ms mỗi câu; chấp nhận được cho vài nghìn câu)."""
    if sr_in == sr_out or not len(x):
        return list(x)
    try:
        import numpy as np
        import soxr
        return soxr.resample(np.asarray(x, dtype=np.float32), sr_in, sr_out, quality="HQ").tolist()
    except ImportError:
        pass
    ratio = sr_in / sr_out
    cutoff = min(1.0, 1.0 / ratio)  # hạ tần số: lọc thông thấp ở tần số Nyquist mới
    a = 3
    half = a / cutoff
    n_out = int(len(x) / ratio)
    n = len(x)
    out = [0.0] * n_out
    sinc = lambda v: 1.0 if v == 0 else math.sin(math.pi * v) / (math.pi * v)  # noqa: E731
    for m in range(n_out):
        t = m * ratio
        lo, hi = int(math.ceil(t - half)), int(math.floor(t + half))
        acc = wsum = 0.0
        for k in range(max(lo, 0), min(hi, n - 1) + 1):
            d = (t - k) * cutoff
            w = sinc(d) * sinc(d / a) if abs(d) < a else 0.0
            acc += x[k] * w
            wsum += w
        out[m] = acc / wsum if wsum else 0.0
    return out


def analyze(x: Sequence[float], sr: int, rel_db: float = -40.0, frame_ms: int = 10) -> Dict:
    """Số đo âm học: duration, peak, rms, clip_frac (mẫu |x| >= 0.999), lead / trail (giây lặng đầu / cuối),
    inner (đoạn lặng liên tục dài nhất giữa hai đầu). Khung 10 ms; khung 'có tiếng' khi RMS > 10^(rel_db/20) * RMS khung lớn nhất."""
    n = len(x)
    if n == 0:
        return dict(duration=0.0, peak=0.0, rms=0.0, clip_frac=0.0, lead=0.0, trail=0.0, inner=0.0, voiced=0.0)
    peak = max(max(x), -min(x))
    fl = max(1, sr * frame_ms // 1000)
    rms_f = []
    for i in range(0, n - fl + 1, fl):
        seg = x[i:i + fl]
        rms_f.append(math.sqrt(sum(v * v for v in seg) / fl))
    if not rms_f:
        rms_f = [math.sqrt(sum(v * v for v in x) / n)]
    top = max(rms_f)
    thr = max(1e-5, top * 10 ** (rel_db / 20))
    voiced = [r > thr for r in rms_f]
    clip = sum(1 for v in x if v >= 0.999 or v <= -0.999) / n
    sec = fl / sr
    if not any(voiced):
        return dict(duration=n / sr, peak=peak, rms=math.sqrt(sum(v * v for v in x) / n), clip_frac=clip,
                    lead=n / sr, trail=n / sr, inner=0.0, voiced=0.0)
    first = voiced.index(True)
    last = len(voiced) - 1 - voiced[::-1].index(True)
    inner = run = 0
    for v in voiced[first:last + 1]:
        run = 0 if v else run + 1
        inner = max(inner, run)
    return dict(duration=n / sr, peak=peak, rms=math.sqrt(sum(v * v for v in x) / n), clip_frac=clip,
                lead=first * sec, trail=(len(voiced) - 1 - last) * sec + (n - len(rms_f) * fl) / sr,
                inner=inner * sec, voiced=sum(voiced) * sec)


def postprocess(x: Sequence[float], sr: int, target_sr: int = TARGET_SR, keep_sil: float = 0.15,
                peak_target: float = 0.9, rel_db: float = -40.0) -> Tuple[List[float], Dict]:
    """resample về target_sr, cắt lặng đầu / cuối (chừa keep_sil giây mỗi đầu), chuẩn hoá đỉnh về peak_target.
    Trả về (mẫu mới, số đo của audio GỐC: raw_peak, clip_frac, lead, trail, inner, sr_in)."""
    raw = analyze(x, sr, rel_db)
    y = resample(x, sr, target_sr)
    st = analyze(y, target_sr, rel_db)
    if st["voiced"] > 0:
        a = max(0, int((st["lead"] - keep_sil) * target_sr))
        b = len(y) - max(0, int((st["trail"] - keep_sil) * target_sr))
        y = y[a:b]
    pk = max(max(y), -min(y)) if y else 0.0
    if pk > 1e-6:
        g = peak_target / pk
        y = [v * g for v in y]
    return y, dict(raw_peak=round(raw["peak"], 4), clip_frac=round(raw["clip_frac"], 6), sr_in=sr,
                   raw_lead=round(raw["lead"], 3), raw_trail=round(raw["trail"], 3))


# ----------------------------------------------------------------------------- backend
class Backend:
    name = "base"
    raw_ext = ".wav"  # đuôi file thô backend ghi ra (edge: .mp3); công cụ tts_synth chuyển về wav 16 kHz

    def __init__(self, **opts):
        self.opts = opts
        if opts.get("raw_ext"):
            self.raw_ext = str(opts["raw_ext"])

    def synth(self, text: str, voice: Voice, out_path: str) -> None:
        raise NotImplementedError


class MockBackend(Backend):
    """Âm thanh giả xác định theo hash(text, voice): mỗi âm tiết ~ sec_per_syl giây tiếng (sin + họa âm, bao hình cos^2),
    đệm lặng đầu / cuối. Dùng cho test; KHÔNG phải lời nói. Tuỳ chọn: sec_per_syl, sample_rate, lead, trail, amp, clip."""
    name = "mock"

    def synth(self, text, voice, out_path):
        sps = float(self.opts.get("sec_per_syl", 0.2))
        sr = int(self.opts.get("sample_rate", 22050))
        lead = float(self.opts.get("lead", 0.35))
        trail = float(self.opts.get("trail", 0.35))
        amp = float(self.opts.get("amp", 0.4))
        n_syl = max(1, count_syllables(text))
        vid = getattr(voice, "id", str(voice))
        h = hashlib.sha256(f"{text}|{vid}".encode("utf-8")).digest()
        f0 = 110 + h[0] % 120
        total = lead + n_syl * sps + trail
        x = [0.0] * int(total * sr)
        for s in range(n_syl):
            i0, i1 = int((lead + s * sps) * sr), int((lead + s * sps + 0.7 * sps) * sr)
            fs = f0 * (1 + (h[1 + s % 20] % 40) / 100)
            w1, w2, w3 = 2 * math.pi * fs / sr, 4 * math.pi * fs / sr, 6 * math.pi * fs / sr
            span = max(1, i1 - i0)
            for i in range(i0, min(i1, len(x))):
                env = math.sin(math.pi * (i - i0) / span) ** 2
                k = i - i0
                x[i] = amp / 1.75 * env * (math.sin(w1 * k) + 0.5 * math.sin(w2 * k) + 0.25 * math.sin(w3 * k))
        if self.opts.get("clip"):  # test: ép clipping
            x = [max(-1.0, min(1.0, v * float(self.opts["clip"]))) for v in x]
        write_wav16(out_path, x, sr)


class CommandBackend(Backend):
    """Lệnh ngoài. command = mẫu có chỗ trống {text_file} {voice} {voice_id} {ref_audio} {out}. Text ghi ra file UTF-8
    tạm cạnh out_path (tránh lỗi quote). Lệnh được tách bằng shlex rồi thay chỗ trống TỪNG token, chạy không qua shell."""
    name = "command"

    def __init__(self, command: str = "", timeout: float = 120.0, **opts):
        super().__init__(**opts)
        if not command or "{out}" not in command:
            raise SystemExit("backend command cần --command có chứa {out} (và thường {text_file}, {voice}), "
                             "vd --command \"tts-cli --text-file {text_file} --voice {voice} --out {out}\"")
        if os.name == "nt":  # posix=False giữ nguyên dấu \ của đường dẫn Windows; bỏ cặp nháy bao quanh từng token
            self.argv = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
                         for t in shlex.split(command, posix=False)]
        else:
            self.argv = shlex.split(command)
        self.timeout = float(timeout)

    def synth(self, text, voice, out_path):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tf = out.with_suffix(".txt")
        tf.write_text(text, encoding="utf-8")
        sub = {"{text_file}": str(tf), "{voice}": str(voice), "{voice_id}": getattr(voice, "id", str(voice)),
               "{ref_audio}": getattr(voice, "ref_audio", ""), "{out}": str(out)}
        argv = []
        for a in self.argv:
            for k, v in sub.items():
                a = a.replace(k, v)
            argv.append(a)
        try:
            r = subprocess.run(argv, capture_output=True, timeout=self.timeout)
        finally:
            try:
                tf.unlink()
            except OSError:
                pass
        if r.returncode != 0:
            raise RuntimeError(f"lệnh trả mã {r.returncode}: {r.stderr.decode('utf-8', 'replace')[-300:]}")
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("lệnh chạy xong nhưng không có file đầu ra")


def _load_callable(spec: str) -> Callable:
    mod, _, fn = spec.rpartition(":")
    if not mod or not fn:
        raise SystemExit(f"--python-fn '{spec}' phải có dạng module:function hoặc đường/dẫn/file.py:function")
    if mod.endswith(".py") and Path(mod).exists():
        s = importlib.util.spec_from_file_location(Path(mod).stem, mod)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    f = getattr(m, fn, None)
    if not callable(f):
        raise SystemExit(f"{spec}: không có hàm gọi được tên '{fn}'")
    return f


class PythonBackend(Backend):
    """Hàm của người dùng: fn(text, voice, out_path) -> None (voice là Voice, lớp con str). Ghi file vào out_path
    (đuôi theo --backend-opt raw_ext=..., mặc định .wav)."""
    name = "python"

    def __init__(self, python_fn: str = "", **opts):
        super().__init__(**opts)
        if not python_fn:
            raise SystemExit("backend python cần --python-fn module:function")
        self.fn = _load_callable(python_fn)

    def synth(self, text, voice, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.fn(text, voice, str(out_path))
        if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
            raise RuntimeError("hàm python chạy xong nhưng không ghi file đầu ra")


class EdgeBackend(Backend):
    """Gói edge-tts (pip install edge-tts). GỬI VĂN BẢN LÊN DỊCH VỤ ONLINE CỦA MICROSOFT -> chỉ dùng khi kịch bản được
    phép rời máy; KHÔNG phải mặc định. Giọng tiếng Việt thường gặp: vi-VN-HoaiMyNeural, vi-VN-NamMinhNeural (chỉ 2 giọng:
    muốn ≥ 3 giọng/thuật ngữ phải thêm engine khác). Ghi mp3 (raw_ext .mp3), tts_synth chuyển sang wav 16 kHz nhờ soundfile /
    ffmpeg. Chưa chạy thử trong repo này."""
    name = "edge"
    raw_ext = ".mp3"

    def __init__(self, rate: str = "+0%", **opts):
        super().__init__(**opts)
        self.rate = str(rate)

    def synth(self, text, voice, out_path):
        try:
            import asyncio
            import edge_tts
        except ImportError:
            raise SystemExit("backend edge cần gói edge-tts: pip install edge-tts (lưu ý: gửi văn bản lên dịch vụ Microsoft)")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(edge_tts.Communicate(text, str(voice), rate=self.rate).save(str(out_path)))


BACKENDS = ("mock", "command", "python", "edge")


def coerce(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def make_backend(name: str, command: str = "", python_fn: str = "", opts: Optional[Dict] = None, timeout: float = 120.0
                 ) -> Backend:
    opts = dict(opts or {})
    if name == "mock":
        return MockBackend(**opts)
    if name == "command":
        return CommandBackend(command=command, timeout=timeout, **opts)
    if name == "python":
        return PythonBackend(python_fn=python_fn, **opts)
    if name == "edge":
        return EdgeBackend(**opts)
    raise SystemExit(f"backend '{name}' không có; chọn: {', '.join(BACKENDS)}")
