"""Phần dùng chung cho các converter: nạp dataset, lưu audio, tự gắn tag <en>, ghi manifest + phiếu duyệt.

Tự gắn tag là BẢN NHÁP cho người duyệt, không phải nhãn cuối:
  C = khớp catalog hoặc danh sách thuật ngữ đi kèm dataset  -> tin cậy cao
  H = token không thể là âm tiết tiếng Việt (không dấu, sai cấu trúc âm tiết) -> cần duyệt
  A = từ vừa hợp lệ tiếng Việt vừa là từ Anh hay gặp (the, to, chat...) nằm cạnh từ Anh -> ghép vào span, cần duyệt
"""
import csv
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from lib.textnorm import Catalog, basic_tokens  # noqa: E402

# ------------------------------------------------------------------ âm tiết tiếng Việt (dạng không dấu)
_VI_SYL = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?"
    r"(oai|oay|uya|ai|ao|au|ay|eo|ia|iu|oa|oe|oi|ua|ui|uy|a|e|i|o|u|y)"
    r"(ch|ng|nh|[cmnpt])?$")
_VI_MARK = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]")
# từ Anh hay gặp nhưng trùng cấu trúc âm tiết Việt: chỉ ghép vào span khi đứng cạnh từ Anh khác
AMBIG_EN = set("""the to in on at it up no so go he be my me do can man men hit kit bit bot net mic pic tip top pop
cup but not get set let put cut tap map gap cap hot chat chip pin pen ten pay pet mac sim ai an long""".split())
# đứng một mình vẫn đáng ngờ là từ Anh -> chỉ gắn cờ cho người duyệt
SUSPECT_ALONE = set("chat bot net mic chip pin sim top hot pay mac ai".split())
_EDGE = re.compile(r"^([^\w<>]*)(.*?)([^\w<>]*)$", re.S)


# vần kết thúc bằng bán nguyên âm: không bao giờ có phụ âm cuối (chain, main, boat... không phải tiếng Việt)
_GLIDE_NUC = {"ai", "ao", "au", "ay", "eo", "ia", "iu", "oi", "ui", "ua", "oai", "oay", "uya"}


def is_vietnamese_token(tok: str) -> bool:
    """True nếu token có thể là một âm tiết tiếng Việt (viết đúng chính tả, có dấu thanh khi cần).

    Token không có dấu nào chỉ hợp lệ khi đúng cấu trúc âm tiết và còn thoả hai luật chính tả:
      - âm tiết tận cùng -p, -t, -c, -ch chỉ mang thanh sắc hoặc nặng, nên bản không dấu
        (chip, kit, mac, net, hot, tech, op) không phải tiếng Việt;
      - vần có bán nguyên âm cuối (ai, ao, oi, ...) không có phụ âm cuối; tiếng Việt viết -inh, không có -ing.
    Giả định text gốc có dấu đầy đủ (G-Group, ViMedCSS, VIVOS, VietSuperSpeech đều vậy).
    """
    t = unicodedata.normalize("NFC", tok.lower())
    if _VI_MARK.search(t) or t.isdigit():
        return True
    m = _VI_SYL.match(t)
    if not m:
        return False
    _, nuc, fin = m.groups()
    if fin and (fin in ("p", "t", "c", "ch") or nuc in _GLIDE_NUC or (nuc == "i" and fin == "ng")):
        return False
    return True


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fuzzy(tok: str, term: str) -> bool:
    if tok == term:
        return True
    if min(len(tok), len(term)) >= 5 and (tok.startswith(term) or term.startswith(tok)):
        return True
    return min(len(tok), len(term)) >= 6 and _edit(tok, term) <= 2


VI_WORD_FILES = [_ROOT / "configs" / "vi_abbrev.txt", _ROOT / "configs" / "vi_loanwords.txt"]
_VI_WORDS_CACHE: Dict[Tuple[str, ...], frozenset] = {}


def load_vi_words(paths: Optional[Sequence] = None, same_surface: bool = False) -> frozenset:
    """Đọc cột đầu của các file danh sách (bỏ dòng '#' và dòng header), viết thường, NFC.

    paths = None -> các file mặc định trong configs/ (file nào chưa có thì bỏ qua); [] -> tập rỗng.
    same_surface=False bỏ các dòng kind = same_surface của vi_loanwords.txt (vitamin, test, gen: trùng mặt chữ tiếng
    Anh, chỉ coi là tiếng Việt trong ViMedCSS; ở các bộ khác chúng theo quy ước B1).
    """
    paths = VI_WORD_FILES if paths is None else paths
    key = tuple(str(p) for p in paths) + (str(same_surface),)
    if key not in _VI_WORDS_CACHE:
        words = set()
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            for line in open(p, encoding="utf-8"):
                cols = line.rstrip("\n").split("\t")
                w = cols[0].strip()
                if not w or w.startswith("#") or w in ("abbr", "word"):
                    continue
                if not same_surface and len(cols) > 1 and cols[1].strip() == "same_surface":
                    continue
                words.add(unicodedata.normalize("NFC", w).lower())
        _VI_WORDS_CACHE[key] = frozenset(words)
    return _VI_WORDS_CACHE[key]


def auto_tag(text: str, catalog: Optional[Catalog] = None, known_terms: Sequence[str] = (),
             vi_words: Optional[Iterable[str]] = None, heuristic: bool = True) -> Dict:
    """Trả về {"text_tagged", "confidence": catalog|heuristic|none, "flags": [...], "spans": [...]}.

    vi_words: từ không bao giờ gắn <en> trừ khi khớp catalog (viết tắt đọc kiểu Việt, từ mượn); None = đọc
    configs/vi_abbrev.txt + configs/vi_loanwords.txt. Thuật ngữ đi kèm dataset (known_terms) mà mặt chữ khớp
    nằm trong vi_words cũng không được gắn. heuristic=False: chỉ gắn theo catalog + known_terms.
    """
    vi_words = load_vi_words() if vi_words is None else vi_words
    raw = text.split()
    parts = [_EDGE.match(w).groups() for w in raw]
    core = [unicodedata.normalize("NFC", p[1]).lower() for p in parts]
    n = len(core)
    lab = [""] * n
    span_id = [-1] * n
    spans: List[Tuple[int, int, str]] = []  # (start, end, kiểu)

    # 1) catalog (n-gram dài nhất, kể cả biến thể chính tả) rồi thuật ngữ đi kèm dataset
    keys = set()
    max_n = 1
    if catalog and catalog.strict_map:
        keys = set(catalog.strict_map)
        max_n = max(len(k) for k in keys)
    known = [tuple(basic_tokens(k)) for k in known_terms if k and k.strip()]
    known = [k for k in known if k]
    i = 0
    while i < n:
        hit = 0
        for L in range(min(max_n, n - i), 0, -1):
            if tuple(core[i:i + L]) in keys:
                hit = L
                break
        if not hit:
            for k in sorted(known, key=len, reverse=True):
                L = len(k)
                if i + L <= n and all(_fuzzy(core[i + j], k[j]) for j in range(L)) and not all(
                        is_vietnamese_token(core[i + j]) and core[i + j] != k[j] for j in range(L)) \
                        and " ".join(core[i:i + L]) not in vi_words:
                    hit = L
                    break
        if hit:
            for j in range(i, i + hit):
                lab[j] = "C"
            spans.append((i, i + hit, "C"))
            i += hit
        else:
            i += 1
    # 1b) viết tắt đọc kiểu Việt, từ mượn có trong từ điển: nhãn V (không bao giờ thành span, không làm mốc ghép)
    for i in range(n):
        if not lab[i] and core[i] in vi_words:
            lab[i] = "V"
    # 2) heuristic theo cấu trúc âm tiết
    for i in range(n):
        if lab[i] or not core[i] or not heuristic:
            continue
        if not is_vietnamese_token(core[i]) and re.search(r"[a-z]", core[i]) and len(core[i]) > 1:
            lab[i] = "H"
    # 2b) chuỗi chữ cái đơn (a p i) có ít nhất một phụ âm
    i = 0
    while i < n and heuristic:
        j = i
        while j < n and not lab[j] and len(core[j]) == 1 and core[j].isalpha() and not _VI_MARK.search(core[j]):
            j += 1
        if j - i >= 2 and any(c not in "aeiouy" for c in core[i:j]):
            for k in range(i, j):
                lab[k] = "H"
        i = max(j, i + 1)
    # 3) từ mơ hồ đứng cạnh từ Anh
    # vi_word:<từ> = từ chữ Latin được coi là tiếng Việt nhờ danh sách: câu B5 có cờ này phải qua người duyệt
    flags = {f"vi_word:{core[i]}" for i in range(n) if lab[i] == "V"}
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not lab[i] and core[i] in AMBIG_EN and (
                    (i > 0 and lab[i - 1] in "HA" and lab[i - 1]) or (i + 1 < n and lab[i + 1] in "HA" and lab[i + 1])):
                lab[i] = "A"
                flags.add("ambiguous_joined")
                changed = True
    for i in range(n):
        if not lab[i] and core[i] in SUSPECT_ALONE:
            flags.add(f"suspect:{core[i]}")
    # gộp H/A liền nhau thành span
    i = 0
    while i < n:
        if lab[i] in ("H", "A"):
            j = i
            while j < n and lab[j] in ("H", "A"):
                j += 1
            spans.append((i, j, "H"))
            i = j
        else:
            i += 1
    spans.sort()
    # viết tắt ngắn đoán bằng heuristic: chưa biết đọc kiểu Anh hay kiểu Việt -> người duyệt nghe rồi quyết
    for s, e, k in spans:
        if k == "H" and e - s == 1 and 2 <= len(core[s]) <= 4 and core[s].isalpha():
            flags.add(f"abbr_reading:{core[s]}")
    out, pos = [], 0
    for s, e, _ in spans:
        out += raw[pos:s]
        pre, post = parts[s][0], parts[e - 1][2]
        inner = " ".join([parts[s][1] + (parts[s][2] if e - s > 1 else "")] +
                         [raw[k] for k in range(s + 1, e - 1)] +
                         ([parts[e - 1][0] + parts[e - 1][1]] if e - s > 1 else []))
        out.append(f"{pre}<en>{inner}</en>{post}")
        pos = e
    out += raw[pos:]
    kinds = {k for _, _, k in spans}
    if "H" in kinds:
        flags.add("heuristic_span")
    conf = "none" if not spans else ("heuristic" if "H" in kinds else "catalog")
    return {"text_tagged": " ".join(out), "confidence": conf, "flags": sorted(flags),
            "spans": [" ".join(core[s:e]) for s, e, _ in spans], "span_kinds": [k for _, _, k in spans]}


# ------------------------------------------------------------------ nạp dataset
def iter_hf_parquet(repo: str, split: str, limit: int, batch_size: int = 32, fs=None,
                    root: str = "datasets/") -> Optional[Iterator[Dict]]:
    """Đọc `limit` dòng đầu của split từ file parquet trên Hugging Face (data/<split>-*.parquet) bằng pyarrow.

    Chỉ tải các row group cần dùng. Generator tự kết thúc sau `limit` dòng, nên không để lại stream đọc dở:
    `datasets` 5.2 + pyarrow 26 bị crash lúc thoát (PyGILState_Release, mã 134) khi dừng giữa một shard lớn.
    Trả về None nếu repo không có file theo mẫu đó. `fs`/`root` chỉ đổi khi kiểm thử (filesystem cục bộ).
    """
    import pyarrow.parquet as pq
    if fs is None:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
    files = sorted(fs.glob(f"{root}{repo}/data/{split}-*.parquet"))
    if not files:
        return None

    def gen():
        n = 0
        for f in files:
            with fs.open(f, "rb") as fh:
                for batch in pq.ParquetFile(fh).iter_batches(batch_size=batch_size):
                    for row in batch.to_pylist():
                        yield row
                        n += 1
                        if n >= limit:
                            return
    return gen()


def load_hf(repo: Optional[str], split: str, parquet_glob: Optional[str] = None, streaming: bool = False,
            limit: int = 0):
    """Nạp bằng `datasets`, audio để nguyên bytes (không cần torchcodec).

    Có `limit` và đọc từ Hugging Face thì chỉ đọc `limit` dòng đầu (xem iter_hf_parquet), không tải cả split.
    """
    if limit and repo and not parquet_glob:
        it = iter_hf_parquet(repo, split, limit)
        if it is not None:
            return it
        streaming = True
    from datasets import Audio, load_dataset
    if parquet_glob:
        ds = load_dataset("parquet", data_files=parquet_glob, split="train", streaming=streaming)
    else:
        ds = load_dataset(repo, split=split, streaming=streaming)
    feats = getattr(ds, "features", None) or {}
    if "audio" in feats and feats["audio"].__class__.__name__ == "Audio":
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def save_audio(audio, out_dir: Path, stem: str, target_sr: int = 0) -> Tuple[str, float, int]:
    """Ghi audio (dict {bytes, path} của datasets, hoặc đường dẫn) -> (tên file, duration, sample rate).

    target_sr > 0: audio bytes khác tần số đó (hoặc nhiều kênh) được resample (soxr HQ) + gộp mono, ghi WAV PCM16.
    Audio dạng đường dẫn có sẵn thì giữ nguyên (không copy).
    """
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, (str, Path)):
        info = sf.info(str(audio))
        return str(audio), info.frames / info.samplerate, info.samplerate
    data = audio.get("bytes")
    src = audio.get("path") or ""
    if data is None:
        info = sf.info(src)
        return src, info.frames / info.samplerate, info.samplerate
    info = sf.info(io.BytesIO(data))
    if target_sr and (info.samplerate != target_sr or info.channels != 1):
        # đổi tần số lấy mẫu (vd ViMedCSS 44.1 kHz -> 16 kHz) và gộp kênh, ghi WAV PCM16
        import soxr
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = soxr.resample(wav, sr, target_sr, quality="HQ")
        path = out_dir / f"{stem}.wav"
        sf.write(str(path), wav, target_sr, subtype="PCM_16")
        return str(path), len(wav) / target_sr, target_sr
    ext = {"WAV": ".wav", "FLAC": ".flac", "MP3": ".mp3", "OGG": ".ogg"}.get(info.format, Path(src).suffix or ".wav")
    path = out_dir / f"{stem}{ext}"
    path.write_bytes(data)
    return str(path), info.frames / info.samplerate, info.samplerate


def youtube_id(url_or_id: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([\w-]{6,})", url_or_id or "")
    return m.group(1) if m else re.sub(r"\W+", "_", url_or_id or "unk")


def safe(s) -> str:
    return re.sub(r"[^\w.-]+", "_", str(s)).strip("_")


# ------------------------------------------------------------------ ghi kết quả
# reading_note: người duyệt ghi cách đọc nghe được cho từ mơ hồ (xem docs/huong_dan_duyet_b1.md)
# text_tagged_orig: nhãn theo dataset gốc (chỉ ViMedCSS), để người duyệt so; sửa vào text_tagged_fixed theo Spec
REVIEW_COLS = ["id", "split", "audio", "duration", "text_tagged", "text_tagged_orig", "confidence", "flags",
               "decision", "text_tagged_fixed", "viet_levels", "reading_note", "region", "gender", "note"]


class Writer:
    """Gom utterance rồi ghi manifest JSONL (theo tên), phiếu duyệt CSV và thống kê."""

    def __init__(self, out_dir: str, dataset: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.rows: Dict[str, List[Dict]] = {}
        self.terms = Counter()
        self.vi_words = None  # tập từ không gắn <en> cho converter dùng chung Writer (None = mặc định configs/)

    def rel(self, path: str) -> str:
        # os.path.abspath chỉ xử lý chuỗi; Path.resolve() stat từng thành phần, rất chậm trên /mnt/<ổ> của WSL
        # (67k câu VietSuperSpeech: > 20 phút chỉ vì resolve)
        p, base = os.path.abspath(str(path)), os.path.abspath(str(self.out))
        return os.path.relpath(p, base) if p == base or p.startswith(base + os.sep) else p

    def add(self, name: str, row: Dict, tag: Optional[Dict] = None):
        if tag is not None:
            row["text_tagged"] = tag["text_tagged"]
            row["auto_tag"] = {"confidence": tag["confidence"], "flags": tag["flags"]}
            for s, k in zip(tag["spans"], tag["span_kinds"]):
                self.terms[(s, k)] += 1
        row["audio"] = self.rel(row["audio"])
        row["duration"] = round(float(row["duration"]), 3)
        self.rows.setdefault(name, []).append(row)

    def close(self, review: Sequence[str] = ()):
        for name, rows in self.rows.items():
            p = self.out / f"{name}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            hrs = sum(r["duration"] for r in rows) / 3600
            print(f"  {p}: {len(rows)} câu, {hrs:.2f} h")
            if name in review:
                rp = self.out / f"{name}_review.csv"
                with open(rp, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
                    w.writeheader()
                    for r in rows:
                        at = r.get("auto_tag", {})
                        w.writerow({"id": r["id"], "split": r["split"], "audio": r["audio"], "duration": r["duration"],
                                    "text_tagged": r["text_tagged"], "text_tagged_orig": r.get("text_tagged_orig", ""),
                                    "confidence": at.get("confidence", ""),
                                    "flags": " ".join(at.get("flags", [])), "decision": "", "text_tagged_fixed": "",
                                    "viet_levels": "", "reading_note": "", "region": r.get("region", ""),
                                    "gender": r.get("gender", ""), "note": ""})
                print(f"  {rp}: phiếu duyệt (điền cột decision = ok / fix / drop)")
        if self.terms:
            tp = self.out / "term_candidates.tsv"
            with open(tp, "w", encoding="utf-8") as f:
                f.write("term\tkind\tcount\n")
                for (t, k), c in self.terms.most_common():
                    f.write(f"{t}\t{'catalog' if k == 'C' else 'heuristic'}\t{c}\n")
            print(f"  {tp}: {len(self.terms)} thuật ngữ gặp được (bổ sung dòng 'heuristic' đúng vào catalog)")


def base_parser(desc: str):
    import argparse
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="thư mục ghi manifest + audio")
    ap.add_argument("--catalog", help="catalog.tsv để gắn tag tin cậy cao")
    ap.add_argument("--parquet", help="glob file parquet đã tải về (thay cho tải từ Hugging Face)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N dòng đầu (thử nhanh)")
    add_vi_words_arg(ap)
    return ap


def add_vi_words_arg(ap):
    ap.add_argument("--vi-words", nargs="*", default=None,
                    help="file danh sách từ không gắn <en> (mặc định configs/vi_abbrev.txt configs/vi_loanwords.txt; "
                         "truyền rỗng để tắt)")
