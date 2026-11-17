"""Phần dùng chung cho các converter: nạp dataset, lưu audio, ghi manifest + phiếu duyệt.

Tự gắn tag <en> (bản nháp cho người duyệt, không phải nhãn cuối) nằm ở gipbench.core.tagging.
"""
import csv
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


# Gắn tag <en>, nhận diện âm tiết tiếng Việt: ở gipbench.core.tagging (core cũng dùng). Import lại để converter giữ nguyên.
from gipbench.core.tagging import (AMBIG_EN, SUSPECT_ALONE, VI_WORD_FILES, add_vi_words_arg, auto_tag,  # noqa: F401
                                   is_vietnamese_token, load_vi_words)


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


