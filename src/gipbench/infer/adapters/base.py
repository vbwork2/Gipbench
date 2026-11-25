"""Hợp đồng adapter ASR dùng chung cho mọi model (Gipformer, VieGem-ASR, Whisper, Qwen3-ASR, ...).

Một adapter nhận danh sách đường dẫn audio, trả về đúng một transcript cho mỗi đầu vào, cùng thứ tự:

    class MyAdapter(ASRAdapter):
        def load(self): ...                                  # nạp model (gọi một lần)
        def transcribe_batch(self, audio_paths) -> list[str]

`run_manifest` lo phần chung: giải mã theo lô, resume theo id đã có, kiểm số hyp == số đầu vào, đo thời gian xử lý
(RTF = giây xử lý / giây audio, KHÔNG gồm thời gian nạp model), ghi run_meta.json.

Output (giữ nguyên SCHEMA.md mục 3):  <out>/hyps.jsonl  {"id": ..., "hyp": ...}  +  <out>/run_meta.json (metadata,
file riêng để không làm vỡ công cụ cũ).

inference_mode (bắt buộc trong config):
  open_vocab   : không có gợi ý nào lúc decode
  lm           : LM ngoài train trên text KHÔNG thuộc test (vd LM đi kèm model)
  context_bias : hotwords / từ vựng ngữ cảnh từ nguồn ngoài test (khai báo `context_vocab.provenance: external`)
  oracle_bias  : hotwords có thông tin từ test (cận trên chẩn đoán, không xếp hạng zero-shot)
"""
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from gipbench.core.fileio import sha256_file_or_none as file_sha256

MODES = ("open_vocab", "lm", "context_bias", "oracle_bias")


@dataclass
class AdapterInfo:
    system: str
    engine: str
    model_id: str
    revision: str
    inference_mode: str
    sample_rate: int = 16000                 # tần số adapter đưa vào model (sau resample nếu có)
    decoding: Dict = field(default_factory=dict)
    device: str = "cpu"
    precision: str = ""
    threads: Optional[int] = None
    batch_size: int = 1
    max_chunk_sec: Optional[float] = None    # audio dài hơn bị cắt thành đoạn liên tiếp (không chồng lấn)
    context_vocab: Optional[Dict] = None     # {"path", "sha256", "n", "provenance": external|test_derived}
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.inference_mode not in MODES:
            raise ValueError(f"inference_mode '{self.inference_mode}' không thuộc {MODES}")
        prov = (self.context_vocab or {}).get("provenance")
        if prov == "test_derived" and self.inference_mode != "oracle_bias":
            raise ValueError("context_vocab lấy từ test thì inference_mode phải là oracle_bias")
        if self.inference_mode in ("context_bias", "oracle_bias") and not self.context_vocab:
            raise ValueError(f"inference_mode {self.inference_mode} cần context_vocab (đường dẫn + provenance)")
        if self.context_vocab and self.inference_mode in ("open_vocab", "lm"):
            raise ValueError(f"có context_vocab nhưng inference_mode = {self.inference_mode}")


class ASRAdapter:
    info: AdapterInfo

    def load(self) -> None:
        pass

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        raise NotImplementedError


def load_audio(path: str, target_sr: Optional[int] = 16000):
    """Đọc audio (wav / flac / mp3 qua libsndfile), trộn về mono float32, resample bằng soxr nếu target_sr khác.
    target_sr=None: giữ tần số gốc. Trả về (mảng, sr)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x.mean(axis=1))
    if target_sr and sr != target_sr:
        import soxr
        x = soxr.resample(x, sr, target_sr).astype(np.float32)
        sr = target_sr
    return x, sr


def split_long(x: np.ndarray, sr: int, max_sec: Optional[float]) -> List[np.ndarray]:
    """Cắt audio dài thành đoạn liên tiếp <= max_sec (không chồng lấn; có thể cắt giữa từ). None = không cắt."""
    if not max_sec or len(x) <= max_sec * sr:
        return [x]
    n = int(max_sec * sr)
    return [x[i:i + n] for i in range(0, len(x), n)]


def hardware() -> Dict:
    hw = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        import torch
        hw["torch"] = torch.__version__
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return hw


def read_done(path: Path) -> Dict[str, str]:
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["id"]] = r.get("hyp", "")
    return done


def run_manifest(adapter: ASRAdapter, rows: Sequence[Dict], out_dir, batch: int = 8, resume: bool = True,
                 progress: Optional[Callable[[int, int], None]] = None, extra_meta: Optional[Dict] = None) -> Dict:
    """Giải mã mọi dòng (cần `id` và `_audio_path`), ghi <out_dir>/hyps.jsonl theo thứ tự manifest và run_meta.json.

    resume=True: giữ hyp của id đã có (hyps.partial.jsonl / hyps.jsonl của lần chạy trước), chỉ giải mã id còn thiếu.
    Lỗi trong một lô -> dừng hẳn (không ghi hyp rỗng thay cho lỗi). Trả về dict run_meta.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hp, part = out / "hyps.jsonl", out / "hyps.partial.jsonl"
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest có id trùng")
    done: Dict[str, str] = {}
    if resume:
        done.update(read_done(hp))
        done.update(read_done(part))
    done = {k: v for k, v in done.items() if k in set(ids)}
    todo = [r for r in rows if r["id"] not in done]
    t_load = time.perf_counter()
    if todo:
        adapter.load()
    load_sec = time.perf_counter() - t_load
    audio_sec = proc_sec = 0.0
    with open(part, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            t0 = time.perf_counter()
            hyps = adapter.transcribe_batch([r["_audio_path"] for r in chunk])
            proc_sec += time.perf_counter() - t0
            if len(hyps) != len(chunk):
                raise RuntimeError(f"adapter trả {len(hyps)} hyp cho {len(chunk)} audio")
            for r, h in zip(chunk, hyps):
                audio_sec += float(r.get("duration") or 0.0)
                done[r["id"]] = h
                f.write(json.dumps({"id": r["id"], "hyp": h}, ensure_ascii=False) + "\n")
            f.flush()
            if progress:
                progress(len(rows) - len(todo) + i + len(chunk), len(rows))
    with open(hp, "w", encoding="utf-8") as f:  # thứ tự manifest, đúng một hyp mỗi id
        for r in rows:
            f.write(json.dumps({"id": r["id"], "hyp": done[r["id"]]}, ensure_ascii=False) + "\n")
    part.unlink(missing_ok=True)
    meta = {"schema": "gipbench.run_meta/1", **asdict(adapter.info), "n_utts": len(rows),
            "n_decoded_this_run": len(todo), "n_resumed": len(rows) - len(todo),
            "audio_sec_this_run": round(audio_sec, 3), "proc_sec_this_run": round(proc_sec, 3),
            "model_load_sec": round(load_sec, 3), "rtf": (proc_sec / audio_sec) if audio_sec else None,
            "rtf_protocol": "giây trong transcribe_batch (gồm đọc + resample audio) / giây audio (trường duration); "
                            "không gồm nạp model; chỉ các câu giải mã trong lần chạy này",
            "hardware": hardware(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hyps_sha256": file_sha256(hp)}
    if extra_meta:
        meta.update(extra_meta)
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
