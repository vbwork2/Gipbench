"""Read the pinned FPT FOSD parquet snapshot and score Gipformer ONNX."""

from collections.abc import Iterable, Iterator
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json

import numpy as np
import yaml

from gipbench.infer.adapters.sherpa_transducer import SherpaTransducerAdapter


ROOT = Path(__file__).resolve().parents[2]


def settings() -> tuple[dict, str]:
    config = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    model = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    return config, model["vocabulary_case"]


def parquet_files(data_dir: Path) -> list[Path]:
    files = [data_dir / f"train-{index:05d}-of-00002.parquet" for index in range(2)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing FPT snapshot shard: {path}")
    return files


def audio_indices(files: list[Path], wanted: set[int]) -> Iterator[tuple[int, dict]]:
    import pyarrow.parquet as pq

    offset = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            selected = sorted(index - offset for index in wanted if offset <= index < offset + size)
            if selected:
                batch = parquet.read_row_group(group, columns=["audio", "transcription"])
                for index, row in zip(selected, batch.take(selected).to_pylist(), strict=True):
                    yield offset + index, row
            offset += size


def read_waveform(raw: bytes) -> tuple[np.ndarray, int]:
    import soundfile

    waveform, rate = soundfile.read(BytesIO(raw), dtype="float32", always_2d=True)
    return np.ascontiguousarray(waveform.mean(axis=1)), rate


def manifest_waveform(row: dict, raw: dict) -> tuple[np.ndarray, int]:
    audio = raw["audio"]
    if audio["path"] != row["audio_file"]:
        raise ValueError(f"Audio filename mismatch for FPT dataset row {row['audio_idx']}.")
    expected = row.get("audio_sha256")
    if expected and sha256(audio["bytes"]).hexdigest() != expected:
        raise ValueError(f"Audio checksum mismatch for FPT dataset row {row['audio_idx']}.")
    return read_waveform(audio["bytes"])


def recognizer(model_dir: Path, work_dir: Path, config: dict, case: str, method: str,
               hotwords: Path | None = None, hotwords_score: float | None = None):
    spec = {
        "system": f"gipformer-{method}",
        "model": {"id": "g-group-ai-lab/gipformer1.5-68M-rnnt", "dir": model_dir},
        "decode": {**config["decode"], "decoding_method": method, "case": case},
        "inference_mode": "context_bias" if hotwords else "open_vocab",
    }
    if hotwords:
        spec["context_vocab"] = {"file": str(hotwords), "score": hotwords_score, "provenance": "external"}
    adapter = SherpaTransducerAdapter(spec, work_dir)
    adapter.load()
    return adapter.rec


def transcribe(rec, items: Iterable[tuple[np.ndarray, int]]) -> list[str]:
    streams = []
    for waveform, rate in items:
        stream = rec.create_stream()
        stream.accept_waveform(rate, waveform)
        streams.append(stream)
    if streams:
        rec.decode_streams(streams)
    return [stream.result.text for stream in streams]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def decode_rows(rows: list[dict], data_dir: Path, rec, batch_size: int) -> dict[int, str]:
    wanted = {item["audio_idx"]: item for item in rows}
    hyps: dict[int, str] = {}
    batch, waves = [], []

    def flush() -> None:
        hyps.update(zip(batch, transcribe(rec, waves), strict=True))
        batch.clear()
        waves.clear()

    for index, raw in audio_indices(parquet_files(data_dir), set(wanted)):
        batch.append(index)
        waves.append(manifest_waveform(wanted[index], raw))
        if len(batch) >= batch_size:
            flush()
            if len(hyps) % (batch_size * 32) == 0:
                print(f"Decoded {len(hyps)} of {len(wanted)} utterances", flush=True)
    flush()
    if hyps.keys() != wanted.keys():
        raise ValueError("Some requested FPT rows do not exist in the snapshot.")
    return hyps
