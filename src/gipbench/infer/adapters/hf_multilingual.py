"""Adapter tham chiếu (không bắt buộc): faster-whisper và Qwen3-ASR, dùng lại đúng hàm nạp của gipbench.infer.asr_hyps
(cùng tham số mặc định: faster-whisper ép ngôn ngữ "vi", beam 5, không VAD; Qwen3-ASR tự nhận ngôn ngữ). Chỉ để chẩn
đoán / so tham chiếu; cần venv gipbench-asr (requirements-asr.txt).

Config: engine: faster_whisper | qwen3_asr; model: {id, revision}; decode: {device, compute_type, dtype, language,
        beam_size, batch_size, max_new_tokens}
"""
import argparse
from typing import Dict, List, Sequence

from gipbench.infer.adapters.base import ASRAdapter, AdapterInfo


class HFMultilingualAdapter(ASRAdapter):
    def __init__(self, cfg: Dict):
        d = cfg.get("decode") or {}
        eng = {"faster_whisper": "faster-whisper", "qwen3_asr": "qwen3-asr"}[cfg["engine"]]
        self.ns = argparse.Namespace(engine=eng, model=cfg["model"].get("path") or cfg["model"]["id"],
                                     device=d.get("device", "auto"), compute_type=d.get("compute_type", "int8"),
                                     dtype=d.get("dtype", "bf16"), language=d.get("language"),
                                     beam_size=int(d.get("beam_size", 5)), batch_size=int(d.get("batch_size", 1)),
                                     max_new_tokens=int(d.get("max_new_tokens", 256)))
        self.info = AdapterInfo(system=cfg["system"], engine=cfg["engine"], model_id=cfg["model"]["id"],
                                revision=cfg["model"].get("revision", "unknown"), inference_mode=cfg["inference_mode"],
                                sample_rate=16000, decoding={k: v for k, v in vars(self.ns).items() if k != "model"},
                                device=self.ns.device, precision=self.ns.compute_type if eng == "faster-whisper" else self.ns.dtype,
                                batch_size=int(cfg.get("batch", 1)))
        self.run = None

    def load(self):
        from gipbench.infer.asr_hyps import load_engine
        self.run = load_engine(self.ns)

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        return list(self.run(list(audio_paths)))
