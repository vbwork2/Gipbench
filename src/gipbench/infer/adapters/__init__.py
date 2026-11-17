"""Adapter ASR không phụ thuộc model: base.ASRAdapter + run_manifest; build_adapter(cfg) chọn theo `engine`."""
from pathlib import Path
from typing import Dict

ENGINES = ("sherpa_transducer", "viegem", "faster_whisper", "qwen3_asr")


def build_adapter(cfg: Dict, work_dir: Path):
    eng = cfg.get("engine")
    if eng == "sherpa_transducer":
        from gipbench.infer.adapters.sherpa_transducer import SherpaTransducerAdapter
        return SherpaTransducerAdapter(cfg, work_dir)
    if eng == "viegem":
        from gipbench.infer.adapters.viegem import VieGemAdapter
        return VieGemAdapter(cfg)
    if eng in ("faster_whisper", "qwen3_asr"):
        from gipbench.infer.adapters.hf_multilingual import HFMultilingualAdapter
        return HFMultilingualAdapter(cfg)
    raise ValueError(f"engine '{eng}' không hỗ trợ; có: {', '.join(ENGINES)}")
