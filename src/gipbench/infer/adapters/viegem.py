"""Adapter VieGem-ASR (5CD-AI/VieGem-ASR): CTC 359M trên audio encoder EmbeddingGemma 2. Chạy ĐÚNG code suy luận
chính thức (thư mục `asr/` trong repo HF), không viết lại model.

Yêu cầu (đã kiểm từ model card + code upstream, revision 95eae70b):
  * snapshot repo VieGem-ASR (viegem_asr.pt fp16 311 MB, asr/*.py, lm/ 71 MB) ở `model.dir`;
  * snapshot google/embeddinggemma-2 (1.5 GB, Apache-2.0, KHÔNG gated; phải tuân Gemma Terms of Use) ở `encoder_dir`:
    CTCModel tự nạp toàn bộ encoder gốc từ đây (checkpoint VieGem chỉ chứa phần đã train + head). Truyền đường dẫn
    local thay cho id HF để khoá revision;
  * transformers đủ mới để có model_type `embedding_gemma2` (config ghi 5.18.0.dev0; venv gipbench-asr có 4.57.6 là
    KHÔNG đủ) -> dùng venv riêng, xem docs/viegem_integration.md;
  * audio 16 kHz mono, mỗi lần <= 30 s (model card). Audio dài hơn bị cắt thành đoạn 30 s liên tiếp rồi nối text.
Giải mã: `greedy` (CTC greedy, như README) hoặc `lm` (KenLM 4-gram đi kèm, alpha 0.35, beta 1.5, beam 32 như README;
cần `pip install kenlm`). LM train trên transcript train của VieGem (theo model card), không phải text test của gipbench.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from gipbench.infer.adapters.base import ASRAdapter, AdapterInfo, file_sha256, load_audio, split_long

PKG = "viegem_upstream_asr"


def import_upstream(repo_dir: Path):
    """Nạp package `asr` của repo VieGem dưới tên riêng (không đụng sys.path); trả về (model, text, lm_decode)."""
    pkg_dir = Path(repo_dir) / "asr"
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = mod
        spec.loader.exec_module(mod)
    return (importlib.import_module(f"{PKG}.model"), importlib.import_module(f"{PKG}.text"),
            importlib.import_module(f"{PKG}.lm_decode") if (pkg_dir / "lm_decode.py").exists() else None)


class VieGemAdapter(ASRAdapter):
    def __init__(self, cfg: Dict):
        d = cfg.get("decode") or {}
        self.cfg = cfg
        self.repo = Path(cfg["model"]["dir"])
        self.encoder_dir = cfg["model"].get("encoder_dir")
        self.method = d.get("method", "greedy")
        if self.method not in ("greedy", "lm"):
            raise ValueError("decode.method: greedy | lm")
        if self.method == "lm" and cfg["inference_mode"] != "lm":
            raise ValueError("decode.method lm cần inference_mode: lm")
        self.lm_cfg = {"alpha": float(d.get("alpha", 0.35)), "beta": float(d.get("beta", 1.5)),
                       "beam_width": int(d.get("beam_width", 32))}
        self.device_req = d.get("device", "auto")
        # pad_sec > 0: đệm im lặng hai đầu audio (ablation CHẨN ĐOÁN độ nhạy với clip cắt sát; mặc định 0 = như upstream)
        self.pad_sec = float(d.get("pad_sec", 0.0))
        self.info = AdapterInfo(system=cfg["system"], engine="viegem", model_id=cfg["model"]["id"],
                                revision=cfg["model"].get("revision", "unknown"), inference_mode=cfg["inference_mode"],
                                sample_rate=16000, decoding={"method": self.method, **(self.lm_cfg if self.method == "lm" else {}),
                                                              "streaming": False, "pad_sec": self.pad_sec},
                                batch_size=int(cfg.get("batch", 8)), max_chunk_sec=float(d.get("max_chunk_sec", 30.0)))
        self.model = None

    def load(self):
        import torch
        M, T, L = import_upstream(self.repo)
        ckpt = self.repo / "viegem_asr.pt"
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        cfg = M.ModelConfig.from_dict(ck["model_config"])
        upstream_encoder = cfg.encoder_id
        if self.encoder_dir:
            cfg.encoder_id = str(self.encoder_dir)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.encoder_id)
        self.units = T.Units(ck["gemma_ids"], cfg.encoder_id)
        model = M.CTCModel(cfg, self.units.gemma_ids)
        res = model.load_state_dict(ck["model"], strict=False)
        bad_missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
        if bad_missing or res.unexpected_keys:
            raise RuntimeError(f"checkpoint không khớp code: thiếu {bad_missing[:5]}, thừa {res.unexpected_keys[:5]}")
        use_cuda = torch.cuda.is_available() and self.device_req in ("auto", "cuda")
        self.device = "cuda" if use_cuda else "cpu"
        self.model = model.to(self.device).eval()
        self.greedy = M.greedy_decode
        self.torch = torch
        self.info.device = self.device
        self.info.precision = "bf16-autocast" if use_cuda else "fp32"
        self.info.threads = torch.get_num_threads()
        self.info.extra.update({"checkpoint_sha256": file_sha256(ckpt), "upstream_encoder_id": upstream_encoder,
                                "encoder_dir": self.encoder_dir, "n_units": len(self.units),
                                "encoder_keys_from_base": len(res.missing_keys)})
        try:
            import transformers
            self.info.extra["transformers"] = transformers.__version__
        except Exception:
            pass
        if self.method == "lm":
            words = (self.repo / "lm" / "viegem_lm.unigrams.txt").read_text(encoding="utf-8").split()
            self.lm = L.LMDecoder(str(self.repo / "lm" / "viegem_lm.bin"), unigrams=words,
                                  alpha=self.lm_cfg["alpha"], beta=self.lm_cfg["beta"])
            self.pieces = ["", ""] + self.tok.convert_ids_to_tokens(self.units.gemma_ids)
            self.info.extra["lm_sha256"] = file_sha256(self.repo / "lm" / "viegem_lm.bin")

    def _decode_wavs(self, wavs: List[np.ndarray]) -> List[str]:
        torch = self.torch
        lens = torch.tensor([len(w) for w in wavs])
        x = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            x[i, :len(w)] = torch.from_numpy(w)
        x, lens = x.to(self.device), lens.to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            logits, olens = self.model(x, lens, streaming=False)
        if self.method == "greedy":
            return [self.units.decode(self.tok, ids) for ids in self.greedy(logits, olens)]
        out = []
        for b in range(logits.shape[0]):
            lp = torch.log_softmax(logits[b, : olens[b]].float(), -1).cpu().numpy()
            out.append(self.units.decode(self.tok, self.lm.decode(lp, self.pieces, beam_width=self.lm_cfg["beam_width"])))
        return out

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        pieces, owner = [], []
        for i, p in enumerate(audio_paths):
            x, _ = load_audio(p, 16000)
            parts = split_long(x, 16000, self.info.max_chunk_sec)
            if self.pad_sec > 0:
                z = np.zeros(int(self.pad_sec * 16000), dtype=np.float32)
                parts = [np.concatenate([z, q, z]) for q in parts]
            if len(parts) > 1:
                self.info.extra["n_chunked_utts"] = self.info.extra.get("n_chunked_utts", 0) + 1
            pieces += parts
            owner += [i] * len(parts)
        texts = self._decode_wavs(pieces)
        out = [[] for _ in audio_paths]
        for i, t in zip(owner, texts):
            out[i].append(t.strip())
        return [" ".join(t for t in ts if t) for ts in out]
