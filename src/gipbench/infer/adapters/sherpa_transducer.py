"""Adapter Zipformer-transducer ONNX qua sherpa-onnx (Gipformer 1.5 hoặc bản FT export lại).

Dùng lại đúng logic của gipbench.infer.decode (chọn file int8/fp32, hotwords từ catalog, bpe.vocab) để kết quả trùng
với E0 / E1 khi cùng tham số: audio đưa vào ở tần số GỐC của file (sherpa-onnx tự resample), trộn về mono.

Config (configs/bench/asr_models/*.yaml):
  engine: sherpa_transducer
  model: {id, revision, dir}
  decode: {quantize: int8|fp32, decoding_method: modified_beam_search|greedy_search, beam: 4, threads: 16,
           blank_penalty: 0.0, case: upper}
  Optional: decode.lm and decode.lm_scale for a compatible RNN LM with inference_mode: lm.
  context_vocab (tuỳ chọn, cần inference_mode context_bias / oracle_bias):
           {catalog: tsv, exclude_b3: true, score: 1.5, provenance: external|test_derived}  hoặc  {file: txt, ...}
"""
from pathlib import Path
from typing import Dict, List, Sequence

from gipbench.infer.adapters.base import ASRAdapter, AdapterInfo, file_sha256, load_audio
from gipbench.infer.decode import pick, write_hotwords


class SherpaTransducerAdapter(ASRAdapter):
    def __init__(self, cfg: Dict, work_dir: Path):
        d = cfg.get("decode") or {}
        self.cfg = cfg
        self.dec = {"quantize": d.get("quantize", "int8"), "decoding_method": d.get("decoding_method", "modified_beam_search"),
                    "beam": int(d.get("beam", 4)), "threads": int(d.get("threads", 4)),
                    "blank_penalty": float(d.get("blank_penalty", 0.0)), "case": d.get("case", "upper")}
        self.model_dir = Path(cfg["model"]["dir"])
        self.work_dir = Path(work_dir)
        self.lm = Path(d["lm"]) if d.get("lm") else None
        if self.lm is not None and not self.lm.is_absolute():
            from gipbench.core.paths import REPO
            self.lm = REPO / self.lm  # config ghi đường dẫn tương đối theo repo, không theo cwd
        if (self.lm is not None) != (cfg["inference_mode"] == "lm"):
            raise ValueError("decode.lm requires inference_mode: lm and vice versa")
        if self.lm is not None:
            if self.dec["decoding_method"] != "modified_beam_search":
                raise ValueError("decode.lm requires modified_beam_search")
            self.dec["lm_scale"] = float(d.get("lm_scale", 0.1))
        cv = cfg.get("context_vocab")
        self.hot = None
        if cv:
            from gipbench.core.textnorm import Catalog
            self.work_dir.mkdir(parents=True, exist_ok=True)
            if cv.get("catalog"):
                self.hot = self.work_dir / "hotwords.txt"
                n = write_hotwords(Catalog.load(cv["catalog"]), self.hot, self.dec["case"], not cv.get("exclude_b3", False))
            else:
                self.hot = Path(cv["file"])
                n = sum(1 for l in open(self.hot, encoding="utf-8") if l.strip())
            cv = {"source": cv.get("catalog") or cv.get("file"), "source_sha256": file_sha256(cv.get("catalog") or cv.get("file")),
                  "path": str(self.hot), "sha256": file_sha256(self.hot), "n": n, "score": float(cv.get("score", 1.5)),
                  "exclude_b3": bool(cv.get("exclude_b3", False)), "provenance": cv["provenance"]}
        self.info = AdapterInfo(system=cfg["system"], engine="sherpa_transducer", model_id=cfg["model"]["id"],
                                revision=cfg["model"].get("revision", "unknown"), inference_mode=cfg["inference_mode"],
                                sample_rate=0, decoding=self.dec, device="cpu", precision=self.dec["quantize"],
                                threads=self.dec["threads"], batch_size=int(cfg.get("batch", 16)), context_vocab=cv,
                                extra={"sample_rate_note": "audio đưa vào ở tần số gốc; sherpa-onnx tự resample về 16 kHz"})
        self.rec = None

    def load(self):
        import sherpa_onnx
        md, q = self.model_dir, self.dec["quantize"]
        kw = dict(encoder=str(pick(md, "encoder", q)), decoder=str(pick(md, "decoder", q)), joiner=str(pick(md, "joiner", q)),
                  tokens=str(md / "tokens.txt"), num_threads=self.dec["threads"], decoding_method=self.dec["decoding_method"],
                  max_active_paths=self.dec["beam"], blank_penalty=self.dec["blank_penalty"])
        self.info.extra["model_files"] = {k: {"path": kw[k], "sha256": file_sha256(kw[k])} for k in ("encoder", "decoder", "joiner")}
        if self.lm is not None:
            if not self.lm.is_file():
                raise FileNotFoundError(f"RNN LM not found: {self.lm}")
            kw.update(lm=str(self.lm), lm_scale=self.dec["lm_scale"])
            self.info.extra["lm"] = {"path": str(self.lm), "sha256": file_sha256(self.lm)}
            self.info.extra["tokens_sha256"] = file_sha256(md / "tokens.txt")
        if self.hot:
            if self.dec["decoding_method"] != "modified_beam_search":
                raise ValueError("hotwords chỉ chạy với modified_beam_search")
            vocab = md / "bpe.vocab"
            if not vocab.exists():
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=str(md / "bpe.model"))
                vocab = self.work_dir / "bpe.vocab"
                with open(vocab, "w", encoding="utf-8") as f:
                    for i in range(sp.get_piece_size()):
                        f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
            kw.update(hotwords_file=str(self.hot), hotwords_score=self.info.context_vocab["score"], modeling_unit="bpe",
                      bpe_vocab=str(vocab))
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)

    def transcribe_batch(self, audio_paths: Sequence[str]) -> List[str]:
        streams = []
        for p in audio_paths:
            x, sr = load_audio(p, target_sr=None)
            if len(x) == 0:  # file chỉ có header (vd 1 câu ViMedCSS test trong E0): sherpa-onnx lỗi shape {0,80}; ra "" như VieGem
                streams.append(None)
                continue
            s = self.rec.create_stream()
            s.accept_waveform(sr, x)
            streams.append(s)
        live = [s for s in streams if s is not None]
        if live:
            self.rec.decode_streams(live)
        return [s.result.text if s is not None else "" for s in streams]
