"""Greedy RNN-T decoding over the Gipformer ONNX triplet, with a blank penalty that a gate can switch per frame.

Gates (decided per frame, before the argmax):
  uniform       every frame; matches sherpa-onnx `blank_penalty` for greedy search
  continuation  the best non-blank piece continues the current word (no word-start marker)
                and at least one token was already emitted
  word_start    the best non-blank piece starts a new word (structured control)
  random        a seeded Bernoulli draw at a fixed rate (rate-matched random control)
"""

from pathlib import Path

import numpy as np

WORD_START = "▁"
GATES = ("uniform", "continuation", "word_start", "random")


def read_pieces(path: Path) -> list[str]:
    pieces = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            piece, index = line.rsplit(maxsplit=1)
            pieces[int(index)] = piece
    if sorted(pieces) != list(range(len(pieces))):
        raise ValueError("tokens.txt must list contiguous token IDs from 0.")
    return [pieces[index] for index in range(len(pieces))]


def continuation_mask(pieces: list[str], blank: int) -> np.ndarray:
    """True for pieces that continue a word: no word-start marker and not a special <...> symbol."""
    mask = np.array([
        not piece.startswith(WORD_START) and not (piece.startswith("<") and piece.endswith(">")) for piece in pieces
    ])
    mask[blank] = False
    return mask


def condition(gate: str, logit: np.ndarray, emitted: int, continuation: np.ndarray, blank: int) -> bool:
    """Whether the structural condition of `gate` holds at this frame (random gates use `fires`)."""
    if gate == "uniform":
        return True
    masked = logit.copy()
    masked[blank] = -np.inf
    best = int(masked.argmax())
    if gate == "continuation":
        return emitted > 0 and bool(continuation[best])
    if gate == "word_start":
        return not continuation[best]
    raise ValueError(f"Gate {gate} has no structural condition.")


def pieces_to_text(ids: list[int], pieces: list[str]) -> str:
    return "".join(pieces[index] for index in ids).replace(WORD_START, " ").strip()


class OnnxTransducer:
    def __init__(self, model_dir: Path, quantize: str = "int8", threads: int = 4, sample_rate: int = 16000,
                 feature_dim: int = 80):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        suffix = ".int8.onnx" if quantize == "int8" else ".onnx"

        def session(name):
            return ort.InferenceSession(str(model_dir / f"{name}{suffix}"), options, providers=["CPUExecutionProvider"])

        self._session = session
        self._encoder = None
        self.decoder, self.joiner = session("decoder"), session("joiner")
        self.context_size = int(self.decoder.get_modelmeta().custom_metadata_map["context_size"])
        self.pieces = read_pieces(model_dir / "tokens.txt")
        self.blank = 0
        self.unk = self.pieces.index("<unk>") if "<unk>" in self.pieces else -1
        self.continuation = continuation_mask(self.pieces, self.blank)
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self._decoder_cache: dict[tuple[int, ...], np.ndarray] = {}

    def features(self, waveform: np.ndarray, rate: int) -> np.ndarray:
        from gipformer_mech.fbank import fbank
        from gipformer_mech.rnnt_loader import resample

        return fbank(resample(waveform, rate, self.sample_rate), self.sample_rate, self.feature_dim)

    def encode(self, features: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            self._encoder = self._session("encoder")
        out, lengths = self._encoder.run(None, {"x": features[None], "x_lens": np.array([len(features)], dtype=np.int64)})
        return out[0, :int(lengths[0])]

    def decoder_out(self, context: tuple[int, ...]) -> np.ndarray:
        if context not in self._decoder_cache:
            y = np.array([context], dtype=np.int64)
            self._decoder_cache[context] = self.decoder.run(None, {"y": y})[0]
        return self._decoder_cache[context]

    def greedy(self, encoder_out: np.ndarray, penalty: float = 0.0, gate: str = "uniform",
               rng: np.random.Generator | None = None, rate: float = 0.0, initial: tuple[int, ...] | None = None) -> dict:
        """One symbol per frame, as in sherpa-onnx offline greedy search."""
        if gate not in GATES:
            raise ValueError(f"Unknown gate {gate}.")
        if gate == "random" and penalty and rng is None:
            raise ValueError("A random gate needs a seeded generator.")
        context = tuple(initial) if initial is not None else (-1,) * (self.context_size - 1) + (self.blank,)
        decoder_out = self.decoder_out(context)
        ids, held, fired = [], 0, 0
        for frame in range(len(encoder_out)):
            logit = self.joiner.run(None, {"encoder_out": encoder_out[frame:frame + 1], "decoder_out": decoder_out})[0][0]
            if gate == "random":
                active = bool(penalty) and rng.random() < rate
            else:
                active = condition(gate, logit, len(ids), self.continuation, self.blank)
            held += active
            if penalty and active:
                logit = logit.copy()
                logit[self.blank] -= penalty
                fired += 1
            token = int(logit.argmax())
            if token != self.blank and token != self.unk:
                ids.append(token)
                context = context[1:] + (token,)
                decoder_out = self.decoder_out(context)
        return {"ids": ids, "text": pieces_to_text(ids, self.pieces), "frames": len(encoder_out),
                "condition_frames": held, "penalized_frames": fired}
