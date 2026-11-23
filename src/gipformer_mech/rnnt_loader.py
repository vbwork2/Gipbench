"""Load the audited Gipformer checkpoint with the Icefall Zipformer recipe."""

import argparse
from pathlib import Path
import sys

import numpy as np

from gipformer_mech.model_audit import checksum


def load_model(model_dir: Path, icefall: Path, model_map: dict, device: str):
    import sentencepiece
    import torch

    checkpoint = model_dir / "model.pt"
    if checksum(checkpoint) != model_map["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the Phase 0 audit.")
    asr = icefall / "egs/librispeech/ASR"
    recipe = asr / "zipformer"
    if not (recipe / "train.py").is_file():
        raise FileNotFoundError(f"Install the Icefall Zipformer recipe at {recipe}.")
    for directory in (icefall, asr, recipe):
        path = str(directory.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    from train import add_model_arguments, get_model

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    params = parser.parse_args([])
    stacks = model_map["encoder_stacks"]
    params.encoder_dim = ",".join(str(stack["dimension"]) for stack in stacks)
    params.downsampling_factor = ",".join(str(stack["downsampling_factor"]) for stack in stacks)
    params.num_encoder_layers = ",".join(str(stack["block_count"]) for stack in stacks)
    middle_ffn = [
        next(value for key, value in stack["feed_forward"].items() if key.endswith(".feed_forward2"))
        for stack in stacks
    ]
    params.feedforward_dim = ",".join(str(ffn["in_proj.weight"]["shape"][0]) for ffn in middle_ffn)
    params.blank_id = model_map["blank_id"]
    params.vocab_size = model_map["vocabulary_size"]
    params.feature_dim = model_map["feature_dim"]
    params.context_size = model_map["predictor"]["context_size"]
    params.decoder_dim = model_map["predictor"]["tensors"]["decoder.embedding.weight"][1]
    params.joiner_dim = model_map["joiner"]["tensors"]["joiner.output_linear.weight"][1]
    params.use_ctc = model_map["has_ctc_head"]
    params.chunk_size = "-1"
    params.left_context_frames = "-1"

    model = get_model(params)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_data.get("model", checkpoint_data), strict=True)
    model.eval().to(device)
    sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "bpe.model"))
    return model, sp


def resample(waveform: np.ndarray, rate: int, sample_rate: int) -> np.ndarray:
    import soxr

    if rate == sample_rate:
        return waveform
    return soxr.resample(waveform, rate, sample_rate)


def encode(model, waveform: np.ndarray, model_map: dict, device: str):
    import kaldi_native_fbank as kaldi
    import torch

    opts = kaldi.FbankOptions()
    opts.frame_opts.samp_freq = model_map["sample_rate"]
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.high_freq = -400
    opts.mel_opts.num_bins = model_map["feature_dim"]
    fbank = kaldi.OnlineFbank(opts)
    fbank.accept_waveform(model_map["sample_rate"], waveform.tolist())
    fbank.input_finished()
    frames = np.stack([fbank.get_frame(index) for index in range(fbank.num_frames_ready)])
    features = torch.from_numpy(frames).unsqueeze(0).to(device)
    lengths = torch.tensor([len(frames)], device=device)
    return model.forward_encoder(features, lengths)


def greedy(model, output, lengths, sp) -> str:
    from beam_search import greedy_search_batch

    ids = greedy_search_batch(model=model, encoder_out=output, encoder_out_lens=lengths)
    return sp.decode(ids[0])
