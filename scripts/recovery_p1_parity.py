#!/usr/bin/env python3
"""Phase 1 parity checks P1-A, P1-B, P1-C, P1-E of RESEARCH_RECOVERY.md (rules are registered there, section 2.1).

Run in WSL (FT venv, which has torch, Icefall, onnxruntime, Lhotse):
  PYTHONPATH=src:$HOME/jiwer_target ~/.venvs/gipbench-ft/bin/python scripts/recovery_p1_parity.py \
      --icefall ~/icefall --reexport data/work/recovery/p1/models/reexport --out data/work/recovery/p1/parity.json

Nothing here reads a locked bench. Samples come from dev/dev_v2_vi@1 and dev/dev_interim_cs@1, picked by sha256(id).
"""
import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GIP = REPO / "models/gip15"
SAMPLES = (("dev_v2_vi@1", 60), ("dev_interim_cs@1", 40))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_words(text):
    text = re.sub(r"</?[a-z]+>", " ", text)
    return re.sub(r"[^\w\s]", " ", text.upper()).split()


def edits(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


def wer(refs, hyps):
    n = sum(len(norm_words(r)) for r in refs)
    return 100.0 * sum(edits(norm_words(r), norm_words(h)) for r, h in zip(refs, hyps)) / n


def setup_icefall(icefall):
    asr = icefall / "egs/librispeech/ASR"
    for d in (icefall, asr, asr / "zipformer"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def build_model(use_ctc):
    """Same construction as finetune.py / export-onnx.py: train.py parser defaults + the repo's model_args."""
    from train import get_model, get_params, get_parser

    args = get_parser().parse_args(["--use-transducer", "1", "--use-ctc", str(use_ctc)])
    params = get_params()
    params.update(vars(args))
    params.blank_id, params.vocab_size = 0, 2000
    return get_model(params), params


def check_state_dict():
    import torch

    full = torch.load(GIP / "model.pt", map_location="cpu", weights_only=False)
    full = full.get("model", full)
    init = torch.load(GIP / "init_ft.pt", map_location="cpu", weights_only=False)["model"]
    no_ctc = {k: v for k, v in full.items() if not k.startswith("ctc_output.")}
    diff = [k for k in no_ctc if k not in init or not torch.equal(no_ctc[k], init[k])]
    out = {"model_pt_tensors": len(full), "ctc_tensors": sorted(set(full) - set(no_ctc)),
           "init_ft_tensors": len(init), "init_minus_model": sorted(set(init) - set(no_ctc)),
           "tensors_not_bit_equal": diff,
           "sha256": {"model.pt": sha256(GIP / "model.pt"), "init_ft.pt": sha256(GIP / "init_ft.pt")}}
    for name, use_ctc, sd in (("model.pt", 1, full), ("init_ft.pt", 0, init)):
        model, params = build_model(use_ctc)
        res = model.load_state_dict(sd, strict=False)
        out[f"strict_load_{name}"] = {"use_ctc": use_ctc, "missing": list(res.missing_keys),
                                      "unexpected": list(res.unexpected_keys),
                                      "params": sum(p.numel() for p in model.parameters())}
    out["pass"] = bool(not diff and not out["init_minus_model"] and out["ctc_tensors"]
                       and all(not out[f"strict_load_{n}"][k] for n in ("model.pt", "init_ft.pt")
                               for k in ("missing", "unexpected")))
    return out, init


def check_tokenizer(reexport):
    import onnx
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=str(GIP / "bpe.model"))
    rows = [ln.rsplit(maxsplit=1) for ln in (GIP / "tokens.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    table = {int(i): t for t, i in rows}
    bad = [i for i in range(sp.get_piece_size()) if table.get(i) != sp.id_to_piece(i)]
    meta = {}
    for arm, d in (("official", GIP), ("reexport", reexport)):
        for part in ("encoder", "decoder", "joiner"):
            m = onnx.load(str(d / f"{part}.int8.onnx"), load_external_data=False)
            meta[f"{arm}/{part}"] = {p.key: p.value for p in m.metadata_props}
    blanks = {k: v.get("blank_id") for k, v in meta.items() if "blank_id" in v}
    out = {"sp_vocab": sp.get_piece_size(), "tokens_rows": len(table), "mismatched_ids": bad[:20],
           "n_mismatched": len(bad), "blk_id": [i for i, t in table.items() if t == "<blk>"],
           "unk_id": sp.unk_id(), "onnx_blank_ids": blanks}
    out["pass"] = bool(not bad and len(table) == 2000 and out["blk_id"] == [0]
                       and all(str(v) == "0" for v in blanks.values()))
    return out, meta


def onnx_static(reexport):
    import onnx

    out = {}
    for arm, d in (("official", GIP), ("reexport", reexport)):
        for part in ("encoder", "decoder", "joiner"):
            p = d / f"{part}.int8.onnx"
            m = onnx.load(str(p))
            ops = collections.Counter(n.op_type for n in m.graph.node)
            dtypes = collections.Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer)
            out[f"{arm}/{part}"] = {
                "bytes": p.stat().st_size, "sha256": sha256(p), "opset": [(o.domain, o.version) for o in m.opset_import],
                "ir_version": m.ir_version, "producer": f"{m.producer_name} {m.producer_version}",
                "inputs": [i.name for i in m.graph.input], "outputs": [o.name for o in m.graph.output],
                "quant_ops": {k: v for k, v in ops.items() if "Integer" in k or "Quantize" in k},
                "n_nodes": len(m.graph.node), "initializer_dtypes": dict(dtypes)}
    return out


def load_samples():
    import soundfile as sf

    out = []
    for ref, k in SAMPLES:
        path = REPO / "data/releases/dev" / ref / "manifest.jsonl"
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        for r in rows[:k]:
            wav, rate = sf.read(str((path.parent / r["audio"]).resolve()), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if rate != 16000:
                import soxr
                wav = soxr.resample(wav, rate, 16000).astype(np.float32)
            out.append({"set": ref, "id": r["id"], "ref": r["text_tagged"], "wav": wav})
    return out


def knf_fbank(wav):
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, wav.tolist())
    fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)


def lhotse_fbank(wav):
    from lhotse import Fbank, FbankConfig

    return np.asarray(Fbank(FbankConfig(num_mel_bins=80)).extract(wav, 16000), dtype=np.float32)


def torch_encode(model, feats):
    import torch

    with torch.no_grad():
        x = torch.from_numpy(feats)[None]
        enc, lens = model.forward_encoder(x, torch.tensor([len(feats)]))
        return enc, lens


def torch_greedy(model, enc, lens):
    import torch
    from beam_search import greedy_search_batch

    with torch.no_grad():
        return greedy_search_batch(model=model, encoder_out=enc, encoder_out_lens=lens)[0]


def relerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icefall", type=Path, required=True)
    ap.add_argument("--reexport", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch
    import lhotse
    import onnxruntime

    torch.set_num_threads(args.threads)
    setup_icefall(args.icefall.expanduser())
    from gipformer_mech.onnx_greedy import OnnxTransducer, pieces_to_text

    t0 = time.time()
    res = {"versions": {"torch": torch.__version__, "lhotse": lhotse.__version__, "onnxruntime": onnxruntime.__version__},
           "script_sha256": sha256(__file__)}
    res["P1-A"], init = check_state_dict()
    res["P1-B"], res["onnx_metadata"] = check_tokenizer(args.reexport)
    res["onnx_static"] = onnx_static(args.reexport)

    model, _ = build_model(0)
    model.load_state_dict(init, strict=True)
    model.eval()
    arms = {"reexport_fp32": OnnxTransducer(args.reexport, "fp32", args.threads),
            "reexport_int8": OnnxTransducer(args.reexport, "int8", args.threads),
            "official_int8": OnnxTransducer(GIP, "int8", args.threads)}
    pieces = arms["official_int8"].pieces

    per = []
    for s in load_samples():
        fk, fl = knf_fbank(s["wav"]), lhotse_fbank(s["wav"])
        row = {"set": s["set"], "id": s["id"], "frames_knf": len(fk), "frames_lhotse": len(fl)}
        n = min(len(fk), len(fl))
        d = np.abs(fk[:n] - fl[:n])
        row.update(feat_max_abs=float(d.max()), feat_mean_abs=float(d.mean()), feat_p99_abs=float(np.quantile(d, 0.99)))
        enc, lens = torch_encode(model, fk)
        with torch.no_grad():
            proj = model.joiner.encoder_proj(enc)[0, :int(lens[0])].numpy()
        ids = {"torch_knf": torch_greedy(model, enc, lens)}
        enc_l, lens_l = torch_encode(model, fl)
        ids["torch_lhotse"] = torch_greedy(model, enc_l, lens_l)
        for name, arm in arms.items():
            e = arm.encode(fk)
            row[f"encerr_{name}"] = relerr(e, proj) if e.shape == proj.shape else None
            ids[name] = arm.greedy(e)["ids"]
        row["ids"] = ids
        row["ref"] = s["ref"]
        row["hyp"] = {k: pieces_to_text(v, pieces) for k, v in ids.items()}
        per.append(row)
        print(f"{len(per):3d} {s['id'][:50]:50s} encerr fp32 {row['encerr_reexport_fp32']:.2e} "
              f"int8 re {row['encerr_reexport_int8']:.2e} off {row['encerr_official_int8']:.2e} "
              f"feat max {row['feat_max_abs']:.3f}", flush=True)

    refs = [r["ref"] for r in per]
    summ = {"K": len(per)}
    for name in ("torch_knf", "torch_lhotse", "reexport_fp32", "reexport_int8", "official_int8"):
        summ[f"wer_{name}"] = wer(refs, [r["hyp"][name] for r in per])
        summ[f"same_ids_as_torch_knf_{name}"] = sum(r["ids"][name] == r["ids"]["torch_knf"] for r in per) / len(per)
    summ["same_ids_reexport_int8_vs_official_int8"] = sum(
        r["ids"]["reexport_int8"] == r["ids"]["official_int8"] for r in per) / len(per)
    for name in ("reexport_fp32", "reexport_int8", "official_int8"):
        v = [r[f"encerr_{name}"] for r in per if r[f"encerr_{name}"] is not None]
        summ[f"encerr_{name}_median"] = float(np.median(v))
        summ[f"encerr_{name}_max"] = float(np.max(v))
    summ["frames_equal"] = sum(r["frames_knf"] == r["frames_lhotse"] for r in per) / len(per)
    for k in ("feat_max_abs", "feat_mean_abs", "feat_p99_abs"):
        summ[f"{k}_max_over_utts"] = float(max(r[k] for r in per))
        summ[f"{k}_median_over_utts"] = float(np.median([r[k] for r in per]))
    res["P1-C"] = {"encerr_median": summ["encerr_reexport_fp32_median"],
                   "same_ids_rate": summ["same_ids_as_torch_knf_reexport_fp32"]}
    res["P1-C"]["pass"] = bool(res["P1-C"]["encerr_median"] < 1e-4 and res["P1-C"]["same_ids_rate"] == 1.0)
    res["P1-E"] = {"same_ids_rate": summ["same_ids_as_torch_knf_torch_lhotse"],
                   "delta_wer": summ["wer_torch_lhotse"] - summ["wer_torch_knf"],
                   "frames_equal": summ["frames_equal"], "feat_max_abs": summ["feat_max_abs_max_over_utts"],
                   "feat_mean_abs_median": summ["feat_mean_abs_median_over_utts"]}
    res["P1-E"]["pass"] = bool(res["P1-E"]["same_ids_rate"] >= 0.9 and abs(res["P1-E"]["delta_wer"]) <= 0.5)
    res["summary"] = summ
    res["per_utt"] = per
    res["seconds"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("P1-A", "P1-B", "P1-C", "P1-E")}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
