#!/usr/bin/env python3
"""Tính fbank 80 chiều (cấu hình giống recipe librispeech/zipformer của icefall) cho CutSet.

  python ft/compute_fbank.py --cuts data/lhotse/cs_train_cuts_raw.jsonl.gz \
      --out-dir data/fbank --name cs_train --num-jobs 8

Kết quả: data/fbank/<name>_cuts.jsonl.gz — đưa vào finetune_cs.py qua --train-cuts / --valid-cuts.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuts", required=True)
    ap.add_argument("--out-dir", default="data/fbank")
    ap.add_argument("--name", required=True)
    ap.add_argument("--num-jobs", type=int, default=4)
    args = ap.parse_args()

    import torch
    from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cuts = CutSet.from_file(args.cuts)
    cuts = cuts.compute_and_store_features(
        extractor=Fbank(FbankConfig(num_mel_bins=80)),
        storage_path=str(out / f"{args.name}_feats"),
        num_jobs=args.num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    path = out / f"{args.name}_cuts.jsonl.gz"
    cuts.to_file(path)
    print(f"{len(cuts)} cut có fbank -> {path}")


if __name__ == "__main__":
    main()
