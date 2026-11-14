#!/usr/bin/env python3
"""Tạo finetune_cs.py từ egs/librispeech/ASR/zipformer/finetune.py của icefall để đọc dữ liệu của bạn.

  python ft/patch_finetune.py --icefall /path/to/icefall

Thêm 2 tham số:
  --train-cuts  PATH[:WEIGHT] ...  trộn nhiều CutSet bằng CutSet.mux theo trọng số
                                   (vd  data/fbank/cs_train_cuts.jsonl.gz:0.5 data/fbank/vi_replay_cuts.jsonl.gz:0.5)
  --valid-cuts  PATH ...           mỗi file là một tập valid riêng (log loss riêng)
Bỏ phần nạp GigaSpeech/LibriSpeech mặc định. Viết ra cạnh finetune.py: zipformer/finetune_cs.py.
"""
import argparse
import re
import sys
from pathlib import Path

ARGS = '''
    parser.add_argument(
        "--train-cuts",
        type=str,
        nargs="+",
        default=[],
        help="CutSet train dạng PATH[:WEIGHT]; nhiều file được trộn bằng CutSet.mux",
    )

    parser.add_argument(
        "--valid-cuts",
        type=str,
        nargs="+",
        default=[],
        help="Một hoặc nhiều CutSet valid",
    )
'''

TRAIN = '''    # --- gipbench: nạp dữ liệu fine-tune của bạn thay cho GigaSpeech/LibriSpeech
    from lhotse import load_manifest_lazy as _load_lazy

    _sets, _weights = [], []
    for _spec in params.train_cuts:
        _path, _, _w = _spec.partition(":")
        _sets.append(_load_lazy(_path))
        _weights.append(float(_w) if _w else 1.0)
    if not _sets:
        raise ValueError("Cần --train-cuts")
    train_cuts = _sets[0] if len(_sets) == 1 else CutSet.mux(*_sets, weights=_weights, stop_early=True)
'''

VALID = '''    # --- gipbench: tập valid của bạn
    from lhotse import load_manifest_lazy as _load_lazy_v

    if not params.valid_cuts:
        raise ValueError("Cần --valid-cuts")
    valid_sets = [Path(p).name.replace("_cuts.jsonl.gz", "") for p in params.valid_cuts]
    valid_dls = [librispeech.valid_dataloaders(_load_lazy_v(p)) for p in params.valid_cuts]
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icefall", required=True)
    args = ap.parse_args()
    src = Path(args.icefall) / "egs/librispeech/ASR/zipformer/finetune.py"
    s = src.read_text(encoding="utf-8")

    anchor = 'def add_finetune_arguments(parser: argparse.ArgumentParser):\n'
    if anchor not in s:
        sys.exit("Không thấy add_finetune_arguments — phiên bản icefall khác, cần sửa tay (xem Runbook).")
    s = s.replace(anchor, anchor + ARGS, 1)

    m = re.search(r"\n    gigaspeech_cuts = librispeech\.gigaspeech_subset_small_cuts\(\)\n.*?\n    logging\.info\(train_cuts\)\n", s, re.S)
    if not m:
        sys.exit("Không thấy khối nạp train cuts — cần sửa tay (xem Runbook).")
    s = s[:m.start()] + "\n" + TRAIN + "    logging.info(train_cuts)\n" + s[m.end():]

    m = re.search(r"\n    valid_cuts = librispeech\.dev_clean_cuts\(\)\n.*?\n    valid_dls = \[\n.*?\n    \]\n", s, re.S)
    if not m:
        sys.exit("Không thấy khối nạp valid cuts — cần sửa tay (xem Runbook).")
    s = s[:m.start()] + "\n" + VALID + s[m.end():]

    dst = src.with_name("finetune_cs.py")
    dst.write_text(s, encoding="utf-8")
    print(f"Đã tạo {dst}")


if __name__ == "__main__":
    main()
