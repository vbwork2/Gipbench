#!/usr/bin/env python3
"""Tạo finetune_cs.py từ egs/librispeech/ASR/zipformer/finetune.py của icefall để đọc dữ liệu của bạn.

  python -m gipbench.model.train.patch_finetune --icefall /path/to/icefall

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

    parser.add_argument(
        "--mp-start-method",
        type=str,
        default="",
        help="spawn / forkserver cho worker DataLoader (rỗng = mặc định fork). fork + model_avg float64 trên CPU "
        "làm RAM worker phình dần (copy-on-write) khi tính fbank on-the-fly",
    )

    parser.add_argument(
        "--freeze-modules",
        type=str,
        default="",
        help="tiền tố tên tham số cần đóng băng, ngăn bằng dấu phẩy "
        "(vd encoder_embed.,encoder.encoders.0.,encoder.encoders.1.)",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Checkpoint and stop after this many training batches (0 uses the full epoch).",
    )
'''

FREEZE = '''    # --- gipbench: đóng băng một phần model (ScaledAdam coi grad None = 0 nên tham số đứng yên)
    if params.freeze_modules:
        _pref = [x.strip() for x in params.freeze_modules.split(",") if x.strip()]
        _nf = _nt = 0
        for _name, _p in model.named_parameters():
            _nt += _p.numel()
            if any(_name.startswith(x) for x in _pref):
                _p.requires_grad_(False)
                _nf += _p.numel()
        if _nf == 0:
            raise ValueError(f"--freeze-modules {_pref}: không khớp tham số nào")
        logging.info(f"freeze {_pref}: {_nf / 1e6:.2f}M / {_nt / 1e6:.2f}M tham số")
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

    # on-the-fly fbank: cut chưa có features -> num_frames = None; ước tính từ duration (frame shift 10 ms)
    nf_old = "        T = ((c.num_frames - 7) // 2 + 1) // 2\n"
    if nf_old not in s:
        sys.exit("Không thấy dòng T = ((c.num_frames - 7) ...) — cần sửa tay (xem Runbook).")
    s = s.replace(nf_old, "        _nf = c.num_frames if c.num_frames is not None else int(round(c.duration * 100))\n"
                          "        T = ((_nf - 7) // 2 + 1) // 2\n", 1)

    # resume (--start-batch): dict checkpoint (optimizer, sampler, model_avg ~1 GB) bị giữ suốt run() -> giải phóng
    rel_anchor = ('        scaler.load_state_dict(checkpoints["grad_scaler"])\n\n'
                  "    for epoch in range(params.start_epoch, params.num_epochs + 1):\n")
    if rel_anchor not in s:
        sys.exit("Không thấy chỗ nạp grad_scaler trước vòng epoch — cần sửa tay (xem Runbook).")
    s = s.replace(rel_anchor, rel_anchor.replace(
        "\n\n    for epoch", "\n\n    checkpoints = None  # gipbench: giải phóng RAM sau khi nạp xong\n"
        "    import gc as _gc\n    _gc.collect()\n\n    for epoch"), 1)

    # resume (--start-batch): scan_pessimistic_batches_for_oom duyệt hết sampler vừa nạp lại trạng thái -> lhotse coi là
    # đã đi hết epoch, iter() kế tiếp bắt đầu lại từ cut đầu tiên của manifest (E3 v1 16/11: sau resume ở step 15 495,
    # batch 0 / 50 / 100... trùng hệt batch đầu của lần chạy đầu). Chỉ quét khi bắt đầu mới.
    scan_old = "    if not params.print_diagnostics:\n        scan_pessimistic_batches_for_oom(\n"
    if scan_old not in s:
        sys.exit("Không thấy chỗ gọi scan_pessimistic_batches_for_oom — cần sửa tay (xem Runbook).")
    s = s.replace(scan_old, "    if not params.print_diagnostics and params.start_batch <= 0:  # gipbench\n"
                            "        scan_pessimistic_batches_for_oom(\n", 1)

    checkpoint = "            and params.batch_idx_train % params.save_every_n == 0\n"
    after_checkpoint = "        if batch_idx % 100 == 0 and params.use_fp16:\n"
    if checkpoint not in s or after_checkpoint not in s:
        sys.exit("Cannot locate the Icefall batch checkpoint block to set the batch budget.")
    s = s.replace(checkpoint, "            and (params.batch_idx_train % params.save_every_n == 0\n"
                              "                 or params.batch_idx_train == params.max_batches)\n", 1)
    s = s.replace(after_checkpoint, "        if params.max_batches and params.batch_idx_train >= params.max_batches:\n"
                                   "            break\n" + after_checkpoint, 1)

    mp_anchor = "    args = parser.parse_args()\n"
    i = s.find("def main():")
    j = s.find(mp_anchor, i)
    if i < 0 or j < 0:
        sys.exit("Không thấy main() / parser.parse_args() — cần sửa tay phần --mp-start-method (xem Runbook).")
    j += len(mp_anchor)
    s = (s[:j] + "    if args.mp_start_method:  # gipbench\n"
         "        torch.multiprocessing.set_start_method(args.mp_start_method, force=True)\n" + s[j:])

    anchor = "    model.to(device)\n    if world_size > 1:\n"
    if anchor not in s:
        sys.exit("Không thấy chỗ model.to(device) — cần sửa tay phần --freeze-modules (xem Runbook).")
    s = s.replace(anchor, FREEZE + anchor, 1)

    dst = src.with_name("finetune_cs.py")
    dst.write_text(s, encoding="utf-8")
    print(f"Đã tạo {dst}")


if __name__ == "__main__":
    main()
