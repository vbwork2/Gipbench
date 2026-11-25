"""Export checkpoint icefall -> ONNX int8 (như production) và đóng gói artifact model + model_card.json.

export_cmd / collect: trước đây là hàm export_onnx() của ft/run_e3_v1*.sh (stage 5–6). write_artifact: hợp đồng model ->
bench (docs/CONTRACTS.md): <artifact>/onnx/{encoder,decoder,joiner}.int8.onnx + tokens.txt + bpe.model, <artifact>/model_card.json.
"""
import datetime as dt
import json
import shutil
from pathlib import Path

from gipbench.core.fileio import sha256_file as sha256

PARTS = ("encoder", "decoder", "joiner")


def export_cmd(py_ft, exp_dir, model_dir, model_args, select_args):
    """argv chạy trong <icefall>/egs/librispeech/ASR (PYTHONPATH=<icefall>)."""
    return [str(py_ft), "./zipformer/export-onnx.py", "--tokens", str(Path(model_dir) / "tokens.txt"),
            "--exp-dir", str(exp_dir), *model_args.split(), *select_args,
            "--avg", "1", "--use-averaged-model", "0", "--enable-int8-quantization", "1"]


def collect(exp_dir, out_onnx, model_dir):
    """Chuyển <part>-*.int8.onnx vừa export vào out_onnx/<part>.int8.onnx, xoá các .onnx còn lại, chép tokens + bpe."""
    out = Path(out_onnx)
    out.mkdir(parents=True, exist_ok=True)
    for m in PARTS:
        cand = sorted(Path(exp_dir).glob(f"{m}-*.int8.onnx"))
        if not cand:
            raise SystemExit(f"export không ra {m}-*.int8.onnx trong {exp_dir}")
        shutil.move(str(cand[0]), out / f"{m}.int8.onnx")
        for f in Path(exp_dir).glob(f"{m}-*.onnx"):
            f.unlink()
    for f in ("tokens.txt", "bpe.model"):
        shutil.copy(Path(model_dir) / f, out / f)


def write_artifact(artifact_dir, src_model_dir, card):
    """Chép model đã chọn (thư mục có *.onnx + tokens.txt + bpe.model) vào <artifact>/onnx và ghi model_card.json."""
    art = Path(artifact_dir)
    onnx = art / "onnx"
    onnx.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in sorted(Path(src_model_dir).iterdir()):
        if f.suffix == ".onnx" or f.name in ("tokens.txt", "bpe.model"):
            shutil.copy(f, onnx / f.name)
            files[f.name] = sha256(onnx / f.name)
    card = dict(card, files=files, created=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    (art / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return art
