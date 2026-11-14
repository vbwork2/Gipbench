#!/usr/bin/env python3
"""Các bộ tiếng Việt thuần đọc từ thư mục đã tải về: VIVOS, Common Voice, VietSuperSpeech.

  # VIVOS (bản gốc giải nén: vivos/{train,test}/prompts.txt + waves/<SPK>/<ID>.wav) -> B5 (test) + train
  python converters/conv_local.py vivos --root data/dl/vivos --out data/raw/vivos
  # Common Voice vi (thư mục có test.tsv, train.tsv, clips/) -> B5 (test) + train
  python converters/conv_local.py commonvoice --root data/dl/cv-corpus-vi --out data/raw/cv
  # VietSuperSpeech (thư mục có train.json / dev.json hoặc *.parquet, cột audio = đường dẫn tương đối) -> train replay
  python converters/conv_local.py vietsuperspeech --root data/dl/VietSuperSpeech --out data/raw/vss

Audio KHÔNG được copy: manifest trỏ thẳng vào file gốc.
B5 chỉ nhận câu không có từ nào bị nghi là tiếng Anh. VietSuperSpeech là pseudo-label, phần tiếng Anh hay bị méo:
mặc định chỉ giữ câu không có từ Anh (bỏ lọc bằng --keep-english).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converters.common import Writer, auto_tag, safe, save_audio  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402


def emit(w, name, split, uid, audio, text, speaker, source, domain, dataset, cat, keep_english, stats):
    try:
        path, dur, _ = save_audio(audio, w.out, uid)
    except Exception:  # noqa: BLE001
        stats["audio lỗi"] = stats.get("audio lỗi", 0) + 1
        return
    tag = auto_tag(text, cat)
    if tag["spans"] and not keep_english:
        stats["bỏ vì có từ Anh"] = stats.get("bỏ vì có từ Anh", 0) + 1
        return
    row = {"id": uid, "split": split, "audio": path, "duration": dur, "speaker": speaker, "source": source,
           "region": "unk", "gender": "u", "channel": "wideband", "domain": domain, "dataset": dataset}
    if split == "B5":
        w.add(name, row, dict(tag, text_tagged=text))
    else:
        row["text"] = text
        w.add(name, row)


def vivos(args, w, cat, stats):
    root = Path(args.root)
    for part, name, split in (("test", "b5_vivos", "B5"), ("train", "train_vivos", "train")):
        pf = root / part / "prompts.txt"
        if not pf.exists():
            print(f"  thiếu {pf}")
            continue
        for i, line in enumerate(open(pf, encoding="utf-8")):
            if args.limit and i >= args.limit:
                break
            uid, _, text = line.strip().partition(" ")
            spk = uid.split("_")[0]
            emit(w, name, split, f"vivos_{uid}", root / part / "waves" / spk / f"{uid}.wav", text.lower(),
                 f"vivos_{spk}", f"vivos:{spk}", "read", f"vivos:{part}", cat, False, stats)


def commonvoice(args, w, cat, stats):
    root = Path(args.root)
    for part, name, split in (("test", "b5_cv", "B5"), ("train", "train_cv", "train")):
        tsv = root / f"{part}.tsv"
        if not tsv.exists():
            print(f"  thiếu {tsv}")
            continue
        with open(tsv, encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
                if args.limit and i >= args.limit:
                    break
                spk = safe(r["client_id"])[:16]
                emit(w, name, split, f"cv_{safe(Path(r['path']).stem)}", root / "clips" / r["path"], r["sentence"],
                     f"cv_{spk}", f"cv:{spk}", "read", f"commonvoice:{part}", cat, False, stats)


def vietsuperspeech(args, w, cat, stats):
    root = Path(args.root)
    items = []
    for jf, part in (("train.json", "train"), ("dev.json", "dev")):
        p = root / jf
        if p.exists():
            raw = p.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
            items += [(part, d) for d in data]
    if not items:
        import glob
        from datasets import load_dataset
        files = sorted(glob.glob(str(root / "**" / "*.parquet"), recursive=True))
        if not files:
            sys.exit(f"Không thấy train.json/dev.json hay *.parquet trong {root}")
        for f in files:
            part = "dev" if "valid" in Path(f).name or "dev" in Path(f).name else "train"
            items += [(part, d) for d in load_dataset("parquet", data_files=f, split="train")]
    for i, (part, d) in enumerate(items):
        if args.limit and i >= args.limit:
            break
        src = safe(d.get("source") or "unk")
        emit(w, f"{part}_vss", part, f"vss_{i:07d}", root / d["audio"], str(d["text"]).lower(), f"vss_{src}",
             f"vss:{src}", "conversation", "VietSuperSpeech", cat, args.keep_english, stats)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["vivos", "commonvoice", "vietsuperspeech"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--keep-english", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    w, stats = Writer(args.out, args.dataset), {}
    {"vivos": vivos, "commonvoice": commonvoice, "vietsuperspeech": vietsuperspeech}[args.dataset](args, w, Catalog.load(args.catalog), stats)
    w.terms.clear()
    w.close()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
