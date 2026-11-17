#!/usr/bin/env python3
"""Báo cáo nhiễm bẩn: một manifest test so với các manifest train / dev, hotwords, text LM / TTS (core.decontam).

  python -m gipbench.dataset.release.leak_audit --test bench_data/general_v1/manifest.jsonl \
      --train my_ft/train*.jsonl --dev my_ft/dev*.jsonl [--hotwords hw.txt] [--reserved-terms terms.txt] \
      [--text-files lm.txt tts.txt] [--no-audio-sha] [--out leak/x] [--fail-on critical|review|never]

Kiểm: trùng id / speaker / source / session / sha256 audio / text sau chuẩn hoá (critical), text gần trùng
(Jaccard 5-gram ký tự >= --near, cần người xem), thuật ngữ FT-unseen (--reserved-terms) trong train / dev / text LM / TTS /
hotwords, thuật ngữ lấy từ span <en> của test có trong hotwords (biasing có thông tin test).
Thoát mã 1 khi có phát hiện ở mức --fail-on (mặc định critical). Không bao giờ in "sạch hoàn toàn": gần trùng là
heuristic, và trùng âm học (cùng bản ghi cắt khác) không kiểm được bằng hash.
Lưu ý về speaker: G-Group không có id người nói (speaker = id câu) nên trùng speaker với bộ đó luôn bằng 0, không có
nghĩa là không trùng người nói.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

from gipbench.core.decontam import audit, read_lines, test_terms_from_rows
from gipbench.core.manifest import read_jsonl


def expand(patterns):
    out = []
    for p in patterns or []:
        hits = sorted(glob.glob(p))
        out += hits if hits else [p]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--dev", nargs="*", default=[])
    ap.add_argument("--text-files", nargs="*", default=[], help="text LM / kịch bản TTS (1 câu / dòng)")
    ap.add_argument("--hotwords", help="hotwords / từ vựng ngữ cảnh lúc decode (1 cụm / dòng)")
    ap.add_argument("--reserved-terms", help="thuật ngữ FT-unseen (1 cụm / dòng)")
    ap.add_argument("--fields", nargs="*", default=["speaker", "source", "session"])
    ap.add_argument("--near", type=float, default=0.8)
    ap.add_argument("--no-audio-sha", action="store_true", help="bỏ kiểm sha256 audio (nhanh hơn)")
    ap.add_argument("--out", help="thư mục ghi leak_report.json + leak_report.md")
    ap.add_argument("--fail-on", choices=["critical", "review", "never"], default="critical")
    args = ap.parse_args(argv)
    test = read_jsonl(args.test)
    files = expand(args.train) + expand(args.dev)
    other = [r for f in files for r in read_jsonl(f)]
    for f in args.text_files:
        other += [{"id": f"{Path(f).name}:{i}", "text": t} for i, t in enumerate(read_lines(f))]
    rep = audit(test, other, check_audio_sha=not args.no_audio_sha, fields=args.fields, near_threshold=args.near,
                reserved_terms=read_lines(args.reserved_terms), runtime_vocab=read_lines(args.hotwords),
                test_terms=test_terms_from_rows(test))
    rep["inputs"] = {"test": args.test, "against": files, "text_files": args.text_files, "hotwords": args.hotwords,
                     "reserved_terms": args.reserved_terms, "audio_sha_checked": not args.no_audio_sha}
    lines = [f"# Báo cáo rò rỉ: {Path(args.test).parent.name}", "",
             f"Test {rep['stats']['n_test']} câu, so với {rep['stats']['n_train']} dòng từ {len(files)} manifest"
             + (f" + {len(args.text_files)} file text" if args.text_files else "") + ".",
             f"Kết luận: **{rep['critical']} nhóm critical, {rep['review']} nhóm cần xem**. Không kiểm được trùng âm học.", "",
             "| Loại | Mức | Số giá trị | Câu test bị ảnh hưởng | Ví dụ |", "| --- | --- | --- | --- | --- |"]
    for f in rep["findings"]:
        ex = json.dumps(f["examples"][:2], ensure_ascii=False)[:180]
        lines.append(f"| {f['kind']} | {f['severity']} | {f['count']} | {f.get('test_rows', '–')} | `{ex}` |")
    if not rep["findings"]:
        lines.append("| (không phát hiện) | | | | |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        o = Path(args.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "leak_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        (o / "leak_report.md").write_text(text, encoding="utf-8")
    bad = rep["critical"] + (rep["review"] if args.fail_on == "review" else 0)
    sys.exit(1 if args.fail_on != "never" and bad else 0)


if __name__ == "__main__":
    main()
