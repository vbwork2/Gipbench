#!/usr/bin/env python3
"""Báo cáo nhiễm bẩn: một manifest test so với các manifest train / dev, hotwords, text LM / TTS (core.decontam).

  python -m gipbench.dataset.release.leak_audit --test bench_data/general_v1/manifest.jsonl \
      --train my_ft/train*.jsonl --dev my_ft/dev*.jsonl [--hotwords hw.txt] [--reserved-terms terms.txt] \
      [--text-files lm.txt tts.txt] [--no-audio-sha] [--out leak/x] [--fail-on critical|review|never]

Kiểm (core.decontam): trùng id / speaker / source / session / sha256 audio / text sau chuẩn hoá, trùng nguồn chuẩn hoá
(video YouTube, người nói CV / VIVOS dù ghi khác định dạng), trùng clip gốc (CV / VIVOS, kể cả đã resample) (critical);
text gần trùng (Jaccard 5-gram ký tự >= --near), trùng câu chữ trong cùng bộ đọc (VIVOS), cụm hotwords có trong text test,
tỉ lệ câu test không xác định được nguồn (review); thuật ngữ FT-unseen (--reserved-terms) trong train / dev / text LM /
TTS / hotwords, thuật ngữ lấy từ span <en> của test có trong hotwords (critical).
Đóng khi lỗi: với --fail-on critical (mặc định), dòng không xác định được sha256 audio (không có trường audio_sha256, không
đọc được file `_audio_path` / `audio`) là critical (`audio_unverifiable`). Bỏ kiểm audio phải ghi rõ: --no-audio-sha
(báo cáo ghi "không kiểm"), hoặc --no-require-audio-sha (vẫn kiểm, thiếu hash chỉ là review).
--exemptions FILE.yaml: danh sách {kind, value | test_id, reason, approved_by, date}. Finding được miễn vẫn in (mức
`exempted`), không làm thoát mã 1. Thiếu reason / approved_by / date thì từ chối file.
Thoát mã 1 khi có phát hiện ở mức --fail-on (mặc định critical). Không bao giờ in "sạch hoàn toàn": gần trùng là
heuristic, và trùng âm học (cùng bản ghi cắt khác, không cùng mã clip / nguồn) không kiểm được bằng hash.
Lưu ý về speaker: G-Group không có id người nói (speaker = id câu) nên trùng speaker với bộ đó luôn bằng 0, không có
nghĩa là không trùng người nói; báo cáo ghi tỉ lệ này ở `identity_unknown`.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

from gipbench.core.decontam import audit, read_lines, test_terms_from_rows
from gipbench.core.decontam import load_exemptions as _load_exemptions
from gipbench.core.manifest import read_jsonl

def load_rows(path):
    """Đọc manifest; dòng có `audio` mà chưa có `_audio_path` thì gắn đường dẫn (tương đối tính từ thư mục manifest)."""
    rows = read_jsonl(path)
    base = Path(path).parent
    for r in rows:
        a = r.get("audio")
        if a and not r.get("_audio_path") and not r.get("audio_sha256"):
            p = Path(a)
            r["_audio_path"] = str(p if p.is_absolute() else base / p)
    return rows


def load_exemptions(path):
    try:
        return _load_exemptions(path)
    except ValueError as e:
        sys.exit(str(e))


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
    ap.add_argument("--no-audio-sha", action="store_true", help="bỏ kiểm sha256 audio (nhanh hơn; báo cáo ghi 'không kiểm')")
    ap.add_argument("--require-audio-sha", action=argparse.BooleanOptionalAction, default=None,
                    help="dòng thiếu hash audio là critical (mặc định: bật khi --fail-on critical)")
    ap.add_argument("--exemptions", help="yaml miễn trừ có ghi chép (kind, value | test_id, reason, approved_by, date)")
    ap.add_argument("--out", help="thư mục ghi leak_report.json + leak_report.md")
    ap.add_argument("--fail-on", choices=["critical", "review", "never"], default="critical")
    args = ap.parse_args(argv)
    require = args.fail_on == "critical" if args.require_audio_sha is None else args.require_audio_sha
    test = load_rows(args.test)
    files = expand(args.train) + expand(args.dev)
    other = [r for f in files for r in load_rows(f)]
    for f in args.text_files:
        other += [{"id": f"{Path(f).name}:{i}", "text": t} for i, t in enumerate(read_lines(f))]
    exemptions = load_exemptions(args.exemptions)
    rep = audit(test, other, check_audio_sha=not args.no_audio_sha, fields=args.fields, near_threshold=args.near,
                reserved_terms=read_lines(args.reserved_terms), runtime_vocab=read_lines(args.hotwords),
                test_terms=test_terms_from_rows(test), require_audio_sha=require and not args.no_audio_sha,
                exemptions=exemptions)
    rep["inputs"] = {"test": args.test, "against": files, "text_files": args.text_files, "hotwords": args.hotwords,
                     "reserved_terms": args.reserved_terms, "audio_sha_checked": not args.no_audio_sha,
                     "require_audio_sha": require and not args.no_audio_sha, "exemptions": args.exemptions,
                     "fail_on": args.fail_on}
    st = rep["stats"]
    if args.no_audio_sha:
        audio_line = "- sha256 audio: **KHÔNG kiểm** (--no-audio-sha): trùng audio không được loại trừ."
    else:
        audio_line = (f"- sha256 audio: không xác định được sha256 audio cho {st['audio_unreadable_test']} / {st['n_test']} câu "
                      f"test và {st['audio_unreadable_train']} / {st['n_train']} dòng train / dev"
                      + (" (critical: --require-audio-sha)" if rep["inputs"]["require_audio_sha"] else " (review)") + ".")
    lines = [f"# Báo cáo rò rỉ: {Path(args.test).parent.name}", "",
             f"Test {st['n_test']} câu, so với {st['n_train']} dòng từ {len(files)} manifest"
             + (f" + {len(args.text_files)} file text" if args.text_files else "") + ".",
             f"Kết luận: **{rep['critical']} nhóm critical, {rep['review']} nhóm cần xem, {rep['exempted']} nhóm được miễn "
             f"trừ**. Không kiểm được trùng âm học.", "",
             audio_line,
             f"- Nguồn (video / người nói) không xác định được cho {st['identity_unknown_test']} / {st['n_test']} câu test "
             f"(trùng người nói / video chỉ so chuỗi thô với các câu này).",
             f"- Thoát mã 1 khi có finding mức `{args.fail_on}`" + (f"; miễn trừ: `{args.exemptions}`" if args.exemptions else "")
             + ".", "",
             "| Loại | Mức | Số giá trị | Câu test bị ảnh hưởng | Ví dụ |", "| --- | --- | --- | --- | --- |"]
    for f in rep["findings"]:
        ex = json.dumps(f["examples"][:2], ensure_ascii=False)[:180]
        lines.append(f"| {f['kind']} | {f['severity']} | {f['count']} | {f.get('test_rows', '–')} | `{ex}` |")
    if not rep["findings"]:
        lines.append("| (không phát hiện) | | | | |")
    exm = [x for f in rep["findings"] if f["severity"] == "exempted" for x in f["examples"]]
    if exm:
        lines += ["", "## Miễn trừ đã áp dụng", "", "| Loại | Giá trị | Lý do | Người duyệt | Ngày |", "| --- | --- | --- | --- | --- |"]
        for x in exm:
            e = x["exemption"]
            lines.append(f"| {e['kind']} | {x['value']} | {e['reason']} | {e['approved_by']} | {e['date']} |")
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
