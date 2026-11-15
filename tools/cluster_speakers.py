#!/usr/bin/env python3
"""Gom speaker_cluster bằng speaker embedding cho các câu của bench không có speaker id thật (vd G-Group).

  python tools/cluster_speakers.py --bench data/bench_v0_draft/bench_v0_draft.jsonl --source-prefix ggroup: \
      --threshold 0.7565 --prefix ggroup:spk

Dùng lại converters/conv_consensus.cluster_sessions (sherpa-onnx + wespeaker ResNet34, average linkage cosine). Mỗi câu
là một "phiên" (G-Group không có phiên), nên không tự chọn được ngưỡng từ cặp cùng phiên: mặc định lấy ngưỡng của bộ
consensus (data/raw/consensus/consensus_stats.json, phân vị 5% độ giống cùng phiên). speaker_cluster CHỈ dùng cho
bootstrap CI của score.py. Ghi lại bench tại chỗ (bản cũ -> .bak), <bench>.speaker_clusters.tsv và in thống kê.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from converters.conv_consensus import cluster_sessions  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--source-prefix", default="ggroup:")
    ap.add_argument("--prefix", default="ggroup:spk")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ngưỡng cosine; mặc định đọc từ data/raw/consensus/consensus_stats.json")
    ap.add_argument("--model", default=str(G / "models/spk/wespeaker_en_voxceleb_resnet34.onnx"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    th = args.threshold
    if th is None:
        th = json.load(open(G / "data/raw/consensus/consensus_stats.json", encoding="utf-8"))["speaker_embedding"]["threshold"]
    rows = read_jsonl(args.bench)
    sel = [r for r in rows if str(r.get("source", "")).startswith(args.source_prefix)]
    if not sel:
        sys.exit(f"Không có câu nào source bắt đầu bằng {args.source_prefix}")
    lab, st = cluster_sessions([(r["id"], r["_audio_path"]) for r in sel], args.model, args.threads, th, 5, args.seed)
    for r in sel:
        r["speaker_cluster"] = f"{args.prefix}{lab[r['id']]:03d}"
    shutil.copy(args.bench, args.bench + ".bak")
    write_jsonl(args.bench, rows)
    with open(args.bench + ".speaker_clusters.tsv", "w", encoding="utf-8") as f:
        f.write("id\tsplit\tspeaker_cluster\n" + "".join(f"{r['id']}\t{r['split']}\t{r['speaker_cluster']}\n" for r in sel))
    by_split = Counter((r["split"], r["speaker_cluster"]) for r in sel)
    n_split = Counter(s for s, _ in by_split)
    sizes = sorted(Counter(r["speaker_cluster"] for r in sel).values(), reverse=True)
    print(f"{len(sel)} câu -> {st['n_clusters']} cụm (ngưỡng cosine {th}); số cụm theo split: {dict(n_split)}")
    print(f"  cụm lớn nhất: {sizes[:10]}; độ giống giữa câu: {st['between']}; số cụm theo ngưỡng: {st['sweep']}")
    print(f"  bench ghi lại: {args.bench} (bản cũ: {args.bench}.bak)")


if __name__ == "__main__":
    main()
