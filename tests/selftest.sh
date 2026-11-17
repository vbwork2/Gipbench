#!/usr/bin/env bash
# Chạy toàn bộ bộ công cụ trên dữ liệu GIẢ LẬP để kiểm tra cài đặt:  bash tests/selftest.sh [thư_mục_tạm]
set -euo pipefail
G=$(cd "$(dirname "$0")/.." && pwd); T=${1:-/tmp/gipbench_selftest}; rm -rf "$T"
python "$G/tests/fixtures/make_toy_data.py" --out "$T"
python -m gipbench.dataset.release.validate --bench "$T/bench.jsonl" --catalog "$G/configs/dataset/labeling/catalog.example.tsv" --train "$T/train_cs.jsonl" --max-show 2 \
  && { echo "LỖI: validate phải bắt được 3 rò rỉ đã cài sẵn"; exit 1; } || echo "OK: validate bắt được rò rỉ (mong đợi)"
python -m gipbench.bench.report.eda --bench "$T/bench.jsonl" --catalog "$G/configs/dataset/labeling/catalog.example.tsv" --train "$T/train_cs.jsonl" --audio-stats --out "$T/eda"
for s in e0_baseline e1_hotwords e3_ft; do
  python -m gipbench.bench.score --bench "$T/bench.jsonl" --catalog "$G/configs/dataset/labeling/catalog.example.tsv" --hyps "$T/results/$s/hyps.jsonl" --iters 200
done
python -m gipbench.bench.leaderboard --results "$T/results" --baseline e0_baseline --out "$T/leaderboard" --iters 200
echo "Xong. Mở $T/eda/eda_report.html và $T/leaderboard/leaderboard.html"
