#!/usr/bin/env bash
# Chạy mọi test (CPU, dữ liệu giả; không cần GPU / mạng / dữ liệu thật):  bash tests/run_tests.sh [thư_mục_tạm]
# Thoát mã 1 nếu có test hỏng. Golden test trên dữ liệu thật chạy riêng: tests/golden/README.md.
set -uo pipefail
cd "$(dirname "$0")/.."
T=${1:-/tmp/gipbench_tests}; mkdir -p "$T"
export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=
fail=0
run() {
  local name=$1; shift
  local s=$(date +%s)
  if "$@" > "$T/$name.log" 2>&1; then echo "ĐẠT   $name ($(( $(date +%s) - s )) s)"
  else echo "HỎNG  $name: xem $T/$name.log"; tail -5 "$T/$name.log" | sed 's/^/      /'; fail=1; fi
}
run boundaries python tests/test_boundaries.py
run selftest bash tests/selftest.sh "$T/selftest"
run span_metrics python tests/bench/test_span_metrics.py
run dev_tools python tests/integration/test_dev_tools.py
run converters python tests/dataset/test_converters.py "$T/converters"
run tts python tests/dataset/test_tts.py
run hard_mining python tests/dataset/test_hard_mining.py
for f in tests/model/test_*.py; do [ -f "$f" ] && run "$(basename "$f" .py)" python "$f"; done
exit $fail
