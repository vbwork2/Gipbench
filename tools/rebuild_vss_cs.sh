#!/usr/bin/env bash
# Dựng lại vss_cs sạch CHỈ TỪ CACHE (không GPU, không tải audio, không mạng):
#   lọc rẻ (độ dài, --max-en-frac, câu chỉ có thuật ngữ 1 lần, trần thuật ngữ) -> rò rỉ (bench + reserved_b3
#   + dữ liệu test_only + danh sách chặn data/raw/consensus/leak_block.txt của bộ consensus chỉ được test)
#   -> đồng thuận span (hyps đã có, tra theo id + hash audio) -> WER tiếng Việt <= 0.15 -> (gán lại nếu --relabel)
#
#   bash tools/rebuild_vss_cs.sh --bench data/bench_v0.jsonl --catalog configs/catalog_v0.tsv
#
# Khi có data/bench_v0.jsonl thật: chạy lệnh trên để thay bench tạm (data/raw/_proxy). Câu chưa có hyp trong cache
# được đếm riêng ("chưa xét"), không bị coi là bỏ; muốn có hyp cho chúng thì chạy tools/vss_cs_run.py (cần GPU).
#
# Tuỳ chọn: --relabel (thêm nhánh --relabel-spans, mặc định tắt)  --candidates FILE  --out-dir DIR  --seed N
#   --hyps "DIR_TẦNG1 DIR_TẦNG2"  --hash-index FILE  --audio-root DIR  --max-vi-wer X
#   --block-text "FILE..." (danh sách chặn thay cho mặc định; "" = tắt)
#   Biến môi trường PY: python dùng để chạy (mặc định: python).
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH="" CAT="" RELABEL=0 SEED=0 MAXVI=0.15 BLOCK_SET=0 BLOCK=""
CAND=data/raw/vss_full/vss_cs_candidates.jsonl
OUT=data/raw/vss_full
HYPS="results/asr_hyps/qwen3_asr_0.6b results/asr_hyps/fw_large_v3_int8"
IDX=results/asr_hyps/vss_audio_sha256.tsv
ROOT=data/dl/VietSuperSpeech
PY=${PY:-python}
while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH=$2; shift 2 ;;
    --catalog) CAT=$2; shift 2 ;;
    --relabel) RELABEL=1; shift ;;
    --candidates) CAND=$2; shift 2 ;;
    --out-dir) OUT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --hyps) HYPS=$2; shift 2 ;;
    --hash-index) IDX=$2; shift 2 ;;
    --audio-root) ROOT=$2; shift 2 ;;
    --max-vi-wer) MAXVI=$2; shift 2 ;;
    --block-text) BLOCK=$2; BLOCK_SET=1; shift 2 ;;
    *) echo "tham số lạ: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$BENCH" ] || [ -z "$CAT" ]; then
  echo "cần --bench và --catalog" >&2; exit 2
fi
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT"
# shellcheck disable=SC2206
HY=($HYPS)
BLK=()
if [ "$BLOCK_SET" = 1 ]; then
  # shellcheck disable=SC2206
  BLK=(--block-text $BLOCK)
fi

echo "== lọc rẻ + rò rỉ ($CAND, bench $BENCH)"
"$PY" tools/vss_cs_prefilter.py --manifest "$CAND" --catalog "$CAT" --bench "$BENCH" --seed "$SEED" \
  ${BLK[@]+"${BLK[@]}"} --out "$OUT/vss_cs_pre_rebuild.jsonl"

echo "== đồng thuận + WER tiếng Việt (chỉ lọc, từ cache)"
"$PY" tools/agreement_filter.py --manifest "$OUT/vss_cs_pre_rebuild.jsonl" --hyps "${HY[@]}" --catalog "$CAT" \
  --hash-index "$IDX" --audio-root "$ROOT" --tiered --max-vi-wer "$MAXVI" --out "$OUT/vss_cs_clean.jsonl"

if [ "$RELABEL" = 1 ]; then
  echo "== + gán lại span (--relabel-spans), kiểm rò rỉ lại với $BENCH"
  "$PY" tools/agreement_filter.py --manifest "$OUT/vss_cs_pre_rebuild.jsonl" --hyps "${HY[@]}" --catalog "$CAT" \
    --hash-index "$IDX" --audio-root "$ROOT" --tiered --max-vi-wer "$MAXVI" \
    --relabel-spans --bench "$BENCH" --out "$OUT/vss_cs_clean_relabel.jsonl"
fi
echo "== suy domain từ text (decisions §10 / §12.6): domain_source = inferred"
"$PY" tools/infer_domain.py --manifest "$OUT/vss_cs_clean.jsonl" --catalog "$CAT"   --out "$OUT/vss_cs_clean_dom.jsonl" --report "$OUT/vss_cs_clean_dom.jsonl.report.md"
echo "Xong: $OUT/vss_cs_clean.jsonl (+ _dom có domain suy ra)$([ "$RELABEL" = 1 ] && echo ", $OUT/vss_cs_clean_relabel.jsonl")"
