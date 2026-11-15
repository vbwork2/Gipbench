#!/usr/bin/env bash
# Chạy tầng ASR + lọc đồng thuận cho một manifest vss_cs đã qua tools/vss_cs_prefilter.py.
#
#   M=data/raw/vss_full/vss_cs_pilot.jsonl bash tools/vss_cs_pilot.sh
#
# Tầng 1: Qwen3-ASR-0.6B trên mọi câu.
# Tầng 2: faster-whisper large-v3 int8 CHỈ cho câu còn span Qwen chưa xác nhận. --relabel-spans cũng chỉ cần
#         faster-whisper ở đúng các câu này (câu Qwen đã xác nhận hết span thì không có span nào để gán lại),
#         nên bật hay tắt gán lại thì tầng 2 chạy trên cùng một tập.
# Sau đó: lọc (chỉ lọc) và lọc + --relabel-spans (kiểm tra rò rỉ lại, phiếu nghe 30 câu).
# Chạy theo chunk, có resume (asr_hyps bỏ qua id đã có), audio tạm ở $TMP_AUDIO bị xoá sau mỗi chunk,
# dừng (exit 3) nếu ổ chứa $TMP_AUDIO còn < $MIN_FREE_GB GB. Chạy lại đúng lệnh này để chạy tiếp.
#
# Biến môi trường (mặc định trong ngoặc):
#   M (data/raw/vss_full/vss_cs_pilot.jsonl)  PY (python của venv chính)  ASR_PY (python của venv ASR)
#   CAT (data/raw/_proxy/catalog_proxy.tsv)  BENCH (data/raw/_proxy/bench_proxy.jsonl)
#   TMP_AUDIO (data/tmp/vss_audio)  MIN_FREE_GB (3)  CHUNK (200)  HF_HOME (giữ nguyên nếu đã đặt)
#   LISTEN_CSV (reports/vss_relabel_listen.csv)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${M:-data/raw/vss_full/vss_cs_pilot.jsonl}
PY=${PY:-$HOME/.venvs/gipbench/bin/python}
ASR_PY=${ASR_PY:-$HOME/.venvs/gipbench-asr/bin/python}
CAT=${CAT:-data/raw/_proxy/catalog_proxy.tsv}
BENCH=${BENCH:-data/raw/_proxy/bench_proxy.jsonl}
TMP_AUDIO=${TMP_AUDIO:-data/tmp/vss_audio}
MIN_FREE_GB=${MIN_FREE_GB:-3}
CHUNK=${CHUNK:-200}
LISTEN_CSV=${LISTEN_CSV:-reports/vss_relabel_listen.csv}
REPO=thanhnew2001/VietSuperSpeech
ROOT=data/dl/VietSuperSpeech
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$TMP_AUDIO" "$TMP_AUDIO/py"
export TMPDIR="$(cd "$TMP_AUDIO" && pwd)/py"
# faster-whisper (CTranslate2) cần cuBLAS 12 + cuDNN 9 của venv ASR
NV=$("$ASR_PY" -c 'import site; print(site.getsitepackages()[0])')/nvidia
export LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

NAME=$(basename "$M" .jsonl)
OUT=$(dirname "$M")
HQ=results/asr_hyps/qwen3_asr_0.6b/$NAME.jsonl
HW=results/asr_hyps/fw_large_v3_int8/$NAME.jsonl
IDX=${IDX:-results/asr_hyps/vss_audio_sha256.tsv}  # cache hyps theo id + hash audio (lib/hyp_cache.py)
FETCH=(--chunk-size "$CHUNK" --hf-repo "$REPO" --audio-root "$ROOT" --tmp-dir "$TMP_AUDIO" --min-free-gb "$MIN_FREE_GB"
       --hash-index "$IDX")
CACHE=(--hash-index "$IDX" --audio-root "$ROOT" --tiered)

echo "== tầng 1: Qwen3-ASR-0.6B ($M)"
"$ASR_PY" tools/asr_hyps.py --engine qwen3-asr --model Qwen/Qwen3-ASR-0.6B --device cuda:0 \
  --manifest "$M" --out "$HQ" "${FETCH[@]}" 2>&1 | grep -v "pad_token_id"
test "${PIPESTATUS[0]}" -eq 0

echo "== câu còn span Qwen chưa xác nhận"
"$PY" tools/agreement_filter.py --manifest "$M" --hyps "$HQ" --catalog "$CAT" --hash-index "$IDX" --audio-root "$ROOT" \
  --write-unconfirmed-ids "$OUT/${NAME}_need_fw.txt" --out "$OUT/${NAME}_agreed_qwen.jsonl"

echo "== tầng 2: faster-whisper large-v3 int8 ($(wc -l < "$OUT/${NAME}_need_fw.txt") câu)"
"$ASR_PY" tools/asr_hyps.py --engine faster-whisper --model large-v3 --compute-type int8 --device cuda \
  --manifest "$M" --ids "$OUT/${NAME}_need_fw.txt" --out "$HW" "${FETCH[@]}"

echo "== lọc: chỉ lọc"
"$PY" tools/agreement_filter.py --manifest "$M" --hyps "$HQ" "$HW" --catalog "$CAT" "${CACHE[@]}" \
  --out "$OUT/${NAME}_agreed.jsonl"

echo "== lọc + --relabel-spans"
"$PY" tools/agreement_filter.py --manifest "$M" --hyps "$HQ" "$HW" --catalog "$CAT" --bench "$BENCH" "${CACHE[@]}" \
  --relabel-spans --audio-root "$ROOT" --listen-csv "$LISTEN_CSV" --listen-n 30 --seed 0 \
  --out "$OUT/${NAME}_agreed_relabel.jsonl"
rm -rf "$TMP_AUDIO/py"  # file tạm của python/torch (torchinductor)
echo "Xong: $OUT/${NAME}_agreed.jsonl, $OUT/${NAME}_agreed_relabel.jsonl, $LISTEN_CSV"
