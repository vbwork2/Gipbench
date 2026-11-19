#!/usr/bin/env bash
# RNN LM token-level (BPE 2000 của Gipformer 1.5) cho hàng "Gipformer beam 4 + LM" của Phần 4. Chạy trong WSL:
#   bash scripts/diagnostics/train_gip15_rnnlm.sh [OUT_DIR]
# Cần: lm_train.txt / lm_valid.txt từ scripts/diagnostics/gip15_lm_text.py (đã kiểm leak_audit --text-files = 0 critical),
# venv ~/.venvs/gipbench-ft (torch + k2 + lhotse) và icefall ở ~/icefall. Ra OUT_DIR/exp/*.pt, OUT_DIR/*.onnx, train.log.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$REPO/results/phase4/lm/gip15_rnnlm}
ICEFALL=${ICEFALL:-$HOME/icefall}
PY=${PY:-$HOME/.venvs/gipbench-ft/bin/python}
BPE=$REPO/models/gip15/bpe.model
EPOCHS=${EPOCHS:-20}
DIM=${DIM:-512}
LAYERS=${LAYERS:-2}
export PYTHONPATH=$ICEFALL:$ICEFALL/icefall/rnn_lm${PYTHONPATH:+:$PYTHONPATH}
LOCAL=$ICEFALL/egs/librispeech/ASR/local

for s in train valid; do
  "$PY" "$LOCAL/prepare_lm_training_data.py" --bpe-model "$BPE" --lm-data "$OUT/lm_$s.txt" --lm-archive "$OUT/lm_$s.pt"
  "$PY" "$LOCAL/sort_lm_training_data.py" --in-lm-data "$OUT/lm_$s.pt" --out-lm-data "$OUT/sorted_lm_$s.pt" \
    --out-statistics "$OUT/stats_$s.txt"
done

"$PY" "$ICEFALL/icefall/rnn_lm/train.py" --world-size 1 --num-epochs "$EPOCHS" --exp-dir "$OUT/exp" \
  --lm-data "$OUT/sorted_lm_train.pt" --lm-data-valid "$OUT/sorted_lm_valid.pt" --vocab-size 2000 \
  --embedding-dim "$DIM" --hidden-dim "$DIM" --num-layers "$LAYERS" --tie-weights true --batch-size 100 \
  --use-fp16 false --save-every-n 100000 --seed 42 2>&1 | tee "$OUT/train.log"
