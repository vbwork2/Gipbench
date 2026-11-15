#!/usr/bin/env bash
# Mẫu chạy trọn E3: FT Gipformer 1.5 trên code-switching thật + replay tiếng Việt, export ONNX, chấm bench.
# Sửa các biến dưới đây rồi chạy từng stage:  bash ft/run_e3.sh 0 7
set -euo pipefail

stage=${1:-0}; stop=${2:-7}
GIPBENCH=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(pwd)
abs() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
ICEFALL=${ICEFALL:-$HOME/icefall}                 # git clone https://github.com/k2-fsa/icefall
MODEL=${MODEL:-models/gip15}                      # huggingface-cli download g-group-ai-lab/gipformer1.5-68M-rnnt --local-dir models/gip15
BENCH=${BENCH:-data/bench_v1.0.jsonl}
CATALOG=${CATALOG:-configs/catalog.tsv}
CS_TRAIN=${CS_TRAIN:-"data/train_cs.jsonl"}       # CS thật: ViMedCSS train, vss_cs (đã lọc đồng thuận), dữ liệu tự có (KHÔNG chứa bench). SEA: tùy chọn, không dùng
VI_REPLAY=${VI_REPLAY:-"data/train_vi.jsonl"}     # tiếng Việt thuần để chống quên (vd 50–100 h)
DEV=${DEV:-"data/dev_cs.jsonl"}                   # dev CS riêng, không phải bench
EXP=${EXP:-exp/e3_ft}
SYSTEM=${SYSTEM:-e3_ft}
GPUS=${GPUS:-1}; MAXDUR=${MAXDUR:-600}; EPOCHS=${EPOCHS:-8}; AVG=${AVG:-3}; LR=${LR:-0.0045}
CS_W=${CS_W:-0.5}; VI_W=${VI_W:-0.5}              # tỉ lệ trộn CS : tiếng Việt

if [ $stage -le 0 ] && [ $stop -ge 0 ]; then
  echo "== 0. Kiểm tra bench + rò rỉ"
  python $GIPBENCH/validate.py --bench $BENCH --catalog $CATALOG --train $CS_TRAIN $VI_REPLAY $DEV
fi
if [ $stage -le 1 ] && [ $stop -ge 1 ]; then
  echo "== 1. Manifest -> Lhotse (lọc rò rỉ, speed perturb, mô phỏng thoại 8 kHz)"
  python $GIPBENCH/ft/to_lhotse.py --manifest $CS_TRAIN --name cs_train --bench $BENCH --catalog $CATALOG --speed-perturb --phone-sim 0.3
  python $GIPBENCH/ft/to_lhotse.py --manifest $VI_REPLAY --name vi_replay --bench $BENCH --catalog $CATALOG --speed-perturb --phone-sim 0.3
  python $GIPBENCH/ft/to_lhotse.py --manifest $DEV --name cs_dev --bench $BENCH --catalog $CATALOG
fi
if [ $stage -le 2 ] && [ $stop -ge 2 ]; then
  echo "== 2. Fbank"
  for n in cs_train vi_replay cs_dev; do
    python $GIPBENCH/ft/compute_fbank.py --cuts data/lhotse/${n}_cuts_raw.jsonl.gz --out-dir data/fbank --name $n --num-jobs 8
  done
fi
if [ $stage -le 3 ] && [ $stop -ge 3 ]; then
  echo "== 3. Chuẩn bị checkpoint + script finetune"
  python $GIPBENCH/ft/wrap_ckpt.py $MODEL/model.pt $MODEL/init_ft.pt
  python $GIPBENCH/ft/patch_finetune.py --icefall $ICEFALL
fi
if [ $stage -le 4 ] && [ $stop -ge 4 ]; then
  echo "== 4. Fine-tune"
  export PYTHONPATH=$ICEFALL:${PYTHONPATH:-}
  ( cd $ICEFALL/egs/librispeech/ASR && python ./zipformer/finetune_cs.py \
      --world-size $GPUS --num-epochs $EPOCHS --start-epoch 1 --use-fp16 1 \
      --exp-dir $(abs $EXP) \
      --bpe-model $(abs $MODEL)/bpe.model \
      --do-finetune 1 --finetune-ckpt $(abs $MODEL)/init_ft.pt \
      --train-cuts $ROOT/data/fbank/cs_train_cuts.jsonl.gz:$CS_W $ROOT/data/fbank/vi_replay_cuts.jsonl.gz:$VI_W \
      --valid-cuts $ROOT/data/fbank/cs_dev_cuts.jsonl.gz \
      --base-lr $LR --max-duration $MAXDUR \
      --enable-musan False --enable-spec-aug True \
      --manifest-dir $ROOT/data/fbank )
fi
if [ $stage -le 5 ] && [ $stop -ge 5 ]; then
  echo "== 5. Export ONNX (fp32 + int8), trung bình $AVG epoch cuối"
  export PYTHONPATH=$ICEFALL:${PYTHONPATH:-}
  ( cd $ICEFALL/egs/librispeech/ASR && python ./zipformer/export-onnx.py \
      --tokens $(abs $MODEL)/tokens.txt --exp-dir $(abs $EXP) \
      --epoch $EPOCHS --avg $AVG --use-averaged-model 1 --enable-int8-quantization 1 )
  mkdir -p $EXP/onnx
  for m in encoder decoder joiner; do
    cp $EXP/$m-epoch-$EPOCHS-avg-$AVG.onnx $EXP/onnx/$m.onnx
    cp $EXP/$m-epoch-$EPOCHS-avg-$AVG.int8.onnx $EXP/onnx/$m.int8.onnx
  done
  cp $MODEL/tokens.txt $MODEL/bpe.model $EXP/onnx/
fi
if [ $stage -le 6 ] && [ $stop -ge 6 ]; then
  echo "== 6. Giải mã + chấm bench (int8, đúng đường chạy production)"
  python $GIPBENCH/decode.py --bench $BENCH --model-dir $EXP/onnx --system $SYSTEM
  python $GIPBENCH/score.py --bench $BENCH --catalog $CATALOG --hyps results/$SYSTEM/hyps.jsonl
  python $GIPBENCH/decode.py --bench $BENCH --model-dir $EXP/onnx --system ${SYSTEM}_hw \
      --hotwords-catalog $CATALOG --hotwords-score 1.5
  python $GIPBENCH/score.py --bench $BENCH --catalog $CATALOG --hyps results/${SYSTEM}_hw/hyps.jsonl
fi
if [ $stage -le 7 ] && [ $stop -ge 7 ]; then
  echo "== 7. Leaderboard"
  python $GIPBENCH/leaderboard.py --results results --baseline e0_baseline --out reports/leaderboard
fi
