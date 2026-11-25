#!/usr/bin/env bash
# Phase 1 / P1-D: decode official vs re-exported base INT8 with the repo decoder, same environment, same config.
# Usage (WSL): bash scripts/recovery_p1_decode.sh
set -euo pipefail
R=/mnt/d/code/ZAIC/gipbench
W=$R/data/work/recovery/p1
PY=~/.venvs/gipbench-ft/bin/python
export PYTHONPATH=$R/src
M=$W/models/reexport
mkdir -p $M
for p in encoder decoder joiner; do
  ln -f $W/base_exp/$p-epoch-1-avg-1.int8.onnx $M/$p.int8.onnx
  ln -f $W/base_exp/$p-epoch-1-avg-1.onnx $M/$p.onnx
done
cp -f $R/models/gip15/tokens.txt $R/models/gip15/bpe.model $M/
cd $R
for set in dev_interim_cs@1 dev_v2_vi@1; do
  for arm in official reexport; do
    md=$R/models/gip15; [ $arm = reexport ] && md=$M
    echo "$(date -Is) start $arm $set"
    $PY -m gipbench.infer.decode --bench data/releases/dev/$set/manifest.jsonl --model-dir $md \
      --system $arm --results $W/decode/$set --quantize int8 --decoding-method modified_beam_search --beam 4 \
      --threads 16 --batch 1 --case upper
    echo "$(date -Is) done $arm $set"
  done
done
