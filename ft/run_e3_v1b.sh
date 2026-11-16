#!/usr/bin/env bash
# E3 v1b (16/11/2026): giống hệt v1 nhưng trần domain 50%.
# E3 v1 (16/11/2026): FT Gipformer 1.5 trên RTX 3050 4 GB, dữ liệu curate + mix theo configs/train_mix.tsv.
#   bash ft/run_e3_v1.sh <stage_đầu> <stage_cuối>
# 0 validate (bench vs train / dev) · 1 curate_train · 2 tách nhóm + to_lhotse · 3 mix_cuts · 4 finetune
# 5 ứng viên (checkpoint mỗi epoch ảo + WiSE-FT) -> export -> decode dev · 6 select_ckpt · 7 bench (E3, E7) + leaderboard
# Feature tính on-the-fly (D: không đủ chỗ cho fbank); checkpoint ghi sang C: (EXP). Dừng an toàn khi D: < 3 GB hoặc RAM
# WSL < 1 GB (watchdog trong stage 4).
set -euo pipefail
stage=${1:-0}; stop=${2:-7}
G=/mnt/d/code/ZAIC/gipbench; cd $G
SRC=data/e3_v1; D=data/e3_v1b   # v1b: trần domain 50%; đầu vào (dev, manifest train) dùng chung với v1
ICEFALL=${ICEFALL:-$HOME/icefall}
MODEL=$G/models/gip15
BENCH=data/bench_v0_draft/bench_v0_draft.jsonl
CATALOG=data/bench_v0_draft/catalog_v0_draft.tsv
EXP=${EXP:-/mnt/c/Users/goose/gipbench_exp/e3_v1b}
LR=${LR:-0.002}; MAXDUR=${MAXDUR:-150}; EPOCHS=${EPOCHS:-6}; EPOCH_H=${EPOCH_H:-160}
FREEZE=${FREEZE:-"encoder_embed.,encoder.encoders.0.,encoder.encoders.1."}
MODEL_ARGS="--use-transducer 1 --use-ctc 0"
FT=$HOME/.venvs/gipbench-ft/bin/python
PY=$HOME/.venvs/gipbench/bin/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
TRAIN_IN="$SRC/train_vimedcss_abs.jsonl $SRC/train_vss_cs.jsonl $SRC/train_vivos.jsonl data/raw/cv/train_cv.jsonl $SRC/distill.jsonl $SRC/train_confusable.jsonl"

if [ $stage -le 0 ] && [ $stop -ge 0 ]; then
  echo "== 0. dev CS = 600 câu ViMedCSS validation + vss_cs held-out; dev VI = VIVOS train held-out; validate"
  $PY - <<'EOF'
import json, random
rows = [json.loads(l) for l in open("data/raw/vimedcss_full/dev_vimedcss.jsonl", encoding="utf-8")]
random.Random(0).shuffle(rows)
base = "/mnt/d/code/ZAIC/gipbench/data/raw/vimedcss_full/"
out = [dict(r, audio=base + r["audio"]) for r in rows[:600]]
out += [json.loads(l) for l in open("data/e3_v1/dev_cs_vss.jsonl", encoding="utf-8")]
out = [r for r in out if float(r["duration"]) >= 0.5]  # ViMedCSS validation có câu audio rỗng (0 s): sherpa-onnx lỗi
with open("data/e3_v1/dev_cs.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("dev_cs:", len(out), "câu", round(sum(float(r["duration"]) for r in out) / 3600, 2), "h")
with open("data/e3_v1/train_vimedcss_abs.jsonl", "w", encoding="utf-8") as f:  # audio tuyệt đối (manifest gốc: tương đối)
    for l in open("data/raw/vimedcss_full/train_vimedcss.jsonl", encoding="utf-8"):
        r = json.loads(l)
        f.write(json.dumps(dict(r, audio=base + r["audio"]), ensure_ascii=False) + "\n")
EOF
  $PY validate.py --bench $BENCH --catalog $CATALOG --train $TRAIN_IN --dev $D/dev_cs.jsonl $D/dev_vi.jsonl > $D/validate.log 2>&1 || true
  grep -E "== LỖI|KẾT QUẢ" $D/validate.log
fi
if [ $stage -le 1 ] && [ $stop -ge 1 ]; then
  echo "== 1. curate_train (trần domain 50% trên giờ CS, miễn domain other; holdout finance)"
  $PY ft/curate_train.py --manifest $TRAIN_IN --catalog $CATALOG --out $D/train_curated.jsonl \
      --holdout-domain finance --max-domain-share 0.50 --cap-exempt other
fi
if [ $stage -le 2 ] && [ $stop -ge 2 ]; then
  echo "== 2. tách nhóm theo configs/train_mix.tsv + to_lhotse (chặn rò rỉ bench, speed perturb, 8 kHz 30%)"
  $PY - <<'EOF'
import json
groups = {}
for l in open("configs/train_mix.tsv", encoding="utf-8"):
    if l.startswith("#") or l.startswith("group\t") or not l.strip():
        continue
    g, w, _, ds, _ = l.rstrip("\n").split("\t")
    for d in ds.split(","):
        groups[d.strip().lower()] = g
out = {}
for l in open("data/e3_v1b/train_curated.jsonl", encoding="utf-8"):
    r = json.loads(l)
    d = r["dataset"].split(":")[0].lower()
    g = groups.get(d) or ("distill" if r.get("distill") else None)
    if g is None:
        raise SystemExit(f"không biết nhóm của dataset {r['dataset']}")
    out.setdefault(g, []).append(l)
for g, ls in out.items():
    open(f"data/e3_v1b/group_{g}.jsonl", "w", encoding="utf-8").writelines(ls)
    print(g, len(ls), "câu", round(sum(json.loads(x)["duration"] for x in ls) / 3600, 2), "h")
EOF
  for f in $D/group_*.jsonl; do
    n=$(basename $f .jsonl); n=${n#group_}
    $FT ft/to_lhotse.py --manifest $f --name $n --bench $BENCH --catalog $CATALOG --out $D/lhotse --speed-perturb --phone-sim 0.3
  done
  $FT ft/to_lhotse.py --manifest $SRC/dev_cs.jsonl --name dev_cs --bench $BENCH --catalog $CATALOG --out $D/lhotse
  $FT ft/to_lhotse.py --manifest $SRC/dev_vi.jsonl --name dev_vi --bench $BENCH --catalog $CATALOG --out $D/lhotse
fi
if [ $stage -le 3 ] && [ $stop -ge 3 ]; then
  echo "== 3. mix_cuts: $EPOCHS epoch ảo × $EPOCH_H h cut"
  args=()
  while IFS=$'\t' read -r g w env ds note; do
    case "$g" in \#*|group|"") continue;; esac
    w=${!env:-$w}
    [ -f $D/lhotse/${g}_cuts_raw.jsonl.gz ] && args+=(--group "$g=$D/lhotse/${g}_cuts_raw.jsonl.gz:$w")
  done < configs/train_mix.tsv
  $FT ft/mix_cuts.py "${args[@]}" --epoch-hours $EPOCH_H --epochs $EPOCHS --out $D/lhotse/train_mix_cuts.jsonl.gz
fi
if [ $stage -le 4 ] && [ $stop -ge 4 ]; then
  echo "== 4. finetune (LR $LR, freeze $FREEZE, bỏ CTC, on-the-fly fbank) -> $EXP"
  STEPS=$($FT -c "
import json; s = json.load(open('$D/lhotse/train_mix_cuts.jsonl.gz.json'))
print(int(s['epoch_hours'] * 3600 / ($MAXDUR * 0.93)))")
  # RAM tiến trình chính tăng dần (~0.1–0.9 MB / batch) -> chạy theo đoạn: checkpoint mỗi SEG = 1/4 epoch ảo; watchdog
  # dừng khi RAM < 1 GB thì resume từ checkpoint mới nhất (--start-batch, sampler nạp lại vị trí). D: < 3 GB = dừng hẳn.
  SEG=$(( (STEPS + 3) / 4 )); EPS=$(( SEG * 4 ))
  echo "  ước tính $STEPS step / epoch ảo -> checkpoint mỗi $SEG step, epoch ảo = $EPS step"
  mkdir -p $EXP
  $FT ft/wrap_ckpt.py $MODEL/model.pt $MODEL/init_ft.pt --drop-ctc
  $FT ft/patch_finetune.py --icefall $ICEFALL
  export PYTHONPATH=$ICEFALL:${PYTHONPATH:-}
  cleanup() {  # giữ checkpoint ở ranh giới epoch ảo + checkpoint mới nhất
    local last=$(ls $EXP/checkpoint-*.pt 2>/dev/null | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
    for f in $EXP/checkpoint-*.pt; do
      [ -e "$f" ] || continue
      n=$(basename $f .pt); n=${n#checkpoint-}
      [ "$n" = "$last" ] || [ $((n % EPS)) -eq 0 ] || rm -f "$f"
    done
  }
  fails=0
  while true; do
    rm -f $D/watchdog.log
    latest=$(ls $EXP/checkpoint-*.pt 2>/dev/null | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
    if [ -z "$latest" ]; then
      RES=(--do-finetune 1 --finetune-ckpt $MODEL/init_ft.pt --start-epoch 1 --num-epochs 1)
    else
      RES=(--do-finetune 0 --start-epoch 2 --num-epochs 2 --start-batch $latest)
    fi
    echo "$(date '+%F %T') == đoạn mới: ${RES[*]}" | tee -a $D/finetune.log $D/segments.log
    ( while sleep 60; do
        cleanup
        f=$(df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9); m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$f" -lt 3 ] || [ "$m" -lt 1024 ]; then
          echo "$(date '+%F %T') WATCHDOG: D: ${f} GB, RAM ${m} MB -> dừng finetune" >> $D/watchdog.log
          pkill -INT -f "[z]ipformer/finetune_cs.py"; break
        fi
      done ) & WD=$!
    rc=0
    ( cd $ICEFALL/egs/librispeech/ASR && $FT ./zipformer/finetune_cs.py \
        --world-size 1 --use-fp16 1 $MODEL_ARGS "${RES[@]}" \
        --exp-dir $EXP --bpe-model $MODEL/bpe.model --freeze-modules "$FREEZE" \
        --train-cuts $G/$D/lhotse/train_mix_cuts.jsonl.gz \
        --valid-cuts $G/$D/lhotse/dev_cs_cuts_raw.jsonl.gz $G/$D/lhotse/dev_vi_cuts_raw.jsonl.gz \
        --on-the-fly-feats True --num-workers ${WORKERS:-2} --num-buckets ${BUCKETS:-10} --mp-start-method spawn --shuffle True \
        --base-lr $LR --max-duration $MAXDUR --enable-musan False --enable-spec-aug True \
        --save-every-n $SEG --keep-last-k 40 --manifest-dir $G/$D/lhotse ) >> $D/finetune.log 2>&1 || rc=$?
    kill $WD 2>/dev/null || true
    cleanup
    cat $D/watchdog.log >> $D/segments.log 2>/dev/null || true
    echo "$(date '+%F %T') đoạn kết thúc rc=$rc" >> $D/segments.log
    if tail -5 $D/finetune.log | grep -q "Done!"; then echo "finetune XONG"; break; fi
    if grep -q "WATCHDOG: D: [0-2] GB" $D/watchdog.log 2>/dev/null; then echo "DỪNG: D: < 3 GB"; break; fi
    if [ -s $D/watchdog.log ]; then sleep 20; continue; fi  # RAM < 1 GB: resume
    fails=$((fails + 1)); echo "finetune lỗi rc=$rc (lần $fails)"
    [ $fails -ge 2 ] && { echo "DỪNG: lỗi 2 lần liên tiếp, xem $D/finetune.log"; break; }
    sleep 20
  done
fi

# ---- stage 5–7: chọn trên dev, rồi chấm bench (decode CPU 16 luồng, int8 như production)
RD=results/e3_v1b_dev
export_onnx() {  # export_onnx <exp_dir> <tên> <args chọn checkpoint...>: int8 vào $EXP/cand/<tên>/onnx
  local ed=$1 name=$2; shift 2
  local od=$EXP/cand/$name/onnx; mkdir -p $od
  ( cd $ICEFALL/egs/librispeech/ASR && PYTHONPATH=$ICEFALL $FT ./zipformer/export-onnx.py --tokens $MODEL/tokens.txt \
      --exp-dir $ed $MODEL_ARGS "$@" --avg 1 --use-averaged-model 0 --enable-int8-quantization 1 ) > $EXP/cand/$name/export.log 2>&1
  for m in encoder decoder joiner; do
    mv $(ls $ed/$m-*.int8.onnx | head -1) $od/$m.int8.onnx; rm -f $ed/$m-*.onnx
  done
  cp $MODEL/tokens.txt $MODEL/bpe.model $od/
}
decode_dev() {  # decode_dev <model_dir> <tên>
  for s in dev_cs dev_vi; do
    [ -f $RD/$2/$s/hyps.jsonl ] && continue
    $PY decode.py --bench $SRC/$s.jsonl --model-dir $1 --system $s --results $RD/$2 --threads 16 --batch 32 > /dev/null
  done
}
if [ $stage -le 5 ] && [ $stop -ge 5 ]; then
  echo "== 5. ứng viên: model gốc + checkpoint mỗi epoch ảo -> export int8 -> decode dev"
  decode_dev $MODEL base
  for ck in $(ls $EXP/checkpoint-*.pt | sort -t- -k2 -n); do
    n=$(basename $ck .pt | sed 's/checkpoint-//')
    [ -f $EXP/cand/ck$n/onnx/encoder.int8.onnx ] || export_onnx $EXP ck$n --iter $n
    decode_dev $EXP/cand/ck$n/onnx ck$n
  done
fi
if [ $stage -le 6 ] && [ $stop -ge 6 ]; then
  echo "== 6. chọn checkpoint trên dev, WiSE-FT alpha 0.3 / 0.5 / 0.7 / 1.0 của checkpoint đó, chọn lại"
  $PY ft/select_ckpt.py --catalog $CATALOG --dev-cs $SRC/dev_cs.jsonl --dev-vi $SRC/dev_vi.jsonl --base $RD/base \
      --cand $(ls -d $RD/ck* | sort -V) --out $RD/select_ckpt
  pick=$($PY -c "import json; print(json.load(open('$RD/select_ckpt.json'))['pick'])")
  best=$($PY -c "
import json; r = json.load(open('$RD/select_ckpt.json'))['results']
print(min((k for k in r if k.startswith('ck')), key=lambda k: r[k]['cs_wer']))")
  wck=$([ "$pick" = base ] && echo $best || echo $pick)
  echo "  checkpoint theo luật: $pick; WiSE-FT trên: $wck"
  $FT ft/interpolate_ckpt.py --base $MODEL/init_ft.pt --ft $EXP/checkpoint-${wck#ck}.pt --ft-key model \
      --alphas 0.3 0.5 0.7 1.0 --out-dir $EXP/wise --epoch 1
  for a in 0.3 0.5 0.7 1; do
    [ -f $EXP/cand/wise_$a/onnx/encoder.int8.onnx ] || export_onnx $EXP/wise/alpha_$a wise_$a --epoch 1
    decode_dev $EXP/cand/wise_$a/onnx wise_$a
    rm -f $EXP/wise/alpha_$a/epoch-1.pt
  done
  $PY ft/select_ckpt.py --catalog $CATALOG --dev-cs $SRC/dev_cs.jsonl --dev-vi $SRC/dev_vi.jsonl --base $RD/base \
      --cand $RD/wise_0.3 $RD/wise_0.5 $RD/wise_0.7 $RD/wise_1 --out $RD/select_wise
fi
if [ $stage -le 7 ] && [ $stop -ge 7 ]; then
  pick=$($PY -c "import json; print(json.load(open('$RD/select_wise.json'))['pick'])")
  echo "== 7. bench v0-draft: E3 = $pick, E7 = E3 + hotwords 1.5"
  md=$([ "$pick" = base ] && echo $MODEL || echo $EXP/cand/$pick/onnx)
  for sys in e3_v1b e7_v1b; do
    hw=(); [ $sys = e7_v1b ] && hw=(--hotwords-catalog $CATALOG --hotwords-exclude-b3 --hotwords-score 1.5)
    $PY decode.py --bench $BENCH --model-dir $md --system $sys --results results/v0_draft --threads 16 --batch 32 "${hw[@]}" > results/v0_draft/${sys}_decode.log 2>&1
    $PY score.py --bench $BENCH --catalog $CATALOG --hyps results/v0_draft/$sys/hyps.jsonl | tail -1
  done
  rm -rf /tmp/lb_e3 && mkdir -p /tmp/lb_e3 && cp -r results/v0_draft/{e0_baseline,e1_hw1.5,e3_v1,e7_v1,e3_v1b,e7_v1b} /tmp/lb_e3/
  $PY leaderboard.py --results /tmp/lb_e3 --baseline e0_baseline --out reports/e3_leaderboard | tail -2
fi
