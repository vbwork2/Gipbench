# Dữ liệu code-switching tổng hợp bằng TTS (pipeline E4)

Vì sao: 78% span tiếng Anh của B1 là thuật ngữ không xuất hiện lần nào trong train, nên fine-tune thuần không cải thiện được chúng.
TTS phủ được thuật ngữ, nhưng chỉ có ích khi làm đúng: đúng nhãn, đúng cách đọc, nhiều giọng, có nhóm đối chứng, có kiểm vòng lại.
Toàn bộ chạy được offline, không cần engine thật để thử (backend `mock`). Bảng cách đọc `configs/tts_lexicon.tsv` và mẫu câu
`configs/tts_templates.tsv` là **bản NHÁP do máy soạn, cần người bản ngữ duyệt** trước khi dùng thật.

## Sơ đồ

```
catalog + lexicon + mẫu (+ câu viết sẵn)      tools/tts_make_text.py   ->  tts_script.jsonl, tts_holdout_terms.txt, report
tts_script.jsonl + voices.tsv                 tools/tts_synth.py       ->  train_tts_raw.jsonl + audio/*.wav (16 kHz mono)
train_tts_raw.jsonl                           tools/asr_hyps.py        ->  hyps.jsonl của ASR mạnh (bạn chạy, trên GPU / CPU)
raw + hyps + catalog + lexicon                tools/tts_verify.py      ->  train_tts.jsonl (đã lọc) + report
train_tts.jsonl                               ft/curate_train.py (--max-synthetic-share) -> ft/to_lhotse.py -> ft/mix_cuts.py (nhóm synth)
```

Mỗi câu có hai văn bản: `text_tagged` (nhãn: chính tả Anh trong `<en>…</en>`, số viết dạng đọc, cùng quy ước bench) và `tts_text`
(thứ đưa cho máy đọc: phần Anh thay bằng cách đọc trong lexicon, vd `meeting` -> "mít tinh" / "mi ting" / "mi đinh").
`en_meta[].viet_level` (strong / mid / native) lấy từ cách đọc đã dùng. Thuật ngữ `reserved_b3` không bao giờ vào kịch bản; một phần
thuật ngữ (mặc định 30%, hash ổn định) là **đối chứng**: không bao giờ được tổng hợp, để bench còn lát "chưa thấy".

## Lệnh mẫu

```bash
export PYTHONIOENCODING=utf-8
B=data/bench_v0_draft; O=data/tts
# 1. kịch bản (--require-lexicon: chỉ thuật ngữ đã có cách đọc được duyệt; --sentences cau_viet_san.jsonl: câu người / LLM viết)
python tools/tts_make_text.py --catalog $B/catalog_v0_draft.tsv --bench $B/bench_v0_draft.jsonl --out-dir $O/script \
    --contexts-per-term 5 --readings-per-term 2 --holdout-frac 0.3 --seed 0 --include-lexicon-terms
# 2. tổng hợp (thử đường ống: --backend mock; engine thật: xem mục dưới). voices.tsv: voice, backend_voice, ref_audio, gender, region
python tools/tts_synth.py --script $O/script/tts_script.jsonl --voices data/tts/voices.tsv --backend command \
    --command "mytts --in {text_file} --voice {voice} --out {out}" --bench $B/bench_v0_draft.jsonl --out-dir $O/raw \
    --min-voices-per-term 3 --max-voice-share 0.25
# 3. hyp của ASR mạnh trên audio TTS (venv requirements-asr.txt)
python tools/asr_hyps.py --engine faster-whisper --model large-v3 --manifest $O/raw/train_tts_raw.jsonl \
    --out results/fw_large_v3/tts/hyps.jsonl
# 4. lọc (nhiều hệ: lặp --hyps và đặt --min-agree)
python tools/tts_verify.py --manifest $O/raw/train_tts_raw.jsonl --hyps fw=results/fw_large_v3/tts/hyps.jsonl \
    --catalog $B/catalog_v0_draft.tsv --lexicon configs/tts_lexicon.tsv --out $O/train_tts.jsonl
# 5. vào chuỗi E3: curate chặn trần synthetic, rồi to_lhotse (chặn rò rỉ) và mix nhóm `synth` (datasets = tts)
python ft/curate_train.py --manifest <các manifest train khác> $O/train_tts.jsonl --catalog $B/catalog_v0_draft.tsv \
    --max-synthetic-share 0.2 --out data/train_curated.jsonl
python tools/test_tts.py   # kiểm cả chuỗi bằng backend mock (không cần engine, ASR, mạng)
```
`tts_synth` chạy tiếp được (bỏ qua câu đã có wav hợp lệ), lỗi từng câu ghi vào `synth_errors.jsonl` mà không dừng mẻ, dừng khi
`--abort-after-fails` lỗi liên tiếp. Resample bằng Python thuần mất ~0.5 s/câu; cài `numpy` + `soxr` (đã có trong requirements.txt)
thì nhanh hơn nhiều.

## Cắm engine TTS riêng

- `--backend command --command "… {text_file} … {voice} … {out}"`: chỗ trống `{text_file}` (UTF-8, text đọc), `{voice}` (cột
  `backend_voice`, mặc định = `voice`), `{voice_id}`, `{ref_audio}`, `{out}`. Lệnh được tách theo kiểu shell rồi thay chỗ trống từng
  token, chạy không qua shell. Engine ghi đuôi khác `.wav` (mp3 ...): thêm `--backend-opt raw_ext=.mp3` (cần `soundfile` hoặc `ffmpeg`).
- `--backend python --python-fn my_engine:synth` (hoặc `đường/dẫn/my_engine.py:synth`): hàm `synth(text, voice, out_path) -> None`;
  `voice` là chuỗi (`backend_voice`) có thêm thuộc tính `.id`, `.ref_audio`, `.gender`, `.region`. Nếu engine clone giọng từ
  `ref_audio`, dùng thuộc tính này.
- `--backend edge` (gói `edge-tts`): **gửi văn bản lên dịch vụ online của Microsoft**, không phải mặc định; adapter chưa được chạy thử
  ở repo này; chỉ có 2 giọng Việt nên không đủ `--min-voices-per-term 3` nếu dùng một mình.
- Giọng clone: `ref_audio` KHÔNG được lấy từ audio / người nói của bench. Có `--bench` thì `tts_synth` thoát mã 2 nếu trùng đường dẫn,
  trùng nội dung (md5), hoặc `ref_speaker` / `ref_source` (cột tuỳ chọn của voices.tsv) trùng bench.

## Quét tỉ trọng TTS (E4: 0 / 20 / 50% lượng CS)

Hai chốt độc lập:

1. Trần theo GIỜ có sẵn: `ft/curate_train.py --max-synthetic-share S` (mặc định 0.2) cắt dòng `synthetic: true` còn <= S tổng giờ CS.
   Muốn thử 50% phải nâng `--max-synthetic-share 0.5` và có đủ giờ TTS.
2. Tỉ trọng BATCH: nhóm `synth` của `configs/train_mix.tsv` (cột datasets = `tts`), ghi đè bằng `MIX_SYNTH`. TTS chiếm
   `x = s / (0.45 + s)` lượng CS (0.45 = cs_real 0.35 + cs_pseudo 0.10) nên `s = 0.45x / (1 - x)`; tổng các nhóm phải = 1.00 nên phải
   hạ các nhóm không-CS tương ứng (kiểm bằng `python tools/mix_plan.py`). Ví dụ (giữ tỉ lệ replay_human : distill : confusable = 5 : 4 : 1):

| x (TTS / CS) | MIX_SYNTH | MIX_REPLAY_HUMAN | MIX_DISTILL | MIX_CONFUSABLE |
| --- | --- | --- | --- | --- |
| 0% | 0 | 0.30 | 0.24 | 0.06 (hoặc không có nhóm synth) |
| 20% | 0.11 | 0.22 | 0.175 | 0.045 |
| 50% | 0.45 | 0.05 | 0.04 | 0.01 (thực tế quá lệch: replay còn quá ít, dễ quên tiếng Việt thường) |

Không có giờ TTS thì `MIX_SYNTH` vô nghĩa: `run_e3_v1.sh` bỏ nhóm thiếu file cuts.

## Đọc kết quả

Chấm bench với `score.py --train-manifest <manifest train có cả dòng thật và TTS>` (một agent khác đang thêm các lát này; ở đây chỉ
mô tả). Mỗi span bench được gắn `train_freq_real` / `train_freq_synth`:

- `unseen`: thuật ngữ không có trong train (cả thật lẫn TTS) — gồm **nhóm đối chứng** (`tts_holdout_terms.txt`). Nếu `unseen` cũng khá
  lên khi thêm TTS thì TTS giúp tổng quát hoá (học cách đọc / ghép âm), không chỉ nhớ từ.
- `synth_only`: thuật ngữ chỉ xuất hiện trong dữ liệu TTS (real = 0, synth > 0) — lát đo trực tiếp TTS phủ được gì. Kỳ vọng
  `synth_only` tốt hơn `unseen` của cùng hệ.
- `real_1-4`, `real_5+`: có dữ liệu thật; TTS không được làm tụt các lát này, và B2 / B5 (giọng thật, tiếng Việt thường) không
  được xấu đi (dấu hiệu overfit giọng TTS).

So E3 (không TTS) với E3 + TTS theo từng mức ở bảng trên; kết luận "TTS có giúp" chỉ khi `synth_only` hoặc `unseen` tăng mà
`real_*`, B2, B5 không giảm. Đọc `tts_verify` report trước khi train: thuật ngữ bị loại nhiều với cùng một cách đọc nghĩa là dòng đó
trong lexicon nhiều khả năng SAI (sửa `configs/tts_lexicon.tsv`, chạy lại từ bước 1).

## Giới hạn đã biết

- Lexicon nháp (~107 thuật ngữ, ~220 cách đọc) do máy soạn, chưa ai nghe lại. Catalog thật có ~1 600 thuật ngữ không bị cấm; phần
  không có trong lexicon dùng cách đọc dự phòng (đánh vần viết tắt toàn phụ âm, còn lại đưa nguyên chính tả cho engine): rất dễ đọc sai.
  Báo cáo bước 1 liệt kê chúng; dùng `--require-lexicon` để chỉ giữ thuật ngữ đã có cách đọc, hoặc bổ sung lexicon trước.
- Dòng dự phòng có `viet_level` = `mid` chỉ là giá trị giữ chỗ (schema bắt buộc), không phải mức đọc thật.
- Viết tắt đọc kiểu Việt (`a pê i`) vẫn được gắn `<en>` ở nhãn như `api`; theo SCHEMA mục 1 viết tắt đọc kiểu Việt đáng ra không gắn
  `<en>`: quyết định này cần người duyệt chốt.
- Mẫu câu là văn nói chung chung, chưa cân theo từ loại (động từ như `deploy`, `check` đặt vào chỗ danh từ nghe gượng).
- Kiểm vòng lại phụ thuộc ASR: ASR yếu tiếng Việt-Anh làm loại oan; ASR "nhớ" chính tả làm giữ lầm audio đọc sai chữ nhưng ra đúng chữ.
