# Runbook: chạy bench & fine-tune

Nov 14, 2026 · @God Goose

Dataset bench và dữ liệu train CS chưa có; bộ công cụ gipbench đã sẵn sàng và đã chạy thử trọn quy trình trên dữ liệu giả lập, nên khi bench v1.0 đóng băng là chạy được ngay theo 6 bước dưới đây.

Thứ tự đọc: Spec: dữ liệu, metric & độ tin cậy (dữ liệu phải đạt gì, metric tính thế nào, khi nào kết quả đáng tin) → Kiến trúc (các thành phần nối với nhau ra sao) → tab này (lệnh chạy từng bước). Bench v0 ghép từ dataset công khai theo Spec mục 2 bằng các converter ở Bước 0.

## Cài đặt & tự kiểm tra

Giải nén gipbench.zip, cài thư viện, chạy self-test (khoảng 10 giây). Self-test sinh 480 utterance giả lập, chạy validate → EDA → chấm 3 hệ thống giả → leaderboard.

```bash
unzip gipbench.zip && cd gipbench
pip install -r requirements.txt
bash tools/selftest.sh /tmp/gipbench_selftest
huggingface-cli download g-group-ai-lab/gipformer1.5-68M-rnnt --local-dir models/gip15
python tools/check_gipformer_tokenizer.py --model-dir models/gip15
```

Fine-tune cần thêm máy Linux + CUDA với torch, k2, kaldifeat đúng phiên bản trong pyproject.toml của repo gipformer (`uv sync --extra pytorch`), và `git clone https://github.com/k2-fsa/icefall`.

## Bước 0 — Dựng bench v0 từ dataset công khai

Mỗi converter tải một dataset, tự gắn tag `<en>` nháp và ghi manifest cùng phiếu duyệt CSV; `build_bench_v0.py` chỉ nhận câu đã được người duyệt. Chạy thử `--limit 20` trước: converter mới được kiểm tra trên dataset giả dựng theo schema của bản thật.

```bash
huggingface-cli login     # SEA là dataset gated: đồng ý điều kiện trên trang dataset trước
CAT=configs/catalog.tsv
python converters/conv_ggroup.py   --out data/raw/ggroup   --catalog $CAT
python converters/conv_vimedcss.py --out data/raw/vimedcss --catalog $CAT
python converters/conv_sea.py      --out data/raw/sea      --catalog $CAT --bench-hours 2
python converters/conv_local.py vivos           --root data/dl/vivos            --out data/raw/vivos
python converters/conv_local.py commonvoice     --root data/dl/cv-corpus-vi     --out data/raw/cv
python converters/conv_local.py vietsuperspeech --root data/dl/VietSuperSpeech  --out data/raw/vss
```

| Converter | Ra bench (cần duyệt) | Ra train / dev | Ghi chú |
| --- | --- | --- | --- |
| conv\_ggroup | b1\_candidates (câu có từ Anh), b5\_ggroup (pubadmin không có từ Anh) | — | Không có người nói: mỗi câu một speaker |
| conv\_vimedcss | b3\_candidates (hard), ext\_vimedcss\_test (bench ngoài) | train\_vimedcss, dev\_vimedcss | Sinh catalog\_vimedcss.tsv; thuật ngữ chỉ có ở hard được đánh reserved\_b3 |
| conv\_sea | b1\_candidates\_sea (chọn theo video tới đủ số giờ) | train\_sea | Nhãn máy: mọi câu vào bench phải sửa tay |
| conv\_local vivos / commonvoice | b5\_vivos, b5\_cv (test) | train\_vivos, train\_cv | Tải bản gốc về thư mục trước; không copy audio |
| conv\_local vietsuperspeech | — | train\_vss, dev\_vss | Mặc định bỏ câu có từ Anh (pseudo-label hay méo) |

Duyệt rồi ghép:

1. Mở từng file `*_review.csv` (Excel, Google Sheets), nghe audio, điền `decision` = ok / fix / drop; với fix thì ghi câu đúng vào `text_tagged_fixed`; điền `viet_levels` (vd `strong,mid`), `region`, `gender` nếu nghe ra. Cột `confidence` = heuristic và cột `flags` cho biết câu nào cần xem kỹ. Gắn `<en>` theo **cách đọc** (SCHEMA.md); từ mơ hồ thì nghe audio rồi ghi lý do vào `reading_note`. Hướng dẫn chi tiết: `docs/huong_dan_duyet_b1.md`.
2. Bổ sung các dòng đúng trong `term_candidates.tsv` vào `configs/catalog.tsv`.
3. Ghép bench và gộp catalog:

```bash
python converters/build_bench_v0.py \
  --candidates data/raw/ggroup/b1_candidates.jsonl data/raw/ggroup/b5_ggroup.jsonl \
               data/raw/vimedcss/b3_candidates.jsonl data/raw/sea/b1_candidates_sea.jsonl \
               data/raw/vivos/b5_vivos.jsonl data/raw/cv/b5_cv.jsonl \
  --reviews data/raw/ggroup/b1_candidates_review.csv data/raw/vimedcss/b3_candidates_review.csv \
            data/raw/sea/b1_candidates_sea_review.csv \
  --catalogs configs/catalog.tsv data/raw/vimedcss/catalog_vimedcss.tsv \
  --out data/bench_v0.jsonl --catalog-out configs/catalog_v0.tsv
```

`--accept all` nhận cả câu chưa duyệt, chỉ để chạy thử pipeline; kết quả trên bench nháp đó không dùng để kết luận. Manifest train cho Bước 5: `train_vimedcss` + `train_sea` (CS), `train_vivos` + `train_cv` + `train_vss` (replay tiếng Việt), `dev_vimedcss` (dev).

## Bước 1 — Catalog thuật ngữ & manifest

Mọi dữ liệu (bench, train, dev) dùng chung một định dạng JSONL và một catalog; chi tiết trong SCHEMA.md.

**Catalog** `configs/catalog.tsv` — bắt đầu từ `catalog.example.tsv`, nhắm 1.500–2.000 dòng:

| Cột | Ví dụ | Dùng để |
| --- | --- | --- |
| term | facebook | Dạng chuẩn (chính tả Anh) |
| category | brand | Cắt lát kết quả |
| variants | fb\|face book | Coi là ĐÚNG ở chế độ strict |
| phonetic | phây búc | Chỉ chấp nhận ở lenient (chẩn đoán) |
| reserved\_b3 | 1 | Thuật ngữ chỉ dành cho B3, cấm trong train |

**Manifest** — mỗi dòng một utterance:

```json
{"id": "b1_000123", "split": "B1", "audio": "audio/b1/000123.wav", "duration": 6.42,
 "text_tagged": "hôm nay <en>team</en> mình có <en>meeting</en> lúc ba giờ",
 "speaker": "yt_UCx9_spk2", "source": "youtube:dQw4w9WgXcQ",
 "region": "north", "gender": "f", "channel": "wideband", "domain": "tech",
 "en_meta": [{"viet_level": "mid"}, {"viet_level": "strong"}]}
```

- `speaker` và `source` phải ổn định trên toàn bộ dữ liệu (cả train) — đây là khoá chống rò rỉ và bootstrap.
- Manifest train chỉ cần `text` (không tag) thay cho `text_tagged`; split ghi tuỳ ý.
- Tool gán nhãn (Label Studio, sheet, script) xuất ra đúng định dạng này là đủ.

## Bước 2 — Kiểm tra & đóng băng bench

Bench chỉ được đóng băng khi `validate.py` trả về ĐẠT (mã thoát 0) với mọi manifest train/dev đã biết.

```bash
python validate.py --bench data/bench.jsonl --catalog configs/catalog.tsv \
    --train data/train_cs.jsonl data/train_vi.jsonl data/dev_cs.jsonl --check-audio
```

| Loại | Ví dụ | Xử lý |
| --- | --- | --- |
| LỖI — rò rỉ | speaker/source/audio của bench có trong train; thuật ngữ reserved\_b3 có trong train | Bỏ khỏi train (to\_lhotse.py tự lọc khi FT), không sửa bench |
| LỖI — dữ liệu | thiếu trường, id trùng, tag `<en>` hỏng, B1–B4 không có span, duration lệch audio | Sửa manifest |
| CẢNH BÁO | span ngoài catalog, text còn chữ số, B5 có span, thuật ngữ B3 > 2 lần | Bổ sung catalog / sửa khi có thể |

Đóng băng:

1. Đổi tên thành `data/bench_v1.0.jsonl`, cùng `catalog_v1.0.tsv`.
2. Ghi `sha256sum` của 2 file và của thư mục audio vào README dữ liệu.
3. Mọi thay đổi sau đó (sửa nhãn, thêm câu) = v1.1, chạy lại toàn bộ baseline trước khi so sánh.

## Bước 3 — EDA cho báo cáo

Một lệnh sinh report HTML tự chứa (mở offline, có dark mode, rê chuột xem số), file JSON để lưu vết, và bảng Markdown để dán thẳng vào báo cáo.

```bash
python eda.py --bench data/bench_v1.0.jsonl --catalog configs/catalog_v1.0.tsv \
    --train data/train_cs.jsonl data/train_vi.jsonl --audio-stats --out reports/eda_v1.0
# -> eda_report.html, eda_summary.json, eda_summary.md
```

| Phần trong report | Viết gì vào báo cáo |
| --- | --- |
| Kiểm tra mục tiêu (giờ/split, ≥ 150 người nói B1, ≥ 4.000 span, ≥ 1.000 thuật ngữ, miền ≥ 20%, người nói lớn nhất ≤ 10%) | Bench đạt bao nhiêu mục tiêu; mục chưa đạt ảnh hưởng kết luận nào |
| Tổng quan theo split | Bảng quy mô (copy từ eda\_summary.md) |
| Phân bố thời lượng | Có lệch giữa split không (B4 đọc kịch bản thường ngắn hơn) |
| Miền, giới tính, kênh | Nhóm nào thiếu → kết quả cắt lát nhóm đó kém tin cậy |
| Hồ sơ code-switching (mức CS, vị trí, độ dài span, mức Việt hoá) | Bench mô tả kiểu CS nào: chèn 1 từ giữa câu hay cụm dài |
| Thuật ngữ (category, đuôi dài, top 30, độ phủ catalog) | % thuật ngữ chỉ xuất hiện 1 lần; category nào chiếm ưu thế |
| Mức trùng với train | B3 phải 100% “0 lần”; B1/B2 có bao nhiêu span đã thấy khi train — câu hỏi công bằng khi so FT với baseline |
| Chất lượng audio (SNR ước lượng, clipping, sample rate) | B2/B5 ở kênh thoại kém hơn bao nhiêu dB |

Chạy lại EDA mỗi lần thêm dữ liệu train mới: phần “mức trùng với train” thay đổi theo train, dù bench giữ nguyên.

## Bước 4 — Baseline E0 & hotwords E1

E0 là mốc cho mọi so sánh, nên chạy đúng cấu hình production (int8, modified beam search, beam 4, cùng số luồng CPU). E1 quét điểm hotwords để tìm mức cân bằng giữa recall và chèn sai trên B5.

```bash
# E0
python decode.py --bench data/bench_v1.0.jsonl --model-dir models/gip15 --system e0_baseline
python score.py  --bench data/bench_v1.0.jsonl --catalog configs/catalog_v1.0.tsv --hyps results/e0_baseline/hyps.jsonl

# E1: quét hotwords-score
for s in 1.0 1.5 2.0 3.0; do
  python decode.py --bench data/bench_v1.0.jsonl --model-dir models/gip15 --system e1_hw$s \
      --hotwords-catalog configs/catalog_v1.0.tsv --hotwords-score $s
  python score.py --bench data/bench_v1.0.jsonl --catalog configs/catalog_v1.0.tsv --hyps results/e1_hw$s/hyps.jsonl
done
```

- Chạy thêm `e0_greedy` (`--decoding-method greedy_search`) nếu production dùng greedy, và `e0_fp32` (`--quantize fp32`) để biết int8 mất bao nhiêu.
- Model tham chiếu (Whisper-large-v3, Qwen3-ASR, PhoWhisper) chỉ cần xuất `results/<tên>/hyps.jsonl` dạng `{"id", "hyp"}` rồi chạy score.py như trên.
- Hotwords lấy cả thuật ngữ B3: bias lúc decode được phép biết danh sách từ (đó chính là điểm mạnh của nó so với FT). Dùng `--hotwords-exclude-b3` để đo trường hợp danh sách không đầy đủ.

## Bước 5 — Các thí nghiệm fine-tune

Mọi thí nghiệm FT đi qua cùng một đường `ft/run_e3.sh`; chỉ đổi dữ liệu đầu vào và biến môi trường. Chạy từng stage: `bash ft/run_e3.sh 0 7`.

| Stage | Việc | Ghi chú |
| --- | --- | --- |
| 0 | validate bench với mọi manifest train/dev | Dừng nếu có LỖI |
| 1 | to\_lhotse.py: lọc rò rỉ, speed perturb 0.9/1.0/1.1, 30% mô phỏng thoại 8 kHz, text IN HOA | In ra số câu bị bỏ và lý do |
| 2 | compute\_fbank.py: fbank 80 chiều | Giống recipe librispeech/zipformer |
| 3 | wrap\_ckpt.py + patch\_finetune.py | Tạo `zipformer/finetune_cs.py` trong icefall |
| 4 | finetune\_cs.py trộn CS : tiếng Việt theo CS\_W : VI\_W | Valid trên dev CS riêng, không phải bench |
| 5 | export-onnx.py, trung bình AVG epoch cuối, fp32 + int8 | Copy tokens.txt, bpe.model vào `exp/.../onnx` |
| 6 | decode + score bản int8, có và không hotwords | E7 ra cùng lúc |
| 7 | leaderboard.py | So với e0\_baseline |

Cấu hình khởi đầu (biến trong run\_e3.sh): `LR=0.0045` (mặc định FT của icefall), `EPOCHS=8`, `AVG=3`, `MAXDUR=600` (giảm nếu GPU < 24 GB), `CS_W=VI_W=0.5`. Giữ nguyên tham số kiến trúc mặc định của recipe — Gipformer dùng đúng mặc định đó.

| Thí nghiệm | Đổi gì so với E3 | Cách chạy |
| --- | --- | --- |
| E2 — LM text CS | Không train ASR; LM ngoài từ text CS | Train RNN-LM/n-gram bằng recipe LM của icefall, giải mã PyTorch (`zipformer/decode.py` với shallow fusion/LODR), xuất `{id, hyp}` rồi score.py |
| E3 — FT CS thật | — | `SYSTEM=e3_ft bash ft/run_e3.sh 0 7` |
| E3 biến thể | Tỉ lệ trộn, LR | `CS_W=0.3 VI_W=0.7 SYSTEM=e3_mix37 EXP=exp/e3_mix37 bash ft/run_e3.sh 4 7` (bỏ qua stage 0–3 vì dữ liệu đã có) |
| E4 — + TTS | Thêm CS tổng hợp: câu LLM sinh chứa thuật ngữ (trừ reserved\_b3), TTS đa giọng Việt | Viết manifest `train_tts.jsonl` (speaker = id giọng TTS), thêm thành tập thứ 3 trong `--train-cuts`; quét tỉ lệ 0 / 20 / 50% |
| E5 — ghép thuật ngữ | Chèn audio thuật ngữ (thu hoặc TTS) vào câu tiếng Việt tại ranh giới từ | Cần forced alignment để cắt ranh giới; xuất manifest như E4 |
| E6 — mở rộng BPE | Chỉ làm khi check tokenizer ra ĐỎ hoặc E3 chững lại | Train BPE mới giữ nguyên piece cũ + thêm piece tiếng Anh; mở rộng `decoder.embedding`, `joiner.output_linear`, `simple_lm_proj` theo vocab mới (hàng mới khởi tạo nhỏ), FT lâu hơn E3 |
| E7 — FT + hotwords | Không train thêm | Stage 6 của run\_e3.sh đã chạy sẵn (`<system>_hw`) |
| E8 — hậu xử lý | Sửa phây búc → facebook sau ASR | AdaCS hoặc LLM đọc N-best; nhận hyps.jsonl, xuất hyps.jsonl mới dưới tên `e8_...` |

Mỗi thí nghiệm một `SYSTEM` và một `EXP` riêng; không ghi đè kết quả cũ. Đặt tên `eN_mô-tả-ngắn` (ví dụ `e4_tts20`) để leaderboard đọc được.

## Bước 6 — Leaderboard & chọn phương án

Leaderboard tự áp luật chọn ở tab chính và ghi một câu kết luận: hệ thống nào tốt nhất, tốt hơn baseline bao nhiêu, kèm CI95.

```bash
python leaderboard.py --results results --baseline e0_baseline --out reports/leaderboard
# -> leaderboard.md (dán vào báo cáo), leaderboard.csv, leaderboard.html (biểu đồ CS-WER + CI)
```

| Luật | Ngưỡng mặc định | Tham số |
| --- | --- | --- |
| Loại nếu WER trên B5 tăng | > 0.3 điểm | `--max-b5-delta` |
| Loại nếu N-WER (B1–B4) tăng | > 0.3 điểm | `--max-nwer-delta` |
| Loại nếu chậm hơn | RTF > 1.2× baseline | `--max-rtf-ratio` |
| Chỉ tính là cải thiện | CI95 của ΔCS-WER (B1+B2) nằm dưới 0 | bootstrap cặp theo người nói, `--iters` |
| Xếp hạng | CS-WER B1+B2; hoà thì CS-WER B3 | — |

Cách đọc các cột:

- **CS-WER lenient − strict** = lỗi “nghe đúng, viết sai” (ra phây búc thay vì facebook). Khoảng này lớn thì E8 (hậu xử lý) đáng làm; nhỏ thì lỗi nằm ở phần nghe, phải FT.
- **CS-WER B3** = khả năng tổng quát sang thuật ngữ chưa thấy. FT thường giảm mạnh B1/B2 nhưng ít ở B3; hotwords thì ngược lại — đây là lý do E7 thường thắng.
- **Term recall / precision và “B5 chèn sai /100 utt”** = dấu hiệu over-biasing. Hotwords-score cao làm recall tăng nhưng B5 chèn sai cũng tăng.
- **slices.md** trong mỗi `results/<system>/`: CS-WER theo miền, kênh, giới, domain, category, mức Việt hoá, và 25 thuật ngữ sai nhiều nhất — dùng để viết “phương án thắng còn yếu ở đâu”.

Trên dữ liệu giả lập của self-test, leaderboard loại hệ thống hotwords vì B5 WER tăng và chọn hệ thống FT — đây chỉ là minh hoạ cách luật vận hành, không phải kết quả thật.

## Bẫy thường gặp

- **Kiểu chữ của tokenizer.** Gipformer xuất IN HOA; text train phải cùng kiểu chữ với bpe.model (to\_lhotse.py mặc định `--case upper`). Kiểm tra bằng check\_gipformer\_tokenizer.py trước khi FT.
- **Checkpoint.** `load_model_params` của icefall đòi `{"model": ...}`; dùng wrap\_ckpt.py. Chiều đầu của `decoder.embedding.weight` phải bằng số piece của bpe.model.
- **MUSAN.** Recipe bật `--enable-musan` mặc định và cần `musan_cuts.jsonl.gz`; run\_e3.sh tắt đi. Muốn dùng thì chạy `local/compute_fbank_musan.py` của icefall vào `data/fbank`.
- **Chấm bản int8 qua sherpa-onnx**, không chấm bản PyTorch: số liệu phải là số của đường chạy production.
- **Dữ liệu tổng hợp và B3.** Câu TTS (E4) không được chứa thuật ngữ reserved\_b3, nếu không B3 mất ý nghĩa; to\_lhotse.py lọc giúp nếu catalog đánh dấu đủ.
- **Chọn tham số trên bench.** Quét hotwords-score, LR, tỉ lệ trộn trên dev CS, không trên bench; bench chỉ chạy cho cấu hình cuối của mỗi hướng để tránh overfit vào bench.
- **Số đọc khác nhau.** “hai mươi tư” và “hai mươi bốn”, “linh” và “lẻ” bị tính lệch; quy ước gán nhãn nên chốt một dạng (score.py đọc chữ số thành: mốt, lăm, bốn, linh).
