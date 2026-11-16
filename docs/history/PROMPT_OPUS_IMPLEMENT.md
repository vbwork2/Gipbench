# Prompt: implement các phần đã đề xuất cho gipbench

> Dán toàn bộ nội dung dưới đường kẻ cho Claude Opus (Claude Code chạy trên máy có dữ liệu, có mạng tới Hugging Face, lý tưởng là có GPU). Đặt `gipbench/` (giải nén từ gipbench.zip) trong thư mục làm việc trước khi bắt đầu.

---

## Bối cảnh

Mình đang cải thiện **Gipformer 1.5** (`g-group-ai-lab/gipformer1.5-68M-rnnt`, Zipformer-RNNT 68M, icefall + sherpa-onnx, output IN HOA không dấu câu) để nhận đúng **từ tiếng Anh nói giọng Việt** chen trong câu tiếng Việt (code-switching, CS), và **sau khi fine-tune vẫn phải tốt nhất trên tiếng Việt tổng quát**.

Tài liệu thiết kế nằm trong `gipbench/docs/research/` — đọc hết trước khi làm: `01_ke_hoach_research_pipeline.md` (papers, pipeline E0–E9), `02_spec_du_lieu_metric_do_tin_cay.md` (spec dữ liệu, metric, độ tin cậy; mục 10 là đề xuất chống overfit mà các phase dưới đây implement), `03_kien_truc.md` (+ `so_do_kien_truc.png`), `04_runbook_bench_ft.md`. Nếu chỗ nào trong tài liệu mâu thuẫn với prompt này thì prompt này đúng, và ghi chỗ mâu thuẫn vào báo cáo.

Thư mục `gipbench/` là bộ công cụ đã có, đã chạy thử trên dữ liệu giả lập. Đọc tiếp, theo thứ tự: `README.md`, `SCHEMA.md`, `lib/textnorm.py`, `lib/align.py`, `score.py`, `leaderboard.py`, `converters/common.py`, `ft/to_lhotse.py`, `ft/patch_finetune.py`, `ft/run_e3.sh`.

Những gì đã có:
- Manifest JSONL thống nhất (`id, split, audio, duration, text_tagged|text, speaker, source, region, gender, channel, domain, en_meta`). Span tiếng Anh được đánh dấu `<en>…</en>`. Catalog TSV gồm các cột `term, category, variants, phonetic, reserved_b3`.
- Bench có 5 split. B1 là CS tự nhiên, B2 là CS trong domain sản phẩm, B3 là thuật ngữ chưa thấy (cấm có trong train), B4 là câu đọc có kiểm soát, B5 là tiếng Việt thuần để chặn suy giảm.
- Metric. Chính là CS-WER strict trên B1+B2. Ràng buộc: N-WER và WER trên B5 không được tăng quá 0.3 điểm, WER của từng domain trong B5 không tăng quá 0.5 điểm, RTF không vượt 1.2× baseline. Chỉ tính là cải thiện khi CI95 của ΔCS-WER (bootstrap cặp theo người nói) nằm hoàn toàn dưới 0.
- Converter: `conv_ggroup`, `conv_vimedcss`, `conv_sea`, `conv_local` (vivos, commonvoice, vietsuperspeech), cùng `build_bench_v0.py` có phiếu duyệt CSV. **Các converter chỉ mới chạy trên dataset giả dựng theo schema đọc từ trang dataset.**
- Fine-tune: `to_lhotse.py` (lọc rò rỉ) → `compute_fbank.py` → `wrap_ckpt.py` → `patch_finetune.py` (tạo `zipformer/finetune_cs.py`, nhận `--train-cuts PATH:WEIGHT` và `--valid-cuts`) → `export-onnx.py`. Ngoài ra có `interpolate_ckpt.py` (WiSE-FT).

## Quy tắc làm việc

1. Trước khi sửa bất kỳ thứ gì, chạy `bash tools/selftest.sh` và `python tools/test_converters.py`; cả hai phải đạt. Sau mỗi task chạy lại, và **thêm kiểm thử cho phần mới** vào một trong hai script (dùng dữ liệu giả, không cần mạng).
2. **Không đoán schema dataset.** Tải vài dòng thật (`--limit 20`, `datasets` streaming, hoặc đọc trực tiếp file parquet/json) rồi in ra tên cột và kiểu dữ liệu trước khi viết code. Dataset bị gated hoặc không truy cập được thì dừng task đó và báo lại; không viết code dựa trên phỏng đoán.
3. Không đổi định nghĩa metric, bộ chuẩn hoá hay quy ước transcript nếu chưa hỏi mình. Những thứ này gắn với phiên bản bench.
4. Giữ phong cách sẵn có: mỗi script là một CLI độc lập, docstring ở đầu file bằng tiếng Việt, thông báo cho người dùng bằng tiếng Việt, chạy được trên Python 3.10 trở lên (không dùng f-string lồng cùng loại nháy), các script giao tiếp với nhau chỉ qua file.
5. Ngưỡng nào là đề xuất chưa kiểm chứng thì đặt thành tham số CLI có giá trị mặc định, đừng hard-code.
6. Mỗi task commit riêng một lần. Commit message nêu đã kiểm thử gì, và **chưa** kiểm thử gì.
7. Phần nào không chạy thật được (không có GPU, không có dữ liệu) thì ghi rõ trong báo cáo cuối. Không được báo "đã chạy" khi chưa chạy.
8. **Chỉ làm phase mình yêu cầu, xong thì DỪNG.** Ghi `reports/phase<N>_report.md` (theo mẫu báo cáo cuối) rồi chờ mình duyệt. Kết quả của phase trước (schema thật, chất lượng tự gắn tag) có thể làm thay đổi phase sau; nếu thấy cần đổi kế hoạch thì đề xuất trong báo cáo, không tự đổi.

## Phase 0 — Xác minh converter trên dữ liệu thật

**T0.1** Chạy từng converter với `--limit 20` trên dataset thật: `g-group-ai-lab/vi-asr-{tech,edu,finance,pubadmin}-test`, `tensorxt/ViMedCSS`, `gijs/sea_speech` (gated; nếu mình chưa được duyệt quyền thì bỏ qua và báo lại), và VIVOS, Common Voice, VietSuperSpeech nếu đã có trong `data/dl/`.
- Sửa mọi chỗ lệch schema. Đặc biệt kiểm tra cấu trúc thật của cột `spans` trong SEA và cập nhật `english_spans()` trong `conv_sea.py`.
- Kiểm tra các nhãn gốc trong VIVOS và VietSuperSpeech có viết hoa không, và sau `basic_tokens` có đúng chữ thường không.
- **Chấp nhận khi:** chạy đủ các bộ không lỗi; in ra 5 câu `text_tagged` mẫu của mỗi bộ để mình xem; ghi số câu, số giờ và phân bố `auto_tag.confidence` vào `reports/converter_check.md`.

**T0.2** Đo chất lượng của bộ tự gắn tag (`auto_tag`): lấy ngẫu nhiên 100 câu G-Group có từ Anh, gắn tag tay vào một file vàng do bạn tạo, rồi tính precision và recall ở mức span. Nếu recall < 0.9, đề xuất cách sửa heuristic (danh sách từ mơ hồ, cấu trúc âm tiết) và sửa luôn. Báo lại con số trước và sau khi sửa.

## Phase 1 — Dữ liệu train chống overfit

**T1.1 `ft/curate_train.py`**: đọc nhiều manifest train, xuất một manifest đã lọc kèm báo cáo. Có tham số cho từng luật:
- `--max-per-term 50`: mỗi thuật ngữ catalog xuất hiện tối đa N câu. Khi lấy mẫu, ưu tiên đa dạng ngữ cảnh (n-gram quanh thuật ngữ) và đa dạng người nói.
- `--min-speakers-per-term 3`: thuật ngữ nào chưa đủ số người nói thì ghi vào báo cáo (không loại).
- `--max-speaker-share 0.02`: một người nói tối đa 2% số giờ CS.
- `--max-domain-share 0.35`: một domain tối đa 35% số giờ CS. Giảm bằng cách lấy mẫu xuống; ViMedCSS sẽ bị ảnh hưởng.
- `--max-synthetic-share 0.2`: với các dòng có `synthetic: true` (dữ liệu TTS hoặc ghép).
- `--holdout-domain finance`: loại khỏi train mọi câu thuộc domain này (dùng làm domain CS hoàn toàn ngoài train).
- `--agreement-hyps A.jsonl B.jsonl`: chỉ giữ các câu CS có nhãn máy (SEA) khi span tiếng Anh trong nhãn gốc khớp với hyp của ít nhất một trong hai hệ thống ASR (so sau `normalize_hyp`).

Báo cáo gồm: số giờ trước/sau theo domain, nguồn và người nói; số thuật ngữ bị cắt; danh sách thuật ngữ thiếu người nói.
**Chấp nhận khi:** có kiểm thử trên manifest giả chứng minh từng luật hoạt động độc lập.

**T1.2 `ft/make_distill.py`** (replay tự chưng cất): nhận manifest audio tiếng Việt chưa có nhãn, gọi `decode.py` của model **gốc** (greedy và modified beam search), và chỉ giữ câu mà hai cách giải mã cho kết quả giống hệt nhau (dùng làm proxy cho độ tin cậy). Lọc thêm theo tỉ lệ số token trên giây và theo thời lượng. Xuất manifest train với `text` = hyp của model gốc, `distill: true`.
- Nếu sherpa-onnx trả được log-prob của token thì thêm ngưỡng confidence; kiểm tra API thật rồi mới dùng.
- **Chấp nhận khi:** chạy được trên vài file audio thật với Gipformer 1.5 thật, và báo cáo tỉ lệ câu giữ lại.

**T1.3 `tools/make_confusables.py`** (câu tiếng Việt gần âm với thuật ngữ):
- Từ catalog, lấy các dạng `phonetic` và các âm tiết tiếng Việt trùng hoặc gần âm với thuật ngữ (vd tim/team, ga/gas, sóp/shop).
- Quét transcript tiếng Việt thuần (vivos, Common Voice, VietSuperSpeech đã lọc) để tìm câu chứa các âm tiết đó **với nghĩa tiếng Việt**.
- Xuất 2 file. File thứ nhất là ứng viên cho B5 domain `confusable` (200–300 câu, kèm phiếu duyệt). File thứ hai là danh sách cho train, dùng làm ví dụ âm. Hai file không được trùng nhau: tách theo người nói và nguồn.
- Danh sách cặp gần âm phải là tham số/file cấu hình; đừng hard-code mấy chục từ.

## Phase 2 — Bench v1 và đánh giá

**T2.1 Converter cho B5 nhiều domain.** Mục tiêu là B5 có đúng các tập mà Gipformer công bố số liệu. Tìm trên Hugging Face, đọc trang dataset và kiểm tra license trước khi viết code:
- VLSP 2020 Task 1
- VietMed
- MultiMed (phần tiếng Việt)
- LSVSC
- FLEURS vi

Mỗi bộ chỉ lấy split test, gán `domain` rõ ràng, và đặt `speaker`/`source` theo quy ước chung (`<dataset>:<id>`). Bộ nào không tìm được, bị gated hoặc license không cho phép thì ghi vào báo cáo và bỏ qua.

**T2.2 `build_bench_v0.py`**:
- Thêm `--max-per-term 10` (cân bằng thuật ngữ trong bench, giữ đa dạng người nói).
- Thêm `--min-domain-tokens 500` để cảnh báo khi một domain B5 quá nhỏ.
- Thêm `--version vX.Y`: ghi file `bench_vX.Y.jsonl`, `bench_vX.Y.sha256` và `catalog_vX.Y.tsv`.

**T2.3 `leaderboard.py`**:
- Thêm biểu đồ SVG hai trục, mỗi hệ thống là một điểm: trục x là WER tổng quát (B5), trục y là CS-WER (B1+B2). Đánh dấu biên Pareto và vạch ngưỡng ràng buộc.
- Thêm bảng WER B5 theo từng domain so với baseline.
- Báo cáo riêng CS-WER của domain hold-out (tham số `--holdout-domain`).
- Biểu đồ theo đúng phong cách SVG trong `lib/svgcharts.py`, có dark mode và tooltip. Chụp màn hình bằng Playwright rồi tự xem lại trước khi báo xong.

## Phase 3 — Huấn luyện chống overfit

**T3.1 `ft/select_ckpt.py`**: với mỗi epoch đã lưu (sau khi lấy trung bình `--avg`) và mỗi alpha WiSE-FT (`interpolate_ckpt.py`):
- export ONNX int8;
- giải mã **dev CS** và **dev tiếng Việt** (hai file riêng, cả hai không trùng bench);
- chấm bằng `score.py`.
Chọn điểm có CS-WER dev thấp nhất trong số các điểm có WER dev tiếng Việt tăng không quá ngưỡng (tham số). Xuất bảng tất cả các điểm đã thử cùng điểm được chọn.
**Chấp nhận khi:** chạy được với một checkpoint thật (hoặc nêu rõ đã chạy tới đâu nếu không có GPU).

**T3.2 Đóng băng một phần encoder**: thêm vào `patch_finetune.py` tham số `--freeze-modules` (danh sách prefix tên tham số, vd `encoder.encoders.0,encoder.encoders.1,encoder_embed`).
- In tên module thật của Zipformer trong icefall hiện tại trước khi chọn prefix mặc định; không đoán tên.
- Đặt `requires_grad=False` cho các tham số khớp prefix, và log số tham số được train.
- Kiểm tra optimizer `ScaledAdam` của icefall có xử lý đúng tham số bị đóng băng không; nếu không thì lọc chúng ra khỏi optimizer.

**T3.3 Cập nhật `ft/run_e3.sh`**:
- thêm stage chạy `curate_train.py` trước `to_lhotse`;
- trộn thêm replay chưng cất trong `--train-cuts`;
- truyền hai tập valid;
- thêm stage WiSE-FT cộng `select_ckpt.py`, đặt sau export.
Mọi tỉ trọng trộn là biến môi trường, mặc định: CS thật 0.40, tổng hợp 0.10, replay có nhãn người 0.25, chưng cất 0.20, câu gần âm 0.05.

## Phase 4 — Chỉ thiết kế, chưa implement (viết `docs/design_*.md` rồi hỏi mình trước khi code)

- **E4 TTS**: tìm hiểu các TTS hiện có hỗ trợ tiếng Việt chen tiếng Anh, có nhiều giọng và có thể dùng thương mại; so sánh license, chất lượng và chi phí. Đề xuất pipeline sinh câu (từ catalog, loại trừ thuật ngữ `reserved_b3`), số giọng, số giờ.
- **E6 mở rộng BPE**: chỉ cần nếu `tools/check_gipformer_tokenizer.py` ra mức ĐỎ. Thiết kế cách thêm piece mới và resize `decoder.embedding`, `joiner.output_linear` và `simple_lm_proj` mà không phá các piece cũ.
- **E8 hậu xử lý**: AdaCS hoặc LLM đọc N-best. Nhận vào và xuất ra `hyps.jsonl`.

## Báo cáo cuối (bắt buộc)

Tạo `reports/IMPLEMENTATION_REPORT.md`, gồm:
1. Bảng task: trạng thái (xong / một phần / chưa làm), file đã đổi, kiểm thử đã chạy, **những gì chưa chạy trên dữ liệu hoặc model thật**.
2. Schema thật của từng dataset đã đọc được, và chỗ nào khác với giả định trong code cũ.
3. Mọi quyết định đã tự đưa ra mà mình nên xem lại.
4. Lệnh chạy lại toàn bộ, từ dữ liệu thô tới leaderboard.
