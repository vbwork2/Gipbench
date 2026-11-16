# Ma trận thí nghiệm fine-tune (đăng ký trước), 16/11/2026

**Chưa chạy thí nghiệm nào trong tài liệu này.** Mọi ô kết quả là `N/A (chưa chạy)`. Đây là kế hoạch có giả thuyết, metric
và luật quyết định cố định TRƯỚC khi có số, để tránh chọn kết quả sau khi xem. Giao thức đánh giá:
[benchmark_protocol_v1.md](benchmark_protocol_v1.md).

## 1. Cổng trước khi train (Phase E gate)

| Điều kiện | Trạng thái 16/11/2026 | Bằng chứng |
| --- | --- | --- |
| (i) benchmark đã kiểm, có phiên bản | **CHƯA ĐẠT**: G0 / C0 / v0 đều `draft`; nhãn B1 + B3 chờ người dùng duyệt | `configs/bench/benchmarks/*.yaml` |
| (ii) dev độc lập với test | **CHƯA ĐẠT**: dev_cs của E3 lấy từ ViMedCSS validation, mà ViMedCSS không tách theo video | `docs/dataset_registry.md` §3 |
| (iii) báo cáo rò rỉ sạch | **CHƯA ĐẠT**: C0 trùng 201 video với ViMedCSS train; G0 (VIVOS) trùng text 194 câu với VIVOS train | `leak_audit` |
| (iv) baseline chạy được | ĐẠT một phần: Gipformer G0 + C0 đã chạy; VieGem xem báo cáo tiến độ | `reports/general_asr_progress_vi.md` |
| (v) giao thức đánh giá không dùng test | ĐẠT (code): `evaluate --mode final`, inference mode, chặn oracle | `src/gipbench/bench/` |
| (vi) đủ compute / dữ liệu / license | **CHƯA ĐẠT** cho VieGem (RTX 3050 4 GB); Gipformer: chỉ chạy thử nhỏ | mục 4 |

Kết luận: **không chạy FT đầy đủ** cho tới khi (i)–(iii) đạt.

## 2. Chẩn đoán trước khi chọn cách FT

Chạy trên **dev**, không trên test khoá. Mục tiêu: biết lỗi chủ yếu đến từ đâu trước khi đổ thêm dữ liệu CS.

| Câu hỏi | Cách đo (code đã có) |
| --- | --- |
| Phát âm / âm học | lát `pronunciation_style` (khi có nhãn nghe), so CER theo span (`span_cer`) với CS-WER |
| Ngữ cảnh code-switch | CS-WER theo vị trí span, theo `category`; N-WER quanh span |
| Tokenizer / vocab đầu ra | `python -m gipbench.model.systems.tokenizer_check` (độ phủ BPE Gipformer với thuật ngữ catalog) |
| Giải mã / LM | M5: cùng checkpoint, đổi greedy ↔ beam ↔ LM / biasing **ngoài test** |
| Nhiễu nhãn | lấy mẫu 100 câu pseudo-label (vss_cs, distill), nghe và đo tỉ lệ lỗi span EN |

## 3. Ma trận (ngân sách bằng nhau: cùng số update hiệu dụng / giờ audio, cùng dev, cùng cách giải mã, ≥ 2 seed nếu kịp)

| ID | Gipformer (RNN-T, icefall) | VieGem (CTC, code upstream) | Giả thuyết | Metric chính (dev) | Kết quả |
| --- | --- | --- | --- | --- | --- |
| M0 | checkpoint gốc, mbs4, open_vocab | checkpoint gốc, greedy | mốc tái lập | WER G0 theo bộ, CS-WER C0 | xem báo cáo tiến độ (G0 / C0) |
| M1 | FT dữ liệu Việt phổ thông + replay VI, LR thấp | chỉ train head / reader (encoder đóng băng) | cải thiện tiếng Việt mà không quên | ΔWER G0-dev ≤ 0 (CI) | N/A (chưa chạy) |
| M2 | M1 + CS thật (ViMedCSS train **tách theo video**, dữ liệu tự thu) | M1 + cùng hỗn hợp | học chuyển ngữ từ CS thật | CS-WER C0-dev, không kém G0 | N/A (chưa chạy) |
| M3 | M2 + lấy mẫu cân theo thuật ngữ / cách đọc | M2 + cùng cách lấy mẫu | tăng recall từ Anh đọc giọng Việt, thuật ngữ hiếm | term_acc_macro, FT-unseen recall | N/A (chưa chạy) |
| M4 | M2 + TTS CS đã duyệt, tỉ lệ synthetic 0 / vừa / cao, cùng ngân sách | như trái nếu code train hỗ trợ | synthetic có giá trị hơn chỉ tăng lượng | CS-WER, G0 regression | N/A (chưa chạy) |
| M5 | cùng checkpoint, LM / biasing từ nguồn ngoài test | cùng checkpoint, KenLM đi kèm | tách lợi ích giải mã khỏi lợi ích train | Δ do giải mã | N/A (chưa chạy) |
| M6 (nghiên cứu) | LID phụ trợ / contrastive near-miss nếu có lý do | biến thể tương thích CTC | chỉ thử sau khi M1–M5 rõ | – | N/A (chưa chạy) |

Luật chọn (cố định trước): trên **dev**, chọn theo Pareto (WER G0, CS-WER, FT-unseen recall, RTF) với ràng buộc
không kém hơn: cận trên CI95 của ΔWER cho từng bộ G0-dev ≤ +0.5 điểm (giữ ngưỡng B5 cũ của leaderboard, là lựa chọn
của dự án, không phải chuẩn ngành). Test khoá chỉ chạy ở mốc đã khai báo, một lần mỗi mốc.

## 4. Tài nguyên (dựa trên số đã đo)

| Việc | Đo được | Ước tính / cần |
| --- | --- | --- |
| Gipformer FT trên RTX 3050 4 GB | A4: VRAM 3 912 / 4 096 MiB ở max-duration 150, 0.66 step/s, ~93 giờ audio / GPU-giờ | M1–M4 với ~160 h / epoch × 6 epoch: ~10 GPU-h mỗi epoch trên 3050; GPU 16–24 GB: chưa đo |
| VieGem suy luận | xem `docs/viegem_integration.md` (VRAM, RTF đo thật) | – |
| VieGem FT | **không chạy được trên 4 GB**: encoder EmbeddingGemma 2 + reader 359M, train top-k + activation 30 s | thuê GPU ≥ 24 GB (ước tính, chưa đo); bắt đầu M1 bằng `encoder_mode: frozen` |

## 5. Rủi ro đã biết từ các lần FT trước (A4, E3)

- A4 (194 step, ~1 h nhãn giả): CS-WER không đổi (Δ −0.44, CI chứa 0), quên tiếng Việt (N-WER +2.8, B5 +1.9).
- E3 v1 (LR 0.002, đóng băng `encoder_embed` + 2 stack đầu, max-duration 150, 6 × 160 h, bỏ CTC): CS-WER B1+B2 92.44
  (E0 71.66), B5 12.81 (E0 9.52). Cần kiểm trước khi chạy lại: LR (0.002 cao cho FT; thử ≤ 5e-4), tỉ lệ replay
  tiếng Việt có nhãn người, chất lượng nhãn giả (vss_cs, distill), lấy mẫu theo domain (medical 77.8% phần CS),
  checkpoint chọn trên dev_cs không tách video.
- E3 dùng `catalog_v0_draft.tsv` (có span lấy từ bench) để giới hạn tần suất thuật ngữ khi chọn dữ liệu train: đây là
  thông tin từ test lọt vào bước chọn dữ liệu. Lần sau dùng catalog không chứa span của bench (refactor plan §10 câu 2).
- `run_e3_v1b.sh` gọi `validate.py ... || true` (nay ở `docs/history/`): kiểm rò rỉ bị bỏ qua. Recipe mới phải dừng khi
  `leak_audit` có critical.

## 6. Lệnh (khi cổng đạt)

```bash
# 1. kiểm rò rỉ dữ liệu FT so với mọi benchmark (phải ra 0 critical)
python -m gipbench.dataset.release.leak_audit --test ${GIPBENCH_DATA}/benchmarks/general_v1_draft/manifest.jsonl \
    --train <pool...> --dev <dev...> --reserved-terms configs/.../reserved_terms.txt --fail-on critical
# 2. Gipformer: recipe theo refactor (configs/model/experiments/*.yaml, gipbench.model.run) của nhánh refactor/pipelines
# 3. chấm checkpoint trên dev bằng cùng scorer
python -m gipbench.bench.transcribe --model <asr_models/x.yaml> --manifest <dev.jsonl> --out results/dev/<x>
python -m gipbench.bench.evaluate --manifest <dev.jsonl> --hyps results/dev/<x>/hyps.jsonl --baseline-hyps results/dev/m0/hyps.jsonl
```
