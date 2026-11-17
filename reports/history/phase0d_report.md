# Phase 0d: chỉ lọc `vss_cs`, cache theo hash, nhóm `cs_pseudo`, danh sách từ, công cụ duyệt bench

**Trạng thái: XONG ở mục tiêu 4 h** (15/11/2026, 20:18). Theo `decisions.md` §12.1, mục tiêu đã đổi lại thành 6 h, và một job chạy tiếp với `--target-clean-h 6` đã tự khởi động lúc 20:18:32 (`~/chain6h.sh`, log `results/asr_hyps/vss_cs_run_s1_b8_6h.log`). Mọi con số dưới đây là ở mốc 4 h.

Mỗi task một commit. Sau mỗi task em đều chạy `tools/test_converters.py` và `tools/selftest.sh`, cả hai đều đạt (lần cuối sau commit `b91d850`).

| Task | Commit | Tóm tắt |
| --- | --- | --- |
| 1. Luật gán lại span | `da69a4a` | Cấm chữ có dấu tiếng Việt, không dài hơn span cũ quá 1 token |
| 2. Lọc rẻ: câu chỉ có thuật ngữ 1 lần | `4215e0f` | Bỏ, giữ ngẫu nhiên 10% (seed 0) |
| 3. WER tiếng Việt nhánh chỉ lọc | `1096e9f` | Lấy thấp nhất theo các ASR, ≤ 0.15 |
| 4. Cache + rebuild | `d6ce4b4` | Tra theo id + hash audio; `tools/rebuild_vss_cs.sh` |
| 5. `train_mix.tsv`, `cs_pseudo` | `dcca857` | `tools/mix_plan.py` kiểm trần domain 35% |
| 6. Chạy tiếp ASR (chỉ lọc) | `6b33540` (code) | `tools/vss_cs_run.py`, Qwen lô 8 xếp theo độ dài |
| 7. `vi_abbrev` / `vi_loanwords` | `486bb92` | `reports/top_review.csv`; P/R tập chính |
| 8. `tools/review_bench.html` | `203d34e` | Duyệt offline, xuất CSV cho `build_bench_v0.py` |

## Kết quả chính

- **`vss_cs` sạch: 4.15 h, 1 133 câu** → `data/raw/vss_full/vss_cs_clean.jsonl` (chỉ lọc, `--relabel-spans` tắt).
  - Pilot (hyp lô 1, không chạy lại): 226 câu, 0.82 h.
  - Seed 1: 902 câu, 3.31 h, giữ 902 / 3 600 câu đã chạy (25.1%).
  - 5 câu còn lại lấy từ lần thử 20 câu ở Phase 0b.
- **Rebuild chỉ từ cache khớp hoàn toàn với lần chạy.** Lệnh `bash tools/rebuild_vss_cs.sh --bench data/raw/_proxy/bench_proxy.jsonl --catalog data/raw/_proxy/catalog_proxy.tsv` chạy 16 s trên CPU, không dùng GPU, cho đúng 1 133 id như lần chạy.
- **Lô 8 không làm đổi quyết định nào.** Trên 50 câu so sánh, xếp theo độ dài: 13/50 câu có hyp khác lô 1, khoảng 2.0% token khác, **0 câu đổi quyết định giữ / bỏ**. RTF của Qwen giảm từ 0.24 xuống 0.075.
- **Không phải hạ lô xuống 4 lần nào.** RAM trống của WSL luôn ở mức 4.9–5.2 GB ở đầu mỗi chunk.
- **Cảnh báo trần domain:** medical (ViMedCSS) vẫn chiếm 77.8% phần CS, vượt trần 35%.

## 1. Luật gán lại span (T1)

Nhánh gán lại vẫn tắt mặc định. Hai luật mới:

- span mới không được chứa token có dấu tiếng Việt;
- span mới không được dài hơn span cũ quá 1 token.

Chạy lại từ cache trên pilot: số span được thay giảm **120 → 114**, và **13 span bị luật chặn**:

| id | span cũ | span ASR đề xuất | lý do |
| --- | --- | --- | --- |
| vss_0000888 | expe | expectation cái | có dấu tiếng Việt |
| vss_0008636 | website | yêu bếp | có dấu tiếng Việt |
| vss_0008636 | vnana | vân nana | có dấu tiếng Việt |
| vss_0016535 | teddy | tết đi | có dấu tiếng Việt |
| vss_0016535 | test | tết | có dấu tiếng Việt |
| vss_0027957 | scottle | scope của | có dấu tiếng Việt |
| vss_0032477 | st id | sti đi | có dấu tiếng Việt |
| vss_0036007 | rocket | rap việt | có dấu tiếng Việt |
| vss_0038468 | ag | ở agency | có dấu tiếng Việt |
| vss_0038962 | vietnam innovator | việt nam innovator | có dấu tiếng Việt |
| vss_0048732 | top | tốt | có dấu tiếng Việt |
| vss_0052830 | ipellog | yves saint laurent | dài hơn span cũ quá 1 token |
| vss_0054918 | what weis | we are what we eat | dài hơn span cũ quá 1 token |

Đa số là ASR nghe nhầm sang âm tiết Việt hoặc nuốt thêm từ Việt bên cạnh vào span, đúng loại lỗi luật này nhắm tới. Có 2 trường hợp luật chặn nhưng đề xuất có thể đúng: `vietnam innovator` và `yves saint laurent`. Vì nhánh gán lại đang tắt, chúng không ảnh hưởng tới `vss_cs`.

Báo cáo đầy đủ: `data/raw/vss_full/vss_cs_pilot_agreed_relabel_0d.jsonl.report.md`.

## 2. Lọc rẻ trước ASR (T2)

Tần suất thuật ngữ tính trên cả 31 949 câu ứng viên. Câu mà mọi thuật ngữ chỉ có trong 1 câu bị bỏ, trừ 10% giữ ngẫu nhiên (seed 0).

| Bước | Câu | Giờ |
| --- | --- | --- |
| `vss_cs_candidates` | 31 949 | 117.32 |
| độ dài 1–20 s | 31 949 | 117.32 |
| `--max-en-frac 0.5` | 20 228 | 74.41 |
| rò rỉ B1–B3 + reserved_b3 (bench tạm) | 20 228 | 74.41 |
| bỏ câu chỉ có thuật ngữ 1 lần (giữ 10%) | 17 249 | 63.48 |
| `--term-cap 50` | **12 351** | **45.49** |

Bước rò rỉ chưa bỏ câu nào vì đang dùng bench tạm (`bench_proxy`). Khi có `data/bench_v0.jsonl` thì chạy lại `tools/rebuild_vss_cs.sh --bench data/bench_v0.jsonl --catalog configs/catalog_v0.tsv` (runbook trong `docs/research/04_*`).

## 3. Ngưỡng WER tiếng Việt cho nhánh chỉ lọc (T3)

- WER phần tiếng Việt là giá trị thấp nhất theo các ASR có hyp, tính giống `relabel_spans`. Câu được giữ khi WER ≤ 0.15. Nhánh gán lại giữ ngưỡng ≤ 0.10.
- Trên toàn bộ 4 412 câu đã có hyp, có 1 374 câu qua lọc span. Trong số đó:
  - **241 câu (0.88 h) bị loại** vì WER tiếng Việt > 0.15;
  - **215 câu nằm trong khoảng 0.10–0.15**, tức là được giữ nhờ nới ngưỡng từ 0.10 lên 0.15.

| WER tiếng Việt ≤ | Câu |
| --- | --- |
| 0.05 | 508 |
| 0.10 | 410 |
| 0.15 | 215 |
| 0.20 | 117 |
| 0.30 | 87 |
| ∞ | 37 |

Riêng pilot (1 000 câu, chạy lại từ cache với luật mới): **242 / 1 000 câu (0.88 h)**. Có 279 câu qua lọc span, trong đó 37 câu bị loại vì WER > 0.15 và 44 câu nằm trong khoảng 0.10–0.15.

Pilot trong rebuild cuối chỉ còn 226 câu (0.82 h), vì sau commit `53c45f2` (consensus) bước rò rỉ chặn thêm `leak_block.txt`. Tập pilot trong pre giảm từ 794 xuống 783 câu.

## 4. Cache và rebuild (T4)

- Hyp được tra theo `(id, sha256 audio)`. sha256 lấy từ hash LFS của HuggingFace và khớp với sha256 của file tải về. Bảng tra nằm ở `results/asr_hyps/vss_audio_sha256.tsv`.
- Mỗi dòng hyp ghi `audio_sha256`, `audio_rel`, `engine`, `model`, `batch_size`.
- `tools/rebuild_vss_cs.sh` chạy cả chuỗi (prefilter → `agreement_filter --tiered`) chỉ từ cache và nhận `--bench`, `--catalog`. Kiểm tra lúc 20:19: 1 133 / 1 133 id giống hệt, 16 s trên CPU.
- **Thiếu so với yêu cầu "ghi batch_size vào mỗi dòng":** 800 dòng Qwen và 355 dòng faster-whisper của seed 1 chưa có trường này. Đó là các dòng chunk 0–1, chạy lô 1 trước khi thêm trường. Em chưa bổ sung `batch_size: 1` vì job 6 h đang ghi tiếp vào đúng các file này; làm sau khi job dừng nếu anh/chị muốn.

## 5. Trộn dữ liệu (T5)

`configs/train_mix.tsv`:

| Nhóm | Tỉ trọng |
| --- | --- |
| cs_real (ViMedCSS) | 0.35 |
| cs_pseudo (vss_cs, "audio thật, nhãn giả") | 0.10 |
| synth | 0.05 |
| replay có nhãn người | 0.25 |
| distill | 0.20 |
| confusable | 0.05 |

`tools/mix_plan.py` chạy lại với `vss_cs` 4.15 h. Phần CS (cs_real + cs_pseudo) dự kiến gồm **medical 77.8%, conversation 22.2%**, nên script **cảnh báo vượt trần 35%**.

Tỉ lệ này không phụ thuộc số giờ `vss_cs` vì tỉ trọng nhóm cố định (0.35 : 0.10). Muốn medical ≤ 35% thì phải đổi tỉ trọng hoặc thêm nguồn CS thật ngoài y khoa. `decisions.md` §12.2 ghi sẽ làm 2 biến thể E3, trần 35% và 50%.

`vss_cs` đang mang domain cố định `conversation`. Việc suy domain từ text (§10 / §12.6) chưa làm.

## 6. Chạy ASR chỉ lọc (T6)

| | Giá trị |
| --- | --- |
| Mẫu seed 1 | 6 000 câu từ phần chưa chạy (`data/raw/vss_full/vss_cs_run_s1.jsonl`), chunk 400, có resume |
| Dừng | sau chunk 9 (3 600 câu): 4.15 h ≥ mục tiêu 4 h |
| Tầng 1 | Qwen3-ASR-0.6B, lô 8 xếp theo độ dài (chunk 0–1: lô 1, chạy trước khi đổi) |
| Tầng 2 | faster-whisper large-v3 int8, beam 5, lô 1, chỉ cho câu chưa xác nhận (~320 / 400 câu mỗi chunk) |
| Thời gian | phần lô 8: 3.46 h thực, 4.03 GPU-h; mỗi chunk khoảng 26–28 phút, thêm 0.32–0.44 h sạch |
| D: | 32.6–36.3 GB trống ở đầu các chunk (ngưỡng dừng 3 GB) |
| RAM WSL | 4.88–5.16 GB trống (ngưỡng hạ lô 1 GB), không phải hạ lô |
| Tiến độ | `data/raw/vss_full/vss_cs_run_s1.progress.jsonl`, `vss_cs_clean.jsonl.run.md` |

### So sánh lô 8 với lô 1 (50 câu pilot đã có cache)

| Lần đo | Câu có hyp khác | Token khác | Đổi quyết định | RTF Qwen |
| --- | --- | --- | --- | --- |
| lô 1 chạy lại | 0 / 50 | 0 | – | 0.24 |
| lô 8, không xếp | 12 / 50 | ~1.5% | 0 | – |
| **lô 8, xếp theo độ dài** | **13 / 50** | **~2.0%** | **0** (xác nhận span, giữ tầng 1, giữ cuối) | **0.075** |

Mức khác > 1% số câu đã được báo trước khi chạy. Anh/chị chọn lô 8 với điều kiện không câu nào đổi quyết định, và điều kiện đó đạt.

### Sự cố trong lúc chạy

- D: từng tụt còn 0.38 GB do pagefile Windows phình (21.5 GB; tiến trình `dllhost` 6.6 GB). Em dừng toàn bộ và báo; anh/chị dọn D:, sau đó còn khoảng 33 GB trống.
- `memory=6GB` đã có trong `.wslconfig`. Sau khi khởi động lại WSL, MemTotal là 6.07 GB.

## 7. Danh sách từ (T7)

- `configs/vi_abbrev_project.tsv` thêm `vn, ubnd, bhxh, bhyt, cccd, cmnd, thpt, thcs, csgt, tphcm, hcm` với nguồn "quy ước dự án, đọc kiểu Việt". Bật lại `tp`, bỏ `cdn`, mã chứng khoán chỉ lấy HOSE + HNX.
- Luật tự động: mục `vi_abbrev` trùng thuật ngữ EN của catalog thì bị loại, và `build_wordlists.py` in ra danh sách bị loại.
- `reports/top_review.csv`: top 30 mỗi file, mỗi dòng kèm 1 câu ví dụ.
- P/R tập chính (161 câu G-Group, `reports/t0d7_eval_main.md`), trên 100 câu có từ Anh: P 0.977 (0.982 nếu bỏ qua viết tắt VN), R 0.977, F1 0.977. Trên mọi câu: P 0.960, R 0.977.
- Con số trên đo với danh sách ở commit `486bb92`. Sau đó phiên khác đã sửa `vi_abbrev` / `vi_loanwords` (`9e98d0e`, `d79914f`: thêm sjc / tv / vtv…, loại mã chứng khoán hay gặp trong text thường, `same_surface`), nên cần đo lại khi khoá v1.
- Tập kiểm tra 60 câu: **chưa chấm**, đúng dặn, chờ khoá v1. Phiếu hỏi hiện gộp vào `reports/term_decisions_ask.csv` (§12.4).

## 8. `tools/review_bench.html` (T8)

- Mở offline bằng trình duyệt, không cần server.
- Đọc phiếu `*_review.csv`, phát audio từ thư mục máy hoặc link HF; span EN in đậm.
- Phím: 1 = ok, 2 = drop, 3 = sửa (mở ô sửa `text_tagged`). Xong một câu thì tự sang câu chưa duyệt kế tiếp và tự phát audio.
- Tự lưu vào localStorage và hiện số giây trung bình mỗi câu.
- Xuất CSV với các cột `decision, text_tagged_fixed, viet_levels, region, gender`.
- Đã test: phần lõi JS (parse / xuất CSV, kiểm tra tag) chạy bằng Node trong `test_converters.py`, và `build_bench_v0.py` đọc được CSV xuất ra.
- **Chưa test:** thao tác thật trong trình duyệt (phím, autoplay, chọn thư mục audio). Mục tiêu ≤ 10 s / câu cũng chưa đo với người thật.

## Việc còn chờ anh/chị

1. Khoá `vi_abbrev` / `vi_loanwords` v1 qua `reports/term_decisions_ask.csv`; sau đó em chấm tập kiểm tra 60 câu.
2. Trần domain: medical 77.8% phần CS; xem §12.2 (2 biến thể E3).
3. Có muốn bổ sung `batch_size: 1` cho 1 155 dòng hyp cũ sau khi job 6 h dừng không.
