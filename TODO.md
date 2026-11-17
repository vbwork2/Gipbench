# TODO: đường tới một lần fine-tune đáng tin

Cập nhật: 17/11/2026 · Nguồn: [audit 17/11](reports/audits/full_research_audit_vi.md) (mã rủi ro `Rxx`) và [implementation_plan.md](reports/audits/implementation_plan.md) (mã `IP-xx`, có đặc tả sửa + test chi tiết).

Cách dùng: làm **từ trên xuống**. Thứ tự các phần là thứ tự phụ thuộc. Tick `[x]` khi đạt điều kiện "Xong khi". Mục gắn **👤** là việc chỉ bạn quyết định được.

---

## Đang ở đâu (tóm tắt 1 phút)

- **Đã có:** 4 bench khoá (G0 `general_v1`, C0 `code_switch_v1`, C1 `cs_ggroup_v1`, E0 `external_v1`), hash tái lập 4/4. Scorer chạy độc lập từ gói share v1.1.2 ra đúng số. Baseline Gipformer 1.5 và VieGem greedy đã chấm trên G0/C0/C1.
- **2 lần FT trước (A4, E3 v1) đều làm hại.** Nguyên nhân có bằng chứng mạnh nhất: **dev hỏng** (dev_cs báo CS-WER −23 điểm trong khi bench +20.8 điểm). 91% câu ViMedCSS trong dev chung video với train; dev_vi là giọng đọc VIVOS.
- ~~Cổng chống rò rỉ không chặn được gì~~ → **đã đóng khi lỗi (2a, 17/11)**: `model.run` stage 0 dừng (exit 3) khi trùng với mọi bench khoá; bắt được 1 514 câu E0 ViMedCSS mà bản cũ báo 0.
- ~~Từ vựng bench lọt vào chọn dữ liệu / checkpoint / TTS~~ → **catalog `from_bench` bị từ chối ở mọi bước (2b)**; còn chờ 👤 chốt `reserved_terms_v1` để dựng `catalog/train-v1`.
- **Bước 0, 1a, 1b, 2a, 2b (công cụ) xong (17/11)** trên nhánh `fix/scorer-leak-audit` (chưa commit): test xanh 19/19, golden 100/100 không đổi, số cũ không đổi. Tiếp theo: 👤 `reserved_terms_v1`, rồi 2c (pool không chứa video bench) và Phần 3 (dev_v2).

- **Mục tiêu "general" (đăng ký trước):** CS-WER C1 giảm (CI95 < 0) ở ≥ 2 domain ngoài y khoa **và** `vi_only_wer` G0 không tăng (cận trên CI ≤ +0.5); xác nhận cùng chiều trên Z0 khi có. C0 / E0-ViMedCSS không dùng để claim general. Các hướng cải tiến ngoài X0–X6: Phần 7.

**Kết luận:** chưa được FT tiếp. Còn thiếu (1) dev độc lập (Phần 3), (2) pool train không chứa video bench (2c; cổng đang chặn pool hiện tại), (3) `reserved_terms_v1` được chốt. Sau đó mới tới thí nghiệm X0.

---

## Bước 0 · Dọn để test xanh (≤ 1 giờ)

- [x] Sửa vi phạm tầng ở `package.py`: danh sách module chuyển ra [share_code.yaml](configs/dataset/share_code.yaml), `package.py` đọc lúc gọi (`share_code()`). Không thêm ngoại lệ vào `test_boundaries`. *(17/11: 16/16 test ĐẠT; giá trị và bao đóng code 29 module khớp bản cũ)*
- [x] Chụp mốc golden trước khi sửa scorer. *(17/11: `results/golden/base_8ad0f9b`, chụp 2 lần ra 100/100 file giống nhau)*
  - Mỗi PR sau: `python tests/golden/capture.py --layout new --out results/golden/run && mv results/golden/run results/golden/<tên>`, rồi `python tests/golden/compare.py results/golden/base_8ad0f9b results/golden/<tên>`. Phải ghi vào đúng `results/golden/run` rồi mới đổi tên, vì vài file chứa đường dẫn tương đối theo thư mục output.

---

## Phần 1 · CHECK BENCHMARK & CHẤM ĐIỂM

Mục đích: số chấm ra phải đúng và không đảo dấu mà không ai biết. Không cần GPU.

### 1a. Scorer (IP-03, IP-04 · R07 R08 R13) ✅ 17/11, nhánh `fix/scorer-leak-audit`, scorer 1.1.0
- [x] `evaluate`: chạy `check_hyps` cho `--baseline-hyps`. Final mode thoát mã 2 khi baseline thiếu/thừa id.
  - *Đã kiểm:* baseline C1 cắt còn 630/1 260 dòng → thoát 2 ("thiếu 630").
- [x] `summary.json` ghi thêm `label_field`, `overrides`, `scorer_code_sha256`, `profile.code_sha256`, `baseline_hyps_sha256`, `baseline_hyp_check`, `metric_definitions`. Final mode từ chối `--label-field` không phải nhãn chính / `alt_label_fields`.
- [x] Metric thêm (không đổi metric cũ): `cs_errors` (`S_en / D_en / I_en_left / I_en_right_only / boundary_ins`), `cs_wer_sd`, `boundary_ins_rate`, `vi_only_wer`, `en_hallucination_per100`.
  - *Đã kiểm:* C1 `cs_wer_sd` Gipformer 56.37, VieGem 71.56 (khớp audit); 6 kết quả v1 + cặp C1: 3 013 giá trị cũ (gồm CI, Δ cặp) khớp 100%; golden 100/100 file giống.
  - "Câu thuần Việt" = không span `<en>` **và** ref không có token giống từ Anh (G0 G-Group có từ Anh không tag). G0: `vi_only_wer` Gipformer 9.79 / VieGem 15.09.
- [x] Chốt quy ước: số chính = `cs_wer` theo `text_tagged`, luôn báo kèm `cs_wer_sd`. Ghi ở [evaluation_metrics_v1.md](docs/evaluation_metrics_v1.md) mục 2b / 2c.

### 1b. Bộ kiểm rò rỉ (IP-01 · R03) ✅ 17/11, cùng nhánh
- [x] `core/decontam.py`: `source_key()` (YouTube ID từ `yt_` / `youtube:` / URL, CV, VIVOS; không bao giờ dùng id câu) → `source_norm_overlap` (critical). Thêm `clip_key()` → `clip_overlap` (cùng clip CV / VIVOS dù đã resample).
- [x] Thiếu hash audio → `audio_unverifiable` (critical khi require) và in số câu vào báo cáo. Thêm `identity_unknown` (review, % câu test không xác định được nguồn).
- [x] Trùng text trong cùng bộ đọc (VIVOS) → `lexical_overlap_read_corpus` (review), không còn critical. Hotwords chứa cụm có trong text test → `test_ngram_in_runtime_vocab` (review).
- [x] `leak_audit.py`: `--require-audio-sha` (mặc định bật khi `--fail-on critical`), `--no-require-audio-sha`, `--exemptions` (bắt buộc reason / approved_by / date), đọc được trường `audio` làm đường dẫn.
- [x] Test D.3 ca 1–5 + ca 7 (+ miễn trừ, bộ đọc, CLI) viết trước, HỎNG trên code cũ, giờ ĐẠT. **Ca 6** (gate trong `model.run`) chuyển sang 2a.
  - *Đã kiểm trên dữ liệu thật* (`--no-audio-sha`): E0 vimedcss_test × ViMedCSS train **1 514 câu / 898 video**, thoát 1 (code cũ báo 0). C0 × train 388 / 248; C0 × dev_cs 71 / 48; G0 VIVOS × VIVOS train 194 câu → review, thoát 0; G0 CV27 × E0 CV17 **1 024 clip** + 143 người nói.

### 1c. Diễn giải đúng các bench hiện có (R04 R12 R18 R20) · 17/11, gần xong
- [x] Ghi rõ **C0 và E0-ViMedCSS không đo tổng quát** trong [datasheet](docs/datasheet_bench_v1.md) mục 7.2 / 8b / 9, `dataset_registry.md`, README gói share.
- [x] Bench phiên bản mới (IP-07, **không sửa bản khoá**), dựng bằng `python -m gipbench.dataset.release.derive_v1_1`:
  - [x] `code_switch_v1_1` (draft): `train_video_overlap`, nhóm báo cáo theo trường này, `alt_label_fields: [text_tagged_orig]`. 388 / 248 trùng, **270 câu / 156 video chưa thấy**.
    - *Kết quả lát video chưa thấy:* WER VieGem − Gipformer **−0.03 [−1.26, +1.27]** (trên video đã thấy: −1.46); term recall VieGem 20.75 → **10.09**. Lợi thế WER / thuật ngữ của VieGem trên C0 là in-domain.
  - [x] `external_v1_1` (draft): `audio` + `audio_sha256` + `duration` cho 30 467 câu (5 951 sha cũ khớp lại), speaker ViMedCSS test `yt_<ID>` + `train_video_overlap` (1 514), `boot_unit: per_dataset` (FLEURS theo `sentence_id`), `require_audio_sha`.
  - [ ] **Chờ ViMD tải xong** (1 736 câu, job evalsets PID 2354, ~5 h từ 15:00 17/11) → chạy lại `derive_v1_1 --which external` → điền `manifest_sha256`.
  - [ ] 👤 Duyệt khoá `code_switch_v1_1` / `external_v1_1` (đổi `evidence_level: locked`).
- [x] Đính chính 308/201 → 388/248, 37 261 → 36 237 (datasheet mục 9, `dataset_registry.md`, `experiment_matrix.md`, `share_readme.md`).
- Code thêm: `evaluate` có Δ so cặp theo mọi khoá `group_by`, `boot_unit: per_dataset`; `registry.check_frozen` từ chối khoá bench `require_audio_sha` còn thiếu sha.

### 1d. Hệ tham chiếu cho so sánh công bằng (IP-10 · R17) · đang chạy nền từ 17/11 15:24
- [ ] 👤 **VieGem + KenLM bị chặn**: cần `sudo apt install build-essential cmake` trong WSL (cần mật khẩu), rồi `uv pip install --python ~/.venvs/gipbench-viegem/bin/python kenlm`. `pypi-kenlm` không có wheel cho Python 3.11. File LM `models/viegem/lm/viegem_lm.bin` đã có. Sau đó chạy `viegem_lm` trên G0 / C0 / C1.
- [ ] Whisper-large-v3 và Qwen3-ASR trên G0 / C1: đang chạy (hàng đợi `results/general_asr/logs_1d/queue.log`). Whisper cần `LD_LIBRARY_PATH` tới cuBLAS / cuDNN trong venv (RUNBOOK 3b).
- [ ] E0 v1.1 (phần đã có audio, 30 467 câu) cho VieGem rồi Gipformer: trong cùng hàng đợi, sau Whisper / Qwen. Chạy lại với manifest đủ khi ViMD xong (resume theo id).
- [ ] Khi hàng đợi xong: điền bảng kết quả vào datasheet mục 8 (hệ thứ ba + E0).
- Lưu ý: WSL RAM 6 GB, không chạy Gipformer và VieGem song song. Gipformer giữ batch 1 (batch làm đổi hyp).

---

## Phần 2 · CHECK DATA

Mục đích: không có test trong train, không có từ vựng bench trong khâu chọn, và replay đủ đa dạng.

### 2a. Cổng rò rỉ trong train (IP-02 · R02) ✅ 17/11, nhánh `fix/scorer-leak-audit`
- [x] `core/release.py`: `locked_bench_refs()` (đọc `releases.yaml`, mọi `bench/*` có `flags.locked`). Đã `register` 4 bench v1.0 + `catalog/vimedcss-v1.0` vào `data/releases/` (hash khớp bản khoá).
- [x] Cổng ở `core/leakgate.py` (model không được import dataset), gọi từ `model.run` stage 0: kiểm `leak_guard ∪ bench khoá` với mọi pool / dev / `leak_gate.text_files`. Critical chưa miễn trừ thì **thoát mã 3 trước curate / train**; bench khoá chưa register cũng dừng. Stage 6 từ chối tạo artifact nếu thiếu báo cáo cổng hoặc còn blocking; artifact kèm `leak_report.{json,md}` + `leak_report_sha256` + `leak_gate` trong `model_card.json`.
  - *Đã kiểm trên dữ liệu thật* (`e3_v1.yaml`, bỏ kiểm catalog bằng `catalog_use=oracle_diagnostic` để xem riêng cổng; 37 s, 280 MB): **thoát 3, 11 critical**. C0 ← ViMedCSS train **388 câu / 248 video** (+ dev_cs 71); E0 vimedcss_test ← ViMedCSS train **1 514 câu / 898 video** (+ dev_cs 518 — code cũ báo 0); G0 ← pool confusable 4 + VIVOS/CV text; 0 critical với C1 (G-Group không có id người nói: `identity_unknown` 100%).
  - `e3_v1.yaml` nguyên bản: thoát 3 ngay vì catalog `v0-draft` (xem 2b).
- [x] Test D.3 ca 6: `tests/model/test_gate_run.py` (release giả: bench khoá ngoài `leak_guard` vẫn được kiểm; bench khoá chưa register → "CHƯA KIỂM"; miễn trừ từng loại; `mode: report`; stage 6 thiếu báo cáo).
- ⚠️ **Mặc định `audio_sha: metadata`** (không đọc file audio): pool train không có `audio_sha256`, nên trùng audio **không được kiểm** ở cổng mặc định (báo cáo ghi rõ; hash cũng không bắt được audio resample). Trùng nguồn / clip / text mới là kiểm chính. Bật `leak_gate.audio_sha: hash` (đọc file, chậm hơn) và `require_audio_sha: true` khi cần chặt.
- ⚠️ **Hệ quả:** mọi experiment dùng `pool/vimedcss_train@full` sẽ bị cổng chặn (trùng video với C0 / E0). Cần pool đã loại video bench (2c), **không** nên miễn trừ.
- Sửa lỗi có sẵn: stage 0 của `model.run` vỡ khi chạy thật (`Catalog(str(path))` thay vì `Catalog.load`; refactor 16/11 chỉ kiểm bằng `--dry-run`).

### 2b. Chặn từ vựng bench (IP-05 · R05 R23) · công cụ xong, chờ 👤
- [ ] 👤 **Chốt `configs/dataset/reserved_terms_v1.txt`**: danh sách thuật ngữ Anh giữ lại làm FT-unseen, **trước** khi dựng bất kỳ dữ liệu nào. Đã có **đề xuất** [reserved_terms_v1.proposal.txt](configs/dataset/reserved_terms_v1.proposal.txt) (820 thuật ngữ; `python -m gipbench.dataset.release.propose_reserved_terms --fraction 0.5`):
  - 2 476 thuật ngữ <en> của bench; 843 đã có trong text train (không thể giữ riêng nếu không loại câu train) → 1 633 ứng viên → giữ riêng 50% mỗi domain (edu 150, finance 143, medical 217, pubadmin 39, tech 271).
  - Bạn quyết: `--fraction` (nhiều = lát FT-unseen to và chắc, nhưng cấm dạy model chính các thuật ngữ tech / finance mục tiêu), và sửa tay danh sách rồi đổi tên thành `reserved_terms_v1.txt`.
- [x] `core/release.py`: `catalog_flags()`, `forbid_bench_catalog()`. `model.run` (mọi stage, cả `--dry-run`), `curate`, `select_ckpt`, `make_text` từ chối catalog `from_bench` (exit 3 / 1); chẩn đoán oracle phải khai báo `catalog_use: oracle_diagnostic` / `--allow-bench-catalog` (artifact + system YAML gắn `oracle_bias`, `rankable: false`). `e3_v1` / `e3_v1b` / `e3_v2` bị từ chối (giữ nguyên làm lịch sử).
- [x] `select_ckpt`: `--rule term_acc_macro|cs_wer` (đăng ký trước, `select.rule`), ghi `rule` / `dev_refs` / `dev_flags` / `catalog_flags` vào select.json khi chạy từ `model.run`; từ chối dev không có `flags.independent: true` (trừ `--allow-legacy-dev`).
- [x] `make_text`: `--reserved-terms` bắt buộc (không rỗng), thuật ngữ giữ riêng không bao giờ vào kịch bản (kiểm lại ở cuối), ghi `tts_text.txt` cho `leak_gate.text_files`.
- [x] `dataset/release/build_train_catalog.py`: catalog = thuật ngữ <en> của text train − span mọi bench khoá − `reserved_terms`. **Chưa dựng bản thật** (cần reserved_terms chốt; `reserved_terms` rỗng bị từ chối).
  - *Xem trước với đề xuất* (ViMedCSS train `text_tagged_orig` + vss_cs): 1 479 thuật ngữ train, **716 (48%) cũng là thuật ngữ bench** nên bị loại → còn **763**. Bench ↔ train: C0 79 / 582 thuật ngữ, C1 191 / 1 443, E0 (ViMedCSS test) 503 / 516.
  - Khi 👤 xong: chạy lệnh ở RUNBOOK mục 1, bỏ comment mục `catalog/train-v1` trong `releases.yaml`, `register`.
  - Xong khi: `leak_audit --reserved-terms` = 0 trên mọi pool (đã có test; chạy thật khi có file chốt).

### 2c. Dọn dữ liệu train (IP-06 · R14 R15 R16)
- [ ] **Pool ViMedCSS không chứa video có trong bench khoá** (`pool/vimedcss_train@v2`, loại theo `core.decontam.source_key`): cổng 2a chặn pool hiện tại (388 câu C0, 1 514 câu E0). Đây là điều kiện để X3 chạy được.
- [ ] Dedupe theo audio trong `curate.py` (590/590 câu confusable trùng audio với `vivos_train`).
- [ ] `mix_cuts.py --max-views 8`: trần số lượt lặp mỗi audio gốc (hiện cs_real ~55 lượt / audio gốc).
- [ ] Thay replay CV27 train (chỉ **1 người nói**) bằng `pool/cv_validated@v2` nhiều người nói, loại client_id có trong test/dev CV27 và CV17 test. Đo số người nói; nếu quá ít thì bỏ CV khỏi replay.
- [ ] Release mới `pool/confusable_train@e3_v1b` với `label_source` đúng. Thêm `errata:`, không sửa mục cũ.

### 2d. Chất lượng nhãn giả (IP-10 · R11 R24)
- [ ] Viết `dataset/pseudolabel/listen_sample.py`: lấy mẫu phân tầng 200–300 câu vss_cs, xuất CSV cho trang duyệt `term_review_page.py`.
- [ ] 👤 Nghe và chấm mẫu đó.
  - Xong khi: có tỉ lệ lỗi span kèm CI. **Bỏ vss_cs nếu lỗi span > 10%.**

### 2e. Độ phủ domain CS (R10)
- [ ] Thống kê CS train theo domain (hiện: medical 1 567, tech 54, edu 29, pubadmin 4, finance 0).
- [ ] 👤 Tìm nguồn **CS tự nhiên ngoài y khoa, có id người nói** (tech / edu / business / finance). Đây cũng là nguồn cho dev_v2 ở Phần 3.

---

## Phần 3 · DEV ĐỘC LẬP (IP-07 · R01 R14) · cần 1b + 2a

**Đây là rào cản số 1.** Không có nó thì không FT nào chứng minh được là tốt lên.

- [ ] `make_dev_split.py --exclude-from <mọi pool train + mọi bench khoá>`, loại theo `source_key`.
- [ ] `dev_v2_cs`: ViMedCSS validation sau khi loại video train / C0 / E0, cộng CS tự nhiên ngoài y khoa (từ 2e).
- [ ] `dev_v2_vi`: tiếng Việt **nói tự nhiên** có id người nói (không dùng VIVOS).
- [ ] Release ghi `flags: {independent: true, exclusion_sets: [...]}`.
- Xong khi:
  - [ ] gate báo 0 critical giữa dev_v2 và mọi pool / bench;
  - [ ] ≥ 1 500 câu CS, ≥ 4 domain, ≥ 100 người nói; dev VI ≥ 1 h tự nhiên;
  - [ ] **kiểm đại diện:** chấm checkpoint `ck4132` (E3 v1) trên dev_v2 thì phải thấy **xấu đi** như trên bench. Nếu dev_v2 vẫn báo "tốt lên" thì dev_v2 chưa dùng được.
- ⛔ **NO-GO:** nếu 2e không tìm được đủ CS tự nhiên ngoài y khoa, dừng ở đây và chuyển sang thu dữ liệu.

---

## Phần 4 · CHECK PHƯƠNG PHÁP (chẩn đoán rẻ, không cần train)

Làm song song với Phần 2–3 được.

- [ ] `tokenizer_check` BPE Gipformer trên span C1: số token / thuật ngữ, % thuật ngữ ≥ 4 token. (Trả lời: có cần đụng tokenizer / backbone không. Audit: hiện **chưa có bằng chứng** cần đổi backbone.)
- [ ] Decode các checkpoint ứng viên E3 (`results/e3_v1_dev/*`) trên 787 câu G-Group thuần Việt + C1. Xác nhận dev cũ đảo thứ hạng (chỉ chẩn đoán, **không** dùng để chọn).
- [ ] So ma trận giải mã greedy / beam / LM cho cả hai model (gộp với 1d). Mục đích: tách phần lợi do giải mã khỏi phần lợi do train.
- [ ] Ghi lại luật chọn checkpoint **trước** khi chạy (`select.rule`). Lần trước luật bị đổi sau khi chạy (R21).
- [ ] Kiểm CTC loss ở fp32 / bf16 trên vài trăm step: còn phân kỳ không? Quyết định có giữ head CTC làm bộ điều chuẩn (Phần 7, L4) hay tiếp tục `--drop-ctc`.
- [ ] Lát lỗi theo môi trường thu / kênh trên G0 (theo bộ: đọc / tự nhiên / điện thoại). Quyết định có bật augment kênh / nhiễu (Phần 7, D5; `model/run.py` đang `--enable-musan False`).

---

## Phần 5 · FINE-TUNE CÓ KIỂM SOÁT (IP-09 · F.4) · cần Phần 2 + 3

Luật cho **mọi** lần chạy:
- Mỗi thí nghiệm chỉ đổi 1 yếu tố, cùng ngân sách update, **2 seed**.
- Gate rò rỉ = 0 critical; `reserved_terms` = 0.
- Giữ tiếng Việt: cận trên CI95 ΔWER `dev_v2_vi` ≤ +0.5, `en_hallucination` không tăng.
- Không domain nào trên `dev_v2_cs` xấu đi quá ngưỡng đăng ký trước trong `select.rule` (chống negative transfer, audit mục 7 #11: pubadmin +10.23 ở E3).
- Test khoá chỉ chấm ở mốc đã khai báo trước, **một lần**.

- [ ] Viết `configs/model/experiments/e4_x0.yaml` … `e4_x6.yaml` (`catalog/train-v1`, `leak_gate.mode: fail`, `dev_v2`, `seeds: [0,1]`, `mix.max_views: 8`).
  - Xong khi: `--stages 0-6 --dry-run` in lệnh; stage 0 thoát 0.
- [ ] **X0** · FT chỉ replay tiếng Việt nhãn người, LR 3e-4, đóng băng embed + 2 stack. Câu hỏi: pipeline có tự gây hại không?
  - Đạt nếu ΔWER dev VI ≤ +0.3. ⛔ Không đạt thì lỗi nằm ở recipe (LR / scheduler / drop_ctc): **dừng mọi thí nghiệm CS**, sửa recipe trước.
- [ ] **X1** · sweep LR 1e-4 / 3e-4 / 1e-3. Chọn LR lớn nhất còn đạt X0.
- [ ] **X2** · chỉ train decoder + joiner (đóng băng encoder). Nếu đạt ≥ 50% lợi ích của X3 thì vấn đề chính là thiên lệch ngôn ngữ, không phải âm học.
- [ ] **X3** · X1 + CS thật đã lọc video, trần lặp ≤ 8. Đạt nếu CI95 Δ CS-WER < 0 ở ≥ 2 domain ngoài y khoa.
- [ ] **X4** · X3 + vss_cs (chỉ khi 2d đạt). Bỏ vss_cs nếu không tốt hơn X3.
- [ ] **X5** · X3 + TTS 10 / 30% (lexicon từ `catalog/train-v1`, không từ bench). Đo `term_recall` trên thuật ngữ `ft_unseen`.
- [ ] **X6** · checkpoint tốt nhất × greedy / beam / LM.
- [ ] Mỗi lần chạy: model_card có leak report, số lượt lặp, seed, sampler state.

---

## Phần 6 · SHARE BENCH (không chặn FT, làm khi rảnh)

- [ ] 👤 Xác định gói v1.0.1 / v1.1 / v1.1.1 đã gửi cho ai. Gửi bản thay thế (các gói này dặn `git clone` repo riêng tư).
- [ ] 👤 **Không** cấp quyền repo `vbwork2/Gipbench` cho người ngoài.
- [ ] `share_policy.yaml` dạng allowlist + regex cấm; `package.py` quét gói, có vi phạm thì dừng; bỏ `consensus_private` khỏi `datasets.yaml` của gói (IP-08).
- [ ] Sửa docstring `infer/decode.py` và comment `asr_models/gip15_mbs4_open.yaml` (đang nhắc E1, `results/v0_draft`).
- [ ] 👤 Rà license (bảng G.4 của audit): ViMD (ND), LSVSC / CV17 (bản đăng lại), Bud500 / Speech-MASSIVE test (gated). Chưa chắc thì không kèm text.
- [ ] Chạy thử `fetch` trên một máy sạch khác.

---

## Phần 7 · Hướng cải tiến ngoài X0–X6 (không phải biến thể)

Mỗi hướng tác động vào một chỗ **khác cơ chế** của hệ. Biến thể (LR, epoch, trọng số mix, độ sâu freeze, alpha WiSE) không nằm ở đây, chúng là sweep bên trong một hướng (X1). Mỗi hướng chỉ chạy sau X0 đạt, theo đúng luật Phần 5, so với checkpoint X3 cùng ngân sách update. Nguồn: audit F.2 / E.3–E.5 / mục 7, [research/01](docs/research/01_ke_hoach_research_pipeline.md) (bảng paper), [research/02](docs/research/02_spec_du_lieu_metric_do_tin_cay.md).

| Mã | Nhóm | Hướng | Cơ chế (1 câu) | Mở khoá khi | Repo 17/11 |
| --- | --- | --- | --- | --- | --- |
| D1 | Dữ liệu | CS tự nhiên ngoài y khoa có id người nói | đúng phân phối mục tiêu; lệch domain là nguyên nhân mạnh nhất khiến E3 hỏng | 👤 2e | chưa có nguồn |
| D2 | Dữ liệu | Nhãn giả có kiểm soát (teacher khác họ, vote nhiều model, lọc ngữ âm) | tăng CS thật rẻ, sửa lệch "chỉ giữ từ Anh dễ" | 2d đạt | `pseudolabel/` có, chưa nghe mẫu |
| D3 | Dữ liệu | TTS CS (≥ 20 giọng, ≥ 3 cách đọc / thuật ngữ) | phủ thuật ngữ hiếm / `ft_unseen` | `catalog/train-v1`; = X5 | `dataset/synth/` có, chưa có audio |
| D4 | Dữ liệu | Ghép cụm audio thuật ngữ vào câu Việt thật | rẻ hơn TTS cả câu, giữ âm học Việt thật | so trực tiếp với D3 | chưa có |
| D5 | Dữ liệu | Augment kênh / nhiễu (MUSAN, codec 8 kHz, RIR) | general = môi trường thu đa dạng | lát kênh G0 (Phần 4) | `--enable-musan False` |
| D6 | Dữ liệu | SSL / continual pretrain trên audio CS chưa nhãn | học âm "Anh giọng Việt" không cần nhãn | có ≫ 100 h audio CS | chưa có, chi phí cao |
| L1 | Loss | KL / distillation tới model gốc trên audio Việt (LwF) | giữ phân phối đầu ra tiếng Việt, mạnh hơn chỉ replay nhãn | X3 vẫn quên | chưa có (pool `distill` là nhãn, khác bản chất) |
| L2 | Loss | Kéo trọng số về gốc (L2-SP / EWC) | phạt lệch tham số quan trọng cho tiếng Việt | thay / so với L1 | chưa có |
| L3 | Loss | WiSE-FT (nội suy trọng số) | hậu kỳ, không train thêm | mọi lần chạy | ✅ stage 6 |
| L4 | Loss | Giữ head CTC phụ (fp32 / bf16) | điều chuẩn alignment, E3 phải bỏ vì phân kỳ fp16 | kiểm CTC (Phần 4) | `wrap_ckpt --drop-ctc` |
| L5 | Loss | LID phụ trợ theo frame / token | biết ranh giới ngôn ngữ, giảm `boundary_ins` / `en_hallucination` | `boundary_ins_rate` là lỗi chính | chưa có |
| L6 | Loss | Keyword-aware / contrastive near-miss | tăng trọng số lỗi span, phạt nhầm sang âm Việt gần | dùng kèm D3 / D4 | chưa có |
| P1 | Tham số | Chỉ prediction network + joiner | sửa thiên lệch ngôn ngữ, không động âm học; cũng là chẩn đoán | = X2 | freeze theo prefix có sẵn |
| P2 | Tham số | Adapter / LoRA trong Zipformer | model gốc nguyên vẹn, tắt được nên gần như không quên | X3 quên mà L1 không đủ | chưa có cho Zipformer |
| P3 | Tham số | Nhánh / expert theo ngôn ngữ (bi-encoder, MoE gating LID) | tách biểu diễn Việt / Anh | có bằng chứng kiến trúc là nút cổ chai | không khuyến nghị (audit F.2-G) |
| T1 | Text-only | Pretrain predictor trên text CS rồi FT speech (kiểu AsyncSwitch) | predictor RNN-T là LM nội, học chuyển ngữ từ text | X2 cho thấy lỗi ở ngôn ngữ | chưa có |
| T2 | Text-only | LM ngoài + shallow fusion / LODR (text CS ngoài bench) | tách lợi ngôn ngữ khỏi lợi âm học | = X6 | chưa có trong repo (icefall có) |
| B-a | Giải mã | Hotwords Aho-Corasick từ catalog **ngoài bench** | bias tĩnh rẻ; đo chèn sai trên G0 | `catalog/train-v1` | ✅ (E1 / E7 cũ là oracle) |
| B-b | Giải mã | Bias học được (TCPGen phoneme-aware) | bias theo ngữ cảnh + phát âm | B-a vẫn yếu trên `ft_unseen` | chưa có, nghiên cứu |
| O1 | Đầu ra | Mở rộng BPE + resize head | giảm phân mảnh thuật ngữ thành chuỗi ký tự | `tokenizer_check` C1 chỉ ra nút cổ chai | chưa chạy check |
| H1 | Hậu xử lý | Chuẩn hoá CS sau ASR (kiểu AdaCS) | sửa từ Anh bị viết kiểu Việt; canh `vi_only_wer` | sau khi chọn model | chưa có |
| H2 | Hậu xử lý | LLM sửa N-best (GER có ngữ âm) | server-side, không hợp edge | tuỳ chọn | chưa có |

Thứ tự khuyến nghị:

1. Chẩn đoán rẻ: `tokenizer_check`, P1 (X2), T2 (X6).
2. Nền dữ liệu: D1 + D5.
3. Chống quên cùng X3: L1, L4 (L3 luôn có).
4. Thêm dữ liệu: D2 / D3 / D4 (X4, X5).
5. B-a trên checkpoint tốt nhất.
6. P2, L5, T1, O1 chỉ khi chẩn đoán chỉ ra đúng nút cổ chai.
7. Nghiên cứu: H1 / H2, B-b, D6, P3.

---

## Để sau (P2–P3)

- [ ] Nhãn **cách đọc** cho span tiếng Anh (`viet_level`) trên dev / Z0 (R09). Cần cho câu hỏi cốt lõi "từ Anh đọc giọng Việt".
- [ ] Thu Z0 zero-shot (`zero_shot_v1.yaml` mới có giao thức) (R23).
- [ ] CI cho chỉ số macro (R20).
- [ ] Đo lại `duration` thật cho C0 / VietMed (R25).
- [ ] FT VieGem (upstream không có code train; cần GPU ≥ 24 GB).

---

## 👤 Quyết định đang chờ bạn

| Quyết định | Chặn mục |
| --- | --- |
| Danh sách `reserved_terms_v1` | 2b → toàn bộ Phần 5 |
| Nguồn CS tự nhiên ngoài y khoa có id người nói | 2e → Phần 3 (NO-GO nếu không có) |
| Nguồn tiếng Việt nói tự nhiên có id người nói | `dev_v2_vi`, replay |
| Nghe 200–300 câu nhãn giả | 2d → X4 |
| GPU cho X0–X6 (2 seed × 7 cấu hình) | Phần 5 |
| Hướng ưu tiên sau X3 (D3 TTS / D4 ghép cụm / L1 KL / P2 LoRA) | Phần 7 |
| Gói share cũ đã gửi cho ai | Phần 6 |

## Làm ngay hôm nay

1. ~~Bước 0, 1a, 1b, 2a, 2b (công cụ)~~ ✅ 17/11.
2. 👤 Duyệt [reserved_terms_v1.proposal.txt](configs/dataset/reserved_terms_v1.proposal.txt) (đổi `--fraction`, sửa tay) rồi đổi tên thành `reserved_terms_v1.txt`; mình dựng `catalog/train-v1` ngay sau đó.
3. **2c** dọn dữ liệu train, trước hết `pool/vimedcss_train@v2` loại video có trong bench khoá (đang bị cổng chặn).
4. **1d** (chạy nền): KenLM cần 👤 `sudo apt install build-essential cmake`; hàng đợi Whisper / Qwen3 / E0 đang chạy.
5. 👤 Nguồn CS tự nhiên ngoài y khoa có id người nói (cho `dev_v2`, Phần 3).
