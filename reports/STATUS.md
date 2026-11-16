# STATUS — cập nhật 15/11/2026, 23:00 (giờ VN): ưu tiên E0/E1 trên bench NHÁP

## 0. Job ASR (vss_cs) và lệnh resume

- Job 6 h **đã tự dừng lúc 22:29, exit 0**: đạt **6.09 h sạch / 1 657 câu** (chunk 14; mẫu 6 000 câu, seed 1). Lúc nhận prompt "ĐỔI ƯU TIÊN" thì
  không còn tiến trình ASR nào chạy, nên không phải kill. Log: `results/asr_hyps/vss_cs_run_s1_b8_6h.log`, `results/asr_hyps/chain6h.log`.
- Kiểm tra cache sau khi dừng: `qwen3_asr_0.6b/vss_cs_run_s1.jsonl` 5 600 dòng, `fw_large_v3_int8/vss_cs_run_s1.jsonl` 4 334 dòng:
  **0 dòng JSON hỏng, 0 id trùng**. `*.runs.jsonl` có id trùng, đúng thiết kế (mỗi lần chạy ghi 1 dòng).
- Lệnh resume (WSL; đọc lại cache theo id + hash audio, chỉ giải mã câu chưa có hyp). Nâng `--target-clean-h` / `--max-new` nếu cần quá 6 h:

```bash
cd /mnt/d/code/ZAIC/gipbench && source ~/.venvs/gipbench/bin/activate && export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
nohup python tools/vss_cs_run.py --pre data/raw/vss_full/vss_cs_pre_rebuild.jsonl --catalog data/raw/_proxy/catalog_proxy.tsv \
  --seed 1 --max-new 6000 --target-clean-h 6 --chunk 400 --qwen-batch 8 > results/asr_hyps/vss_cs_run_s1_b8_resume.log 2>&1 &
```

## 0b. Bench NHÁP v0-draft (không chờ duyệt)

- Lệnh và log: `data/bench_v0_draft/build.log`. `--accept all --version v0-draft`; mỗi dòng có `bench_version: v0-draft`.
- Nguồn: B1 G-Group · B2 consensus agreed · B3 ViMedCSS hard · B5 G-Group Việt thuần + VIVOS test + consensus product. Không lấy disputed / pending.
- Kết quả: **4 238 câu**: B1 1 330 (1.38 h), B2 218 (0.40 h), B3 524 (1.04 h), B5 2 166 (2.37 h). Bỏ 134 câu B3 không còn span `<en>`
  theo nhãn Spec (từ mượn như `test`, `vitamin`; validate coi là LỖI).
- `validate.py`: **ĐẠT, 0 LỖI**, 894 CẢNH BÁO (bỏ qua theo dặn; log `data/bench_v0_draft/validate.log`).
- Catalog nháp: `data/bench_v0_draft/catalog_v0_draft.tsv` = catalog ViMedCSS + 1 618 span B1/B2 của chính bench → hotwords E1 là
  danh sách "biết trước" thuật ngữ của bench (cận trên lạc quan).
- Kết quả decode / chấm: `results/v0_draft/<system>/`, bảng: `reports/v0_draft_tables.md`.

## 0c. Phase 1 A (16/11/2026, 01:10)

| Mục | Trạng thái |
| --- | --- |
| A1 icefall / k2 / lhotse | XONG: `~/.venvs/gipbench-ft` (torch 2.15.0+cu132, CUDA OK, k2 1.24.4, lhotse 1.33.0, scipy), icefall `f8c8578`; `tools/setup_ft_env.sh` |
| A2 bench nháp v0-draft | XONG (mục 0b) |
| A3 E0 / E1 hotwords 1.0 / 1.5 / 2.0 | XONG: E1 thắng E0 khoảng 10–12 điểm CS-WER B1+B2, cả ba mức đều đạt luật chọn; `reports/v0_draft_tables.md`, `reports/v0_draft_leaderboard/` |
| A4 FT thử trên GPU | XONG: 194 step, VRAM 3.9 / 4 GB, 0.66 step/s; bị leaderboard loại (quên tiếng Việt). Xem `reports/a4_trial_report.md` |
| Job ASR tới 6 h | Đã đạt sẵn (6.09 h, mục 0), không phải chạy tiếp |

## 0d. Đêm 16/11/2026: bước 1 + E3 v1 (đang chạy, xem `reports/e3_v1_report.md` khi xong)

| Việc | Trạng thái |
| --- | --- |
| 1. Bảng E0 / E1 theo split, phân loại lỗi thuật ngữ, đề xuất variants (chưa áp), speaker_cluster G-Group | XONG (`ea9c57e`): `reports/v0_draft_tables.md`. G-Group: B1 208 cụm, B5 69 cụm (ngưỡng consensus 0.7565); CI B1 tính lại |
| 2. ViMedCSS train đầy đủ | XONG: 11 832 câu / 22.95 h + validation 1 714 / 3.37 h (`data/raw/vimedcss_full/`). **D: chỉ còn 3.1 GB lúc bắt đầu (< 20 GB yêu cầu)**: đã xoá checkpoint A4 + `vivos.tar.gz` (đã giải nén đủ 12 420 wav), convert `--streaming` (không cache 14 GB parquet) |
| 2. distill VSS không có từ Anh | XONG: nguồn 9.44 h (giới hạn bởi số video không trùng vss_cs), giữ 1 889 câu / 6.78 h (greedy = beam). Decode trên **CPU** (sherpa-onnx pip là bản CPU). HF giới hạn tốc độ (429) khi tải audio VSS: tải chậm, đủ 2 944 file |
| 2. VIVOS train đầy đủ | Có sẵn (11 658 câu), tách 4 người nói (0.96 h) làm dev VI |
| 3. E3 v1 | **ĐANG CHẠY** (`ft/run_e3_v1.sh`, checkpoint ở `C:\Users\goose\gipbench_exp\e3_v1` vì D: hết chỗ) |
| D: / RAM | D: còn ~5 GB (ngưỡng dừng 3 GB). RAM WSL 6 GB: watchdog đã dừng FT 4 lần khi RAM < 1 GB (xem report) |

---

# STATUS — 15/11/2026, 19:20 (giờ VN)

Chỉ kiểm tra, không sửa code, không động vào job. Nguồn đối chiếu: các prompt anh/chị đã gửi (đọc lại từ transcript các phiên), `docs/PROMPT_OPUS_IMPLEMENT.md`, `reports/*.md`, git log.

## 1. Job đang chạy

| | Giá trị |
| --- | --- |
| Job | Phase 0d T6: `tools/vss_cs_run.py --seed 1 --max-new 6000 --target-clean-h 4 --chunk 400 --qwen-batch 8` (nohup, WSL) |
| PID (WSL) | 335 (bash nohup) → 438 (`vss_cs_run.py`) → 35642 (`asr_hyps.py` faster-whisper, đang chạy) |
| Bắt đầu | khoảng 16:51; đã chạy 2 h 23 phút |
| Log | `results/asr_hyps/vss_cs_run_s1_b8.log` |
| Tiến độ | **chunk 6** (0-index; chunk 0–1 lấy từ cache). Qwen 400/400 xong; faster-whisper 100/304 |
| Giờ sạch | **3.01 h / 825 câu** (tính cả pilot), ghi lúc bắt đầu chunk 6. Mỗi chunk thêm 0.33–0.44 h, mất khoảng 28,5 phút |
| Mục tiêu | **job chạy với 4 h** (theo prompt "Chọn A + hạ mục tiêu" và prompt batch 8). Prompt hôm nay ghi 6 h, xem mục 4 |
| ETA (ước tính) | đạt 4 h sau chunk 8 hoặc 9, tức **khoảng 20:20–20:50**. Nếu mục tiêu là 6 h: thêm khoảng 9 chunk, xong khoảng 23:00–23:30 (vẫn dưới trần 6 000 câu = 15 chunk) |
| Cache | Qwen `vss_cs_run_s1.jsonl` 2 800 dòng (có `batch_size: 8`); faster-whisper 2 081 dòng |
| D: | **33.8 GB trống** (Windows); log báo 36.3 GB ở `data/tmp/vss_audio`; ngưỡng dừng 3 GB |
| C: | 28.9 GB trống |
| RAM WSL | tổng 5.9 GB, used 2.8, **available 3.1 GB** (ngưỡng hạ batch: 1 GB); swap 16 GB, đang dùng 0.1 GB. Log: RAM trống 5.1 GB mỗi đầu chunk |
| RAM Windows | trống 4.2 / 15.7 GB; `vmmemWSL` 3.2 GB; pagefile D: cấp 15.1 GB, dùng 0.9 GB |
| GPU | 2 630 / 4 096 MiB, util 86%, chỉ có tiến trình 35642 |
| Job khác | **Không có** (không có tmux, không có tiến trình python nào khác) |

## 2. Bảng việc

| Phase / mục | Trạng thái | Commit | File chính | Ghi chú |
| --- | --- | --- | --- | --- |
| 0 · T0.1 converter trên dữ liệu thật | XONG | `b22cc45` | `converters/*`, `reports/converter_check*.md` | Chỉ `--limit 20` (đúng dặn). Common Voice `--limit 20` làm ở 0b. SEA: BỎ |
| 0 · T0.2 đo auto_tag | XONG, **CHỜ TÔI** | `89e8498`, `09e80dc` | `tools/eval_autotag.py`, `reports/t02_*` | Phiếu `t02_spotcheck.csv`: **0/30 câu đã điền** |
| 0 · báo cáo | XONG | `79cd202` | `reports/phase0_report.md` | Giữ `09e80dc` (anh/chị đã duyệt) |
| 0b.1 quy ước theo cách đọc | XONG | `7d647ce` | `SCHEMA.md`, `docs/huong_dan_duyet_b1.md` | |
| 0b.2 `vi_abbrev.txt` | XONG code, **CHỜ TÔI** | `95d8060` | `configs/vi_abbrev.txt`, `tools/build_wordlists.py` | Chưa khoá v1 |
| 0b.3 ViMedCSS hai nhãn, `vi_loanwords`, `--label-field` | XONG code, **CHỜ TÔI** | `ac53aa8` | `conv_vimedcss.py`, `configs/vi_loanwords.txt`, `score.py` | ViMedCSS mới chạy `--limit 20`; B3 chưa chấm thật |
| 0b.4 `vss_cs_candidates`, bộ lọc đồng thuận | XONG | `c117cc3` | `lib/agreement.py`, `tools/agreement_filter.py`, `tools/asr_hyps.py` | Nhóm trộn đã đổi sang `cs_pseudo` ở 0d.5 |
| 0b.5 sửa tài liệu, resample 16 kHz | XONG | `8d68d79` | `docs/research/*` | |
| 0b.6 SEA | BỎ | – | – | Bỏ theo quyết định sau 0b |
| 0c (bỏ SEA, `--relabel-spans`, `--max-en-frac`, lọc rẻ, ASR theo tầng và chunk, pilot 1 000) | XONG | `705edb0` | `tools/vss_cs_prefilter.py`, `tools/vss_cs_pilot.sh`, `reports/phase0c_report.md` | `vss_relabel_listen.csv`: BỎ (anh/chị dặn không nghe) |
| 0d.1 luật gán lại span | XONG | `da69a4a` | `lib/agreement.py` | 120 → 114 span |
| 0d.2 bỏ câu có thuật ngữ chỉ 1 lần | XONG | `4215e0f` | `tools/vss_cs_prefilter.py` | còn 12 351 câu, 45.5 h |
| 0d.3 WER tiếng Việt ≤ 0.15 | XONG | `1096e9f` | `lib/agreement.py` | pilot 279 → 242 câu (0.88 h) |
| 0d.4 cache id + hash, `rebuild_vss_cs.sh` | XONG | `d6ce4b4` | `lib/hyp_cache.py`, `tools/rebuild_vss_cs.sh` | Chưa chạy rebuild sau T6 (chờ T6 xong) |
| 0d.5 `train_mix.tsv`, `cs_pseudo` | XONG | `dcca857` | `configs/train_mix.tsv`, `tools/mix_plan.py` | **Cảnh báo:** medical 77.8% phần CS, vượt trần 35% |
| 0d.6 chạy ASR chỉ lọc | **ĐANG CHẠY** | `6b33540` (code) | `tools/vss_cs_run.py` | Batch 8 xếp theo độ dài: 13/50 câu hyp khác, ~2.0% token, **0 câu đổi quyết định giữ/bỏ**. Con số này mới có trong commit message, **chưa có trong report** (prompt dặn ghi vào report) |
| 0d.7 `vi_abbrev` / `vi_loanwords` | XONG code, **CHỜ TÔI** | `486bb92` | `configs/vi_abbrev*.tsv/txt`, `reports/top_review.csv` | Tập kiểm tra 60 câu **chưa chấm**, đúng dặn (chờ khoá v1) |
| 0d.8 `review_bench.html` | XONG | `203d34e` | `tools/review_bench.html` | Test lõi JS chạy bằng Node |
| 0d · báo cáo | **BẢN NHÁP, chưa commit** | – | `reports/phase0d_report.md` | Ghi "TASK 6 ĐANG CHẠY", phần sau còn trống |
| 1 A1–A4 | **CHƯA LÀM, không rõ định nghĩa** | – | – | Không tìm thấy "A1–A4" trong prompt hay docs. Chỉ có "Phase 1 A (E0/E1 và FT thử, cần GPU)", được dặn làm **sau khi** ASR xong và đã rebuild |
| 1 B · T1.1 `curate_train.py` | XONG | `7a5a94f` | `ft/curate_train.py` | |
| 1 B · T1.2 `make_distill.py` | XONG | `b91d850` | `ft/make_distill.py`, `decode.py --save-logprobs` | Nhận 2 file hyps, không gọi `decode.py` (theo decisions mục 8; PROMPT gốc ghi "gọi decode.py"). Smoke 20 câu VIVOS. Decode smoke chạy trên CPU hay GPU: không rõ |
| 1 B · T1.3 `make_confusables.py` | XONG | `a76378c` | `tools/make_confusables.py` | Chạy trên `train_vss` (nhãn giả VSS): bench 227 câu cần duyệt, train 729 câu |
| consensus (converter, chặn rò rỉ, category name, speaker cluster) | XONG | `53c45f2` | `converters/conv_consensus.py`, `reports/consensus_report.md` | Đủ mục 1–8 của prompt |
| consensus · chốt theo thuật ngữ | **CHƯA LÀM** | – | – | **Không có prompt nào** nhắc tới (đã tìm trong mọi transcript) |
| consensus · `--accept trusted-text` | **CHƯA LÀM** | – | – | Như trên. `build_bench_v0.py` hiện chỉ có `--accept all`, và dòng `review_required` vẫn phải qua duyệt |
| `consensus_terms_ask.md`, `consensus_b5_flags.md` | **CHƯA CÓ** | – | – | Như trên. 19 câu B5 có cờ nằm ở `data/raw/consensus/b5_consensus_product_review.csv` |
| domain report | **CHƯA LÀM** | – | – | Không có prompt riêng. Liên quan: decisions mục 10 (suy domain từ text, `configs/domain_keywords.tsv`, `domain_source: inferred`, báo 20 câu mẫu cho `--holdout-domain finance`) **chưa được làm**: không có file, không có code. `vss_cs` đang mang domain cố định `conversation` |
| B3 ngoài y khoa, `b3_nonmed_terms.csv` | **CHƯA LÀM** | – | – | Không có prompt nào nhắc tới |
| Phase 2 (T2.1 B5 nhiều domain, T2.2 `build_bench_v0`, T2.3 `leaderboard`) | CHƯA LÀM | – | `converters/build_bench_v0.py` (bản gốc, đã sửa nhỏ ở 0b / 0d / consensus) | Chưa có `data/bench_v0.jsonl` |
| Phase 3 (select_ckpt, freeze, `run_e3.sh`) | CHƯA LÀM | – | – | |
| Phase 4 (chỉ thiết kế) | CHƯA LÀM | – | – | |

## 3. CHỜ TÔI (theo ưu tiên)

| # | Việc | File | Cần làm | Phút (ước) | Đang chặn |
| --- | --- | --- | --- | --- | --- |
| 1 | Chốt mục tiêu giờ sạch: **4 h hay 6 h** | – | Trả lời một dòng. Job đang chạy với 4 h và sẽ tự dừng khoảng 20:20–20:50 | 1 | Rebuild + báo cáo 0d → Phase 1 A |
| 2 | Duyệt danh sách từ, rồi báo khoá v1 | `reports/top_review.csv` (60 dòng) | Xoá dòng sai. Nên xem kỹ các mã chứng khoán trùng từ tiếng Anh trong text VSS, vd `tot` (362 lần), `fir` (348 lần) | 15–20 | Chấm tập kiểm tra 60 câu; tag nháp ổn định trước khi duyệt bench (tránh làm lại) |
| 3 | Gửi prompt cho các mục chưa từng được giao | – | Chốt theo thuật ngữ, `--accept trusted-text`, `consensus_terms_ask.md`, `consensus_b5_flags.md`, domain report, B3 ngoài y khoa / `b3_nonmed_terms.csv`, định nghĩa A1–A4 | 5–10 | Các mục đó |
| 4 | Duyệt B1 | `data/raw/ggroup/b1_candidates_review.csv` (1 361 câu, 0 đã duyệt) qua `tools/review_bench.html` | ok / drop / fix | khoảng 230 (10 s/câu) | `bench_v0` → `catalog_v0` → rebuild `vss_cs` thay bench tạm → Phase 1 A trên bench thật, Phase 2 |
| 5 | Duyệt consensus | `data/raw/consensus/b2_consensus_candidates_review.csv` (231), `b5_consensus_product_review.csv` (19 câu có cờ), `disputed_candidates_review.csv` (585) | như trên; B2 chú ý cờ `abbr_reading`, `name:` | khoảng 40 + 5 + 100 | B2 / B5-product trong `bench_v0` |
| 6 | Duyệt B5 confusable | `data/raw/confusable/b5_confusable_review.csv` (227) | như trên. Text gốc là nhãn giả VSS | khoảng 40 | B5 confusable trong `bench_v0` |
| 7 | Duyệt spotcheck T0.2 | `reports/t02_spotcheck.csv` (0/30) | điền `dong_y` y/n, `sua_lai` | 10 | Số P/R auto_tag hiện vẫn là "vàng do Claude gắn, chưa duyệt" |
| 8 | Quyết trần domain | `reports/mix_plan_0d.md` | medical 77.8% phần CS > trần 35%: chấp nhận, hạ tỉ trọng ViMedCSS, hay chờ thêm cs_real khác | 5 | Cấu hình trộn cho Phase 3 |
| 9 | Duyệt pilot / báo cáo 0d sau khi T6 xong | `reports/phase0d_report.md` | duyệt | 10 | Phase 1 A |
| 10 | Dữ liệu replay | Common Voice (anh/chị lo), VIVOS đầy đủ | tải về | không rõ | Nhóm `replay_human` 0.25 hiện có 0 h |
| 11 | Dọn thư mục tạm | `D:\tmp_gitbash` (vẫn còn) | xoá | 1 | – |

## 4. Quyết định đã chốt

| Quyết định | Ghi ở đâu |
| --- | --- |
| Git, `.gitignore` (data/, models/, exp/, results/, *.onnx, *.pt …) | `decisions.md` §6 |
| WSL2, venv Python 3.11 bằng uv, kiểm cú pháp 3.10, không dùng 3.13 | `decisions.md` §6 |
| GPU 3050: Phase 3 chỉ chạy thử 1 epoch nhỏ | `decisions.md` §6 |
| Đồng thuận: faster-whisper large-v3 int8 + Qwen3-ASR-0.6B, `--min-agree 1` | `decisions.md` §7 |
| `make_distill` nhận 2 hyps, nguồn: call center của anh/chị, nếu không có thì VSS | `decisions.md` §8 |
| `--min-contexts-per-term 5` chỉ báo cáo; thứ tự ưu tiên B5; spotcheck 30 câu | `decisions.md` §9 |
| Bộ domain chung, suy domain cho train, `--holdout-domain` | `decisions.md` §10 (**chưa implement phần suy domain**) |
| **Bỏ SEA**; nguồn CS thật = ViMedCSS train + vss_cs + dữ liệu tự có | `decisions.md` §11, README, SCHEMA |
| `vss_cs` **chỉ lọc**, `--relabel-spans` mặc định tắt (gán lại: WER ≤ 0.10) | `decisions.md` §11; luật mới ở 0d.1: chỉ commit `da69a4a` |
| `--max-en-frac 0.5` trước ASR; lọc rẻ; ASR theo tầng và chunk; dừng khi D: < 3 GB | `decisions.md` §11 |
| Bỏ câu có mọi thuật ngữ chỉ 1 lần, giữ 10% (seed 0) | **chưa ghi vào `decisions.md`**: chỉ có ở commit `4215e0f` và report 0d (nháp) |
| Ngưỡng WER tiếng Việt nhánh chỉ lọc ≤ 0.15 (min theo các ASR) | **chưa ghi**: commit `1096e9f` |
| Cache theo id + hash audio; `rebuild_vss_cs.sh` | **chưa ghi**: commit `d6ce4b4`, runbook |
| `cs_pseudo` (vss_cs) và trọng số cs_real 0.35 / cs_pseudo 0.10 / synth 0.05 / replay 0.25 / distill 0.20 / confusable 0.05; trần domain 35% | `configs/train_mix.tsv`; **chưa ghi vào `decisions.md`** |
| Qwen batch 8, xếp theo độ dài; ghi `batch_size` vào cache; hạ batch 4 khi RAM WSL < 1 GB; faster-whisper beam 5 | **chưa ghi**: chỉ có commit `6b33540` |
| Mục tiêu giờ sạch 4 h (tính cả pilot), tối đa 6 000 câu, seed 1 | **chưa ghi**: chỉ có trong prompt và lệnh job |
| vi_abbrev: thêm 11 mục "quy ước dự án", bật lại `tp`, bỏ `cdn`, chỉ HOSE + HNX, luật loại mục trùng catalog EN | `configs/vi_abbrev*`, commit `486bb92`; **chưa ghi vào `decisions.md`** |
| Quy ước gắn `<en>` theo cách đọc (brand Latin = EN; viết tắt đọc kiểu Việt = không; viết tắt Anh đọc kiểu Anh = EN; mơ hồ thì người duyệt quyết, cột `reading_note`) | `SCHEMA.md` §1, `docs/huong_dan_duyet_b1.md` |
| Từ mượn: có trong từ điển Việt = không EN; `same_surface` (vitamin, protein…) chỉ áp trong ViMedCSS | `SCHEMA.md` §1 |
| ViMedCSS hai nhãn (`text_tagged` theo Spec, `text_tagged_orig` theo dataset) | `SCHEMA.md` |
| Tên người nước ngoài = EN, `category: name` | `SCHEMA.md` (`term_category`), `consensus_report.md` |
| consensus = `license: test_only`, không tách dev, `domain: product`, chặn rò rỉ bằng `leak_block.txt` | `SCHEMA.md`, `consensus_report.md`; **chưa ghi vào `decisions.md`** |

**Mâu thuẫn hoặc chưa rõ:**

1. **Mục tiêu giờ sạch: 6 h hay 4 h.** Prompt 0d ghi 6 h. Hai prompt sau ("Chọn A + hạ mục tiêu" và prompt batch 8) ghi 4 h. Prompt hôm nay lại ghi 6 h. Job chạy 4 h. `vss_cs_run.py` có mặc định `--target-clean-h 6.0`, và docstring ví dụ cũng ghi 6.
2. `SCHEMA.md`: bảng quy ước ghi `vitamin` = không EN (áp chung), nhưng ghi chú dưới bảng nói `same_surface` chỉ áp trong ViMedCSS. Ở bộ khác thì `vitamin` theo quy ước B1, tức là EN. Hai câu này nên viết lại cho khớp.
3. Luật tên người trùng âm tiết Việt (`kim`, `chen`, `sam` chỉ nhận khi đứng cạnh một tên không phải âm tiết Việt) là luật Claude tự đặt, **chưa được anh/chị duyệt**.
4. `decisions.md` §10 (suy domain) chưa được làm, nhưng Phase 0b → 0d đều không báo chuyện này.
5. `decisions.md`, phần "Ghi nhận khi dựng môi trường", vẫn ghi "Đang chặn: C: còn 0 GB". Thông tin này đã cũ: C: hiện còn 28.9 GB. Vị trí hiện tại của `ext4.vhdx`: không rõ (chưa kiểm).
6. Từ §11 trở đi, `decisions.md` không còn được cập nhật. Quyết định của 0d, batch 8 và consensus chỉ nằm rải rác ở commit, config và report.

## 5. Rủi ro và lưu ý

| Rủi ro | Chi tiết |
| --- | --- |
| **Bench tạm** `data/raw/_proxy/` (`bench_proxy.jsonl`, `catalog_proxy.tsv`) | Đang dùng cho: chặn rò rỉ và lọc `vss_cs` (`vss_cs_prefilter`, `vss_cs_run`, `agreement_filter`), category và gộp variants của consensus (`conv_consensus --catalog`), confusables. Khi có `bench_v0` thì phải chạy `tools/rebuild_vss_cs.sh --bench data/bench_v0.jsonl --catalog configs/catalog_v0.tsv`, chạy lại `conv_consensus.py` với `catalog_v0`, và chạy lại `make_confusables` nếu catalog đổi |
| Tag nháp phụ thuộc `vi_abbrev` / `vi_loanwords` chưa khoá | Duyệt bench trước khi khoá v1 thì có thể phải làm lại một phần. Mã chứng khoán trùng từ Anh (`tot`, `fir`…) có thể chặn gắn `<en>` sai |
| Trần domain | medical 77.8% phần CS (cs_real chỉ có ViMedCSS 24.3 h; vss_cs mới 0.83 h trong `mix_plan_0d.md`, số cũ) |
| Đĩa | D: còn 33.8 GB, ổn. Lần trước `D:\pagefile.sys` từng phình tới 21.5 GB khi commit charge cao, làm D: còn 0.4 GB. Hiện pagefile D: cấp 15.1 GB. Job tự dừng khi D: < 3 GB |
| RAM | WSL giới hạn 6 GB, available 3.1 GB; Windows còn trống 4.2 GB. Chạy thêm việc nặng trong WSL (vd chạy song song với job) có thể chạm ngưỡng 1 GB |
| Test bỏ qua hoặc thiếu | `vss_cs_run.py` không có test tự động (cần GPU), mới chạy thử 12 câu thật (seed 99). Nhánh "speaker embedding BỎ QUA" trong test là có chủ đích. Test `review_bench.html` chỉ kiểm lõi JS bằng Node, giao diện chưa được kiểm bằng trình duyệt. `selftest.sh` chạy trên dữ liệu đồ chơi |
| Dữ liệu chỉ nằm trong `data/` / `results/` (gitignore, không backup) | Toàn bộ manifest, phiếu duyệt `*_review.csv`, `leak_block.txt`, `speaker_clusters.tsv`, cache hyps ASR (khoảng 4 900 dòng, nhiều GPU-giờ), `vss_audio_sha256.tsv`. Mất `results/asr_hyps/` thì phải chạy lại ASR |
| Số liệu nháp | `phase0d_report.md` (chưa commit); giờ sạch T6 (đang chạy); P/R auto_tag (vàng chưa duyệt); `speaker_cluster` consensus (16 cụm, ước lượng thô, nhạy theo ngưỡng); `mix_plan_0d.md` (vss_cs 0.83 h, số cũ); selftest CS-WER (dữ liệu đồ chơi) |
| Nhãn giả | B5 confusable lấy câu từ VSS (nhãn Zipformer-30M, nhiều nhiễu), nên bắt buộc phải duyệt kỹ |

## 6. Bước tiếp theo đề xuất

| # | Việc | Cần |
| --- | --- | --- |
| 1 | Chờ T6 đạt mục tiêu (4 h, hoặc 6 h nếu anh/chị đổi). Không chạy gì khác trên GPU | GPU (đang chạy) |
| 2 | Chạy `rebuild_vss_cs.sh` từ cache (bench tạm), hoàn tất và commit `phase0d_report.md` (gồm số so sánh batch 8, tỉ lệ giữ, giờ sạch, `mix_plan` mới) | CPU |
| 3 | Sau khi anh/chị khoá v1: chấm tập kiểm tra 60 câu đúng 1 lần, rồi cập nhật `decisions.md` với các quyết định chưa ghi ở mục 4 | CPU |
| 4 | Phase 1 A (E0/E1, FT thử 1 epoch nhỏ), nếu anh/chị gửi định nghĩa A1–A4 | GPU |
| 5 | Sau khi duyệt bench: `build_bench_v0.py`, rồi chạy lại rebuild `vss_cs`, `conv_consensus` và confusables với `bench_v0` / `catalog_v0` | CPU |

## 7. Sức khoẻ repo

| | Kết quả |
| --- | --- |
| Nhánh | `master` (chỉ có nhánh này; không có `main` local) |
| git status | 1 file chưa track: `reports/phase0d_report.md` (bản nháp, cố ý chưa commit tới khi T6 xong). Không có thay đổi chưa commit nào khác |
| `tools/test_converters.py` (WSL, `CUDA_VISIBLE_DEVICES=""`) | **ĐẠT**, exit 0, 24 s ("TẤT CẢ KIỂM THỬ CONVERTER ĐẠT"). Các CẢNH BÁO trong log là của chính các ca test |
| `tools/selftest.sh` (CPU) | **ĐẠT**, exit 0, 8 s |
