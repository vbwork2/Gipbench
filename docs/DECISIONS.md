# Quyết định đã chốt (trả lời câu hỏi trước Phase 0)

Ngày 14/11/2026. Các quyết định này thay cho phần tương ứng trong `docs/PROMPT_OPUS_IMPLEMENT.md` và `docs/research/` nếu có khác biệt.

## 6. Môi trường

- Dùng git: `git init`, `.gitignore` bỏ qua `data/`, `models/`, `exp/`, `results/`, `reports/**/*.html`, `*.onnx`, `*.pt`.
- Làm toàn bộ dự án trong WSL2. Lý do: các script `.sh` và icefall/k2 chỉ chạy trên Linux, và WSL2 dùng được GPU NVIDIA.
- Python: venv Python 3.11 tạo bằng uv. Kiểm tra cú pháp thêm bằng 3.10 (`uv run --python 3.10 python -m py_compile ...`). Không dùng Python 3.13 cho dự án.
- GPU RTX 3050 4 GB: Phase 0–2 không cần GPU. Phase 3 chỉ cần code chạy đúng và chạy thử 1 epoch rất nhỏ (`--max-duration 100`, fp16) để chứng minh pipeline thông; huấn luyện thật chạy trên GPU thuê. Báo cáo phải ghi rõ phần nào đã chạy trên 3050, phần nào chưa.
- Dữ liệu: Claude được tự tải các bộ công khai không gated (G-Group, ViMedCSS, VietSuperSpeech, VIVOS nếu tải được). Common Voice và SEA do người dùng lo. Phase 0 chỉ cần `--limit 20`, không tải toàn bộ.

## 7. Lọc SEA bằng đồng thuận

- Không dùng Gipformer gốc để đồng thuận. Dùng 2 model đa ngôn ngữ ra được chính tả Anh: faster-whisper large-v3 (int8) và Qwen3-ASR (0.6B nếu 1.7B không vừa VRAM).
- Định nghĩa: giữ câu SEA khi các span tiếng Anh trong nhãn gốc (MOSS) xuất hiện trong hyp của ít nhất K model, so sau `normalize_hyp`. Tham số `--min-agree`, mặc định 1. Nhãn gốc + 1 model xác nhận = 2 nguồn đồng thuận, khớp Spec mục 5.
- Chạy 2 model là bước riêng: một script xuất `hyps.jsonl` cho từng model. `curate_train.py` chỉ đọc các file đó.

## 8. make_distill

- `make_distill.py` nhận 2 file `hyps.jsonl` (greedy, beam) do `decode.py` chạy sẵn; không gọi `decode.py`.
- Nguồn audio không nhãn, theo thứ tự ưu tiên: (a) audio call center của người dùng nếu được cung cấp; (b) audio VietSuperSpeech, bỏ nhãn cũ và gán lại bằng Gipformer gốc. Loại mọi `source` có trong bench.
- Thêm `--source-manifest` để sau này cắm nguồn khác.

## 9. Lệch nhỏ

- `curate_train.py` thêm `--min-contexts-per-term 5`: chỉ báo cáo, không loại (giống `--min-speakers-per-term`). "Ngữ cảnh" = cặp (từ trước, từ sau) quanh thuật ngữ.
- B5: danh sách trong prompt là chuẩn. Ưu tiên VLSP-T1, VietMed, FLEURS, Common Voice, vivos, vi-asr-pubadmin; MultiMed và LSVSC là tuỳ chọn. Lọc trùng giữa các bộ bằng hash audio và text sau chuẩn hoá; nếu MultiMed chứa VietMed thì giữ VietMed.
- T0.2: gắn tag vàng từ text là đúng cách đo, vì `auto_tag` cũng chỉ nhìn text. Xuất 30 câu ngẫu nhiên vào `reports/t02_spotcheck.csv` để người dùng duyệt. Ghi rõ trong báo cáo: "vàng do Claude gắn, người duyệt 30 câu".

## 10. Bộ giá trị domain chung

- Giá trị: `tech, finance, edu, medical, pubadmin, business, daily, entertainment, product, callcenter, conversation, read, confusable, other`. Đưa vào `lib/manifest.py` và `SCHEMA.md`.
- Mỗi converter phải map domain gốc về bộ này.
- Dữ liệu train không có nhãn domain (VietSuperSpeech / vss_cs, dữ liệu tự có, crawl; SEA nếu dùng lại) thì suy ra từ text: từ khoá theo domain (`configs/domain_keywords.tsv`) cộng category của thuật ngữ catalog trong câu; không chắc thì `other`. Ghi `domain_source: inferred`.
- `--holdout-domain finance` loại các câu train có domain finance (gán sẵn hoặc suy ra). Báo cáo số câu bị loại và 20 câu mẫu để người dùng kiểm tra bộ suy luận.

## 11. Sau Phase 0b (15/11/2026)

- **Bỏ SEA**: SEA Code-Switching là tùy chọn, không dùng; không có trong lệnh mặc định. Mục 7 chỉ còn áp cho vss_cs. Nguồn CS thật cho train = ViMedCSS train + vss_cs + dữ liệu tự có; B1 = G-Group + dữ liệu tự thu.
- vss_cs mặc định **chỉ lọc**. `--relabel-spans` (mặc định tắt) chỉ thay từng span EN khi cả hai ASR cho cùng chữ sau chuẩn hoá và WER phần tiếng Việt so với mỗi ASR ≤ 0.10; không gán lại cả câu. Câu gán lại: `relabeled: true`, text gốc ở `text_orig`, kiểm tra rò rỉ lại.
- `--max-en-frac 0.5` áp dụng trước ASR.
- Không chạy hết 31 949 câu. Trình tự:
  1. lọc rẻ trước ASR (max-en-frac, độ dài 1–20 s, rò rỉ B1–B3 + reserved_b3, trần thuật ngữ ≤ 50, ưu tiên thuật ngữ hiếm, có seed);
  2. chạy ASR theo tầng: Qwen3-ASR-0.6B trước, faster-whisper chỉ cho câu Qwen chưa xác nhận;
  3. chạy theo chunk, có resume, xoá audio tạm sau mỗi chunk, dừng nếu D: < 3 GB;
  4. pilot 1 000 câu, rồi dừng chờ duyệt. Mục tiêu sau đó: tối đa ~10k câu, 5–10 h sạch.
- vi_abbrev, vi_loanwords: vẫn chờ duyệt.

## Ghi nhận khi dựng môi trường (Claude, chờ người dùng xem lại)

- WSL2 đã có sẵn, nhưng distro là **Ubuntu 26.04** (không phải 22.04), Python hệ thống 3.14. Dự án dùng Python 3.11 từ uv nên không phụ thuộc Python hệ thống; icefall/k2 cài theo bản wheel của torch nên khả năng cao không cần đúng 22.04. Chưa kiểm chứng tới Phase 3.
- venv đặt trong filesystem Linux (`~/.venvs/gipbench`) thay vì trên `/mnt/d`, vì I/O qua `/mnt/d` chậm; code vẫn nằm ở `D:\code\ZAIC\gipbench`.
- ~~Đang chặn: ổ đĩa ảo của Ubuntu nằm trên C:, C: còn 0 GB.~~ **Đã xử lý (15/11):** distro Ubuntu chuyển sang `D:\WSL\Ubuntu`; swap WSL ở `D:\WSL\swap.vhdx` (16 GB); `.wslconfig` giới hạn `memory=6GB`. Ngày 15/11, 19:20: C: còn 28.9 GB, D: còn 33.8 GB.

## 12. Sau STATUS 15/11/2026 (gồm các quyết định trước đó chưa ghi)

### 12.1 vss_cs (Phase 0d), trước đây chỉ có ở commit / config

- Lọc rẻ: bỏ câu mà mọi thuật ngữ chỉ có trong 1 câu (tính trên cả 31 949 câu), giữ ngẫu nhiên 10% số câu đó (seed 0). (`4215e0f`)
- Nhánh chỉ lọc: WER phần tiếng Việt = min theo các ASR có hyp, giữ khi ≤ 0.15. Nhánh gán lại vẫn ≤ 0.10. (`1096e9f`)
- Luật gán lại span (vẫn tắt): span mới không chứa token có dấu tiếng Việt, không dài hơn span cũ quá 1 token. (`da69a4a`)
- Cache hyps ASR tra theo id + sha256 audio; `tools/rebuild_vss_cs.sh` chạy lại cả chuỗi chỉ từ cache, không dùng GPU. Khi có `bench_v0` thật thì chạy lệnh này để thay bench tạm. (`d6ce4b4`)
- Qwen3-ASR `--batch-size 8`, câu xếp theo độ dài trước khi chia lô, ghi `batch_size` vào từng dòng cache. Hạ xuống lô 4 nếu RAM trống WSL < 1 GB. faster-whisper giữ beam 5. Pilot giữ hyp lô 1, không chạy lại. Trên 50 câu so sánh: 13/50 câu hyp khác, ~2.0% token, 0 câu đổi quyết định giữ / bỏ. (`6b33540`)
- Lần chạy seed 1: tối đa 6 000 câu mới, kiểm tra D: trước mỗi chunk (dừng < 3 GB). Mục tiêu giờ sạch (tính cả pilot) **6 h**. Lịch sử: 6 h → hạ 4 h (prompt "Chọn A") → **đổi lại 6 h** (15/11, sau STATUS). Job 4 h không bị dừng; khi nó dừng ở 4 h thì chạy tiếp cùng lệnh với `--target-clean-h 6` (resume từ cache).
- Trong lúc ASR nền chạy: không chạy gì khác trên GPU. Việc CPU giới hạn thread (`~/.local/bin/gbcpu`: 2 thread, ẩn GPU) và dừng nếu RAM trống WSL < 1.5 GB.

### 12.2 Trộn dữ liệu

- `configs/train_mix.tsv`: nhóm `cs_pseudo` = vss_cs ("audio thật, nhãn giả"), không thuộc `cs_real`. Trọng số mặc định: cs_real 0.35, cs_pseudo 0.10, synth 0.05, replay có nhãn người 0.25, distill 0.20, confusable 0.05 (tổng 1.00). (`dcca857`)
- Trần domain trên phần CS (cs_real + cs_pseudo): xuất 2 phương án, **35% và 50%**, làm 2 biến thể E3.

### 12.3 Danh sách từ và quy ước gắn nhãn

- vi_abbrev: thêm `vn, ubnd, bhxh, bhyt, cccd, cmnd, thpt, thcs, csgt, tphcm, hcm` (nguồn "quy ước dự án, đọc kiểu Việt"); bật lại `tp`; chỉ HOSE + HNX (không UPCOM); bỏ `cdn`; mục nào trùng term EN của catalog thì tự loại và ghi báo cáo. (`486bb92`)
- vi_abbrev: tự loại mã chứng khoán xuất hiện ≥ 20 lần trong text không phải tài chính (VSS, VIVOS), vd `tot`, `fir`; danh sách đã loại ghi vào report.
- `sjc, tv, vtv, htv, sctv` = **vi** (đưa vào vi_abbrev). `sms, ok, email, check, zalo, app, online` = **en**.
- `same_surface` (`vitamin`, `protein`, `insulin`…) = **không EN ở MỌI bộ**, không còn chỉ áp cho ViMedCSS. Đã sửa `SCHEMA.md` và `load_vi_words`.
- Luật tên người của Claude: **đã duyệt**. Tên nước ngoài = EN, `category: name`; tên trùng âm tiết Việt (`kim`, `chen`, `sam`) chỉ nhận khi đứng cạnh một tên không phải âm tiết Việt, đứng một mình thì chỉ gắn cờ `maybe_name:*`.
- Quy ước theo cách đọc (brand Latin = EN; viết tắt đọc kiểu Việt = không; viết tắt Anh đọc kiểu Anh = EN; mơ hồ thì người duyệt quyết) giữ như `SCHEMA.md` §1.

### 12.4 Chốt tag theo thuật ngữ (thay cho duyệt từng câu) với nguồn có text do người gõ

- Nguồn: G-Group (B1) và consensus `agreed`. Dòng mang `trusted_text: true`.
- `configs/term_decisions.tsv` (cột `term`, `decision` = `en` / `vi` / `drop_tag`, `note`) là nơi chốt cho từng thuật ngữ. `build_bench_v0.py --accept trusted-text` nhận dòng `trusted_text` khi mọi cờ còn lại của câu đã được chốt trong file này. Câu chứa thuật ngữ chưa chốt thì chưa vào bench.
- Phiếu hỏi duy nhất: `reports/term_decisions_ask.csv` (một dòng mỗi thuật ngữ còn cờ, gộp cả `top_review.csv`). Claude điền sẵn đề xuất; người dùng chỉ sửa dòng sai.

### 12.5 consensus

- `license: test_only`, chỉ dùng để test; không tách dev; `domain: product`; `speaker = consensus:<id phiên>`. Text chuẩn hoá vào `data/raw/consensus/leak_block.txt` (chặn rò rỉ ở curate_train, vss_cs_prefilter, rebuild_vss_cs); `validate.py` báo LỖI nếu dòng test_only có trong train / dev. (`53c45f2`)
- Không lấy thuật ngữ consensus làm reserved_b3 (giữ ở B2).

### 12.6 Confusable, domain, replay, backup

- Bỏ `b5_confusable_review.csv` cũ (câu lấy từ nhãn giả VSS). Chạy lại `make_confusables` trên nguồn có nhãn người: VIVOS và consensus B5 product. Phần train chỉ lấy từ VIVOS train. Phần bench không trùng người nói / nguồn với phần train. Phiếu chỉ cần ĐỌC text, có cột "nghĩa Việt? y/n".
- Implement §10 (suy domain từ text, `configs/domain_keywords.tsv`, `domain_source: inferred`), áp cho vss_cs và các train khác chưa có domain.
- reserved_b3 ngoài y khoa: đề xuất 40–80 thuật ngữ (`reports/b3_nonmed_terms.csv`) từ G-Group tech / edu / pubadmin (không lấy finance) và vss_cs; thuật ngữ có 3–20 câu, nhiều người nói, bỏ từ quá phổ biến. **Chưa áp dụng**; báo trước số giờ vss_cs sẽ mất nếu áp dụng.
- Replay: `conv_local.py vivos` trên toàn bộ VIVOS train (CPU). Common Voice thêm sau, nếu người dùng tải về.
- Backup: `tools/backup.sh` nén manifest `data/raw/**/*.jsonl`, `*_review.csv`, `configs/`, `results/asr_hyps/`, `leak_block.txt`, `speaker_clusters.tsv` (không nén audio) vào `D:\gipbench_backup\<ngày>.tar.gz`.

### 12.7 Phase 1 A

- A1: icefall + k2 + lhotse trong venv riêng; chỉ kiểm tra CUDA bằng lệnh nhỏ.
- A2: bench nháp `v0-draft` (`--accept trusted-text` cho các thuật ngữ đã chốt, cộng B3 ViMedCSS hard, cộng B5).
- A3: E0 + E1 (`--hotwords-score` 1.0 / 1.5 / 2.0, `--hotwords-exclude-b3`) decode trên CPU, score, leaderboard. Ghi rõ "bench nháp"; RTF đo trên CPU khi máy bận, không dùng để so.
- A4: FT thử nhỏ trên GPU (~200 step, ~2 h dữ liệu), CHỈ sau khi job ASR xong: wrap_ckpt → to_lhotse → fbank → finetune_cs → export int8 → decode → score. Báo VRAM, step/s, ước tính GPU-h cho FT thật trên GPU 16–24 GB.

## 13. Refactor 3 pipeline (16/11/2026, nhánh `refactor/pipelines`)

Đường dẫn file trong các mục 6–12 ở trên là đường dẫn TRƯỚC refactor. Bảng chuyển file: `docs/history/REFACTOR_PLAN.md` §5;
bản đồ hiện tại: `README.md`, `docs/ARCHITECTURE.md`.

- Tách thành `gipbench.{core, infer, dataset, model, bench}` (package `src/gipbench/`, chạy bằng `python -m`). Luật import
  giữa các tầng được kiểm tự động (`tests/test_boundaries.py`). Code dùng chung đưa xuống `core`: `core.scoring`
  (metric, chung cho chọn checkpoint và chấm bench), `core.tagging` (auto_tag), `core.agreement_rules`, `core.sampling`.
- Ba pipeline nói chuyện qua 4 hợp đồng (`docs/CONTRACTS.md` §4–7): data release bất biến (sha256), experiment YAML (thay
  `ft/run_e3_*.sh`), model artifact + `model_card.json`, system YAML (thay việc gọi `decode.py` bằng tay).
- `configs/train_mix.tsv` -> các experiment YAML: `e3_v1` / `e3_v1b` dùng trọng số lúc chạy (cs_real 0.35, synth 0.05;
  commit `5171980`); `e3_v2` (NHÁP, chưa chạy) dùng trọng số cuối của tsv (cs_real 0.20, synth 0.20). Bản tsv cuối:
  `docs/history/train_mix.tsv`.
- Đợt release đầu đăng ký đúng các file mà E3 v1 / v1b và bench nháp đã dùng (`configs/dataset/releases.yaml`). Các biến
  thể khác trong `data/raw/vss_full/`, `ggroup_l20`, `vimedcss` / `vimedcss_hard` chưa đăng ký: người dùng chốt bản nào
  được phát hành.
- Không đổi hành vi: golden test (`tests/golden/`) cho 97/97 file output giống code trước refactor. Hai khác biệt hợp lệ:
  dòng "Cấu hình: `<file>`" của `mix_plan.md`; trường audio của pool confusable được chuẩn hoá đường dẫn (cùng file).
- Chưa đổi (cần người dùng chốt): bỏ `--agreement-hyps` khỏi curate (lọc đồng thuận thuộc dataset); dựng catalog không
  chứa span của bench; đổi tên `data/raw` -> `data/interim`, `data/dl` -> `data/sources`; giữ hay xoá `ingest/sea.py`;
  giữ `reports/dataset/eda_current/*.csv` trong git hay không.
