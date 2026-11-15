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
- **Đang chặn:** ổ đĩa ảo của Ubuntu (`ext4.vhdx`) nằm trên C:, và C: còn 0 GB trống. Xem báo cáo trong hội thoại.
