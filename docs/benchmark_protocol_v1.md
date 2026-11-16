# Giao thức benchmark v1 (bản nháp, 16/11/2026)

Tài liệu này định nghĩa cách gipbench đánh giá **mọi** model ASR tiếng Việt (Gipformer 1.5, VieGem-ASR, Whisper,
Qwen3-ASR, bản FT…) cho công bằng và tái lập được. Không thay thế `docs/research/02_spec_du_lieu_metric_do_tin_cay.md`
(đặc tả bench v0, giữ làm lịch sử). Mã nguồn: `src/gipbench/bench/{registry,transcribe,evaluate}.py`,
`src/gipbench/infer/adapters/`, `src/gipbench/core/{normprofiles,evalstats,decontam}.py`. Định nghĩa metric:
[evaluation_metrics_v1.md](evaluation_metrics_v1.md). Nguồn dữ liệu: [dataset_registry.md](dataset_registry.md).

## 1. Các track

| Track | Câu hỏi | Benchmark (configs/bench/benchmarks/) | Trạng thái |
| --- | --- | --- | --- |
| **G0** General | Nhận dạng tiếng Việt phổ thông (đọc + tự nhiên) tốt tới đâu? | `general_v1_draft`: VIVOS test, Common Voice vi 27 test, 4 bộ G-Group test (4 400 câu, 4.47 h) | active, **draft** |
| **C0** Code-switching | Từ tiếng Anh trong câu tiếng Việt? | `code_switch_v1_draft`: ViMedCSS hard (658 câu, 1.29 h); B1/B2 ở `v0_draft` | active, **draft** |
| **R0** Regression | FT có làm hỏng tiếng Việt? | G0 + B5 của `v0_draft`; so cặp với model gốc | dùng G0 + v0 |
| **Z0** Zero-shot / OOD sau FT | Tổng quát trên người nói / phiên / thuật ngữ không có trong dữ liệu FT? | `zero_shot_v1` | **planned** (chưa có dữ liệu thu mới) |
| **E0** External | So với số đã công bố theo giao thức của từng bộ | dùng lại G0/C0, báo từng bộ | không bộ nào công bố scorer chính thức (xem registry) |
| legacy | Bench nháp B1–B5 | `v0_draft` (4 238 câu) | giữ để so lịch sử |

B1–B5 cũ vẫn chấm được y như trước (`gipbench.bench.score`, hoặc `evaluate --benchmark v0_draft` với hồ sơ `legacy_v0`,
cho đúng số lịch sử tới từng chữ số).

## 2. Mức bằng chứng và khoá phiên bản

- `evidence_level`: `draft` (nhãn chưa được dự án duyệt) → `human_reviewed` → `locked`. Hiện **mọi benchmark là draft**.
- Một benchmark chỉ được `locked` khi có `manifest_sha256` (sha256 nội dung manifest, bỏ đường dẫn audio, kèm
  `audio_sha256` từng câu), hồ sơ chuẩn hoá và scorer có phiên bản. `evaluate` từ chối chạy nếu manifest khác hash đã khoá.
- Đổi nhãn / đổi câu / đổi chuẩn hoá = **phiên bản mới**; không sửa bản đã khoá. Số cũ (v0-draft) không bị ghi đè.
- `--mode final` của `evaluate`: thoát mã 2 nếu hyps thiếu / thừa / trùng id, benchmark chưa khoá (trừ khi
  `--allow-draft`, khi đó kết quả vẫn ghi draft), inference mode chưa rõ hoặc không được phép, hồ sơ chuẩn hoá chỉ để
  chẩn đoán, hoặc nguồn chưa rõ license. `--mode diagnostic` (mặc định) chỉ cảnh báo.

## 3. Chế độ suy luận (inference mode), bắt buộc khai báo

| Mode | Được dùng gì lúc decode | Xếp hạng ở |
| --- | --- | --- |
| `open_vocab` | không gợi ý gì | G0, C0, Z0 |
| `lm` | LM train trên text **không** thuộc test (vd LM đi kèm VieGem) | G0, C0, Z0 (báo riêng) |
| `context_bias` | hotwords từ nguồn ngoài test, khai báo `provenance: external` | C0 (báo riêng) |
| `oracle_bias` | hotwords có thông tin từ test (vd catalog v0-draft chứa 1 618 span của chính bench) | **chỉ chẩn đoán**, bị chặn ở G0/Z0 |

`transcribe` chặn trước khi chạy nếu mode không thuộc `allowed_inference_modes` của benchmark, hoặc hotwords lấy từ
catalog có `catalog_from_bench: true` mà không khai báo `test_derived` + `oracle_bias`. E1 lịch sử (hotwords từ catalog
v0-draft) là **oracle_bias**, không phải zero-shot.

Không so Gipformer + oracle hotwords với VieGem không gợi ý trong cùng một bảng xếp hạng.

## 4. Zero-shot, OOD và nhiễm bẩn

- "Zero-shot base" = model gốc chưa FT trên test set. "Sau FT" = model FT chạy trên domain / người nói / thuật ngữ
  không có trong dữ liệu FT. Báo **riêng**.
- Thuật ngữ không có trong dữ liệu FT gọi là **FT-unseen**, không gọi là "unseen" vì không biết pretraining của model
  gốc (Gipformer không công bố dữ liệu train).
- Mỗi nguồn có `pretraining_overlap` theo model (`confirmed / known_possible / not_listed / unknown`). Bộ có
  `known_possible` (vd ViMedCSS với VieGem, G-Group với Gipformer) **không** dùng để kết luận khả năng tổng quát độc lập.
- Trước mọi FT: `python -m gipbench.dataset.release.leak_audit --test <bench> --train <pool...> --dev <dev...>
  [--reserved-terms ...] [--hotwords ...]` phải ra 0 nhóm critical (hoặc từng nhóm được giải thích trong report).

## 5. Thống kê và báo cáo

- WER báo theo **từng bộ**; con số gộp duy nhất được phép là **macro theo bộ** (trung bình đều, đăng ký trước trong
  yaml `aggregate: macro_by_dataset`), luôn kèm số từng bộ. Không báo một micro-WER gộp mọi audio làm số chính.
- CI 95%: bootstrap theo cụm (`boot_unit`, mặc định speaker), seed 0, 1 000 lần. G-Group không có id người nói nên
  bootstrap thực chất theo câu (ghi rõ trong báo cáo).
- So hai hệ: Δ ghép cặp trên **cùng id câu**, bootstrap cặp theo cụm (`evaluate --baseline-hyps`).
- Một hàng kết quả = (bộ dữ liệu, model + revision, cách giải mã, inference mode): số câu, giờ, định nghĩa WER, WER, CER,
  CS-WER (nếu có), term recall, CI, RTF + phần cứng, cảnh báo nhiễm bẩn. Ô chưa đo ghi `N/A (chưa chạy)`.
- Tốc độ (RTF) báo tách khỏi độ chính xác vì phần cứng khác nhau (Gipformer chạy CPU, VieGem chạy GPU).

## 6. Tái lập

- Mỗi lần chạy model ghi `run_meta.json` cạnh `hyps.jsonl`: model id + revision + sha256 file, cách giải mã, inference
  mode, hotwords (đường dẫn, sha256, provenance), phần cứng, precision, thread, batch, RTF và cách đo.
- Mỗi lần chấm ghi `summary.json`: scorer + phiên bản, git commit, hồ sơ chuẩn hoá + sha256, manifest_sha256,
  hyps_sha256, catalog_sha256, định nghĩa WER / CER, seed, đơn vị bootstrap, kết quả kiểm hyps.
- **Lô giải mã sherpa-onnx ảnh hưởng kết quả**: cùng model, cùng audio, đổi thành phần lô làm đổi hyp (đo trên 140
  câu: lô 1 vs lô 32 khác 30 câu; `decode.py` lô 32 trên tập con chỉ khớp E0 lịch sử 95/140 câu). Config Gipformer mới
  dùng **lô 1** để kết quả không phụ thuộc thứ tự / tập con. Số E0 lịch sử chỉ tái lập được với đúng manifest v0-draft,
  đúng thứ tự, lô 32.

## 7. Lệnh

```bash
# WSL, tại gốc repo; dữ liệu và model nằm ở cây chính
export PYTHONPATH=$PWD/src GIPBENCH_DATA=/mnt/d/code/ZAIC/gipbench/data GIPBENCH_MODELS=/mnt/d/code/ZAIC/gipbench/models
python -m gipbench.bench.registry                                            # kiểm registry
python -m gipbench.dataset.release.build_general --which general             # dựng G0 (không tải gì)
python -m gipbench.dataset.release.build_general --which code_switch         # dựng C0
python -m gipbench.bench.transcribe --model configs/bench/asr_models/gip15_mbs4_open.yaml --benchmark general_v1_draft \
    --out results/general_asr/general_v1_draft/gip15_mbs4_open                # ~/.venvs/gipbench
python -m gipbench.bench.transcribe --model configs/bench/asr_models/viegem_greedy.yaml --benchmark general_v1_draft \
    --out results/general_asr/general_v1_draft/viegem_greedy                  # ~/.venvs/gipbench-viegem (xem viegem_integration.md)
python -m gipbench.bench.evaluate --benchmark general_v1_draft \
    --hyps results/general_asr/general_v1_draft/viegem_greedy/hyps.jsonl \
    --baseline-hyps results/general_asr/general_v1_draft/gip15_mbs4_open/hyps.jsonl
```
