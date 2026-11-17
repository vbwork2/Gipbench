# Audit benchmark gipbench (16/11/2026)

Kiểm trên nhánh `feat/general-asr` (tách từ `refactor/pipelines` @ `8a5d20c`, đã gồm `main` @ `1570a42`). Tài liệu giao
việc mô tả bố cục cũ (`score.py`, `lib/`, `ft/`); repo hiện đã package hoá thành `src/gipbench/{core,infer,dataset,model,bench}`.
Mọi đối chiếu dưới đây theo bố cục mới.

## 1. Phân loại tính năng

| Tính năng | Trạng thái | Ghi chú |
| --- | --- | --- |
| Chấm WER / CS-WER strict + lenient / N-WER / term / span, CI bootstrap theo người nói (`bench/score.py`, `core/scoring.py`) | đã cài, đã test trên dữ liệu thật | số v0-draft tái lập đúng tới mọi chữ số |
| Leaderboard + luật chọn (`bench/leaderboard.py`) | đã cài, đã chạy thật | mặc định `--main-metric term_acc_macro`; báo cáo cũ xếp theo `cs_wer`: phải ghi rõ metric khi so |
| Giải mã Gipformer (`infer/decode.py`, sherpa-onnx) | đã cài, đã chạy thật | **kết quả phụ thuộc thành phần lô** (mục 3.4) |
| faster-whisper / Qwen3-ASR (`infer/asr_hyps.py`) | đã chạy thật (cho lọc đồng thuận vss_cs) | chưa chạy trên bench |
| validate rò rỉ (`dataset/release/validate.py`) | đã cài, đã test (selftest) | thoát mã 1 khi có lỗi; bị bỏ qua bằng `|| true` trong `run_e3_v1b.sh` (nay ở `docs/history/`) |
| Converter G-Group / ViMedCSS / VIVOS / CV / consensus | đã test trên dữ liệu thật | |
| FT Gipformer (icefall) | đã chạy thật (A4, E3 v1), kết quả kém (mục 3.6) | |
| **Mới:** registry nguồn / benchmark có phiên bản | đã cài, đã test | `configs/bench/`, `bench/registry.py` |
| **Mới:** hồ sơ chuẩn hoá có phiên bản | đã cài, đã test | `core/normprofiles.py` |
| **Mới:** evaluate đa bộ, final mode, CER, macro theo bộ, Δ cặp | đã cài, đã test, đã chạy thật | `bench/evaluate.py` |
| **Mới:** adapter ASR + runner, run_meta | đã cài, đã test; Gipformer + VieGem chạy thật | `infer/adapters/`, `bench/transcribe.py` |
| **Mới:** kiểm nhiễm bẩn (hash, text, gần trùng, speaker / source, thuật ngữ FT-unseen, hotwords) | đã cài, đã test, đã chạy thật | `core/decontam.py`, `dataset/release/leak_audit.py` |
| VieGem + KenLM | **bị chặn** | WSL không có trình biên dịch C/C++ để build `kenlm` |
| Whisper / Qwen3 trên G0 / C0 | đã cài adapter, chưa chạy | không bắt buộc |
| Z0 zero-shot sau FT | chưa có (cần dữ liệu thu mới) | giao thức `zero_shot_v1.yaml` (planned) |
| FT VieGem | chưa cài (upstream không có code train) | `docs/experiment_matrix.md` |

## 2. Kiểm lại các khẳng định trong tài liệu giao việc

| Khẳng định | Kết quả kiểm |
| --- | --- |
| v0-draft 4 238 câu ≈ 5.19 h; B1 1 330, B2 218, B3 524, B5 2 166; không có B4 | **đúng** (`audio_sec` 18 671 s = 5.19 h) |
| E0: CS-WER B1+B2 71.66, B5 9.52; E1: 60.21 / 9.63; E3 v1: 92.44 / 12.81 | **đúng** (tái lập bằng `evaluate --benchmark v0_draft`, hồ sơ legacy_v0) |
| Hotwords E1 lấy từ thuật ngữ của test | **đúng**: catalog v0-draft = catalog ViMedCSS + 1 618 span của chính bench → đã xếp vào `oracle_bias` |
| score.py tính hyp thiếu như rỗng, chỉ cảnh báo | **đúng**, giữ nguyên để tương thích; `evaluate --mode final` thoát mã 2 |
| Mặc định leaderboard là `term_acc_macro` | **đúng** |
| `run_e3_v1b.sh` có `validate.py ... || true` | **đúng** (dòng stage 0; file nay ở `docs/history/`) |
| Gipformer ~68M Zipformer RNN-T, sherpa-onnx | **đúng**; file int8 ở `models/gip15` trùng sha256 với HF @ `dd9227dc` |
| VieGem ~359M CTC | **đúng** (model card @ `95eae70b`); thêm: cần encoder `google/embeddinggemma-2` (1.5 GB) và transformers ≥ 5.18 |
| Phần cứng: WSL2, RTX 3050 4 GB, ~16 GB RAM | **đúng**; WSL giới hạn 6 GB RAM; D: còn ~48 GB, C: ~23 GB |

## 3. Rủi ro khoa học / tái lập lớn nhất (theo mức độ)

1. **Nhãn bench chưa được duyệt.** Mọi benchmark là `draft`; B1 / B3 chờ người dùng duyệt (≈ 230 + 90 phút).
2. **ViMedCSS không tách theo video.** 308 / 658 câu hard đến từ 201 video có trong ViMedCSS train. VieGem và E3 đều
   train trên ViMedCSS train → C0 không đo tổng quát sang người nói mới. dev_cs của E3 (ViMedCSS validation) cũng vậy.
3. **VIVOS dùng lại câu đọc**: ~194 / 760 câu test có cùng text với VIVOS train. Model FT trên VIVOS train có lợi thế text.
4. **Giải mã theo lô của sherpa-onnx không bất biến**: trên 140 câu, lô 1 vs lô 32 khác 30 hyp; `decode.py` lô 32 trên
   tập con chỉ khớp E0 lịch sử 95 / 140 câu. Hệ quả: E0 chỉ tái lập được với đúng manifest, đúng thứ tự, lô 32; so hai
   hệ chạy với lô khác nhau có thêm nhiễu. Adapter mới dùng lô 1 cho Gipformer (khớp từng byte với `decode.py` cùng đầu vào).
5. **Thông tin test lọt vào chọn dữ liệu FT**: E3 dùng `catalog_v0_draft.tsv` (có span của bench) cho trần thuật ngữ khi
   curate train.
6. **CS-WER strict = lenient** trên v0-draft: catalog có 0 dòng `phonetic` và 0 dòng `variants` → hai bảng trùng nhau.
   Không phải lỗi scorer. Số lenient chỉ có nghĩa khi soạn phiên âm từ nguồn ngoài test.
7. **Chuẩn hoá phạt cách viết hợp lệ** (hoà / hòa, kỹ / kĩ, A.I., email, colour): đã thêm `gipbench_strict_v1`
   (B5 E0 9.52 → 9.38). Mặc định cũ không đổi.
8. **Không có id người nói ở G-Group** → CI theo câu, không kiểm được trùng người nói giữa các bộ.
9. **G-Group do tổ chức phát hành Gipformer công bố** và model card Gipformer báo số trên các bộ này → không phải bằng
   chứng tổng quát độc lập cho Gipformer. Common Voice và ViMedCSS có thể trùng nguồn với dữ liệu train VieGem.
10. `<en>` gộp ba thuộc tính (từ vựng Anh / brand Latin / cách đọc): đã thêm trường tuỳ chọn `language_spans`
    (`core/langspans.py`); dữ liệu hiện có chưa điền.

## 4. Lệnh tái lập audit

```bash
export PYTHONPATH=$PWD/src GIPBENCH_DATA=/mnt/d/code/ZAIC/gipbench/data GIPBENCH_MODELS=/mnt/d/code/ZAIC/gipbench/models
python -m gipbench.bench.registry
python -m gipbench.bench.evaluate --benchmark v0_draft --hyps /mnt/d/code/ZAIC/gipbench/results/v0_draft/e0_baseline/hyps.jsonl --out /tmp/e0
D=$GIPBENCH_DATA; python -m gipbench.dataset.release.leak_audit --test $D/benchmarks/code_switch_v1_draft/manifest.jsonl \
   --train $D/e3_v1b/train_curated.jsonl --dev "$D/e3_v1/dev_*.jsonl" --no-audio-sha --fail-on never
```
