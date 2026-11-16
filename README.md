# gipbench — bench code-switching Việt–Anh cho Gipformer 1.5

Bộ công cụ để kiểm tra dữ liệu, làm EDA, chấm điểm và chọn phương án fine-tune. Hướng dẫn từng bước nằm ở tab **Runbook** trong doc dự án.

```
pip install -r requirements.txt
bash tools/selftest.sh            # chạy thử toàn bộ trên dữ liệu giả lập (~10 s)
```

**Graphify (tùy chọn):** Dùng `$graphify` trong Codex để hỏi về quan hệ giữa các
file và hàm. Để cập nhật và truy vấn đồ thị mã nguồn từ PowerShell tại thư mục gốc:

```powershell
$graphifyPython = '.\.venv\graphify\tools\graphifyy\Scripts\python.exe'
& $graphifyPython -m graphify update .
& $graphifyPython -m graphify query 'score decode' --budget 300
```

Nếu checkout mới chưa có CLI, cài lại vào môi trường riêng của repo:

```powershell
$env:UV_TOOL_DIR = Join-Path $PWD '.venv\graphify\tools'
$env:UV_TOOL_BIN_DIR = Join-Path $PWD '.venv\graphify\bin'
uv tool install graphifyy==0.9.111
```

Kết quả nằm trong `graphify-out/`. Lệnh `update .` tạo đồ thị lần đầu và cập nhật
mã nguồn cục bộ ở các lần sau, không cần API key.

**Nguồn dữ liệu** (quyết định sau Phase 0b, 15/11/2026):

- B1 = G-Group (vi-asr-tech/edu/finance/pubadmin-test) + dữ liệu tự thu.
- Nguồn CS thật cho train = ViMedCSS train + `vss_cs` (VietSuperSpeech, lọc đồng thuận) + dữ liệu tự có.
- SEA Code-Switching (`gijs/sea_speech`): **tùy chọn, không dùng**. `conv_sea.py` còn giữ để dùng sau nếu cần.

| File | Việc |
| --- | --- |
| `SCHEMA.md` | Định dạng manifest JSONL, catalog thuật ngữ, file hyp |
| `configs/catalog.example.tsv` | Mẫu catalog (term, category, variants, phonetic, reserved_b3) |
| `validate.py` | Kiểm tra manifest + chống rò rỉ bench ↔ train (thoát mã 1 nếu có lỗi) |
| `eda.py` | Report HTML + `eda_summary.json` + `eda_summary.md` (bảng dán vào báo cáo) |
| `decode.py` | Giải mã bằng sherpa-onnx (Gipformer hoặc model FT đã export), có hotwords, đo RTF |
| `score.py` | WER, CS-WER strict/lenient, N-WER, term recall/precision, CI bootstrap theo người nói, cắt lát |
| `leaderboard.py` | So cặp với baseline, áp luật chọn, xuất `.md/.csv/.html` |
| `converters/conv_ggroup.py` | G-Group vi-asr-*-test → ứng viên B1 + B5 |
| `converters/conv_vimedcss.py` | ViMedCSS → B3 (hard), bench ngoài (test), train, dev, catalog y tế |
| `converters/conv_sea.py` | SEA Code-Switching en-vi → train + ứng viên B1. **Tùy chọn, không dùng** (không có trong lệnh mặc định) |
| `converters/conv_local.py` | VIVOS, Common Voice, VietSuperSpeech từ thư mục đã tải → B5 + train replay |
| `converters/build_bench_v0.py` | Ghép bench từ ứng viên + phiếu duyệt, gộp catalog |
| `tools/test_converters.py` | Kiểm thử converter bằng dataset giả đúng schema |
| `ft/to_lhotse.py` | Manifest → Lhotse, lọc rò rỉ, speed perturb, mô phỏng thoại 8 kHz |
| `ft/compute_fbank.py` | Fbank 80 chiều như recipe icefall |
| `ft/wrap_ckpt.py` | Bọc `model.pt` của Gipformer thành `{"model": …}` cho `finetune.py` |
| `ft/patch_finetune.py` | Tạo `zipformer/finetune_cs.py` đọc `--train-cuts`/`--valid-cuts` của bạn |
| `ft/interpolate_ckpt.py` | Trộn trọng số model gốc và model FT (WiSE-FT) để giữ chất lượng tổng quát |
| `ft/run_e3.sh` | Mẫu chạy trọn E3: dữ liệu → FT → export ONNX → chấm → leaderboard |
| `tools/check_gipformer_tokenizer.py` | Kiểm tra BPE của Gipformer với từ tiếng Anh (xanh/vàng/đỏ) |
| `docs/research/` | Tài liệu: kế hoạch + papers, Spec, Kiến trúc, Runbook (xuất từ doc) |
| `docs/PROMPT_OPUS_IMPLEMENT.md` | Prompt giao việc implement cho Opus |
| `tools/make_toy_data.py`, `tools/selftest.sh` | Dữ liệu giả lập + tự kiểm tra |

Đã kiểm thử: toàn bộ trên dữ liệu giả lập (Python 3.11 & 3.13); `decode.py` trên một model Zipformer sherpa-onnx thật
(int8/fp32, greedy/beam, hotwords); `ft/*` trên lhotse 1.33 và icefall commit f8c8578 (12/11/2026).
Converter: đã chạy trên dataset giả dựng theo schema đọc từ trang dataset (14/11/2026), CHƯA chạy trên dữ liệu thật —
chạy `--limit 20` trước để bắt lệch schema.
Chưa chạy: huấn luyện thật (cần GPU + k2) và model Gipformer thật (không tải được từ môi trường build).
