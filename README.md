# gipbench: bench code-switching Việt–Anh cho Gipformer 1.5

Ba pipeline độc lập, nói chuyện với nhau chỉ qua file có hợp đồng. Đổi hướng ở một pipeline (engine TTS, cách FT, model
ngoài, metric) không phải sửa hai pipeline còn lại. Kiến trúc: `docs/ARCHITECTURE.md`. Lệnh: `docs/RUNBOOK.md`.

```
data/dl ─► [ DATASET ] ─► data/releases/{bench,catalog,pool,dev}/   (bất biến, sha256)
                              │ pool, dev, catalog        │ bench, catalog
                              ▼                           ▼
                         [ MODEL ] ─► artifact (onnx + model_card.json) ─► configs/bench/systems/<system>.yaml
                                                                                  │
                         [ BENCH ] ◄──────────────────────────────────────────────┘ (hoặc hyps.jsonl của model ngoài)
                              └─► results/<bench>/<system>/ ─► leaderboard
```

| Thư mục | Nội dung |
| --- | --- |
| `src/gipbench/core/` | Dùng chung: chuẩn hoá text, căn chỉnh, metric (`scoring`), chống rò rỉ, gắn tag, release, đường dẫn |
| `src/gipbench/infer/` | Engine ASR -> `hyps.jsonl`: sherpa-onnx (`decode`), faster-whisper / Qwen3-ASR (`asr_hyps`), cache |
| `src/gipbench/dataset/` | `ingest` (converter), `label` (duyệt, wordlist, domain), `pseudolabel` (vss_cs, đồng thuận, distill), `synth` (TTS), `derive` (confusable), `release` (bench, dev, đăng ký release, validate), `report` (EDA dữ liệu) |
| `src/gipbench/model/` | `recipe` (experiment YAML, curate, tách nhóm, mix_plan), `features` (Lhotse), `train` (icefall + watchdog), `select` (chọn checkpoint, WiSE-FT), `export`, `systems` (hotwords, tokenizer), `run.py` |
| `src/gipbench/bench/` | `run.py` (system -> decode -> chấm), `score`, `leaderboard`, `report` (EDA bench, bảng) |
| `configs/` | `paths.example.yaml`; `dataset/` (quy ước gắn tag, TTS, `releases.yaml`); `model/experiments/`; `bench/` (`systems/`, `rules.yaml`) |
| `tests/` | `run_tests.sh` (mọi test, CPU), `test_boundaries.py` (luật import), `golden/` (so output với code trước refactor) |
| `docs/` | `ARCHITECTURE.md`, `CONTRACTS.md` (định dạng file), `RUNBOOK.md` (lệnh), `DECISIONS.md`, `STATUS.md`, `history/` |

## Cài đặt

```bash
pip install -e .                       # venv tạo bằng uv không có pip: ghi đường dẫn src/ vào <venv>/lib/python3.11/site-packages/gipbench_src.pth
cp configs/paths.example.yaml configs/paths.yaml   # sửa đường dẫn exp / icefall / venv của máy này (không có file thì dùng mặc định)
bash tests/run_tests.sh                # mọi test trên dữ liệu giả, ~2 phút CPU
```

Ba venv trong WSL: `~/.venvs/gipbench` (CPU: dataset, bench, decode), `~/.venvs/gipbench-ft` (torch, k2, lhotse, icefall:
`scripts/setup_ft_env.sh`), `~/.venvs/gipbench-asr` (faster-whisper, Qwen3-ASR: `requirements-asr.txt`).

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

**Nguồn dữ liệu** (quyết định sau Phase 0b, 15/11/2026): B1 = G-Group (vi-asr-tech/edu/finance/pubadmin-test) + dữ liệu
tự thu. CS thật cho train = ViMedCSS train + `vss_cs` (VietSuperSpeech, lọc đồng thuận) + dữ liệu tự có. SEA
Code-Switching: tùy chọn, không dùng (`gipbench.dataset.ingest.sea` còn giữ).
