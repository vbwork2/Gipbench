# Kiến trúc: 3 pipeline (dataset · model · bench)

Cập nhật 16/11/2026, sau refactor (nhánh `refactor/pipelines`). Bản trước refactor: `docs/research/03_kien_truc.md`.

## Mục tiêu

Đổi hướng tiếp cận ở một phần (engine TTS, cách lọc nhãn giả, cách FT, model ngoài, metric, luật chọn) chỉ phải sửa đúng
pipeline đó. Hai pipeline còn lại không sửa code, không chạy lại.

## Các tầng và luật phụ thuộc

```
gipbench.core      textnorm · align · manifest · leak · tagging · term_decisions · domain · sampling · scoring ·
                   agreement_rules · release · paths · svgcharts ·
                   normprofiles (chuẩn hoá có phiên bản) · evalstats (bootstrap cụm / ghép cặp) · decontam · langspans
gipbench.infer     decode (sherpa-onnx) · asr_hyps (faster-whisper, Qwen3-ASR) · cache ·
                   adapters (Gipformer / VieGem / faster-whisper / Qwen3; model bất kỳ)   -> hyps.jsonl + run_meta.json
gipbench.dataset   ingest (+ evalsets: bộ test công khai) · label · pseudolabel · synth · derive ·
                   release (+ build_general, leak_audit, fetch, package) · report   -> data release, benchmark v1, gói share
gipbench.model     recipe · features · train · select · export · systems · run               -> artifact + system YAML
gipbench.bench     run · score · leaderboard · report ·
                   registry · transcribe · evaluate · compare (benchmark v1)        -> results, leaderboard, report.md
```

| Tầng | Được import | Cấm import |
| --- | --- | --- |
| `core` | stdlib, thư viện ngoài | mọi tầng khác |
| `infer` | `core` | `dataset`, `model`, `bench` |
| `dataset` | `core`, `infer` | `model`, `bench` |
| `model` | `core`, `infer` | `dataset`, `bench` |
| `bench` | `core`, `infer` | `dataset`, `model` |

`tests/test_boundaries.py` kiểm luật này bằng AST, gồm cả import nằm trong hàm và chuỗi `python -m gipbench.<tầng>`
truyền qua subprocess. Nó cũng cấm `sys.path.insert` và import module cũ (`lib`, `tools`, `ft`, `converters`). Code
dùng chung cho hai pipeline thì đưa xuống `core`, không import chéo. Ví dụ: công thức metric nằm ở `core.scoring` vì model
dùng nó để chọn checkpoint trên dev, còn bench dùng nó để chấm. Nhờ vậy hai bên không thể lệch nhau.

## Ranh giới trách nhiệm

| Câu hỏi | Pipeline |
| --- | --- |
| Nhãn đúng chưa, có rò rỉ / test_only không, có qua lọc đồng thuận / WER không, thuật ngữ nào thuộc catalog | dataset |
| Audio TTS có đọc đúng không (`synth.verify`) | dataset |
| Bench / dev / catalog / pool gồm những câu nào, phiên bản nào | dataset (release) |
| Lấy bao nhiêu từ mỗi pool, trần domain / người nói / thuật ngữ, holdout domain, trọng số trộn | model (experiment YAML) |
| Đặc trưng, speed perturb, mô phỏng 8 kHz, freeze, LR, chọn checkpoint trên dev, WiSE-FT, export | model |
| Hotwords (catalog nào, score, distractor) | model, ghi vào system YAML; bench chỉ thực thi |
| Decode bench, chấm, CI bootstrap, so cặp, luật chọn, leaderboard, EDA bench | bench |

Quy tắc phân định: **lọc chất lượng thuộc dataset; chọn và cân bằng dữ liệu thuộc model.**

## Hợp đồng giữa các pipeline

Định dạng chi tiết: `docs/CONTRACTS.md`.

| Hợp đồng | Ghi | Đọc |
| --- | --- | --- |
| Data release `data/releases/<kind>/<name>@<ver>/` (manifest + catalog + `release.json`, bất biến, sha256) | `gipbench.dataset.release.register` | model (`run.py` stage 0), bench (`run.py`) |
| Experiment YAML `configs/model/experiments/<exp>.yaml` | người làm thí nghiệm | `gipbench.model.run` |
| Model artifact `<exp_root>/artifacts/<exp>/` (onnx + `model_card.json`) | `gipbench.model.run` stage 6 | bench qua system YAML |
| System YAML `configs/bench/systems/<system>.yaml` (model hoặc hyps + tham số decode) | model stage 6 / người dùng | `gipbench.bench.run` |
| `hyps.jsonl`, `utt_scores.jsonl`, `decode_summary.json`, manifest JSONL, catalog TSV, CutSet Lhotse | như trước refactor | như trước refactor |
| Benchmark v1 `configs/bench/benchmarks/<tên>.yaml` + manifest khoá `manifest_sha256` (CONTRACTS mục 8) | `gipbench.dataset.release.build_general` | `gipbench.bench.{transcribe,evaluate}`; đăng ký thêm thành release `bench/<tên>-v1.0` cho `leak_guard` |
| `run_meta.json` cạnh `hyps.jsonl` (model, revision, inference mode, decode) | `gipbench.bench.transcribe` / người dùng | `gipbench.bench.evaluate` |
| Gói share (manifest không audio + datasheet + yaml khoá + SHA256SUMS) | `gipbench.dataset.release.package` | người nhận: `gipbench.dataset.release.fetch`, rồi `bench.evaluate` |

## Đổi hướng thì sửa ở đâu

| Đổi hướng | Sửa / chạy | Không đụng |
| --- | --- | --- |
| Đổi engine TTS, lexicon, tỉ trọng TTS | `dataset/synth` -> release mới `pool/tts@<ver>` (thêm vào `configs/dataset/releases.yaml`); experiment YAML thêm ref vào `pools.synth.refs`, chỉnh `weight` | bench, `core.scoring`, các pool khác |
| Lọc vss_cs chặt / lỏng hơn | `dataset/pseudolabel` -> `pool/vss_cs@<ver mới>`; experiment trỏ sang ver mới | code model, bench |
| Đổi cách FT (LR, freeze, CTC, số epoch, trần curate) | experiment YAML mới (hoặc `--set khoá=giá_trị`); code `model/train` nếu đổi thuật toán | dataset, bench |
| Bỏ FT, dùng Whisper / Qwen / API | system YAML với `hyps:` (hoặc adapter mới trong `infer`) | dataset, model |
| Hậu xử lý sửa thuật ngữ (E8) | script ghi `hyps.jsonl` mới + system YAML `hyps:` | dataset, model, code bench |
| Thêm metric | `core/scoring` + `bench/leaderboard`; `bench.run --hyps-only` chấm lại | dataset, model; không decode lại |
| Đổi luật chọn | `configs/bench/rules.yaml` | mọi thứ khác |
| Bench v1 thay v0-draft (đã làm 17/11/2026) | `dataset/release/build_general` -> `configs/bench/benchmarks/{general,code_switch,cs_ggroup,external}_v1.yaml` (khoá sha256) + release `bench/{general,code-switch,cs-ggroup,external}-v1.0`; chấm bằng `bench.transcribe` + `bench.evaluate` (hồ sơ `gipbench_strict_v1`); `bench.run` / `score.py` giữ cho v0-draft | model (chỉ đổi `leak_guard` khi train lại) |
| Thêm một bộ test công khai | `dataset/ingest/evalsets.SPECS` (repo, revision, sha256 file) + `configs/bench/datasets.yaml`; benchmark mới hoặc version mới | code bench, model |

## Thư mục

```
src/gipbench/{core,infer,dataset,model,bench}/
configs/paths.example.yaml        configs/paths.yaml (local, gitignore)
configs/dataset/{labeling,synth}/  configs/dataset/releases.yaml
configs/model/experiments/         configs/bench/{systems/,rules.yaml}
data/raw/        output converter + phiếu duyệt (tên cũ; nên là data/interim)
data/dl/         dataset gốc tải về (tên cũ; nên là data/sources)
data/releases/   release (gitignore)
data/<exp>/      thư mục làm việc của experiment (output.work_dir)
results/<bench>/<system>/   results/<exp>_dev/
<exp_root>/<exp>/  checkpoint + ứng viên        <exp_root>/artifacts/<exp>/  artifact
reports/{dataset,model,bench}/  reports/history/ (báo cáo trước refactor)
scripts/   tiện ích máy (setup_ft_env, backup, chuỗi lệnh vss_cs)
tests/     run_tests.sh, test_boundaries.py, golden/, {core,dataset,model,bench,integration}/
```
