# REFACTOR PLAN: tách gipbench thành 3 pipeline độc lập (dataset · model · benchmark)

Tài liệu giao việc cho model implement. Đọc hết trước khi sửa bất kỳ file nào. Ngày lập: 16/11/2026, trên commit `1570a42`.

## 0. Mục tiêu và luật cứng

**Mục tiêu.** Khi đổi hướng tiếp cận ở một phần (vd đổi engine TTS, đổi cách FT, thêm model ngoài, đổi metric), chỉ phải
sửa đúng pipeline đó. Hai pipeline còn lại không phải sửa code và không phải chạy lại.

**Luật cứng cho người implement (vi phạm thì dừng lại và hỏi):**

1. **Không đổi hành vi.** Đây là refactor cấu trúc. Metric, ngưỡng, giá trị mặc định của mọi tham số, seed, thứ tự lọc,
   định dạng output phải giữ nguyên. Kiểm bằng golden test ở Phase 0 (so từng byte).
2. **Không động vào dữ liệu.** Không xoá, sửa, di chuyển bất cứ thứ gì trong `data/`, `results/`, `models/`, `exp/`,
   `C:\Users\goose\gipbench_exp\`. Chỉ ĐỌC để chạy golden test. Output tạm ghi vào thư mục scratch ngoài repo.
3. **Không chạy GPU, không train, không decode bench đầy đủ, không tải dữ liệu.** Mọi kiểm thử chạy trên CPU, dùng dữ liệu đồ
   chơi hoặc hyps đã có.
4. **Mỗi phase là một hoặc vài commit riêng.** Dùng `git mv` để giữ lịch sử. Không gộp việc chuyển file với việc sửa logic
   trong cùng một commit.
5. Python: venv 3.11 trong WSL. Cú pháp phải chạy được trên 3.10 (không dùng `tomllib`).
6. Không tự quyết các mục ở **§10 Câu hỏi mở**. Gặp chỗ không rõ thì làm theo cách giữ nguyên hành vi và ghi lại.
7. Không thêm framework (Hydra, DVC, Prefect…). Chỉ thêm `PyYAML` để đọc config YAML.

---

## 1. Chẩn đoán hiện trạng (vì sao repo "loạn")

| # | Vấn đề | Bằng chứng |
| --- | --- | --- |
| 1 | **Phụ thuộc ngược chiều:** thư viện chung `lib/` import từ `converters/` | `lib/agreement.py:214`, `lib/term_decisions.py:94-96`, `lib/tts_backends.py:211` đều `from converters.common import ...` |
| 2 | **Pipeline dữ liệu import code FT** | `converters/conv_consensus.py:54` `from ft.curate_train import pick_diverse` |
| 3 | **Tool import lẫn nhau giữa các tầng** | `tools/cluster_speakers.py:21` import `converters.conv_consensus`; `leaderboard.py:31` import `eda.CSS`; `ft/select_ckpt.py:26` và `tools/eda_current.py:29` import `score`; `tools/eda_current.py:30` import `tools.eda_lineage` |
| 4 | **`tools/` chứa đủ thứ:** 45 file trộn mọi tầng (lọc VSS, TTS, duyệt, EDA, test, backup, cài môi trường) | `ls tools/` |
| 5 | **Dữ liệu không có phiên bản.** Thí nghiệm trỏ thẳng vào file trung gian. Nhiều biến thể cùng tồn tại mà không có metadata | `data/raw/vss_full/` có khoảng 30 file `vss_cs_*_0d_t3`, `_relabel_0d`, `_agreed_a1_f0.5`…; `data/raw/_proxy/` là "bench tạm" vẫn đang dùng để chặn rò rỉ |
| 6 | **Thí nghiệm được tạo bằng cách chép shell.** `ft/run_e3_v1.sh` và `ft/run_e3_v1b.sh` chỉ khác 1 tham số (`--max-domain-share`) nhưng là 2 file 220 dòng | `diff ft/run_e3_v1.sh ft/run_e3_v1b.sh` |
| 7 | **Logic nằm trong heredoc Python của shell**, không test được | `ft/run_e3_v1b.sh` stage 0 (dựng `dev_cs`, đổi audio sang đường dẫn tuyệt đối), stage 2 (tách nhóm theo `train_mix.tsv`), stage 4 và 6 (`python -c` đọc JSON) |
| 8 | **Đường dẫn gắn với một máy** | `G=/mnt/d/code/ZAIC/gipbench`, `EXP=/mnt/c/Users/goose/...`, `$HOME/.venvs/...` trong `ft/run_e3_v1*.sh`; manifest E3 ghi audio tuyệt đối `/mnt/d/...` |
| 9 | **Đường dẫn dữ liệu cứng trong thư viện** | `lib/leak.py:15` `DEFAULT_BLOCK_FILES = data/raw/consensus/leak_block.txt`; `converters/common.py:80` `VI_WORD_FILES` |
| 10 | **Model pipeline phụ thuộc bench:** FT dùng catalog của bench nháp (`catalog_v0_draft.tsv` = catalog ViMedCSS + span lấy từ chính bench) cho curate, to_lhotse và select_ckpt | `ft/run_e3_v1b.sh`, biến `CATALOG` |
| 11 | **Tên thư mục gây hiểu nhầm:** `data/dl/` mới là dữ liệu gốc. `data/raw/` thực ra chứa output của converter và phiếu duyệt | docstring `conv_local.py` |
| 12 | **Quyết định nằm rải rác** ở commit message, STATUS và report | `reports/STATUS.md` §4 có 8 dòng ghi "chưa ghi vào `decisions.md`" |
| 13 | Hầu hết mọi file dùng `sys.path.insert(0, ...)` thay vì package | `grep -rn sys.path.insert` |
| 14 | Lọc chất lượng (đồng thuận ASR) nằm lẫn trong bước chọn dữ liệu train | bước 2 `--agreement-hyps` của `ft/curate_train.py` |

---

## 2. Kiến trúc đích

Ba pipeline và hai lớp dùng chung. Các phần nói chuyện với nhau **chỉ qua file có hợp đồng** (§3).

```
                    ┌──────────────────────── gipbench.core ────────────────────────┐
                    │ textnorm · align · manifest · leak · tagging · term_decisions │
                    │ domain · sampling · speaker · scoring · release · paths       │
                    │ report_html · svgcharts                                       │
                    └───────────────────────────────────────────────────────────────┘
                    ┌──────────────────────── gipbench.infer ───────────────────────┐
                    │ engines: sherpa_onnx · faster_whisper · qwen3_asr → hyps.jsonl│
                    │ cache (id + audio_sha256)                                     │
                    └───────────────────────────────────────────────────────────────┘
 data/sources ──► [ DATASET ] ──► data/releases/{bench,catalog,pool,dev}/<name>@<ver>/
                                         │                │
                                         │ pool, dev,     │ bench, catalog
                                         │ catalog        │
                                         ▼                ▼
                                    [ MODEL ] ──► artifacts/models/<system>/   (onnx + model_card.json)
                                                          │
                                                          ▼
                                    [ BENCH ] ◄── configs/bench/systems/<system>.yaml
                                         │        (model artifact HOẶC hyps.jsonl ngoài)
                                         ▼
                              results/<bench_ver>/<system>/  →  reports/bench/<bench_ver>/leaderboard.*
```

### 2.1 Luật phụ thuộc (bắt buộc, có test tự động ở Phase 7)

| Module | Được import | Cấm import |
| --- | --- | --- |
| `gipbench.core` | stdlib, numpy | mọi thứ khác trong `gipbench` |
| `gipbench.infer` | `core` | `dataset`, `model`, `bench` |
| `gipbench.dataset` | `core`, `infer` | `model`, `bench` |
| `gipbench.model` | `core`, `infer` | `dataset`, `bench` |
| `gipbench.bench` | `core`, `infer` | `dataset`, `model` |

Ngoài ra: không module nào dùng `sys.path.insert`. Không module nào dùng `subprocess` gọi script của pipeline khác. Gọi
`gipbench.infer` qua Python hoặc `python -m gipbench.infer...` thì được.

### 2.2 Ranh giới trách nhiệm

| Câu hỏi | Pipeline |
| --- | --- |
| Nhãn câu này có đúng không? Có rò rỉ hay test_only không? Có qua lọc đồng thuận / WER không? Thuật ngữ nào thuộc catalog? | **dataset** |
| Audio TTS của câu này có đọc đúng không (`tts_verify`)? | **dataset** |
| Bench / dev / catalog gồm những câu nào, phiên bản nào? | **dataset** |
| Lấy bao nhiêu từ mỗi pool? Trần domain / người nói / thuật ngữ, holdout domain, trọng số trộn? | **model** (recipe) |
| Đặc trưng, speed perturb, mô phỏng 8 kHz, freeze, LR, chọn checkpoint trên dev, WiSE-FT, export? | **model** |
| Hotwords (catalog nào, score, distractor) | **model**, ghi vào system config; bench chỉ thực thi |
| Decode bench, chấm, CI bootstrap, so cặp, luật chọn, leaderboard, EDA bench | **bench** |
| Công thức metric (`score_utt`, `agg`, …) | **core.scoring**: model dùng để chọn checkpoint trên dev, bench dùng để chấm, nên hai bên phải chạy cùng một code |

Quy tắc phân định: **lọc chất lượng thuộc dataset; chọn và cân bằng dữ liệu thuộc model.**

---

## 3. Hợp đồng giữa các pipeline

Giữ nguyên các định dạng đang có trong `SCHEMA.md`: manifest JSONL, catalog TSV, `hyps.jsonl`, `utt_scores.jsonl`,
`decode_summary.json`, CutSet Lhotse. Thêm 4 hợp đồng mới dưới đây. Viết tất cả vào `docs/CONTRACTS.md` (gộp `SCHEMA.md`
vào đó; `SCHEMA.md` chỉ còn 1 dòng trỏ sang).

### 3.1 Data release: `data/releases/<kind>/<name>@<version>/`

`kind` là một trong `bench`, `catalog`, `pool`, `dev`. Thư mục release **không bao giờ bị sửa**: muốn đổi thì ra version mới.

```
data/releases/bench/v0-draft/          (riêng bench và catalog: tên = version)
  manifest.jsonl
  release.json
data/releases/catalog/v0-draft/
  catalog.tsv
  release.json
data/releases/pool/vss_cs@2026-11-16/
  manifest.jsonl
  release.json
```

`release.json`:

```json
{
  "kind": "pool", "name": "vss_cs", "version": "2026-11-16",
  "created": "2026-11-16T10:00:00+07:00", "git_commit": "1570a42",
  "builder": "gipbench.dataset.release.build_pool", "builder_args": {"...": "..."},
  "files": {"manifest.jsonl": {"sha256": "...", "rows": 1657, "hours": 6.09}},
  "parents": [{"path": "data/interim/vss_full/vss_cs_clean_dom.jsonl", "sha256": "..."}],
  "catalog": "catalog/v0-draft",
  "stats": {"by_split": {}, "by_dataset": {}, "by_domain": {}, "cs_rows": 0},
  "flags": {"train_only": true, "label_source": "pseudo:Zipformer-30M-RNNT-6000h", "synthetic": false},
  "notes": "đã lọc đồng thuận fw large-v3 + qwen3-asr, WER tiếng Việt <= 0.15"
}
```

Quy định:
- Đường dẫn audio trong manifest của release là **đường dẫn tương đối theo file manifest**, đúng như `SCHEMA.md` đang ghi.
  Cấm đường dẫn tuyệt đối. Release builder ghi đường dẫn tương đối tới audio nằm ở `data/interim/...`, không copy audio.
- Bench release phải ghi rõ catalog mà nó dùng. Nếu catalog có span lấy từ chính bench thì ghi
  `"flags": {"catalog_from_bench": true}` để leaderboard in cảnh báo "hotwords = cận trên lạc quan".
- Tham chiếu release bằng chuỗi `"<kind>/<name>@<version>"` (riêng bench/catalog: `"bench/v0-draft"`). Chỉ có một hàm phân
  giải: `gipbench.core.release.resolve(ref) -> Release(path, manifest, catalog, meta)`. Hàm này kiểm sha256 khi đọc.

### 3.2 Recipe + experiment: `configs/model/experiments/<exp>.yaml`

File này thay `ft/run_e3_v1.sh` / `ft/run_e3_v1b.sh`. Ví dụ dưới đây phải **tương đương đúng** `run_e3_v1b.sh` hiện tại:

```yaml
name: e3_v1b
base_model: models/gip15                 # có model.pt, bpe.model, tokens.txt
catalog: catalog/v0-draft                # dùng cho trần theo thuật ngữ và chuẩn hoá
leak_guard: [bench/v0-draft]             # CHỈ dùng để chặn rò rỉ, không đọc gì khác từ bench
dev: {cs: dev/dev_cs@e3_v1, vi: dev/dev_vi@e3_v1}
pools:                                    # group -> pool release + trọng số (thay configs/train_mix.tsv)
  cs_real:      {weight: 0.20, refs: [pool/vimedcss_train@full]}
  cs_pseudo:    {weight: 0.10, refs: [pool/vss_cs@e3_v1]}
  synth:        {weight: 0.20, refs: []}           # chưa có TTS: nhóm rỗng bị bỏ qua, giống hiện tại
  replay_human: {weight: 0.25, refs: [pool/vivos_train@e3_v1, pool/cv_train@v1]}
  distill:      {weight: 0.20, refs: [pool/distill@e3_v1]}
  confusable:   {weight: 0.05, refs: [pool/confusable_train@e3_v1]}
curate:                                   # = tham số ft/curate_train.py hiện tại
  holdout_domain: [finance]
  max_domain_share: 0.50
  cap_exempt: [other]
  max_per_term: 50
  max_speaker_share: 0.02
  max_synthetic_share: 0.5
  min_speakers_per_term: 3
  seed: 0
features: {speed_perturb: true, phone_sim: 0.3, on_the_fly: true}
mix: {epochs: 6, epoch_hours: 160}
train:
  icefall: ${ICEFALL}
  lr: 0.002
  max_duration: 150
  freeze: "encoder_embed.,encoder.encoders.0.,encoder.encoders.1."
  model_args: "--use-transducer 1 --use-ctc 0"
  drop_ctc: true
  workers: 2
  buckets: 10
  watchdog: {min_free_disk_gb: 3, min_ram_mb: 1024, hang_s: 600}
select:
  max_vi_delta: <giữ đúng mặc định hiện tại của ft/select_ckpt.py>
  wise_alphas: [0.3, 0.5, 0.7, 1.0]
export: {int8: true}
output: {exp_dir: ${GIPBENCH_EXP}/e3_v1b, work_dir: data/work/model/e3_v1b}
```

Các biến môi trường `MIX_*` hiện dùng để ghi đè trọng số vẫn được giữ, nhưng chuyển thành tham số ghi đè chung của
orchestrator: `--set pools.synth.weight=0.35`. Giữ alias `MIX_SYNTH`… thêm một thời gian và in cảnh báo deprecated.

### 3.3 Model artifact: `artifacts/models/<system>/`

```
artifacts/models/e3_v1b/
  onnx/encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt bpe.model
  model_card.json
```

`model_card.json` gồm: `system`; `base_model` (đường dẫn + sha256 của `model.pt`); `experiment` (đường dẫn yaml + sha256 của
nội dung sau khi phân giải); `releases_used` (mọi ref kèm sha256); `checkpoint_picked` (vd `wise_0.5`, `ck4000`, `base`);
`dev_metrics` (bảng của select_ckpt); `git_commit`; `created`. Model gốc Gipformer cũng có một artifact
(`artifacts/models/gip15/`, có thể là symlink hoặc file trỏ tới `models/gip15`) để bench xử lý mọi model như nhau.

`artifacts/` mặc định nằm ở `${GIPBENCH_EXP}` (hiện là `C:\Users\goose\gipbench_exp` vì D: thiếu chỗ). Không ghi cứng trong code.

### 3.4 System config: `configs/bench/systems/<system>.yaml`

Một "system" = một artifact + cách decode. Bench chỉ đọc file này.

```yaml
system: e7_v1b
model: artifacts/models/e3_v1b        # hoặc: hyps: path/to/hyps.jsonl  (model ngoài / hậu xử lý E8)
decode:
  engine: sherpa_onnx
  method: greedy_search
  threads: 16
  batch: 32
  hotwords: {catalog: catalog/v0-draft, exclude_b3: true, score: 1.5, distractors: null, distractors_n: 0}
```

Tham số decode phải khớp 1-1 với tham số đang có của `decode.py` (không thêm, không đổi mặc định).

---

## 4. Cây thư mục đích

```
gipbench/
├── pyproject.toml            # package "gipbench" (src layout); extras: [model] lhotse/torch, [asr] faster-whisper/qwen, [dev]
├── README.md                 # ngắn: cài đặt, 3 pipeline, link docs
├── src/gipbench/
│   ├── core/
│   │   ├── textnorm.py align.py manifest.py leak.py term_decisions.py domain.py svgcharts.py   (từ lib/)
│   │   ├── tagging.py        # auto_tag, is_vietnamese_token, AMBIG_EN, load_vi_words (từ converters/common.py)
│   │   ├── agreement_rules.py# hàm so span thuần của lib/agreement.py (xem §10 câu 1)
│   │   ├── sampling.py       # pick_diverse (từ ft/curate_train.py)
│   │   ├── speaker.py        # cluster_sessions (từ converters/conv_consensus.py)
│   │   ├── scoring.py        # score_utt, agg, bootstrap_many, freq_slice(s), term_weights, train_term_freq, SPAN_METRICS (từ score.py)
│   │   ├── report_html.py    # CSS + khung HTML dùng chung (từ eda.py)
│   │   ├── release.py        # resolve / write_release / sha256 / kiểm bất biến
│   │   └── paths.py          # đọc configs/paths.yaml + biến môi trường
│   ├── infer/
│   │   ├── sherpa_onnx.py    # logic decode.py (bỏ phần riêng của bench)
│   │   ├── asr_hyps.py       # faster-whisper, qwen3-asr (từ tools/asr_hyps.py)
│   │   ├── cache.py          # từ lib/hyp_cache.py
│   │   └── cache_cli.py      # từ tools/hyp_cache.py
│   ├── dataset/
│   │   ├── ingest/           # common.py (HF, Writer, save_audio), ggroup.py, vimedcss.py, local.py, consensus.py, sea.py
│   │   ├── label/            # wordlists.py, top_review.py, term_decisions_cli.py, eval_autotag.py, review_priority.py,
│   │   │                     # review_bench.html, b3_nonmed_terms.py, infer_domain.py, domain_report.py, cluster_speakers.py
│   │   ├── pseudolabel/      # agreement.py (phần còn lại của lib/agreement), agreement_filter.py, distill.py,
│   │   │   └── vss_cs/       # prefilter.py, run.py, pilot.sh, rebuild.sh
│   │   ├── synth/            # backends.py (lib/tts_backends), make_text.py, synth.py, verify.py
│   │   ├── derive/           # confusables.py
│   │   ├── release/          # build_bench.py, build_catalog.py, build_pool.py, build_dev.py, validate.py
│   │   └── report/           # inventory.py (eda_current), lineage.py, notebook.py, converter_check.py
│   ├── model/
│   │   ├── recipe/           # experiment.py (đọc yaml), curate.py, split_groups.py, mix_plan.py
│   │   ├── features/         # to_lhotse.py, compute_fbank.py, mix_cuts.py
│   │   ├── train/            # wrap_ckpt.py, patch_finetune.py, icefall_runner.py (vòng lặp + watchdog của stage 4)
│   │   ├── select/           # select_ckpt.py, interpolate_ckpt.py, learning_curve.py
│   │   ├── export/           # export_onnx.py, model_card.py
│   │   ├── systems/          # hotword_distractors.py (tools/make_distractors), tokenizer_check.py
│   │   └── run.py            # orchestrator theo stage
│   └── bench/
│       ├── run.py            # system yaml -> decode (infer) -> results/<bench_ver>/<system>/ -> score
│       ├── score.py          # CLI; logic nằm ở core.scoring
│       ├── leaderboard.py
│       └── report/           # eda.py, draft_tables.py
├── configs/
│   ├── paths.example.yaml    # data_root, exp_root (GIPBENCH_EXP), icefall, venv; paths.yaml là bản local, gitignore
│   ├── dataset/
│   │   ├── labeling/         # vi_abbrev.txt vi_abbrev_project.tsv vi_loanwords.txt en_names.tsv term_decisions.tsv
│   │   │                     # confusables.tsv domain_keywords.tsv catalog.example.tsv
│   │   ├── synth/            # tts_lexicon.tsv tts_templates.tsv voices.example.tsv
│   │   └── releases/         # <release>.yaml: công thức dựng từng release (nguồn, tham số builder)
│   ├── model/experiments/    # e3_v1.yaml e3_v1b.yaml
│   └── bench/
│       ├── rules.yaml        # ngưỡng leaderboard (main_metric, b5_margin, nwer_margin, max_rtf_ratio, ...)
│       └── systems/          # e0_baseline.yaml e1_hw1.0.yaml e1_hw1.5.yaml e1_hw2.0.yaml e3_v1.yaml e7_v1.yaml ...
├── tests/
│   ├── fixtures/make_toy_data.py
│   ├── core/ dataset/ model/ bench/      # các test_* hiện có, chia theo pipeline
│   ├── golden/                           # script so output trước/sau refactor (Phase 0)
│   ├── test_boundaries.py                # kiểm luật import §2.1
│   └── selftest.sh
├── scripts/                  # setup_ft_env.sh, backup.sh, migrate_data_layout.sh (tiện ích máy, không thuộc pipeline)
├── docs/
│   ├── ARCHITECTURE.md       # viết lại từ docs/research/03_kien_truc.md theo kiến trúc mới
│   ├── CONTRACTS.md          # SCHEMA.md + §3 ở trên
│   ├── RUNBOOK.md            # lệnh chạy 3 pipeline (thay phần lệnh của docs/research/04)
│   ├── DECISIONS.md          # từ reports/decisions.md + các quyết định "chưa ghi" trong STATUS §4
│   ├── STATUS.md             # từ reports/STATUS.md
│   ├── tts_pipeline.md  huong_dan_duyet_b1.md
│   ├── research/             # giữ nguyên
│   └── history/              # PROMPT_OPUS_IMPLEMENT.md, REFACTOR_PLAN.md (khi xong), run_e3*.sh, train_mix.tsv (chỉ tham khảo)
└── reports/
    ├── dataset/ model/ bench/          # output mới ghi vào đây
    └── history/                        # toàn bộ reports/*.md, *.csv, *.json hiện có (phase0..0d, t02, a4, v0_draft …)

data/                          (gitignore; KHÔNG di chuyển trong refactor, xem §7)
├── sources/   = data/dl hiện tại
├── interim/   = data/raw hiện tại
├── releases/  (mới)
└── work/model/<exp>/  = data/e3_v1, data/e3_v1b hiện tại
results/<bench_ver>/<system>/  (giữ như hiện tại: results/v0_draft/...)
```

---

## 5. Bảng chuyển file (toàn bộ 182 file đang được track)

Cột "Ghi chú" là phần phải sửa ngoài `git mv`. Với mọi file Python: bỏ `sys.path.insert` và `G = Path(__file__).parents[...]`,
đổi import sang `gipbench.<...>`, giữ `main()` + argparse, chạy được bằng `python -m gipbench.<...>`.

### 5.1 Gốc repo

| Cũ | Mới | Ghi chú |
| --- | --- | --- |
| `validate.py` | `dataset/release/validate.py` | đường dẫn block-file lấy từ tham số / `paths.yaml`, không cứng trong code |
| `eda.py` | `bench/report/eda.py` | `CSS` chuyển sang `core/report_html.py` |
| `decode.py` | `infer/sherpa_onnx.py` | `bench/run.py` gọi hàm này; giữ CLI `python -m gipbench.infer.sherpa_onnx` với **đúng** tham số cũ |
| `score.py` | `core/scoring.py` (hàm) + `bench/score.py` (CLI) | CLI giữ nguyên tham số và output |
| `leaderboard.py` | `bench/leaderboard.py` | ngưỡng mặc định đọc từ `configs/bench/rules.yaml`, giá trị = mặc định hiện tại |
| `SCHEMA.md` | `docs/CONTRACTS.md` | `SCHEMA.md` chỉ còn 1 dòng trỏ sang |
| `README.md` | viết lại cho ngắn | thay bảng "File / Việc" bằng sơ đồ 3 pipeline + link RUNBOOK |
| `requirements*.txt` | extras trong `pyproject.toml` | có thể giữ 2 file requirements làm bản lock tham khảo |

### 5.2 `lib/`

| Cũ | Mới | Ghi chú |
| --- | --- | --- |
| `lib/textnorm.py`, `align.py`, `manifest.py`, `svgcharts.py`, `domain.py` | `core/`, giữ tên | |
| `lib/leak.py` | `core/leak.py` | bỏ `DEFAULT_BLOCK_FILES` cứng; caller truyền danh sách block-file. **Giá trị mặc định cũ được giữ qua `core/paths.py`** để hành vi không đổi |
| `lib/term_decisions.py` | `core/term_decisions.py` | import `core.tagging` thay cho `converters.common` |
| `lib/agreement.py` | hàm so span thuần (`agree`, `vi_wer`, `pair_wer`, `has_span`, …) → `core/agreement_rules.py`; phần còn lại (`mine_hard`, `relabel_spans`, `HARD_STAGES`) → `dataset/pseudolabel/agreement.py` | import `core.tagging`. Xem §10 câu 1 |
| `lib/hyp_cache.py` | `infer/cache.py` | |
| `lib/tts_backends.py` | `dataset/synth/backends.py` | import `core.tagging` |

### 5.3 `converters/`

| Cũ | Mới | Ghi chú |
| --- | --- | --- |
| `converters/common.py` | tách làm hai: `core/tagging.py` (`auto_tag`, `is_vietnamese_token`, `AMBIG_EN`, `load_vi_words`, `VI_WORD_FILES`, `add_vi_words_arg`, `REVIEW_COLS`) và `dataset/ingest/common.py` (`Writer`, `base_parser`, `load_hf`, `iter_hf_parquet`, `safe`, `save_audio`) | `VI_WORD_FILES` trỏ tới `configs/dataset/labeling/` |
| `conv_ggroup.py`, `conv_vimedcss.py`, `conv_local.py`, `conv_consensus.py`, `conv_sea.py` | `dataset/ingest/{ggroup,vimedcss,local,consensus,sea}.py` | `consensus.py`: lấy `pick_diverse` từ `core.sampling`, chuyển `cluster_sessions` ra `core/speaker.py`. `sea.py`: dòng đầu docstring ghi "không dùng (decisions §11)" |
| `build_bench_v0.py` | `dataset/release/build_bench.py` | ghi thêm thư mục release + `release.json` (§3.1) **bên cạnh** output cũ; không đổi nội dung manifest |

### 5.4 `ft/`

| Cũ | Mới | Ghi chú |
| --- | --- | --- |
| `curate_train.py` | `model/recipe/curate.py` | `pick_diverse` → `core/sampling.py`; `agree` lấy từ `core.agreement_rules`. Bước 2 agreement: xem §10 câu 1 |
| `to_lhotse.py`, `compute_fbank.py`, `mix_cuts.py` | `model/features/` | |
| `wrap_ckpt.py`, `patch_finetune.py` | `model/train/` | |
| `select_ckpt.py`, `interpolate_ckpt.py`, `learning_curve.py` | `model/select/` | `select_ckpt` import `core.scoring` |
| `make_distill.py` | `dataset/pseudolabel/distill.py` | distill là một pool dữ liệu; hyps của teacher lấy qua `infer` |
| `prep_e3_v1.py` | tách làm hai: `dataset/release/build_dev.py` (tách dev theo người nói / video, lọc `duration < 0.5`, dựng `dev_cs` = 600 câu ViMedCSS validation + vss_cs held-out) và `dataset/release/build_pool.py` (pool distill, vss_cs train, vivos train, confusable train) | gộp cả heredoc stage 0 của `run_e3_v1b.sh` vào đây. Audio ghi đường dẫn **tương đối** thay cho `/mnt/d/...` tuyệt đối |
| `run_e3.sh`, `run_e3_v1.sh`, `run_e3_v1b.sh` | `docs/history/` | thay bằng `model/run.py` + `configs/model/experiments/e3_v1{,b}.yaml` |

### 5.5 `tools/`

| Cũ | Mới |
| --- | --- |
| `agreement_filter.py` | `dataset/pseudolabel/agreement_filter.py` |
| `vss_cs_prefilter.py`, `vss_cs_run.py`, `vss_cs_pilot.sh`, `rebuild_vss_cs.sh` | `dataset/pseudolabel/vss_cs/{prefilter.py,run.py,pilot.sh,rebuild.sh}` (`run.py` gọi `python -m gipbench.infer.asr_hyps`) |
| `asr_hyps.py` | `infer/asr_hyps.py` |
| `hyp_cache.py` | `infer/cache_cli.py` |
| `tts_make_text.py`, `tts_synth.py`, `tts_verify.py` | `dataset/synth/{make_text,synth,verify}.py` |
| `make_confusables.py` | `dataset/derive/confusables.py` |
| `build_wordlists.py`, `wordlist_top_review.py`, `term_decisions.py`, `eval_autotag.py`, `review_priority.py`, `review_bench.html`, `b3_nonmed_terms.py`, `infer_domain.py`, `domain_report.py`, `cluster_speakers.py` | `dataset/label/` (`term_decisions.py` đổi tên thành `term_decisions_cli.py` để không trùng với `core/term_decisions.py`) |
| `make_dev_split.py` | gộp vào `dataset/release/build_dev.py` |
| `converter_check.py`, `eda_current.py`, `eda_lineage.py`, `eda_notebook.py` | `dataset/report/{converter_check,inventory,lineage,notebook}.py` (`inventory` import `core.scoring.train_term_freq`) |
| `mix_plan.py` | `model/recipe/mix_plan.py` (đọc experiment yaml thay cho `train_mix.tsv`) |
| `make_distractors.py` | `model/systems/hotword_distractors.py` |
| `check_gipformer_tokenizer.py` | `model/systems/tokenizer_check.py` |
| `draft_tables.py` | `bench/report/draft_tables.py` (import `core.scoring`) |
| `make_toy_data.py` | `tests/fixtures/make_toy_data.py` |
| `test_converters.py`, `test_tts.py`, `test_hard_mining.py` | `tests/dataset/` |
| `test_span_metrics.py` | `tests/bench/` (phần metric thuần chuyển sang `tests/core/`) |
| `test_dev_tools.py` | tách sang `tests/model/` và `tests/bench/` theo thứ nó kiểm |
| `selftest.sh` | `tests/selftest.sh` |
| `setup_ft_env.sh`, `backup.sh` | `scripts/` |

Lưu ý: `tools/test_hard_mining.py:174` đọc `git show HEAD:{rel}` theo đường dẫn cũ, phải cập nhật `rel`.
`tools/test_converters.py:484` và `:996` gọi `rebuild_vss_cs.sh` và `backup.sh` theo đường dẫn cũ, phải cập nhật.

### 5.6 `configs/`, `docs/`, `reports/`, `examples/`

| Cũ | Mới |
| --- | --- |
| `configs/{vi_abbrev.txt, vi_abbrev_project.tsv, vi_loanwords.txt, en_names.tsv, term_decisions.tsv, confusables.tsv, domain_keywords.tsv, catalog.example.tsv}` | `configs/dataset/labeling/` |
| `configs/{tts_lexicon.tsv, tts_templates.tsv}` | `configs/dataset/synth/` |
| `configs/train_mix.tsv` | mục `pools` của `configs/model/experiments/e3_v1.yaml` và `e3_v1b.yaml`; file tsv cũ chuyển vào `docs/history/` |
| `docs/research/*` | giữ nguyên (kể cả `03_kien_truc.md`, coi là lịch sử; bản mới là `docs/ARCHITECTURE.md`) |
| `docs/PROMPT_OPUS_IMPLEMENT.md` | `docs/history/` |
| `reports/STATUS.md`, `reports/decisions.md` | `docs/STATUS.md`, `docs/DECISIONS.md` |
| mọi file khác trong `reports/` | `reports/history/` (giữ nguyên tên, kể cả `reports/eda_current/`) |
| `examples/*_GIA_LAP.html` | `docs/examples/` |

Mọi đường dẫn mặc định trong code đang trỏ tới `configs/...` hoặc `reports/...` phải cập nhật theo bảng này. Tìm bằng
`grep -rn "configs/\|reports/" src tests`.

---

## 6. Đưa logic ra khỏi shell (stage của `run_e3_v1b.sh` → module)

| Stage cũ | Module mới | Ghi chú |
| --- | --- | --- |
| 0, heredoc dựng `dev_cs`, `train_vimedcss_abs` | `dataset/release/build_dev.py`, `build_pool.py` | Dựng một lần thành release; experiment chỉ tham chiếu |
| 0, `validate.py` | stage `check` của `model/run.py`: gọi `core.leak` để kiểm mọi pool + dev so với `leak_guard` | Model không import `dataset.release.validate`; logic kiểm rò rỉ phải nằm ở `core.leak` để hai bên dùng chung |
| 1, `curate_train` | `model/recipe/curate.py` | tham số lấy từ `curate:` |
| 2, heredoc tách nhóm theo `train_mix.tsv` | `model/recipe/split_groups.py` | nhóm lấy từ `pools:` (mỗi pool ref thuộc đúng 1 group), thay cho việc đoán theo tiền tố `dataset` + `r.get("distill")`. **Golden check:** số câu / giờ mỗi nhóm phải khớp output cũ |
| 2, `to_lhotse` cho từng nhóm + dev | `model/features/to_lhotse.py` | |
| 3, vòng `while read` đọc `train_mix.tsv` | `run.py` gọi `model/features/mix_cuts.py` | |
| 4, vòng lặp finetune + watchdog + dọn checkpoint | `model/train/icefall_runner.py` | Viết lại bằng Python (`subprocess.Popen`, `/proc/meminfo`, `shutil.disk_usage`). Giữ **đúng** luật: kiểm mỗi 60 s; bỏ qua 3 phút đầu; RAM < 1024 MB 2 lần liên tiếp thì SIGINT rồi resume; disk < 3 GB thì dừng hẳn; log đứng ≥ 600 s sau 10 phút đầu thì TERM, chờ 20 s, KILL; lỗi 2 lần liên tiếp thì dừng; resume bằng `--do-finetune 0 --start-epoch 2 --num-epochs 1 --start-batch N`; dọn checkpoint nhưng giữ checkpoint ở bội của `EPS` và checkpoint mới nhất. Giữ lại comment về `--start-epoch` của icefall. **Không chạy thật.** Test bằng tiến trình giả (`tests/model/test_icefall_runner.py`: script giả in log / ngủ / thoát) |
| 5, `export_onnx`, `decode_dev` | `model/export/export_onnx.py` + `infer.sherpa_onnx` | decode dev gọi `infer`, không gọi bench |
| 6, `select_ckpt` + `python -c` đọc JSON + WiSE-FT | `model/select/` + `run.py` | giữ nguyên logic "nếu pick = base thì làm WiSE-FT trên checkpoint có CS-WER thấp nhất" |
| sau 6 | `model/export/model_card.py` | ghi `artifacts/models/<system>/` (§3.3) |
| 7, decode bench + score + leaderboard | **bỏ khỏi model pipeline.** Thay bằng: sinh `configs/bench/systems/e3_v1b.yaml`, `e7_v1b.yaml`, rồi người dùng chạy `python -m gipbench.bench.run` | Đây là điểm cắt chính giữa model và bench |

Orchestrator: `python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 0-6 [--set key=value ...] [--dry-run]`.
`--dry-run` in lệnh và tham số đã phân giải của từng stage mà không chạy. Dùng nó để so với `run_e3_v1b.sh` ở Phase 4.

Bench: `python -m gipbench.bench.run --bench bench/v0-draft --system configs/bench/systems/e7_v1b.yaml` ghi ra
`results/v0_draft/e7_v1b/{hyps.jsonl,decode_summary.json,utt_scores.jsonl,summary.json,slices.md}`.
Leaderboard: `python -m gipbench.bench.leaderboard --bench bench/v0-draft --systems e0_baseline e1_hw1.5 e3_v1b e7_v1b --baseline e0_baseline`.
Danh sách system truyền thẳng vào, thay cho việc copy thư mục sang `/tmp/lb_e3` như hiện nay.

---

## 7. Đường dẫn và môi trường

- `configs/paths.example.yaml` (track) và `configs/paths.yaml` (gitignore):

  ```yaml
  repo_root: .
  data_root: data
  exp_root: ${GIPBENCH_EXP:-/mnt/c/Users/goose/gipbench_exp}
  icefall: ${ICEFALL:-~/icefall}
  python: {cpu: ~/.venvs/gipbench/bin/python, ft: ~/.venvs/gipbench-ft/bin/python, asr: <venv asr>}
  leak_block_files: [data/interim/consensus/leak_block.txt]
  ```
  `core/paths.py` đọc file này. Không có file thì dùng giá trị mặc định (= hành vi hiện tại). Biến môi trường được ghi đè.
- **Không di chuyển `data/` trong refactor.** `core/paths.py` có bảng alias: `data/interim` → `data/raw`, `data/sources` →
  `data/dl`, `data/work/model/e3_v1` → `data/e3_v1`, áp dụng khi thư mục mới chưa tồn tại. Viết sẵn
  `scripts/migrate_data_layout.sh` (đổi tên thư mục bằng `mv`, có `--dry-run`) nhưng **không chạy**; người dùng tự chạy sau.
- Các release đầu tiên (Phase 3) được **đăng ký từ file đang có**, không dựng lại dữ liệu: ghi `release.json` + manifest với
  đường dẫn audio tương đối. Cụ thể: `bench/v0-draft` từ `data/bench_v0_draft/bench_v0_draft.jsonl`; `catalog/v0-draft` từ
  `catalog_v0_draft.tsv`; các pool / dev của e3_v1 từ `data/e3_v1/*.jsonl`. Đăng ký xong thì kiểm: số dòng, số giờ, sha256 nội
  dung (bỏ trường `audio`) khớp file gốc.
- `.gitignore`: thêm `configs/paths.yaml`, `artifacts/`, `data/work/`.

---

## 8. Các phase và tiêu chí nghiệm thu

Làm tuần tự. Mỗi phase xong phải có: (a) mọi test trong `tests/` đạt, (b) golden test đạt, (c) từ Phase 7 thì
`tests/test_boundaries.py` đạt (trước đó chạy ở chế độ chỉ báo cáo).

### Phase 0: lưới an toàn (chưa sửa code)

1. Chạy và lưu log (CPU, `CUDA_VISIBLE_DEVICES=""`): `bash tools/selftest.sh`, `python tools/test_converters.py`,
   `python tools/test_tts.py`, `python tools/test_span_metrics.py`, `python tools/test_dev_tools.py`,
   `python tools/test_hard_mining.py`. Ghi trạng thái từng test (đạt / đã hỏng từ trước) vào `tests/golden/BASELINE.md`.
   Test nào đã hỏng từ trước thì chỉ ghi lại, không sửa.
2. Viết `tests/golden/capture.py`, lưu output vào thư mục ngoài repo (tham số `--out`). Gồm:
   - `score.py` cho mọi `results/v0_draft/<system>/hyps.jsonl`. Copy hyps sang scratch trước, `--out` vào scratch,
     `--bench data/bench_v0_draft/bench_v0_draft.jsonl --catalog data/bench_v0_draft/catalog_v0_draft.tsv`.
   - `leaderboard.py` trên bản copy các thư mục kết quả đó, `--baseline e0_baseline`.
   - `validate.py` trên bench + train + dev của e3_v1b (y như stage 0 của `run_e3_v1b.sh`), lưu stdout.
   - `ft/curate_train.py` với đúng tham số stage 1 của `run_e3_v1b.sh`, `--out` vào scratch.
   - heredoc stage 2 (tách nhóm): số câu / số giờ mỗi nhóm.
   - `tools/mix_plan.py` trên output của curate.
   - `tools/draft_tables.py`, và `ft/select_ckpt.py` trên `results/e3_v1b_dev/` nếu đủ input.
   - `eda.py` trên bench nháp (so `eda_summary.json`).
3. Viết `tests/golden/compare.py`: chạy lại các bước trên bằng lệnh mới (bảng ánh xạ lệnh cũ → mới nằm trong file) và so
   từng byte. JSON thì so sau khi `sort_keys`. Được bỏ qua trường thời gian / đường dẫn tuyệt đối, nhưng phải liệt kê rõ trường
   nào bị bỏ qua.
4. Commit `tests/golden/` + `BASELINE.md`. Không đụng file nào khác.

**Nghiệm thu:** `capture.py` chạy xong dưới 15 phút CPU; `compare.py` chạy trên chính code cũ ra 0 khác biệt.

### Phase 1: package hoá + `core`

1. Tạo `pyproject.toml` (src layout, `pip install -e .`) và `src/gipbench/__init__.py`.
2. `git mv` `lib/*` theo §5.2; tách `converters/common.py` theo §5.3; tạo `core/sampling.py`, `core/speaker.py`,
   `core/scoring.py`, `core/report_html.py`, `core/agreement_rules.py`.
3. Sửa 7 phụ thuộc sai ở §1, hàng 1–3.
4. Bỏ `sys.path.insert` trong các file đã chuyển. File chưa chuyển (còn ở `tools/`, `ft/`, `converters/`) tạm import
   `gipbench.core` (đã cài editable nên chạy được).

**Nghiệm thu:** test + golden đạt. `grep -rn "from lib\.\|import lib\b" .` (trừ `.venv`, `docs/history`) ra 0 kết quả.

### Phase 2: `infer` + `bench`

1. Chuyển `decode.py`, `asr_hyps.py`, `hyp_cache` vào `infer/`; chuyển `score.py`, `leaderboard.py`, `eda.py`,
   `draft_tables.py` vào `bench/`.
2. Viết `bench/run.py` + `configs/bench/systems/*.yaml` cho mọi system đang có trong `results/v0_draft/`
   (`e0_baseline`, `e1_hw1.0`, `e1_hw1.5`, `e1_hw2.0`, `a4_trial`, `a4_trial_hw1.5`, `e3_v1`, `e7_v1`, và v1b nếu có).
   Tham số decode lấy **từ `*_decode.log` / `decode_summary.json`** của từng system, không đoán.
3. `configs/bench/rules.yaml` = các mặc định hiện tại của `leaderboard.py`.
4. `bench/run.py --hyps-only`: chỉ chấm lại từ hyps có sẵn, không decode. Dùng cho golden và cho việc đổi metric về sau.

**Nghiệm thu:** golden score + leaderboard khớp từng byte; `bench/run.py --system e1_hw1.5.yaml --dry-run` in đúng lệnh decode
như trong `results/v0_draft/e1_hw1.5_decode.log`.

### Phase 3: `dataset`

1. Chuyển `converters/*`, các tool dữ liệu (§5.5), `validate.py`, `make_distill.py`, TTS, confusables.
2. Viết `core/release.py` + `dataset/release/{build_bench,build_catalog,build_pool,build_dev}.py`.
3. Đăng ký các release đầu tiên từ file đang có (§7). Ghi công thức vào `configs/dataset/releases/*.yaml`.
4. `dataset/release/validate.py --release bench/v0-draft --against pool/... dev/...` cho kết quả như validate cũ.

**Nghiệm thu:** golden validate khớp; mọi release đã đăng ký có sha256 khớp; `test_converters.py`, `test_tts.py`,
`test_hard_mining.py` đạt ở vị trí mới.

### Phase 4: `model`

1. Chuyển phần còn lại của `ft/*` theo §5.4; viết `experiment.py`, `split_groups.py`, `icefall_runner.py`, `export_onnx.py`,
   `model_card.py`, `run.py` theo §6.
2. Viết `configs/model/experiments/e3_v1.yaml` và `e3_v1b.yaml`, tương đương **đúng** 2 file shell.
3. Test `icefall_runner` bằng tiến trình giả (không cần icefall, không cần GPU).

**Nghiệm thu:** golden curate + tách nhóm + mix_plan khớp; `run.py e3_v1b.yaml --dry-run --stages 0-6` in tham số khớp
từng dòng với `run_e3_v1b.sh` (bảng đối chiếu ghi vào `tests/golden/e3_v1b_dryrun.md`).

### Phase 5: config, đường dẫn, dọn shell

1. `configs/paths.example.yaml`, `core/paths.py`, alias thư mục (§7), `scripts/migrate_data_layout.sh` (không chạy).
2. `git mv` các shell cũ sang `docs/history/`; xoá `ft/`, `tools/`, `converters/`, `lib/` khi đã rỗng.
3. `grep -rn "/mnt/d\|/mnt/c\|C:\\\\Users\|goose" src tests configs` ra 0 kết quả (trừ `paths.example.yaml`).

### Phase 6: docs và reports

1. Viết `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/RUNBOOK.md`. RUNBOOK có lệnh chạy 3 pipeline; mỗi pipeline có mục
   "đổi hướng thì sửa ở đâu".
2. `docs/DECISIONS.md`: gộp `reports/decisions.md`, 8 dòng "chưa ghi" trong `reports/STATUS.md` §4, và mọi quyết định sau
   15/11 nằm trong commit message (`git log --since=2026-11-15`). Mỗi mục ghi: ngày, quyết định, lý do, file / commit.
3. Chuyển `reports/` → `reports/history/`; viết lại README cho ngắn.

### Phase 7: khoá ranh giới

1. `tests/test_boundaries.py`: duyệt AST mọi file trong `src/gipbench/`; kiểm luật §2.1; kiểm không còn `sys.path.insert`;
   kiểm không có `subprocess` gọi `.py` của pipeline khác. Vi phạm thì thoát mã 1.
2. `tests/selftest.sh` chạy theo thứ tự: boundaries → core → dataset → model (dry-run + runner giả) → bench (dữ liệu đồ chơi).
   Dưới 2 phút CPU.
3. Cập nhật `tests/golden/BASELINE.md` với kết quả cuối.

---

## 9. Kiểm tra thiết kế bằng kịch bản đổi hướng

Sau refactor, mỗi kịch bản dưới đây phải làm được **mà không sửa file ở cột "Không đụng"**. Đưa bảng này vào `docs/RUNBOOK.md`.

| Đổi hướng | Sửa / chạy | Không đụng |
| --- | --- | --- |
| Đổi engine TTS, đổi lexicon, nâng tỉ trọng TTS | `dataset/synth` → release mới `pool/tts@<ver>`; experiment yaml thêm ref và chỉnh `pools.synth.weight` | `bench/*`, `core/scoring`, các pool khác |
| Lọc vss_cs chặt hơn | `dataset/pseudolabel` → `pool/vss_cs@<ver mới>` | code model, bench |
| Đổi cách FT (LoRA, freeze khác, CTC, LR) | experiment yaml mới / `model/train` | dataset, bench |
| Bỏ FT, dùng Whisper / Qwen / API | `configs/bench/systems/<x>.yaml` với `hyps:` | dataset, model |
| Hậu xử lý sửa thuật ngữ (E8) | script ghi `hyps.jsonl` mới + system yaml | dataset, model, code bench |
| Thêm metric | `core/scoring` + `bench/leaderboard`; `bench.run --hyps-only` chấm lại mọi system | dataset, model; không phải decode lại |
| Bench v1 (sau khi duyệt) thay v0-draft | `dataset/release/build_bench` → `bench/v1`; giữ nguyên system yaml, chạy lại `bench.run --bench bench/v1` | model (chỉ đổi `leak_guard` sang `bench/v1`) |

---

## 10. Câu hỏi mở (người implement KHÔNG tự quyết; giữ hành vi cũ và ghi chú)

1. **Bước 2 `--agreement-hyps` của `curate_train`.** Theo §2.2, bước này nên chuyển hẳn sang dataset (pool đã được lọc sẵn).
   `run_e3_v1*.sh` không dùng tham số này nên chuyển đi cũng không làm đổi E3. Cách làm trong refactor: giữ tham số trong
   `model/recipe/curate.py`, lấy `agree` từ `core/agreement_rules.py` (model không được import `dataset`). Việc bỏ hẳn tham
   số khỏi model để người dùng chốt sau.
2. **Tách catalog khỏi bench** (`catalog/v0-draft` riêng): làm ở Phase 3 như mô tả. Còn việc dựng một catalog không chứa span
   của bench (để hotwords E1 hết là cận trên lạc quan) là **thay đổi phương pháp**, không làm trong refactor.
3. `conv_sea.py`: giữ lại (đánh dấu không dùng) hay xoá?
4. `data/raw/ggroup_l20`; `vimedcss`, `vimedcss_full`, `vimedcss_hard`; các biến thể `vss_cs_*` trong `data/raw/vss_full/`:
   biến thể nào được đăng ký thành release, biến thể nào bỏ đi. Refactor chỉ đăng ký những file mà `run_e3_v1b.sh` và
   `data/bench_v0_draft/build.log` thực sự dùng; phần còn lại liệt kê vào `docs/STATUS.md` để người dùng quyết.
5. Có chạy `scripts/migrate_data_layout.sh` (đổi `data/raw` → `data/interim`, `data/dl` → `data/sources`) không, và khi nào.
6. `reports/eda_current/*.csv` (đang được track, có vài file lớn): giữ trong git hay chuyển ra ngoài.

---

## 11. Gợi ý chia việc cho model implement

| Gói | Phase | Ghi chú |
| --- | --- | --- |
| A | 0 | Làm trước và làm một mình. Mọi gói sau phụ thuộc golden test |
| B | 1 | Làm một mình (đụng tới gần như mọi import) |
| C | 2 | Chạy song song với D được, sau khi B xong |
| D | 3 | Song song với C |
| E | 4 | Sau khi C + D xong (cần `infer` và release) |
| F | 5 + 6 + 7 | Làm cuối |

Mỗi gói kết thúc bằng một báo cáo gồm: danh sách commit, kết quả `tests/selftest.sh`, kết quả `tests/golden/compare.py`, các
điểm lệch hành vi (nếu có, kèm lý do), và các câu hỏi phát sinh. Sau mỗi gói nên có một lượt review độc lập (đọc diff, chạy lại
golden) trước khi sang gói tiếp theo.
