# Runbook: lệnh chạy 3 pipeline

Chạy trong WSL tại gốc repo. `source ~/.venvs/gipbench/bin/activate && export PYTHONIOENCODING=utf-8`. Mọi script chạy bằng
`python -m gipbench.<...>`, xem `--help` của từng module. Kiến trúc: `docs/ARCHITECTURE.md`. Định dạng: `docs/CONTRACTS.md`.

## 0. Kiểm tra

```bash
bash tests/run_tests.sh                       # mọi test trên dữ liệu giả (~2 phút CPU), gồm luật import giữa pipeline
python tests/golden/capture.py --layout new --out /tmp/g && python tests/golden/compare.py <mốc> /tmp/g   # dữ liệu thật
```

## 1. Dataset: nguồn -> release

| Việc | Lệnh |
| --- | --- |
| Converter | `python -m gipbench.dataset.ingest.ggroup …`, `.vimedcss`, `.local {vivos,commonvoice,vietsuperspeech}`, `.consensus` |
| Gắn tag / wordlist / quyết định thuật ngữ | `gipbench.dataset.label.{wordlists,top_review,term_decisions_cli,eval_autotag}` |
| Duyệt bench | `src/gipbench/dataset/label/review_bench.html`, ưu tiên duyệt: `gipbench.dataset.label.review_priority` (`docs/huong_dan_duyet_b1.md`) |
| Domain, cụm người nói | `gipbench.dataset.label.{infer_domain,domain_report,cluster_speakers}` |
| vss_cs (nhãn giả + đồng thuận ASR) | `bash scripts/dataset/vss_cs_pilot.sh`, `python -m gipbench.dataset.pseudolabel.vss_cs_run …`, dựng lại từ cache: `bash scripts/dataset/rebuild_vss_cs.sh --bench … --catalog …` |
| Hyps ASR (venv ASR) | `~/.venvs/gipbench-asr/bin/python -m gipbench.infer.asr_hyps --engine faster-whisper …` |
| Distill / confusable | `gipbench.dataset.pseudolabel.distill`, `gipbench.dataset.derive.confusables` |
| TTS | `gipbench.dataset.synth.{make_text,synth,verify}` (`docs/tts_pipeline.md`); `make_text` BẮT BUỘC `--reserved-terms` và từ chối catalog `from_bench`; ra thêm `tts_text.txt` cho `leak_gate.text_files` |
| Thuật ngữ giữ riêng (FT-unseen) | đề xuất: `python -m gipbench.dataset.release.propose_reserved_terms [--fraction 0.5]` -> `configs/dataset/reserved_terms_v1.proposal.txt`; người dùng duyệt / sửa rồi đổi tên thành `reserved_terms_v1.txt` (chốt TRƯỚC khi dựng dữ liệu FT) |
| Catalog train (không có span bench) | `python -m gipbench.dataset.release.build_train_catalog --source pool/vimedcss_train@v2:text_tagged_orig --source pool/vss_cs@e3_v1:text_tagged --reserved-terms configs/dataset/reserved_terms_v1.txt --out data/catalogs/train_v1/catalog.tsv` (`--preview` chỉ in số liệu), rồi thêm `catalog/train-v1` (`flags: {from_bench: false}`) vào `releases.yaml` và `register`. **Đã dựng 18/11/2026**: `reserved_terms_v1.txt` chốt (820), `catalog/train-v1` 727 thuật ngữ |
| Bench từ phiếu duyệt | `python -m gipbench.dataset.release.build_bench …` |
| Dev / pool cho FT | `gipbench.dataset.release.{prep_e3_v1,build_dev_cs,make_dev_split}` |
| Kiểm rò rỉ bench ↔ train | `python -m gipbench.dataset.release.validate --bench … --catalog … --train … --dev …` |
| **Phát hành release** | thêm mục vào `configs/dataset/releases.yaml`, rồi `python -m gipbench.dataset.release.register <ref>` (hoặc `--all`); `--list`, `--verify` |
| EDA dữ liệu hiện có | `python -m gipbench.dataset.report.inventory` -> `reports/dataset/eda_current/` |

Release đã có (16/11/2026): `catalog/v0-draft`, `bench/v0-draft`, `pool/{vimedcss_train@full, vss_cs@e3_v1,
vivos_train@e3_v1, cv_train@v1, distill@e3_v1, confusable_train@e3_v1}`, `dev/{dev_cs@e3_v1, dev_vi@e3_v1}`.
Thêm 17/11/2026: `catalog/vimedcss-v1.0`, `bench/{general-v1.0, code-switch-v1.0, cs-ggroup-v1.0, external-v1.0}` (bench khoá: cổng rò rỉ
của `model.run` đọc danh sách này từ `releases.yaml`, nên phải `register` trước khi chạy experiment).

### Dev độc lập

Chuẩn bị manifest nguồn nói tự nhiên có nhãn và id người nói ổn định. Ví dụ dưới đây kiểm trước phần ViMedCSS.
Bổ sung manifest CS ngoài y khoa vào `--candidates` và liệt kê tất cả pool train liên quan trong `--exclude-from`.
Công cụ tự bổ sung mọi bench khoá. Kiểm tra trùng audio dùng SHA-256 có sẵn trong manifest.

```bash
python -m gipbench.dataset.release.make_dev_split \
  --kind cs --candidates data/raw/vimedcss_full/dev_vimedcss.jsonl \
  --exclude-from pool/vimedcss_train@full pool/vimedcss_train@v2 \
  --out-dev data/tmp/dev_vimedcss_filtered.jsonl
```

Đọc `<out-dev>.report.json` để xem số câu, domain, người nói và phát hiện critical. ID video ViMedCSS chỉ được tính là
đơn vị nguồn. Dev CS đạt khi có ít nhất 1.500 câu, 4 domain và 100 người nói. Dùng `--kind vi` với nguồn thuần Việt nói tự
nhiên, ngưỡng ít nhất 1 giờ. Thêm `--release dev/dev_v2_cs@1` hoặc `--release dev/dev_v2_vi@1` để đăng ký release đủ điều
kiện, có audio và kèm `flags.independent` / `flags.exclusion_sets`. Sau đó đối chiếu `ck4132` với model gốc trên dev_v2.

**Dev tạm (18/11, đang dùng cho `e4_x0`).** Chưa có nguồn đạt ngưỡng trên (TODO 2e), nên dev tạm được cắt theo video từ
`vss_cs@v3` (CS) và `distill@v4` (VI). Pool train tương ứng đã bỏ các video này (`vss_cs@v4`, `distill@v5`). Nhãn giả
phải qua người nghe. Dev tạm chỉ dùng cho smoke test / X0 / chọn checkpoint. Nó không đại diện 4 domain, đơn vị người nói
là video, và **không được chấm checkpoint đã train trên `carved_from`** (E3, `ck4132`).

```bash
# đã chạy: data/listen/dev_interim_{cs,vi}/ (heldout.jsonl, candidates.jsonl, listen.csv, audio/)
python -m gipbench.dataset.release.holdout_dev select --pool pool/vss_cs@v3 --kind cs --frac 0.3 \
  --reuse data/listen/vss_cs_v1/listen_A_reviewed.csv --out data/listen/dev_interim_cs
python -m gipbench.dataset.release.holdout_dev select --pool pool/distill@v4 --kind vi --frac 0.2 --out data/listen/dev_interim_vi
# 👤 nghe listen.csv bằng review_bench.html, xuất thành listen_reviewed.csv trong cùng thư mục, rồi:
python -m gipbench.dataset.release.holdout_dev apply --dir data/listen/dev_interim_cs
python -m gipbench.dataset.release.holdout_dev apply --dir data/listen/dev_interim_vi
POOLS="pool/vimedcss_train@v4 pool/vivos_train@v3 pool/confusable_train@v3 pool/cv_other@v3 pool/vss_cs@v4 pool/distill@v5"
python -m gipbench.dataset.release.make_dev_split --candidates data/listen/dev_interim_cs/verified.jsonl --kind cs \
  --interim --carved-from pool/vss_cs@e3_v1 pool/vss_cs@v3 --exclude-from $POOLS \
  --release dev/dev_interim_cs@1 --out-dev data/dev/dev_interim_cs.jsonl
python -m gipbench.dataset.release.make_dev_split --candidates data/listen/dev_interim_vi/verified.jsonl --kind vi \
  --interim --carved-from pool/distill@e3_v1 pool/distill@v3 pool/distill@v4 --exclude-from $POOLS \
  --release dev/dev_interim_vi@1 --out-dev data/dev/dev_interim_vi.jsonl
```

Trước khi nghe (đã chạy 18/11): `holdout_dev triage --hyps <qwen3> <whisper>` nhận luôn câu mà nhãn khớp từng token với
mọi ASR khác họ (VI chỉ có 6 / 365 câu nên giữ lại cả để kiểm mù). `holdout_dev hint --hyps qwen3=… whisper=…` ghi cột
`hint`, trang duyệt hiện cột này dưới câu, từ lệch bọc trong ⟦ ⟧. Hyp chạy trong WSL: `data/listen/dev_interim_{cs,vi}/run_asr.sh`.
Phiếu gộp để nghe một lượt: `data/listen/pending_all/listen.csv`. Nghe phiếu gộp thì chạy `apply` cho từng thư mục với
`--review "$(pwd)/data/listen/pending_all/listen_reviewed.csv"` (mỗi `apply` chỉ lấy id của mình).

`--interim` đòi mọi câu có `label_source: human:*` (hoặc `auto:asr_agree` từ `triage`). Ngưỡng: CS ≥ 300 câu, VI ≥ 1 giờ. Release gắn
`interim: true, representative: false, speaker_unit: video, carved_from`.

## 2. Model: experiment -> artifact

```bash
python -m gipbench.model.run configs/model/experiments/e3_v1b.yaml --stages 0-6 --dry-run    # xem lệnh từng stage
python -m gipbench.model.run configs/model/experiments/e3_v2.yaml --stages 0-3                # dữ liệu + mix
python -m gipbench.model.run configs/model/experiments/e3_v2.yaml --stages 4-6                # GPU: FT -> chọn -> artifact
python -m gipbench.model.run configs/model/experiments/e3_v2.yaml --stages 1-3 --set curate.max_domain_share=0.5
MIX_SYNTH=0.35 python -m gipbench.model.run … --stages 3                                      # quét trọng số như trước
python -m gipbench.model.recipe.mix_plan --mix configs/model/experiments/e3_v2.yaml --manifest … --fit-cap
```

Stage 0 là **cổng rò rỉ đóng khi lỗi** (thoát mã 3 nếu còn critical chưa miễn trừ; xem `docs/CONTRACTS.md` §5 khoá `leak_gate`,
`catalog_use`). Xem riêng cổng, không chạm gì khác: `--stages 0 --set output.work_dir=data/work/model/_probe` rồi đọc
`<work_dir>/leak_report.md`. Catalog `from_bench` (vd `catalog/v0-draft`) bị từ chối: dùng `catalog/train-v1`, hoặc khai báo
`catalog_use: oracle_diagnostic` (kết quả không xếp hạng). `e3_v1` / `e3_v1b` / `e3_v2` giữ nguyên làm lịch sử nên bị từ chối.

Thí nghiệm mới = chép một file trong `configs/model/experiments/`, đổi `name`, `output`, rồi sửa tham số. Không chép
shell. Stage 6 ghi artifact (`output.artifact`) và các system YAML cho bench (`configs/bench/systems/<system>.yaml`).
Các file thuộc `docs/history/run_e3*.sh` chỉ để tham khảo, không chạy được với cấu trúc mới.

### 2b. Fine-tune trên Colab: gói dữ liệu qua Google Drive

Code lấy bằng `git clone` (token GitHub trong Colab Secrets); dữ liệu + model gốc đi qua Drive dưới dạng gói tar ~1 GB giữ
nguyên đường dẫn tương đối theo gốc repo. Không có audio bench trong gói (chỉ manifest text cho cổng rò rỉ).

```bash
# máy local: đóng gói (chạy lại được, chỉ gói file mới / đổi), rồi chép thư mục gói vào thư mục Drive đồng bộ
python scripts/colab/pack_ft_bundle.py --out D:/gipbench_ft_bundle [--refs dev/<dev_v2>@v1 catalog/train-v1]
# Colab: sau khi mount Drive và clone repo vào /content/gipbench
python scripts/colab/unpack_ft_bundle.py --bundle /content/drive/MyDrive/gipbench_ft --dest /content/gipbench
```

Mở `scripts/colab/colab_ft.ipynb` trên Colab GPU. Đặt `BRANCH` trỏ tới nhánh chứa cấu hình X0 và cấp quyền đọc repo bằng
Secret `GH_TOKEN`. Giữ `RUN_EXPERIMENT = False` để kiểm CUDA/k2, giải nén gói Drive và chạy dry-run.
Notebook chép `configs/paths.colab.yaml` thành `configs/paths.yaml`: data ở local, checkpoint / kết quả trên Drive;
ba Python dùng chung môi trường Colab (biến môi trường trỏ tới đường dẫn tuyệt đối của `python3`).

Gói ở `D:/gipbench_ft_bundle` (19/11 đã gộp shard 000–008 cùng một `bundle_index.json`; 008 = `pool/vivos_train@v3`
cho X0). Chép nguyên thư mục lên Drive `My Drive/gipbench_ft`. Shard 006 chứa hai manifest bench v1.1 đã khoá
và model ONNX INT8 cho baseline. Shard 007 chứa catalog train và file thuật ngữ đã chốt. Có thể chạy dry-run cục bộ trước
khi tải gói lên.

```bash
python -m gipbench.model.run configs/model/experiments/e4_x0.yaml --stages 0-6 --dry-run
python -m gipbench.model.run configs/model/experiments/e4_x0.yaml --stages 0-6 --seed 1 --dry-run
```

Lệnh đầu in cả hai seed. Lượt thật chạy lần lượt với `SEED = 0` và `SEED = 1`. Mỗi seed có thư mục làm việc, checkpoint,
kết quả dev và artifact riêng. Bật `RUN_EXPERIMENT` khi cell dry-run không còn báo đầu vào thiếu.
X0 không cần dev (`dev: null`, `select.rule: fixed_last`); catalog train và `reserved_terms_v1.txt` nằm ở shard 007.
Notebook chạy cổng rò rỉ trước stage 1–6. Watchdog dùng `/content` và chừa 5 GiB.

## 3. Bench: system -> kết quả -> leaderboard

```bash
python -m gipbench.bench.run --bench bench/v0-draft --system configs/bench/systems/e3_v2.yaml configs/bench/systems/e7_v2.yaml
python -m gipbench.bench.run --bench bench/v0-draft --system configs/bench/systems/*.yaml --hyps-only   # đổi metric: chấm lại
python -m gipbench.bench.leaderboard --results results/v0_draft --baseline e0_baseline --rules configs/bench/rules.yaml \
    --systems e0_baseline e1_hw1.5 e3_v1 e7_v1 e3_v2 e7_v2 --out reports/bench/v0_draft/leaderboard
python -m gipbench.bench.report.draft_tables --bench data/releases/bench/v0-draft/manifest.jsonl \
    --catalog data/releases/catalog/v0-draft/catalog.tsv --systems results/v0_draft/e0_baseline … --out reports/bench/…
python -m gipbench.bench.report.eda --bench data/releases/bench/v0-draft/manifest.jsonl --catalog … --out reports/bench/eda_v0_draft
```

Model ngoài / hậu xử lý: tạo system YAML với `hyps: <đường dẫn hyps.jsonl>` thay cho `model:`.

## 3b. Bench v1 (ASR tổng quát): benchmark khoá -> transcribe -> evaluate -> gói share

Giao thức: `docs/benchmark_protocol_v1.md`; metric: `docs/evaluation_metrics_v1.md`; datasheet: `docs/datasheet_bench_v1.md`.

```bash
export GIPBENCH_DATA=<data_root> GIPBENCH_MODELS=<models_root>
python -m gipbench.bench.registry                                          # kiểm datasets.yaml + benchmarks/*.yaml
python -m gipbench.dataset.release.build_general --which general --name general_v1 --bench-version general-v1.0
python -m gipbench.dataset.ingest.evalsets --sets fleurs_vi cv17_vi vietmed     # E0; --metadata-only: không tải audio
python -m gipbench.dataset.release.build_general --which external --name external_v1 --bench-version external-v1.0
python -m gipbench.bench.transcribe --model configs/bench/asr_models/gip15_mbs4_open.yaml --benchmark general_v1 --out <dir>
python -m gipbench.bench.evaluate --benchmark general_v1 --hyps <dir>/hyps.jsonl --mode final [--baseline-hyps <hyps gốc>]
python -m gipbench.bench.compare --benchmark general_v1 --runs results/general_asr/general_v1/*
python -m gipbench.dataset.release.leak_audit --test <manifest bench> --train <manifest train ...> --out reports/leak/<x>
python -m gipbench.dataset.release.package --benchmarks general_v1 code_switch_v1 cs_ggroup_v1 external_v1 --name gipbench-bench-v1.1 --baselines <results>/<bench>/<hệ> ...
python -m gipbench.dataset.release.fetch --manifest <gói>/manifests/<x>.jsonl --out bench_data/<x>   # phía người nhận
```

Benchmark đã khoá (`evidence_level: locked`): đổi câu / nhãn / chuẩn hoá = phiên bản mới, không sửa yaml / manifest cũ.

Phiên bản benchmark v1.1 (dựng 17/11/2026, **khoá 18/11/2026**, release `bench/{code-switch,external}-v1.1`; `docs/datasheet_bench_v1.md` mục 9):

```bash
python -m gipbench.dataset.release.derive_v1_1 --which code_switch   # C0 + train_video_overlap (lát video chưa thấy)
python -m gipbench.dataset.release.derive_v1_1 --which external      # E0 + audio_sha256 / duration; chạy lại khi ViMD tải xong
# E0 khi còn bộ đang tải: giải mã phần đã có audio, sau đó chạy lại với manifest đủ (transcribe resume theo id)
python -m gipbench.bench.transcribe --model <m>.yaml --benchmark external_v1_1 --manifest <data>/benchmarks/external_v1_1/manifest_with_audio.jsonl --out <dir>
# Whisper (faster-whisper / ctranslate2, venv gipbench-asr): cần cuBLAS 12 + cuDNN 9 của torch trong venv
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

VieGem + KenLM (`viegem_lm.yaml`) cần module Python `kenlm` trong venv gipbench-viegem; build từ mã nguồn nên cần
trình biên dịch: `sudo apt install build-essential cmake` rồi `uv pip install --python ~/.venvs/gipbench-viegem/bin/python kenlm`
(`pypi-kenlm` không có wheel cho Python 3.11).

## Đổi hướng thì sửa ở đâu

Xem bảng cuối `docs/ARCHITECTURE.md`.
