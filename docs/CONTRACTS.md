# Hợp đồng dữ liệu (định dạng file giữa các pipeline)

Mục 1–3: định dạng có từ trước refactor (trước đây là SCHEMA.md). Mục 4–7: hợp đồng giữa 3 pipeline, thêm khi refactor
(16/11/2026), xem docs/ARCHITECTURE.md.

## 1. Manifest bench / train — JSONL, 1 utterance mỗi dòng

```json
{"id": "b1_000123", "split": "B1", "audio": "audio/b1/000123.wav", "duration": 6.42,
 "text_tagged": "hôm nay <en>team</en> mình có <en>meeting</en> lúc ba giờ",
 "speaker": "yt_UCx9_spk2", "source": "youtube:dQw4w9WgXcQ",
 "region": "north", "gender": "f", "channel": "wideband", "domain": "tech",
 "en_meta": [{"viet_level": "mid"}, {"viet_level": "strong"}],
 "annotator": "a01", "reviewer": "a02"}
```

| Trường | Bắt buộc | Giá trị |
| --- | --- | --- |
| `id` | có | duy nhất trong toàn bộ bench |
| `split` | có | `B1` real-CS general · `B2` real-CS domain (sản phẩm) · `B3` unseen terms · `B4` read-CS · `B5` mono-VI regression; với manifest train: tuỳ ý (vd `train`) |
| `audio` | có | đường dẫn wav/flac, tương đối theo thư mục chứa manifest; mono, ≥ 8 kHz |
| `duration` | có | giây, khớp audio ±0.1 s |
| `text_tagged` | có | transcript theo quy ước (chính tả Anh cho từ Anh, số viết dạng đọc), mỗi từ/cụm tiếng Anh trong `<en>…</en>`; B5 không có tag |
| `speaker` | có | id người nói, ổn định trên toàn bộ dữ liệu (cả train) — dùng để tách split và bootstrap |
| `source` | có | id video/cuộc gọi/phiên thu — dùng chống rò rỉ |
| `region` | nên có | `north` / `central` / `south` / `other` / `unk` |
| `gender` | nên có | `f` / `m` / `u` |
| `channel` | nên có | `wideband` / `phone8k` |
| `domain` | nên có | `tech` / `finance` / `edu` / `daily` / `product` / … |
| `text_tagged_orig` | tuỳ chọn | nhãn `<en>` theo quy ước của dataset gốc khi quy ước đó khác Spec (ViMedCSS: gắn cả từ mượn như `vitamin`, `axit`). Chỉ dùng để chấm số phụ: `score.py --label-field text_tagged_orig` |
| `dataset` | nên có | tên bộ gốc (`ViMedCSS:train`, `vivos:test`, `vss_cs`…). Khoá `pools.<nhóm>.datasets` của experiment YAML (`configs/model/experiments/*.yaml`) map trường này vào nhóm trộn. Nguồn dùng: B1 = G-Group + dữ liệu tự thu; CS thật cho train = ViMedCSS train + `vss_cs` + dữ liệu tự có. `sea_speech:en-vi` (SEA) là **tùy chọn, không dùng** |
| `train_only` | tuỳ chọn | `true` = dòng chỉ được dùng cho train, không bao giờ vào bench (vd `dataset: vss_cs`) |
| `license` | tuỳ chọn | `test_only` = CHỈ được dùng để test (vd bộ `consensus`). `validate.py --train/--dev` báo LỖI nếu dòng test_only (hoặc source / dataset `consensus`) có trong train / dev; `curate_train.py`, `vss_cs_prefilter.py` loại các dòng này và câu trùng text trong `data/raw/consensus/leak_block.txt` |
| `trusted_text` | tuỳ chọn | `true` = text do người gõ (G-Group, consensus `agreed`): `build_bench_v0.py --accept trusted-text` nhận khi mọi thuật ngữ còn cờ đã chốt trong `configs/dataset/labeling/term_decisions.tsv` (decisions §12.4, `src/gipbench/core/term_decisions.py`) |
| `term_category` | tuỳ chọn | `{"<span sau chuẩn hoá>": "<category>"}` cho span không lấy category từ catalog, vd tên người nước ngoài `{"hugo lambert": "name"}`; `score.py` ưu tiên trường này khi cắt lát theo category |
| `speaker_cluster` | tuỳ chọn | cụm người nói ước lượng bằng speaker embedding khi bộ không có speaker id (vd `consensus:spk003`). CHỈ dùng làm đơn vị bootstrap CI trong `score.py` / `leaderboard.py`, không dùng để tách split |
| `review_required` | tuỳ chọn | `true` = `build_bench_v0.py` chỉ nhận khi người duyệt ghi `decision` = ok / fix (kể cả `--accept all`); vd consensus disputed / pending |
| `split_by_tag` | tuỳ chọn | `{"en": "B2", "vi": "B5"}`: khi người duyệt `fix`, split đặt lại theo text đã sửa còn `<en>` hay không |
| `label_source` | tuỳ chọn | nguồn nhãn máy, vd `pseudo:Zipformer-30M-RNNT-6000h` |
| `agreement` | tuỳ chọn | do `src/gipbench/dataset/pseudolabel/agreement_filter.py` ghi: `{"confirmed": [số hyp xác nhận từng span], "n_hyps", "min_agree"}` |
| `en_meta` | nên có | danh sách theo thứ tự span `<en>`; mỗi phần tử `{"viet_level": "strong" \| "mid" \| "native"}` (mức Việt hoá cách phát âm) |

Category của span lấy từ catalog, không cần ghi trong manifest.

### Quy ước gán `<en>`: theo CÁCH ĐỌC

Một từ / cụm viết bằng chữ Latin được gắn `<en>` khi người nói **đọc nó như tiếng Anh**, không xét nguồn gốc của nó.

| Trường hợp | Gắn `<en>`? | Ví dụ |
| --- | --- | --- |
| Từ / cụm tiếng Anh, kể cả khi phát âm kiểu Việt | có | `<en>meeting</en>`, `<en>deadline</en>` |
| Brand / tên riêng viết bằng chữ Latin, kể cả brand Việt | có | `<en>vietcombank</en>`, `<en>shopee</en>`, `<en>zalo</en>` |
| Viết tắt tiếng Anh đọc kiểu Anh | có | `<en>ai</en>` (ây ai), `<en>ceo</en>`, `<en>kpi</en>` |
| Viết tắt / mã đọc kiểu Việt (đánh vần chữ cái tiếng Việt) | không | `vn` (vê en), `bidv`, `plx`, `ubnd`, mã chứng khoán |
| Từ mượn có trong từ điển tiếng Việt | không | `axit`, `canxi`, `vitamin`, `cà phê` |
| Không chắc người nói đọc kiểu nào | **người duyệt nghe audio rồi quyết**, ghi lý do vào cột `reading_note` của phiếu duyệt | `pc` (pê xê / pi xi), `tv` |

- Bản nháp tự gắn dựa trên hai danh sách đang chờ duyệt: `configs/dataset/labeling/vi_abbrev.txt` (viết tắt / mã thường đọc kiểu Việt) và `configs/dataset/labeling/vi_loanwords.txt` (từ mượn có trong từ điển tiếng Việt). Từ nằm trong hai danh sách này không được gắn `<en>`, trừ khi nó có trong catalog. Các mục `same_surface` của `vi_loanwords.txt` (trùng mặt chữ tiếng Anh: `vitamin`, `protein`, `insulin`) là **không EN ở MỌI bộ** (decisions §12). Ngoại lệ chốt theo từng thuật ngữ nằm ở `configs/dataset/labeling/term_decisions.tsv`.
- Tên người nước ngoài viết bằng chữ Latin: gắn `<en>`, `term_category = name`. Tên có ít nhất một token không đọc được như âm tiết Việt (`lucas`, `tanaka`) hoặc danh xưng + tên (`mr tanaka`) thì nhận. Tên trùng âm tiết Việt (`kim`, `chen`, `sam`) chỉ nhận khi đứng cạnh một tên như vậy (`sarah kim`); đứng một mình thì chỉ gắn cờ `maybe_name:*` (decisions §12, đã duyệt).
- Cùng một viết tắt có thể được đọc khác nhau ở hai câu (`ai` = "ây ai" hay "a i"), nên người duyệt quyết theo từng câu chứ không theo danh sách.
- Hướng dẫn duyệt chi tiết: `docs/huong_dan_duyet_b1.md`.

## 2. Catalog thuật ngữ — `configs/catalog.tsv` (TSV, có header)

| Cột | Ý nghĩa |
| --- | --- |
| `term` | dạng chuẩn (chính tả Anh, viết thường): `facebook`, `pull request` |
| `category` | `brand`, `tech`, `business`, `finance`, `edu`, `daily`, `acronym`, `name` … |
| `variants` | biến thể chính tả coi là ĐÚNG ở chế độ strict, ngăn cách `\|`: `fb\|face book` |
| `phonetic` | phiên âm Việt chỉ chấp nhận ở chế độ lenient: `phây búc\|phê búc` |
| `reserved_b3` | `1` = thuật ngữ dành riêng cho B3, cấm có trong mọi dữ liệu train |

## 3. Hypothesis — `results/<system>/hyps.jsonl`

```json
{"id": "b1_000123", "hyp": "HÔM NAY TEAM MÌNH CÓ MEETING LÚC BA GIỜ"}
```

Bất kỳ hệ thống nào (sherpa-onnx, icefall PyTorch, Whisper, API) chỉ cần xuất đúng định dạng này là chấm được.
Hoa/thường, dấu câu, số không cần chuẩn hoá trước — `score.py` tự làm.

Hyps của ASR dùng để lọc đồng thuận (`src/gipbench/infer/asr_hyps.py`) có thêm `engine`, `model`, `audio_sha256` (sha256 của file
audio đã giải mã) và `audio_rel` (đường dẫn audio tương đối với gốc dataset = đường dẫn trong repo HF). Cache tra theo
**id + audio_sha256** (`src/gipbench/infer/cache.py`, chỉ mục `audio_rel → sha256` ở `results/asr_hyps/vss_audio_sha256.tsv`).

## 4. Data release: `data/releases/<kind>/<name>@<version>/` (dataset -> model, bench)

`kind` ∈ `bench`, `catalog`, `pool`, `dev`. Riêng bench và catalog: thư mục là `data/releases/<kind>/<version>/`.
Code: `gipbench.core.release`. Đăng ký: `python -m gipbench.dataset.release.register` theo `configs/dataset/releases.yaml`.

```text
data/releases/pool/vss_cs@e3_v1/
  manifest.jsonl    định dạng mục 1; audio = đường dẫn TƯƠNG ĐỐI theo file manifest (không bao giờ tuyệt đối)
  release.json
data/releases/catalog/v0-draft/
  catalog.tsv       định dạng mục 2
  release.json
```

`release.json`: `kind`, `name`, `version`, `ref`, `created` (ISO-8601), `git_commit`, `builder`, `builder_args`,
`files` (`{tên: {sha256, rows, hours}}`), `source`, `parents` (`[{path, sha256}]`), `catalog` (ref catalog dùng kèm),
`flags` (vd `train_only`, `label_source`, `draft`, `catalog_from_bench`), `notes`, `stats` (giờ theo split / dataset /
domain, số câu CS).

- Tham chiếu: chuỗi `"<kind>/<name>@<version>"`, riêng bench / catalog là `"bench/v0-draft"`. Chỉ đọc qua
  `release.resolve(ref)`: hàm này kiểm sha256 nên sửa tay sẽ báo lỗi.
- **Bất biến:** muốn đổi nội dung thì ra version mới (thêm mục mới trong `releases.yaml`), không sửa thư mục đã có.
- Model đọc release qua `release.materialize(ref, out)`. Hàm này chép manifest ra thư mục làm việc với audio tuyệt đối,
  giữ thứ tự dòng và thứ tự khoá.
- `flags.catalog_from_bench = true`: catalog có span lấy từ chính bench. `bench.run` in cảnh báo: hotwords từ catalog
  này là cận trên lạc quan.

## 5. Experiment: `configs/model/experiments/<exp>.yaml` (người làm thí nghiệm -> model)

Chạy: `python -m gipbench.model.run <exp>.yaml --stages 0-6 [--set khoá.con=giá_trị] [--dry-run]`. Ví dụ đầy đủ:
`e3_v1b.yaml`. Khoá bắt buộc: `name`, `base_model`, `catalog` (ref), `leak_guard` (ref bench, chỉ dùng để chặn rò rỉ),
`dev.{cs,vi}` (ref), `pools`, `output`.

| Khoá | Ý nghĩa |
| --- | --- |
| `pools.<nhóm>` | `weight` (phần của batch), `env` (biến môi trường ghi đè, vd `MIX_SYNTH`), `datasets` (giá trị `dataset` trước ':' thuộc nhóm), `refs` (pool release làm đầu vào, theo thứ tự), `note` |
| `curate` | tham số `gipbench.model.recipe.curate` (snake_case); khoá không ghi = mặc định của tool |
| `features` | `speed_perturb`, `phone_sim` |
| `mix` | `epochs`, `epoch_hours` (epoch ảo) |
| `train` | `lr`, `max_duration`, `freeze`, `model_args`, `drop_ctc`, `workers`, `buckets`, `keep_last_k`, `segments_per_epoch`, `watchdog.{disk_path, min_free_disk_gb, min_ram_mb, ram_grace_s, hang_s, poll_s}` |
| `select` | `max_vi_delta`, `wise_alphas` |
| `decode_dev` | `threads`, `batch` |
| `output` | `exp_dir` (checkpoint), `work_dir` (manifest trung gian, lhotse, log), `dev_results`, `artifact`; nhận `${exp_root}` |
| `systems` | system YAML sinh ở cuối stage 6: `[{system, hotwords?: {catalog, exclude_b3, score}, decode?: {...}}]` |

## 6. Model artifact: `<exp_root>/artifacts/<exp>/` (model -> bench)

```text
onnx/encoder.int8.onnx  onnx/decoder.int8.onnx  onnx/joiner.int8.onnx  onnx/tokens.txt  onnx/bpe.model
model_card.json
```

`model_card.json`: `system`, `experiment` (`path`, `overrides`, `sha256`), `base_model` (`path`, `sha256` của
`model.pt`), `releases_used` (`{ref: files của release.json}`), `checkpoint_picked` (vd `wise_1`, `ck4132`, `base`),
`wise_from`, `dev_metrics` (dòng của select_wise), `git_commit`, `files` (sha256 từng file onnx), `created`.

## 7. System: `configs/bench/systems/<system>.yaml` (model / người dùng -> bench)

```yaml
system: e7_v1                                  # = tên thư mục kết quả, = tên file
model: ${exp_root}/e3_v1/cand/wise_1/onnx      # thư mục có *.onnx (hoặc <dir>/onnx/); HOẶC
# hyps: path/to/hyps.jsonl                     # hyps có sẵn (model ngoài, hậu xử lý E8): bỏ qua decode
decode:                                        # = tham số gipbench.infer.decode (gạch dưới thay gạch ngang)
  decoding_method: modified_beam_search
  beam: 4
  quantize: int8
  threads: 16
  batch: 32
  hotwords_catalog: catalog/v0-draft           # ref catalog release hoặc đường dẫn .tsv
  hotwords_exclude_b3: true
  hotwords_score: 1.5
```

Chạy: `python -m gipbench.bench.run --bench bench/v0-draft --system configs/bench/systems/e7_v1.yaml`. Kết quả ghi vào
`results/<version bench, '-' -> '_'>/<system>/` theo định dạng mục 3. Luật chọn của leaderboard nằm ở
`configs/bench/rules.yaml`.

## 8. Benchmark v1 (thêm 17/11/2026): `configs/bench/benchmarks/<tên>.yaml` (dataset -> bench)

Đọc bằng `gipbench.bench.registry` (kiểm trường bắt buộc, `tests/bench/test_registry.py`). Nguồn dữ liệu từng bộ:
`configs/bench/datasets.yaml` (`hf_id`, `revision`, `license`, `pretraining_overlap` theo model ...).

| Khoá | Ý nghĩa |
| --- | --- |
| `benchmark_id`, `version`, `track` | vd `general`, `v1.0`, `G0` (`G0`, `C0`, `C1`, `E0`, `Z0`, `R0`) |
| `evidence_level` | `draft` -> `human_reviewed` -> `locked`; `locked` bắt buộc có `manifest_sha256` |
| `manifest`, `manifest_sha256` | manifest JSONL; sha256 tính trên mọi trường trừ `audio` và trường bắt đầu bằng `_`, dòng sắp theo `id` |
| `catalog`, `catalog_sha256`, `catalog_from_bench` | catalog chấm CS-WER (null = không catalog); khoá cả catalog nếu có |
| `profile`, `also_report_profiles` | hồ sơ chuẩn hoá (`gipbench.core.normprofiles`): `gipbench_strict_v1` (chính), `legacy_v0` (= score.py cũ) |
| `primary_metric`, `group_by`, `aggregate`, `boot_unit`, `seed`, `iters` | metric chính; nhóm báo cáo; `macro_by_dataset`; đơn vị bootstrap |
| `allowed_inference_modes` | `open_vocab`, `lm`, `context_bias`, `oracle_bias` |
| `sources` | `[{dataset, split, slice}]`, mỗi `dataset` phải có trong `datasets.yaml` |

Manifest v1 theo mục 1, thêm: `dataset`, `upstream_id`, `upstream_split`, `audio_sha256`, `audio_derivation`
(`upstream_bytes` | `soxr_hq_16k_mono_pcm16` | `pinned_source_extract` = bộ E0 khoá bằng metadata, không có
`audio_sha256` / `duration` từng câu), `license`, `annotation_status`, `bench_version`; tuỳ chọn `text_orig`,
`tag_review` (C1), `text_redistributable: false` (lời thoại không được phân phối lại: gói share bỏ, `fetch` lấy lại).
Ngoại lệ so với mục 1: dòng `pinned_source_extract` không có `audio` / `duration` cho tới khi người dùng chạy `fetch`.

`run_meta.json` (cạnh `hyps.jsonl`, schema `gipbench.run_meta/1`): `system`, `engine`, `model_id`, `revision`,
`inference_mode`, `decoding`, `device`, `precision`, `batch_size`, `context_vocab` (`provenance`: `external` |
`test_derived`), `hyps_sha256`, `manifest_sha256`, `rtf`, `hardware`. `gipbench.bench.evaluate --mode final` cần ít
nhất `inference_mode` (hoặc `--inference-mode`).

Kết quả `evaluate`: `<out>/utt_scores.jsonl`, `summary.json` (theo hồ sơ, nhóm, CI, Δ ghép cặp), `report.md`.

Gói share (`gipbench.dataset.release.package`): `manifests/<tên>.jsonl` (không `audio`), `<tên>.benchmark.yaml`,
`<tên>.catalog.tsv`, `baselines/<tên>/<hệ>/{hyps.jsonl,run_meta.json,report.md}`, `docs/`, `configs/`, `README.md`,
`release.json`, `SHA256SUMS`. Package từ chối dòng `test_only`, catalog lấy từ bench, manifest / catalog khác bản khoá.
