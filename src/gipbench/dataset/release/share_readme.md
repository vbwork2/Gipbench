# {name}

Bench ASR tiếng Việt dùng chung để **chấm bất kỳ model ASR nào**, kể cả model bạn tự fine-tune. Mọi người chấm trên cùng bộ câu, cùng cách chuẩn hoá chữ, cùng scorer, có khoảng tin cậy.

Gói này **không chứa audio**. Bạn tự tải audio từ nguồn gốc bằng `fetch`: revision cố định, kiểm sha256 từng file. Đọc `docs/datasheet_bench_v1.md` trước khi dùng (license, giới hạn, mức trùng dữ liệu train của các model).

| Track | Benchmark | Đo gì | Câu / giờ |
| --- | --- | --- | --- |
| G0 | `general_v1` | Tiếng Việt phổ thông: VIVOS, Common Voice 27, G-Group tech / edu / finance / pubadmin | 4 400 / 4.47 h |
| C0 | `code_switch_v1` | Từ tiếng Anh trong câu tiếng Việt, video y khoa (ViMedCSS hard) | 658 / 1.29 h |
| C1 | `cs_ggroup_v1` | Từ tiếng Anh trong câu tiếng Việt nói tự nhiên (các câu G-Group có từ Anh) | 1 260 / 1.31 h |

- Lời thoại là nhãn gốc của bên phát hành từng bộ, không sửa chữ.
- Bản v1.0 đã **khoá** bằng `manifest_sha256` trong `manifests/*.benchmark.yaml`, nên ai cũng chấm trên đúng một bộ câu.
- Phần lớn license là **phi thương mại** (CC-BY-NC / CC-BY-NC-SA). Chỉ dùng cho nghiên cứu hoặc đánh giá nội bộ, và trích dẫn từng bộ nguồn.

## 1. Cài đặt

Repo là riêng tư: xin quyền truy cập từ người gửi gói này. Code bench đang ở nhánh `feat/general-asr` (chưa merge vào `main`).

```bash
git clone -b feat/general-asr https://github.com/vbwork2/Gipbench.git && cd Gipbench
pip install -e ".[audio]" huggingface_hub pyarrow pyyaml soundfile soxr
export HF_HUB_DISABLE_XET=1     # nếu tải từ Hugging Face bị treo
```

## 2. Tải audio (chỉ cần một lần)

```bash
G=<đường dẫn tới thư mục gói này>
python -m gipbench.dataset.release.fetch --manifest $G/manifests/general_v1.jsonl --out bench_data/general_v1 \
    --cv-root /path/cv-corpus-27.0-2026-09-11/vi
python -m gipbench.dataset.release.fetch --manifest $G/manifests/code_switch_v1.jsonl --out bench_data/code_switch_v1
python -m gipbench.dataset.release.fetch --manifest $G/manifests/cs_ggroup_v1.jsonl --out bench_data/cs_ggroup_v1
```

- **Common Voice vi 27.0 phải tải tay** tại commonvoice.mozilla.org, vì Mozilla yêu cầu từng người tự đồng ý điều khoản. Giải nén xong thì trỏ `--cv-root` vào thư mục `vi/`. Thiếu bộ này thì không chấm được G0 ở chế độ final, vì manifest sẽ khác bản khoá.
- `fetch` dừng với mã 1 nếu file tải về khác sha256. Riêng ViMedCSS được resample về 16 kHz, nên sha256 có thể lệch nếu bạn dùng phiên bản soxr / libsndfile khác. Khi đó `fetch` chỉ cảnh báo, miễn là độ dài audio khớp.
- Kết quả: `bench_data/<x>/manifest.jsonl` (có trường `audio`, là đường dẫn tương đối so với manifest) và thư mục `bench_data/<x>/audio/`.

## 3. Kiểm dữ liệu FT không chứa câu test (bắt buộc trước khi báo số)

```bash
python -m gipbench.dataset.release.leak_audit --test bench_data/general_v1/manifest.jsonl \
    --train my_ft/train*.jsonl --dev my_ft/dev*.jsonl --out leak/general_v1
```

Chạy lại lệnh này cho cả `code_switch_v1` và `cs_ggroup_v1`.

- **Định dạng file train / dev:** JSONL. Mỗi dòng ít nhất có `"text"` (hoặc `"text_tagged"`). Nên có thêm:
  - `"audio"` (đường dẫn), để kiểm trùng sha256 audio;
  - `"speaker"` / `"source"`, nếu có.
- **Tuỳ chọn thêm:** `--text-files` (text dùng cho LM / TTS) và `--hotwords`.
- **Kết quả:**
  - Lệnh thoát mã 1 nếu có trùng nghiêm trọng: cùng audio, cùng câu hoặc cùng người nói.
  - Các trường hợp "gần trùng" cần người xem lại trong `leak/<x>/leak_report.md`.
- **Trùng lặp đã biết sẵn:**
  - VIVOS train và test dùng lại khoảng 194 câu đọc.
  - ViMedCSS train và hard có chung 201 video.
  - Nếu bạn FT trên VIVOS train hoặc ViMedCSS train, hãy ghi rõ khi báo số.

## 4. Chạy model để ra hyps.jsonl

Mỗi dòng có dạng `{"id": <id câu>, "hyp": <chữ model nghe ra>}`, và phải có **đúng một dòng cho mỗi id** trong manifest. Scorer (hồ sơ `gipbench_strict_v1`) tự chuẩn hoá chữ hoa / thường, dấu câu và chữ số, bạn không cần xử lý trước.

Có hai cách.

### a) Tự viết script (dùng cho model bất kỳ)

```python
import json, soundfile as sf
base = "bench_data/general_v1/"
rows = [json.loads(l) for l in open(base + "manifest.jsonl", encoding="utf-8")]
with open("my_system/general_v1/hyps.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        wav, sr = sf.read(base + r["audio"], dtype="float32")   # 8-48 kHz, mono / stereo tuỳ bộ: tự resample nếu cần
        f.write(json.dumps({"id": r["id"], "hyp": my_model_transcribe(wav, sr)}, ensure_ascii=False) + "\n")
```

Đặt một file `run_meta.json` cạnh hyps, tối thiểu:

```json
{"system": "ten_model", "inference_mode": "open_vocab", "model_id": "...", "revision": "...", "decoding": {}}
```

Giá trị của `inference_mode`:
- `open_vocab`: không gợi ý gì.
- `lm`: dùng LM ngoài, không train trên text test.
- `context_bias`: hotwords lấy từ nguồn ngoài test.
- `oracle_bias`: hotwords có thông tin từ test. Chỉ dùng để chẩn đoán, không dùng để so.

### b) Dùng adapter có sẵn

Có adapter cho Gipformer / sherpa-onnx transducer, VieGem-ASR, faster-whisper và Qwen3-ASR. `transcribe` tự ghi `run_meta.json` và chạy tiếp được khi bị ngắt.

Copy một file trong `configs/bench/asr_models/` của repo, sửa đường dẫn trỏ tới model của bạn, rồi chạy:

```bash
python -m gipbench.bench.transcribe --model my_model.yaml --benchmark $G/manifests/general_v1.benchmark.yaml \
    --manifest bench_data/general_v1/manifest.jsonl --out my_system/general_v1
```

## 5. Chấm điểm

```bash
python -m gipbench.bench.evaluate --benchmark $G/manifests/general_v1.benchmark.yaml \
    --manifest bench_data/general_v1/manifest.jsonl --hyps my_system/general_v1/hyps.jsonl --mode final \
    --baseline-hyps $G/baselines/general_v1/gip15_mbs4_open/hyps.jsonl
python -m gipbench.bench.evaluate --benchmark $G/manifests/code_switch_v1.benchmark.yaml \
    --manifest bench_data/code_switch_v1/manifest.jsonl --catalog $G/manifests/code_switch_v1.catalog.tsv \
    --hyps my_system/code_switch_v1/hyps.jsonl --mode final
python -m gipbench.bench.evaluate --benchmark $G/manifests/cs_ggroup_v1.benchmark.yaml \
    --manifest bench_data/cs_ggroup_v1/manifest.jsonl --hyps my_system/cs_ggroup_v1/hyps.jsonl --mode final
```

**Đầu ra**, ghi cạnh hyps:
- `report.md`: bảng cho người đọc;
- `summary.json`;
- `utt_scores.jsonl`: điểm từng câu.

**`--mode final` thoát mã 2** nếu:
- hyps thiếu, thừa hoặc trùng id;
- manifest hoặc catalog khác bản khoá;
- `inference_mode` không khai báo hoặc không được phép.

**`--baseline-hyps`** thêm Δ ghép cặp (trên cùng câu) và CI 95% so với model gốc.
- Thư mục `baselines/` có sẵn hyps của hai model gốc: Gipformer 1.5 (`gip15_mbs4_open`) và VieGem-ASR (`viegem_greedy`).
- Cấu hình giải mã của từng model nằm trong `run_meta.json` cạnh hyps.

**Các metric:**

| Metric | Ý nghĩa |
| --- | --- |
| WER | Theo token khoảng trắng (≈ âm tiết) |
| CER | Theo ký tự |
| CS-WER | Lỗi trên từ tiếng Anh (C0, C1) |
| N-WER | Lỗi trên phần tiếng Việt |
| Term recall | Tỷ lệ thuật ngữ được nhận đúng |

Định nghĩa đầy đủ: `docs/evaluation_metrics_v1.md`.

## 6. Báo kết quả

Báo **theo từng bộ**, không chỉ một con số. Kèm CI 95%, hồ sơ chuẩn hoá và inference mode. Mẫu:

| Bộ | Model gốc WER | Model FT WER | Δ (CI 95%) |
| --- | --- | --- | --- |
| vivos | | | |
| commonvoice_vi_27 | | | |
| ggroup_tech / edu / finance / pubadmin | | | |
| C0 ViMedCSS hard: WER / CS-WER | | | |
| C1 G-Group CS: WER / CS-WER | | | |

Ghi kèm:
- dữ liệu FT là gì;
- kết quả `leak_audit`;
- `manifest_sha256` (có in trong `report.md`);
- `run_meta.json`.

Quy tắc:
- Không train, chọn checkpoint, chọn hotwords hay fit LM trên bất kỳ câu nào của bench.
- Không cộng gộp C1 với G0 như hai bằng chứng độc lập, vì C1 là tập con audio G-Group của G0.

## Nội dung gói

{summary}

`SHA256SUMS` chứa sha256 của mọi file trong gói. Kiểm bằng `sha256sum -c SHA256SUMS`.
