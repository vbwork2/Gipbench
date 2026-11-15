# Định dạng dữ liệu

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
| `dataset` | nên có | tên bộ gốc (`ViMedCSS:train`, `vivos:test`, `vss_cs`…). Cột `datasets` của `configs/train_mix.tsv` map trường này vào nhóm trộn. Nguồn dùng: B1 = G-Group + dữ liệu tự thu; CS thật cho train = ViMedCSS train + `vss_cs` + dữ liệu tự có. `sea_speech:en-vi` (SEA) là **tùy chọn, không dùng** |
| `train_only` | tuỳ chọn | `true` = dòng chỉ được dùng cho train, không bao giờ vào bench (vd `dataset: vss_cs`) |
| `label_source` | tuỳ chọn | nguồn nhãn máy, vd `pseudo:Zipformer-30M-RNNT-6000h` |
| `agreement` | tuỳ chọn | do `tools/agreement_filter.py` ghi: `{"confirmed": [số hyp xác nhận từng span], "n_hyps", "min_agree"}` |
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

- Bản nháp tự gắn dựa trên hai danh sách đang chờ duyệt: `configs/vi_abbrev.txt` (viết tắt / mã thường đọc kiểu Việt) và `configs/vi_loanwords.txt` (từ mượn có trong từ điển tiếng Việt). Từ nằm trong hai danh sách này không được gắn `<en>`, trừ khi nó có trong catalog. Riêng các mục `same_surface` của `vi_loanwords.txt` (trùng mặt chữ tiếng Anh: `vitamin`, `test`, `gen`) chỉ áp trong ViMedCSS; ở các bộ khác chúng vẫn theo quy ước B1.
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

Hyps của ASR dùng để lọc đồng thuận (`tools/asr_hyps.py`) có thêm `engine`, `model`, `audio_sha256` (sha256 của file
audio đã giải mã) và `audio_rel` (đường dẫn audio tương đối với gốc dataset = đường dẫn trong repo HF). Cache tra theo
**id + audio_sha256** (`lib/hyp_cache.py`, chỉ mục `audio_rel → sha256` ở `results/asr_hyps/vss_audio_sha256.tsv`).
