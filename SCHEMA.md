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
| `en_meta` | nên có | danh sách theo thứ tự span `<en>`; mỗi phần tử `{"viet_level": "strong" \| "mid" \| "native"}` (mức Việt hoá cách phát âm) |

Category của span lấy từ catalog, không cần ghi trong manifest.

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
