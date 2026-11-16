# Datasheet: gipbench bench v1 (bản nháp, 16/11/2026)

Bench đánh giá nhận dạng tiếng nói (ASR) tiếng Việt, gồm hai track: **G0** tiếng Việt phổ thông và **C0** câu tiếng Việt
có xen từ tiếng Anh (code-switching). Gói share **không chứa audio**; audio được tải lại từ nguồn gốc bằng
`python -m gipbench.dataset.release.fetch` (revision cố định, kiểm sha256). Bố cục theo gợi ý "Datasheets for Datasets"
(Gebru và cs.).

> **Trạng thái: bản nháp (`draft`).** Lời thoại là nhãn của bên phát hành từng bộ, gipbench chưa duyệt lại. Tag `<en>` của
> C0 là nhãn phái sinh (mục 4). Kết quả trên bản này dùng để chẩn đoán, không phải bảng xếp hạng cuối.

## 1. Mục đích

- So sánh **bất kỳ** model ASR tiếng Việt nào trên cùng tập câu, cùng chuẩn hoá, cùng scorer, có khoảng tin cậy.
- Tách hai câu hỏi: nhận dạng tiếng Việt phổ thông (G0) và nhận dạng từ tiếng Anh trong câu tiếng Việt (C0).
- Không tạo "một con số SOTA chung": kết quả luôn báo theo từng bộ dữ liệu.

## 2. Thành phần

| Track | Bộ | Nguồn (revision) | Split | Câu | Giờ | Người nói | License |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | `vivos` | AILAB-VNUHCM/vivos (`3cbfb25`) | test | 760 | 0.75 | 19 | CC-BY-NC-SA-4.0 |
| G0 | `commonvoice_vi_27` | Mozilla Common Voice 27.0 (2026-09-11), tải tay | test | 1 593 | 1.75 | 221 | CC0-1.0 |
| G0 | `ggroup_tech` | g-group-ai-lab/vi-asr-tech-test (`0b8c874`) | test | 508 | 0.49 | không có id | CC-BY-NC-4.0 |
| G0 | `ggroup_edu` | g-group-ai-lab/vi-asr-edu-test (`9a15ad6`) | test | 495 | 0.52 | không có id | CC-BY-NC-4.0 |
| G0 | `ggroup_finance` | g-group-ai-lab/vi-asr-finance-test (`ebadeb0`) | test | 529 | 0.50 | không có id | CC-BY-NC-4.0 |
| G0 | `ggroup_pubadmin` | g-group-ai-lab/vi-asr-pubadmin-test (`40aabb0`) | test | 515 | 0.47 | không có id | CC-BY-NC-4.0 |
| C0 | `vimedcss` | tensorxt/ViMedCSS (`b6959a1`) | hard | 658 | 1.29 | 404 video | CC-BY-4.0 |

Tổng G0: 4 400 câu / 4.47 h; C0: 658 câu / 1.29 h. Lấy **toàn bộ** split công bố, không lọc câu nào; số câu khớp số
công bố của từng bộ.

Mỗi dòng manifest: `id`, `dataset`, `upstream_id`, `upstream_split`, `text_tagged` (lời thoại; C0 có tag `<en>`),
`text_tagged_orig` (C0: tag gốc của ViMedCSS), `duration`, `audio_sha256`, `audio_derivation`, `speaker`, `source`,
`domain`, `region`, `gender` (khi bên phát hành có), `license`, `annotation_status`, `bench_version`.

## 3. Thu thập và nguồn gốc

- Gipbench **không** thu âm hay gõ lại lời thoại. Audio và lời thoại lấy nguyên từ bên phát hành (mục 2).
- G-Group: video công nghệ / giáo dục / tài chính / hành chính công, nói tự nhiên, lời thoại người gõ. Bên phát hành không
  cung cấp id người nói.
- ViMedCSS: video y khoa YouTube (bài báo LREC 2026); split `hard` do tác giả ViMedCSS chọn.
- VIVOS, Common Voice: giọng đọc câu cho sẵn.

## 4. Tiền xử lý và nhãn phái sinh

- **Audio**: VIVOS, Common Voice, G-Group giữ nguyên byte gốc (`audio_derivation: upstream_bytes`). ViMedCSS
  (44.1 kHz) được resample về 16 kHz mono PCM16 bằng soxr HQ (`soxr_hq_16k_mono_pcm16`); sha256 của file này phụ thuộc
  phiên bản soxr / libsndfile (bản gốc: soxr 1.1.0, libsndfile 1.2.2); `fetch` chỉ cảnh báo nếu lệch mà độ dài khớp.
- **Tag `<en>` (C0)**: `text_tagged` theo quy ước của gipbench: tag theo **cách đọc** (viết tắt đọc kiểu Anh = en,
  đọc kiểu Việt = không; brand Latin = en; từ mượn có trong từ điển tiếng Việt như "vitamin" = không). `text_tagged_orig`
  là tag gốc của ViMedCSS. Hai bản lệch nhau ở 339 / 658 câu; báo cáo nên ghi rõ đã chấm theo bản nào
  (`evaluate --label-field text_tagged_orig` để chấm theo tag gốc).
- **Chuẩn hoá khi chấm**: hồ sơ `gipbench_strict_v1`, áp đối xứng cho reference và hypothesis: NFC, chữ thường, số thành
  chữ tiếng Việt, bỏ dấu câu, thống nhất vị trí dấu thanh (hoà = hòa), i/y có dấu (kỹ = kĩ), viết tắt có chấm (A.I. = ai),
  email / tên miền, dấu nháy, chính tả Anh-Anh / Anh-Mỹ. Định nghĩa đầy đủ: `docs/evaluation_metrics_v1.md`.

## 5. Cách dùng được khuyến nghị

- Báo WER **theo từng bộ**, kèm CI 95%. Con số gộp duy nhất được phép là trung bình đều theo bộ (macro), luôn kèm số
  từng bộ.
- Khai báo **inference mode**: `open_vocab` (không gợi ý), `lm` (LM không train trên text test), `context_bias`
  (hotwords từ nguồn ngoài test), `oracle_bias` (hotwords có thông tin test: chỉ để chẩn đoán).
- WER ở đây là **WER theo token khoảng trắng** (≈ âm tiết tiếng Việt), không phải WER theo từ đã tách.
- So hai hệ: dùng Δ ghép cặp trên cùng id câu (`evaluate --baseline-hyps`).

## 6. Không nên dùng cho

- Train, tinh chỉnh, chọn checkpoint, chọn hotwords hay fit LM trên bất kỳ câu / thuật ngữ nào của bench.
- Kết luận "tổng quát sang người nói / domain mới" khi model đã train trên cùng bộ (mục 7).
- Mục đích thương mại: phần lớn nguồn là CC-BY-NC / CC-BY-NC-SA.

## 7. Giới hạn đã biết (đọc trước khi diễn giải kết quả)

1. **Trùng dữ liệu train của model.** Không bộ nào được đảm bảo độc lập với mọi model:
   - G-Group do chính tổ chức phát hành Gipformer công bố, và model card Gipformer báo số trên các bộ này;
   - model card VieGem-ASR ghi đã train trên Common Voice 17 và ViMedCSS train (đã lọc câu test theo lời họ);
   - model không công bố dữ liệu train (vd Gipformer) thì mức trùng là "unknown".
   Mức trùng của từng bộ với từng model: `configs/datasets.yaml` trong gói, trường `pretraining_overlap`.
2. **ViMedCSS không tách theo video**: 308 / 658 câu hard đến từ 201 video YouTube có cả trong ViMedCSS train. Model
   train trên ViMedCSS train đã nghe cùng người nói.
3. **VIVOS dùng lại câu đọc** giữa train và test (~194 / 760 câu test có cùng lời với câu train, người đọc khác).
4. **G-Group không có id người nói**: CI tính theo câu, hẹp hơn thực tế nếu một người nói nhiều câu.
5. Lời thoại chưa được gipbench duyệt lại; tag `<en>` của gipbench là tự động + quy ước, đang được duyệt.
6. Phân bố domain lệch: G0 nghiêng về giọng đọc (VIVOS + CV = 2.5 / 4.47 h); C0 chỉ có y khoa.

## 8. Kết quả tham chiếu (bản nháp, open_vocab, hồ sơ gipbench_strict_v1, WER %)

| Bộ | Gipformer 1.5 (68M, beam 4) | VieGem-ASR (359M, greedy) |
| --- | --- | --- |
| VIVOS | 4.51 | 6.68 |
| Common Voice 27 | 7.47 | 9.50 |
| G-Group tech / edu / finance / pubadmin | 27.70 / 23.52 / 23.11 / 15.28 | 36.46 / 33.70 / 34.60 / 29.02 |
| ViMedCSS hard: WER / CS-WER | 21.92 / 107.12 | 21.05 / 84.54 |

Đọc kèm mục 7: G-Group trùng tổ chức với Gipformer; ViMedCSS trùng dữ liệu train với VieGem.

## 9. Phiên bản và bảo trì

- `bench_version` trong mỗi dòng; `release.json` và `SHA256SUMS` trong gói. Đổi bất kỳ câu / nhãn / chuẩn hoá nào =
  phiên bản mới; bản đã phát hành không bị sửa.
- Dự kiến: v1.0 = G0 + C0 như trên; v1.1 thêm track code-switching G-Group sau khi chốt tag `<en>` theo thuật ngữ. Bộ
  riêng tư (consensus) **không** share, chỉ chấm hộ.
- License của gói = license của từng bộ nguồn (mục 2); phần phái sinh từ VIVOS phân phối theo CC-BY-NC-SA-4.0.
  Trích dẫn từng bộ nguồn khi dùng.
