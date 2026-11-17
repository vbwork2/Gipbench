# Datasheet: gipbench bench v1.1 (khoá 17/11/2026)

Bench đánh giá nhận dạng tiếng nói (ASR) tiếng Việt, gồm bốn track: **G0** tiếng Việt phổ thông, **C0** câu tiếng Việt
có xen từ tiếng Anh (code-switching) trong video y khoa, **C1** code-switching trong lời nói tự nhiên nhiều domain
(G-Group), và **E0** 11 bộ test công khai khác (gồm các bộ model card VieGem-ASR báo số: FLEURS, Common Voice 17,
PhoAudiobook, LSVSC, Bud500, GigaSpeech2, ViMD, ViMedCSS test; thêm VietMed và Speech-MASSIVE). Gói share **không chứa audio**; audio được tải lại từ nguồn gốc bằng
`python -m gipbench.dataset.release.fetch` (revision cố định, kiểm sha256). Bố cục theo gợi ý "Datasheets for Datasets"
(Gebru và cs.).

> **Trạng thái: đã khoá (`locked`; G0 / C0 / C1 từ v1.0, E0 thêm ở v1.1).** Tập câu, nhãn và chuẩn hoá cố định theo `manifest_sha256` (mục 9); đổi bất kỳ
> thứ gì = phiên bản mới. "Khoá" nghĩa là **cố định**, không phải "gipbench đã nghe lại mọi câu": lời thoại vẫn là nhãn của
> bên phát hành từng bộ (gipbench chưa duyệt lại); tag `<en>` của C0 và C1 là nhãn phái sinh (mục 4).

## 1. Mục đích

- So sánh **bất kỳ** model ASR tiếng Việt nào trên cùng tập câu, cùng chuẩn hoá, cùng scorer, có khoảng tin cậy.
- Tách hai câu hỏi: nhận dạng tiếng Việt phổ thông (G0) và nhận dạng từ tiếng Anh trong câu tiếng Việt (C0, C1).
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
| C1 | `ggroup_tech` / `edu` / `finance` / `pubadmin` | như G0 | test, câu có từ Anh | 465 / 346 / 294 / 155 | 0.46 / 0.39 / 0.30 / 0.16 | không có id | CC-BY-NC-4.0 |
| E0 | `fleurs_vi` | google/fleurs vi_vn (`70bb2e8`) | test | 857 | 3.00 | không có id | CC-BY-4.0 |
| E0 | `cv17_vi` | fsicoli/common_voice_17_0 (`8262c16`, mirror CV 17.0) | test | 1 274 | 1.35 | 146 | CC0-1.0 |
| E0 | `vietmed` | leduckhai/VietMed (`f302146`) | test | 3 437 | 6.02 | 27 | MIT (theo HF) |
| E0 | `phoaudiobook` | thivux/phoaudiobook (`f19f241`, gated) | test | 383 | 1.24 | 40 | điều khoản riêng, không phân phối lại |
| E0 | `lsvsc` | doof-ferb/LSVSC (`269de06`) | test | 5 683 | – | không có id | CC-BY-4.0 |
| E0 | `bud500` | linhtran92/viet_bud500 (`4b30571`, gated) | test | 7 500 | – | không có id | CC-BY-NC-SA-4.0 |
| E0 | `gigaspeech2_vi` | speechcolab/gigaspeech2 vi (`8dc0d0e`, gated) | test | 4 422 | – | không có id | điều khoản riêng, không phân phối lại |
| E0 | `vimd` | nguyendv02/ViMD_Dataset (`3a5b301`) | test | 2 026 | – | 1 344 | CC-BY-NC-ND-4.0 |
| E0 | `vimedcss_test` | tensorxt/ViMedCSS (`b6959a1`) | test | 1 614 | – | 995 video | CC-BY-4.0 |
| E0 | `speech_massive_vi` | FBK-MT/Speech-MASSIVE vi-VN (`ff792fe`) | validation | 2 033 | – | 28 | CC-BY-NC-SA-4.0 |
| E0 | `speech_massive_test_vi` | FBK-MT/Speech-MASSIVE-test vi-VN (`2e6d34f`, gated) | test | 2 974 | – | 30 | CC-BY-NC-SA-4.0 |

Tổng G0: 4 400 câu / 4.47 h; C0: 658 câu / 1.29 h; C1: 1 260 câu / 1.31 h. G0 và C0 lấy **toàn bộ** split công bố,
không lọc câu nào; số câu khớp số công bố của từng bộ. C1 là **tập con** của phần G-Group trong G0 (cùng audio, 1 260 /
1 260 sha256 có trong G0): mọi câu G-Group test có ít nhất một từ được gắn `<en>` theo quyết định thuật ngữ (mục 4).

E0: 32 203 câu, lấy **toàn bộ** split công bố của 11 bộ, không lọc câu nào (ViMD: 2 026 câu = tổng số dòng 14 shard
parquet; API thống kê của HF báo 1 550 là sai). Lời thoại = nhãn gốc; riêng ViMedCSS test có thêm tag `<en>` quanh
đúng các thuật ngữ trong `cs_terms_list` của dataset (không đổi chữ, `text_orig` = bản gốc). Giờ "–": bộ được khoá
**chỉ bằng metadata** (mục 4), chưa đo thời lượng.

Mỗi dòng manifest: `id`, `dataset`, `upstream_id`, `upstream_split`, `text_tagged` (lời thoại; C0 có tag `<en>`),
`text_tagged_orig` (C0: tag gốc của ViMedCSS), `text_orig` (C1: lời thoại gốc G-Group, không tag), `tag_review` (C1),
`duration`, `audio_sha256`, `audio_derivation`, `speaker`, `source`,
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
- **Đã kiểm tái dựng (17/11/2026):** `fetch` tải lại từ nguồn gốc cho G0 4 400 / 4 400 câu và C0 658 / 658 câu trùng
  sha256 từng byte (kể cả ViMedCSS resample, cùng soxr 1.1.0 / libsndfile 1.2.2); C1 1 260 / 1 260 câu (từ parquet
  G-Group đã khoá sha256).
- **Tag `<en>` (C0)**: `text_tagged` theo quy ước của gipbench: tag theo **cách đọc** (viết tắt đọc kiểu Anh = en,
  đọc kiểu Việt = không; brand Latin = en; từ mượn có trong từ điển tiếng Việt như "vitamin" = không). `text_tagged_orig`
  là tag gốc của ViMedCSS. Hai bản lệch nhau ở 339 / 658 câu; báo cáo nên ghi rõ đã chấm theo bản nào
  (`evaluate --label-field text_tagged_orig` để chấm theo tag gốc).
- **Tag `<en>` (C1)**: G-Group không có tag; gipbench gắn tag theo **thuật ngữ**, mỗi thuật ngữ một quyết định en / vi /
  bỏ tag do người duyệt chốt sau khi nghe câu mẫu (`configs/dataset/labeling/term_decisions.tsv`, 445 thuật ngữ: 347 en, 98 vi; duyệt
  16–17/11/2026; `tag_review: term_decisions`, 1 251 câu). Thuật ngữ đọc lúc Anh lúc Việt (chỉ có `ai`: "AI" hay đại
  từ "ai") được duyệt **từng câu** (9 câu, `tag_review: ok / fix`). Lời thoại giữ nguyên chữ của G-Group (`text_orig`).
  Ví dụ quyết định: viết tắt đọc kiểu Việt như `vn`, `gov`, `fpt`, `bidv` = vi; `chat`, `hip` = en.
- **Chuẩn hoá khi chấm**: hồ sơ `gipbench_strict_v1`, áp đối xứng cho reference và hypothesis: NFC, chữ thường, số thành
  chữ tiếng Việt, bỏ dấu câu, thống nhất vị trí dấu thanh (hoà = hòa), i/y có dấu (kỹ = kĩ), viết tắt có chấm (A.I. = ai),
  email / tên miền, dấu nháy, chính tả Anh-Anh / Anh-Mỹ. Định nghĩa đầy đủ: `docs/evaluation_metrics_v1.md`.

- **E0, bộ khoá bằng metadata** (`audio_derivation: pinned_source_extract`: LSVSC, Bud500, GigaSpeech2, ViMD,
  ViMedCSS test, Speech-MASSIVE dev / test): lúc khoá, id + lời thoại + meta được đọc thẳng từ file nguồn ở revision cố
  định mà không tải audio (mạng chậm). Không có `audio_sha256` / `duration` từng câu; toàn vẹn audio dựa vào sha256 của
  file nguồn (fetch kiểm trước khi tách) và cách tách cố định (giữ byte, hoặc resample soxr HQ 16 kHz mono khi gốc > 16
  kHz / stereo). Đã đối chiếu trên PhoAudiobook: dòng metadata trùng dòng nhập đầy đủ ở 383 / 383 câu.
- **Lời thoại không được phân phối lại** (PhoAudiobook, GigaSpeech2: `text_redistributable: false`): gói share bỏ lời
  thoại của hai bộ này; người nhận đồng ý điều khoản trên Hugging Face rồi `fetch` lấy lại cả audio lẫn lời thoại từ
  nguồn; manifest sau fetch trùng `manifest_sha256` bản khoá.

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
2. **ViMedCSS không tách theo video, nên C0 và E0 `vimedcss_test` KHÔNG đo tổng quát** cho model đã train trên
   ViMedCSS train (VieGem, mọi FT dùng ViMedCSS):
   - C0: **388 / 658 câu (59.0%) đến từ 248 video YouTube có cả trong ViMedCSS train** (đính chính 17/11/2026: bản
     trước ghi 308 / 201, không tái lập được từ manifest nào; số mới tính bằng khoá nguồn chuẩn hoá
     `core.decontam.source_key` so với `pool/vimedcss_train@full`);
   - E0 `vimedcss_test`: **1 514 / 1 614 câu (93.8%)**, 898 / 995 video;
   - C0 và E0 `vimedcss_test` chung 113 video.
   Chỉ lát **video chưa thấy** đo được tổng quát: C0 v1.1 `train_video_overlap:False` (270 câu / 156 video), E0 v1.1
   (100 câu). Kết quả trên lát này: mục 8b.
3. **VIVOS dùng lại câu đọc** giữa train và test (~194 / 760 câu test có cùng lời với câu train, người đọc khác).
4. **G-Group không có id người nói**: CI tính theo câu, hẹp hơn thực tế nếu một người nói nhiều câu.
5. Lời thoại chưa được gipbench duyệt lại; tag `<en>` của gipbench là tự động + quy ước, đang được duyệt.
6. Phân bố domain lệch: G0 nghiêng về giọng đọc (VIVOS + CV = 2.5 / 4.47 h); C0 chỉ có y khoa.
7. **C1 trùng audio với G0** (tập con G-Group): không cộng gộp C1 với G0 như hai bằng chứng độc lập.
   Tương tự **CV17 (E0) dùng lại 1 024 / 1 274 clip của CV27 test (G0)**: chỉ có để so với số model card VieGem.
8. **C1 chấm không catalog** (strict, không quy đổi biến thể chính tả của từ Anh). Mọi lệch chữ của từ Anh đều tính lỗi.
9. **Kết quả phụ thuộc cách ghép batch.** Với cùng audio, Gipformer (sherpa-onnx) và VieGem (bf16, batch 8) cho hyps
   hơi khác khi câu được ghép batch khác: VieGem trên C1 trùng hyps G0 ở 1 177 / 1 260 câu, WER lệch 0.03 điểm,
   CS-WER 0.24 điểm. Config Gipformer của bench chạy batch 1 (trùng 1 260 / 1 260); báo kèm `run_meta.json`.
10. **E0 trùng nguồn train của VieGem** ở FLEURS, CV17, PhoAudiobook, LSVSC, Bud500, GigaSpeech2, ViMD, ViMedCSS
    (model card VieGem liệt kê các bộ này trong dữ liệu train, "đã lọc theo câu test"). VietMed và Speech-MASSIVE không có
    trong danh sách đó. Mức trùng của Gipformer: unknown (không công bố dữ liệu train).
11. **E0 chưa có kết quả tham chiếu** (17/11/2026: đang chạy Gipformer + VieGem trên E0 v1.1, trừ ViMD đang tải audio).
12. **Bộ trùng nhau giữa các track**: C1 ⊂ G0 (1 260 câu cùng audio); CV17 test (E0) chứa 1 024 / 1 274 clip của CV27 test
    (G0), 143 / 146 người nói; C0 ∩ E0 `vimedcss_test` 113 video. Số câu **không trùng** của cả gói là **36 237**
    (= 4 400 + 658 + 32 203 − 1 024; đính chính: bản trước ghi 37 261, chưa trừ clip CV trùng).

## 8. Kết quả tham chiếu (v1.0, open_vocab, hồ sơ gipbench_strict_v1, `evaluate --mode final`, %)

| Bộ | Gipformer 1.5 (68M, beam 4) | VieGem-ASR (359M, greedy) |
| --- | --- | --- |
| VIVOS | 4.51 | 6.68 |
| Common Voice 27 | 7.47 | 9.50 |
| G-Group tech / edu / finance / pubadmin | 27.70 / 23.52 / 23.11 / 15.28 | 36.46 / 33.70 / 34.60 / 29.02 |
| ViMedCSS hard: WER / CS-WER | 21.92 / 107.12 | 21.05 / 84.54 |
| C1 G-Group CS (1 260 câu): WER / CS-WER | 24.66 / 70.25 | 34.38 / 82.39 |
| C1 term recall | 39.57 | 23.52 |
| C1 CS-WER tech / edu / finance / pubadmin | 75.25 / 71.12 / 66.92 / 52.33 | 85.21 / 80.92 / 82.77 / 72.48 |

Đọc kèm mục 7: G-Group (G0, C1) trùng tổ chức với Gipformer; ViMedCSS trùng dữ liệu train với VieGem. Δ ghép cặp
VieGem − Gipformer trên C1: CS-WER +12.13 [9.24, 14.91], WER +9.73 [8.82, 10.56].

### 8b. C0 tách theo video đã thấy khi train (code_switch v1.1, draft, cùng hyps với mục 8; 17/11/2026)

| Lát C0 | Câu | Gipformer WER / CS-WER / CS-WER S+D / term recall | VieGem WER / CS-WER / CS-WER S+D / term recall |
| --- | --- | --- | --- |
| Video có trong ViMedCSS train | 388 | 22.12 / 107.71 / 88.44 / 9.10 | 20.66 / 84.35 / 71.77 / **20.75** |
| Video chưa thấy khi train | 270 | 21.63 / 105.88 / 87.53 / 10.41 | 21.60 / 84.94 / 77.41 / **10.09** |

Δ ghép cặp VieGem − Gipformer (CI95, bootstrap theo video):

| Lát | WER | CS-WER |
| --- | --- | --- |
| Video có trong train | −1.46 [−3.11, −0.14] | −23.36 [−32.06, −12.97] |
| Video chưa thấy | **−0.03 [−1.26, +1.27]** | −20.94 [−28.61, −11.97] |

Cách đọc: lợi thế WER và term recall của VieGem trên C0 nằm ở video nó đã nghe khi train; trên video chưa thấy, WER
hai model ngang nhau và term recall của VieGem giảm một nửa (20.75 → 10.09). Lợi thế CS-WER còn lại một phần do VieGem
học đúng quy ước viết từ Anh của ViMedCSS (đổi bản tag làm CS-WER đổi ~6 điểm theo hai chiều; `docs/evaluation_metrics_v1.md`
mục 2b), không chỉ do nhận dạng tốt hơn.

## 9. Phiên bản và bảo trì

- `bench_version` trong mỗi dòng; `release.json` và `SHA256SUMS` trong gói. Đổi bất kỳ câu / nhãn / chuẩn hoá nào =
  phiên bản mới; bản đã phát hành không bị sửa.
- v1.0 = G0 + C0 + C1 như trên; v1.1 = v1.0 + E0 (`external_v1`, 32 203 câu, `manifest_sha256`
  `f074d27fb646d2fb376eca1189b3fb05374860276c1463657286d35c26e0a1dd`). Bộ riêng tư (consensus) **không** share,
  chỉ chấm hộ.

  | Benchmark | Track | Câu | manifest_sha256 |
  | --- | --- | --- | --- |
  | `general_v1` | G0 | 4 400 | `e908c75a3f23c949b2ae0f2b82d6ec98e75eca1204f07dd44a50139e352c3dcb` |
  | `code_switch_v1` | C0 | 658 | `744d4773d4a0f01161f7ff5694b8a8e3466f93efaff91ebb74750e70388dc06a` |
  | `cs_ggroup_v1` | C1 | 1 260 | `e3c3cc1b91027fefc529dbfe2bb23d1b52b0948899d17af0b1cbcb9edac496af` |

  C0 khoá cả catalog chấm (`catalog_sha256` `a21c590e…`). manifest_sha256 tính trên mọi trường trừ đường dẫn `audio`,
  nên manifest do `fetch` ghi ra (thêm đường dẫn audio cục bộ) vẫn khớp bản khoá.
- **Phiên bản benchmark v1.1 (draft, 17/11/2026).** Lưu ý tên: "bench v1.1" ở trên là phiên bản **gói**; `code_switch v1.1`
  / `external v1.1` là phiên bản **từng benchmark**, sẽ vào gói kế tiếp khi được khoá. Cùng câu, lời thoại, chuẩn hoá với
  v1.0 (bản khoá v1.0 giữ nguyên); dựng bằng `python -m gipbench.dataset.release.derive_v1_1`.

  | Benchmark | Khác v1.0 | manifest_sha256 lúc dựng | Trạng thái |
  | --- | --- | --- | --- |
  | `code_switch_v1_1` | thêm `train_video_overlap`; nhóm báo cáo theo trường này; `alt_label_fields: [text_tagged_orig]` | `3af637ff…` | draft, chờ duyệt khoá |
  | `external_v1_1` | `audio` + `audio_sha256` + `duration` thật (5 951 sha256 của v1.0 khớp lại sha256 lúc tải); speaker / source ViMedCSS test chuẩn hoá `yt_<ID>` / `youtube:<ID>` + `train_video_overlap`; bootstrap FLEURS theo `sentence_id` | `723d5cac…` (đổi khi ViMD tải xong) | draft: ViMD còn 1 736 câu chưa có audio; khoá cần đủ sha256 (`require_audio_sha`) |

- **Đính chính 17/11/2026** (audit `reports/audits/full_research_audit_vi.md`): số trùng video ViMedCSS 308 / 201 → **388 / 248**;
  số câu không trùng của gói 37 261 → **36 237**. Không bộ khoá nào bị sửa.
- License của gói = license của từng bộ nguồn (mục 2); phần phái sinh từ VIVOS phân phối theo CC-BY-NC-SA-4.0.
  Trích dẫn từng bộ nguồn khi dùng.
