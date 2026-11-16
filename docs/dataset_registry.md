# Registry nguồn dữ liệu đánh giá

File máy đọc: `configs/bench/datasets.yaml` (kiểm bằng `python -m gipbench.bench.registry`). Bảng dưới tóm tắt; mọi
id / revision / license / số câu đã kiểm qua Hugging Face API ngày 16/11/2026. Giao thức: [benchmark_protocol_v1.md](benchmark_protocol_v1.md).

## 1. Đang dùng (có trên máy, đã dựng manifest)

| Khoá | Nguồn (revision) | License | Split | Câu (công bố / trong manifest) | Người nói | Trùng pretraining: Gipformer / VieGem |
| --- | --- | --- | --- | --- | --- | --- |
| `vivos` | AILAB-VNUHCM/vivos (`3cbfb25`) | CC-BY-NC-SA-4.0 | test | 760 / 760 | 19 (id thật) | unknown / not_listed |
| `commonvoice_vi_27` | Common Voice 27.0 (2026-09-11), tải tay | CC0-1.0 | test | 1 593 / 1 593 | 221 (client_id) | unknown / known_possible (VieGem train CV17) |
| `ggroup_tech` | g-group-ai-lab/vi-asr-tech-test (`0b8c874`) | CC-BY-NC-4.0 | test | 508 / 508 | không có id | known_possible (cùng tổ chức) / not_listed |
| `ggroup_edu` | g-group-ai-lab/vi-asr-edu-test (`9a15ad6`) | CC-BY-NC-4.0 | test | 495 / 495 | không có id | known_possible / not_listed |
| `ggroup_finance` | g-group-ai-lab/vi-asr-finance-test (`ebadeb0`) | CC-BY-NC-4.0 | test | 529 / 529 | không có id | known_possible / not_listed |
| `ggroup_pubadmin` | g-group-ai-lab/vi-asr-pubadmin-test (`40aabb0`) | CC-BY-NC-4.0 | test | 515 / 515 | không có id | known_possible / not_listed |
| `vimedcss` | tensorxt/ViMedCSS (`b6959a1`) | CC-BY-4.0 | hard | 658 / 658 | 404 video | unknown / known_possible (VieGem train ViMedCSS train) |
| `consensus_private` | data/raw/consensus | test_only (riêng tư) | – | (B2 + 1 081 câu B5 trong v0-draft) | speaker_cluster ước lượng | unknown / unknown |

Manifest: `${data_root}/benchmarks/general_v1_draft/manifest.jsonl` (4 400 câu, 4.47 h, manifest_sha256
`781fe7a0…`), `${data_root}/benchmarks/code_switch_v1_draft/manifest.jsonl` (658 câu, 1.29 h, `35151e98…`). Không câu nào
bị loại; số câu khớp split công bố. `data/` nằm ngoài git; dựng lại bằng `gipbench.dataset.release.build_general`.

## 2. Chưa dùng (ghi để không bị thay thế âm thầm)

| Khoá | Nguồn | Trở ngại | Cách gỡ |
| --- | --- | --- | --- |
| `fleurs_vi` | google/fleurs (`70bb2e8`), CC-BY-4.0, config vi_vn | chưa tải; dataset_info trên HF không ghi riêng số câu vi_vn | tải split test vi_vn, viết converter, kiểm số câu |
| `vimedcss` test | tensorxt/ViMedCSS split test (1 614 câu) | chỉ có 20 câu (`--limit 20`) | chạy `gipbench.dataset.ingest.vimedcss` không `--limit` cho split test |
| `gigaspeech2_vi` | speechcolab/gigaspeech2 (`8dc0d0e`) | gated (auto-accept), cần token HF đã đồng ý; audio YouTube | người dùng đồng ý điều khoản trên HF rồi tải test vi |
| `bud500` | linhtran92/viet_bud500 (`4b30571`), CC-BY-NC-SA-4.0 | gated (auto-accept); split / số câu chưa đọc được | như trên |
| `vimd` | nguyendv02/ViMD_Dataset (`3a5b301`), CC-BY-NC-ND-4.0 | chưa tải; split chưa kiểm | tải, kiểm split test |
| MU-Bench vi-VN | – | chưa kiểm quyền truy cập audio / license | để tuỳ chọn |

## 3. Phát hiện cần biết khi đọc kết quả

1. **ViMedCSS không tách theo video.** 308 / 658 câu hard đến từ 201 video YouTube **có cả trong ViMedCSS train**
   (leak_audit so với `data/e3_v1b/train_curated.jsonl`). Mọi model đã train trên ViMedCSS train (VieGem, E3) đã nghe
   cùng người nói / cùng video. C0 trên ViMedCSS vì vậy không phải đo tổng quát sang người nói mới.
2. **VIVOS dùng lại câu đọc giữa train và test.** Khoảng 194 / 760 câu test có cùng text (sau chuẩn hoá) với câu VIVOS
   train (người đọc khác). Model FT trên VIVOS train (E3) có lợi thế về text ở các câu này. Common Voice: 3 câu.
3. **G-Group không có id người nói**: bootstrap theo câu; không kiểm được trùng người nói giữa các bộ.
4. **G-Group do chính tổ chức phát hành Gipformer công bố**, và model card Gipformer báo số trên đúng các bộ này. Coi là
   bộ tham chiếu ngoài (external / known_possible), không phải bằng chứng tổng quát độc lập của Gipformer.
5. Model card VieGem ghi dữ liệu train "đã lọc theo mọi câu test và dev" (lọc theo câu, không theo người nói / video).

Báo cáo đầy đủ: `results/general_asr/leak/<benchmark>/leak_report.{md,json}` (ngoài git; lệnh trong
`reports/benchmark_audit_v1.md`).
