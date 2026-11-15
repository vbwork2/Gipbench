# Mục F: replay có nhãn người (VIVOS đầy đủ; Common Voice)

Ngày 15/11/2026, chạy trên CPU. Không có thay đổi code.

| Bộ | Lệnh | Train (replay) | B5 (test) | Bỏ vì có từ Anh |
| --- | --- | --- | --- | --- |
| VIVOS | `conv_local.py vivos --root data/dl/vivos --out data/raw/vivos` | **11 658 câu, 14.92 h** | 760 câu, 0.75 h | 2 |
| Common Voice 27.0 vi | `conv_local.py commonvoice --root data/dl/cv-corpus-27.0-2026-09-11/vi --out data/raw/cv` | **1 807 câu, 2.27 h** | 1 568 câu, 1.72 h | 66 |

Cả hai lệnh đều chạy với `--catalog data/raw/_proxy/catalog_proxy.tsv`.

## VIVOS

- Trước đây máy chỉ có 20 + 20 wav trích ra.
- Lần này tải bản đầy đủ từ HF `AILAB-VNUHCM/vivos`:
  - file `data/vivos.tar.gz`, 1.47 GB, giấy phép CC-BY-NC-SA-4.0;
  - lưu ở `data/dl/vivos.tar.gz`;
  - giải nén vào `data/dl/vivos/` (12 420 wav, 1.8 GB).
- File tar.gz vẫn được giữ, chiếm 1.4 GB trên D:; xoá được nếu cần chỗ.

## Common Voice

- Thư mục `data/dl/cv-corpus-27.0-2026-09-11/vi` đã có đầy đủ (20 605 clip, do anh/chị tải ở Phase 0b), nên em convert luôn theo dặn "thêm sau nếu tôi tải về".
- Split của CV vốn đã tách người nói.

## Nhóm `replay_human` trong `train_mix.tsv`

- Hiện có 17.19 h: VIVOS 14.92 h + CV 2.27 h.
- Người nói VIVOS train không có ai trong bench confusable (mục D dùng `--bench-not-from-train`).
