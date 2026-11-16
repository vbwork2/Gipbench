# Mục D: confusable từ nguồn có nhãn người

Ngày 15/11/2026, chạy trên CPU. Quyết định: `decisions.md` §12.6.

- **Bỏ bản cũ:** bản cũ lấy câu từ nhãn giả VSS (227 câu bench, 729 câu train). Đã bị ghi đè; bản sao nằm ở `data/tmp/prev_confusable/`.
- **Nguồn mới:**
  - VIVOS train (11 658 câu, 14.92 h; đã convert toàn bộ, xem mục F);
  - VIVOS test (760 câu);
  - consensus B5 product (1 081 câu).
- **Phần train chỉ lấy từ VIVOS train** (`--train-from`). Consensus (test_only) và VIVOS test chỉ được vào bench.

```bash
python tools/make_confusables.py --manifest data/raw/vivos/train_vivos.jsonl data/raw/vivos/b5_vivos.jsonl \
    data/raw/consensus/b5_consensus_product.jsonl --train-from data/raw/vivos/train_vivos.jsonl \
    --bench-not-from-train --max-per-group-bench 10 --bench data/raw/ggroup/b1_candidates.jsonl \
    --catalog data/raw/_proxy/catalog_proxy.tsv --out-dir data/raw/confusable
```

| | Câu | Giờ | Ghi chú |
| --- | --- | --- | --- |
| Bench B5 confusable (cần duyệt) | **78** | 0.11 | 40 người nói: VIVOS test 54, consensus 24 |
| Train (ví dụ âm) | **620** | 0.90 | Chỉ VIVOS train |
| Câu train trùng người nói / nguồn với bench | 0 | | Bắt buộc = 0 |

## Phiếu duyệt: chỉ cần ĐỌC text

`data/raw/confusable/b5_confusable_review.csv` có các cột: `id`, `text`, `form`, `term`, `câu có` (vd `[tìm] ~ team`), **`nghĩa Việt? y/n`**, `note`, `dataset`, `speaker`.

- Điền `y` nếu từ đó mang nghĩa tiếng Việt (nhận vào bench); điền `n` nếu thực ra là từ tiếng Anh (bỏ).
- `build_bench_v0.py` đọc cột này như `decision` = ok / drop.
- Khoảng 5 giây mỗi câu, tức khoảng 7 phút cho cả phiếu.

## Lựa chọn em đã làm: bench không lấy người nói VIVOS train (`--bench-not-from-train`)

VIVOS train còn là nguồn replay có nhãn người (`replay_human`, 0.25). Người nói nào lên bench thì phải loại khỏi replay, nếu không thì rò rỉ người nói.

Em đã chạy thử khi cho phép bench lấy cả VIVOS train:
- bench được 191 câu;
- nhưng 38 / 46 người nói VIVOS train lên bench, nên replay mất **9 773 câu, 11.94 / 14.92 h**.

Vì vậy em giữ bench chỉ từ nguồn không dùng để train. Bench nhỏ hơn mức `--bench-n 250`, vì chỉ còn khoảng 85 ứng viên từ VIVOS test và consensus, và mỗi dạng gần âm chỉ được tối đa 10 câu.

Nếu cần bench lớn hơn, có hai cách:
- tách riêng vài người nói VIVOS train cho bench và loại họ khỏi replay;
- thêm nguồn có nhãn người khác, vd Common Voice.

## Thay đổi code

- `make_confusables.py`:
  - thêm `--train-from` và `--bench-not-from-train`;
  - nhóm người nói của nguồn chỉ-bench gom theo `speaker_cluster` / `speaker`, và chỉ gộp theo `source` khi source đó có ở phía train (consensus dùng chung `source = consensus`);
  - chỉ chặn rò rỉ cho câu được phép vào train;
  - phiếu chỉ đọc text.
- `build_bench_v0.py`:
  - hiểu cột `nghĩa Việt? y/n`;
  - bỏ câu trùng audio với câu đã nhận, vì một clip consensus có thể vừa là B5 product vừa là B5 confusable. Câu confusable chưa duyệt không được nhận, nên bản B5 product vẫn được giữ.

Báo cáo chi tiết theo dạng (sinh tự động): `data/raw/confusable/make_confusables.report.md`.
