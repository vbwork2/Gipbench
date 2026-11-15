# Bộ consensus (`data/dl/zac`): converter, chặn rò rỉ, lát cắt category

**Bộ này CHỈ ĐƯỢC DÙNG ĐỂ TEST** (`license: test_only`). Không có speaker id.

```bash
python converters/conv_consensus.py --root data/dl/zac --out data/raw/consensus \
    --catalog data/raw/_proxy/catalog_proxy.tsv --max-per-term 10        # chạy CPU, ~3 phút
```

Báo cáo chi tiết do converter sinh ra: `data/raw/consensus/consensus_report.md`, số liệu ở `consensus_stats.json`.
Catalog dùng tạm là `catalog_proxy.tsv` vì `catalog_v0` chưa có, nên đa số span vẫn nằm "(ngoài catalog)".

## 1. Audio

- Luật ghép do converter tự dò: turn `final` → `audio/<id>/<id>.wav`, các turn khác → `audio/<id>/<id>_<turn>.wav`.
- Ghép được **1933/1933** clip. **Thiếu audio: 0**. Không có file mơ hồ và không có file thừa.
- Toàn bộ audio đã ở dạng 16 kHz mono PCM16, nên không phải resample và manifest trỏ thẳng vào file gốc (đường dẫn tương đối). Nhánh resample + gộp kênh đã có kiểm thử với file 44.1 kHz stereo.
- Tổng 2.76 h; clip dài 1.1–20.2 s.

## 2. Trạng thái và số câu

| Nhóm | Câu | Giờ |
| --- | --- | --- |
| agreed, có `<en>` → **B2** (trước trần) | 290 | 0.51 |
| **B2 sau `--max-per-term 10`** | **231** | **0.42** |
| bị cắt bởi `--max-per-term 10` (ghi ra `b2_consensus_overcap.jsonl`) | **59** | 0.10 |
| agreed, Việt thuần → **B5 domain product** | **1046** | **1.31** |
| disputed (472) + pending (113) → phiếu duyệt | 585 | 0.91 |
| escalate → bỏ | 12 | – |

- Không tách dev. Mọi dòng có `speaker = consensus:<id phiên>`, `source = consensus`, `domain = product`, `license = test_only`, `dataset = consensus`.
- Text đi qua các bước: số → chữ, `auto_tag` (sau `vi_abbrev` / `vi_loanwords`), rồi chuẩn hoá như `normalize_ref` (viết thường, bỏ dấu câu, không áp bảng biến thể của catalog).
- B5: 19/1046 câu có cờ nên phải duyệt (vd `numbers_disagree` khi `numbers_majority = False`, `vi_word:*`). 1027 câu không có cờ, `build_bench_v0.py` nhận thẳng.
- Disputed / pending: đoán split là 148 B2, 437 B5. Phiếu `disputed_candidates_review.csv` có cột `hint` chứa `listener_texts`, và `review_bench.html` hiện cột này làm gợi ý. Các dòng này mang `review_required: true`, nên `build_bench_v0.py` chỉ nhận chúng khi decision = ok / fix, kể cả khi chạy `--accept all`. Khi người duyệt `fix`, split được đặt lại theo text đã sửa (`split_by_tag`).
- Phiếu duyệt B2 (`b2_consensus_candidates_review.csv`, 231 câu) vẫn cần thiết, vì `<en>` chỉ là nhãn nháp do `auto_tag` gắn. Ví dụ `sjc`, `sms`, `ok`, `st`, `rv`, `tv` có cờ `abbr_reading`: người duyệt nghe rồi quyết đọc kiểu Anh hay kiểu Việt.

## 3. Thuật ngữ B2

137 thuật ngữ khác nhau sau trần. Top 30, kèm số câu sau trần:

zalo 10 · sjc 10 · sms 10 · ok 10 · email 10 · check 9 · pizza 6 · st 5 · metro 5 · sarah 5 · rv 4 · shop 4 · online 4 ·
pancake 4 · zkn 4 · vitech 4 · mr tanaka 4 · app 3 · book 3 · list 3 · note 3 · workshop 3 · deadline 3 · check in 3 ·
lucas 3 · homestay 3 · laptop 2 · sctv 2 · jetxanh 2 · tv 2

`--max-per-term 10` cắt 5 thuật ngữ, tổng **59 câu**. Câu giữ lại được chọn xoay vòng theo phiên và ưu tiên ngữ cảnh chưa gặp (dùng lại `pick_diverse` của `curate_train`):

| Thuật ngữ | Câu trước | Câu sau |
| --- | --- | --- |
| sjc | 32 | 10 |
| email (gộp cả `mail`, theo variants của catalog) | 24 | 10 |
| zalo | 23 | 10 |
| sms | 18 | 10 |
| ok | 12 | 10 |

## 4. Category (span B2 sau trần)

| Category | Span |
| --- | --- |
| (ngoài catalog) | 197 |
| **name** | **34** |
| tech | 18 |
| brand | 10 |
| business | 8 |
| daily | 3 |
| edu | 3 |
| finance | 3 |

- Tên người nước ngoài lấy từ `configs/en_names.tsv` (đây là tham số, có thể sửa).
- Tên được gắn `<en>` nếu có ít nhất một token không đọc được như âm tiết Việt (`lucas`, `tanaka`, `lambert`…), hoặc là danh xưng + tên (`mr tanaka`, `mrs watanabe`).
- Tên trùng âm tiết Việt (`kim`, `chen`, `sam`) chỉ được nhận khi đứng cạnh một tên như vậy (vd `michael chen`, `sarah kim`). Một mình thì chỉ nhận cờ `maybe_name:*`. Ví dụ `sam kim` hiện là `maybe_name`, có 2 câu.
- Kết quả được ghi vào `term_category {"<span>": "name"}`, kèm cờ `name:<span>` để người duyệt xác nhận. Có 33 câu B2 và 30 câu disputed mang tên người.
- Các brand giả lập (`brightpath holdings`, `vertex software`…) chưa có category và nằm trong "(ngoài catalog)". Muốn cắt lát brand thì cần thêm chúng vào catalog.
- `score.py`: category lấy từ `term_category` trước, rồi tới catalog. `slices.md` có thêm bảng "Category thuật ngữ (B2, theo span)" bên cạnh bảng B1–B4.

## 5. Chặn rò rỉ

- `data/raw/consensus/leak_block.txt` chứa **4209 câu** đã chuẩn hoá, lấy từ `consensus_text` và mọi `listener_texts` của mọi trạng thái, kể cả escalate. Đây là đường dẫn mặc định trong `lib/leak.py` (`DEFAULT_BLOCK_FILES`).
- `ft/curate_train.py` có bước 0 luôn chạy: loại dòng `license` / `source` / `dataset` test_only (không tắt được), và loại câu trùng danh sách chặn (`--block-text` không kèm file để tắt riêng phần text).
- `tools/vss_cs_prefilter.py` và `tools/rebuild_vss_cs.sh` (thêm `--block-text`) chặn thêm test_only và danh sách chặn. `LeakGuard` dùng mặc định, nên `agreement_filter --relabel-spans` và `make_confusables` cũng được chặn.
- `validate.py`: báo **LỖI** "RÒ RỈ: dữ liệu test_only có trong train / dev" khi gặp `license = test_only`, hoặc source / dataset `consensus`, hoặc source của các dòng test_only trong bench. Có thêm `--dev`. Câu train / dev trùng danh sách chặn chỉ bị **cảnh báo**, vì câu ngắn như "ok" có thể trùng một cách vô hại.
- Kiểm tra trên dữ liệu thật (chỉ đọc): `train_vss` (28 551 câu), `dev_vss` (3 050), `vss_cs_candidates` (31 949), `vss_cs_clean` (229), `train_confusable` (729), `train_vimedcss` (20) **không có câu nào trùng**. `validate.py` trên bench nháp consensus (1277 câu, `--check-audio`) với `--train vss_cs_clean train_confusable --dev dev_vss` cho kết quả: **0 LỖI**.
- Cảnh báo dự kiến khi validate bench có consensus: "speaker có mặt ở nhiều split bench" (113 phiên có turn ở cả B2 và B5) và "source có mặt ở nhiều split bench" (`source = consensus` dùng chung, theo yêu cầu).

## 6. Speaker embedding (tuỳ chọn, CPU)

- Đã chạy bằng sherpa-onnx 2.0.1 (có sẵn trong venv) với model `wespeaker_en_voxceleb_resnet34.onnx` (tải từ release công khai của sherpa-onnx vào `models/spk/`, đã gitignore). Chạy 2 luồng CPU, `CUDA_VISIBLE_DEVICES=""`, không đụng GPU của ASR nền.
- **16 cụm từ 626 phiên** (1921 clip; 8 phiên chỉ có câu escalate). Ngưỡng cosine 0.757 = phân vị 5% của độ giống giữa các clip **cùng phiên**. Cách này cố ý gom mạnh tay: thà ít cụm, CI rộng, còn hơn chia sai làm CI hẹp giả.
- Độ giống cùng phiên: trung vị 0.888 (p5 0.756). Khác phiên: trung vị 0.655, nhưng p75 0.778 và p95 0.911. Nhiều cặp khác phiên giống nhau như cùng người, nên nhiều khả năng bộ này chỉ có ít người nói.
- Kích thước cụm lệch mạnh: 296, 186, 57, 51 phiên, còn lại ≤ 10. Số cụm nhạy với ngưỡng: 0.5 → 2, 0.6 → 4, 0.7 → 10, 0.8 → 22, 0.9 → 66 cụm. Không có nhãn thật để kiểm tra, nên hãy coi `speaker_cluster` là ước lượng thô.
- `speaker_cluster` chỉ dùng làm đơn vị bootstrap (`boot_unit`) trong `score.py` và `leaderboard.py`. Dòng không có trường này vẫn bootstrap theo `speaker`. Ánh xạ phiên → cụm nằm ở `data/raw/consensus/speaker_clusters.tsv`.
- Nếu không có thư viện hoặc model, `--spk-embed auto` bỏ qua bước này và ghi lý do vào báo cáo (đã có kiểm thử). `--spk-embed off` tắt hẳn.

## 7. Kiểm thử

- `tools/test_converters.py` có thêm `test_consensus`, dùng CSV giả cùng schema. Các trường hợp được kiểm:
  - dò luật ghép audio, gồm cả clip thiếu audio và file thừa;
  - resample 44.1 kHz stereo;
  - trạng thái agreed / disputed / pending / escalate;
  - tên người (category name), `numbers_disagree`, trần thuật ngữ;
  - phiếu duyệt có `hint`, nội dung `leak_block`;
  - `build_bench_v0` không nhận dòng `review_required` khi chưa duyệt, và đổi split sau khi fix;
  - chặn rò rỉ ở `curate_train` / `vss_cs_prefilter`; LỖI của `validate` qua cả `--train` và `--dev`;
  - lát cắt `name` và bootstrap theo cụm ở `score.py`.
- **`tools/test_converters.py` ĐẠT, `tools/selftest.sh` ĐẠT** (chạy trong WSL, CPU).
