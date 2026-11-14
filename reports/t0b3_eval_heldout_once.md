# Đánh giá auto_tag — reports/t02_gold_heldout.jsonl

Danh sách từ không gắn <en>: `/mnt/d/code/ZAIC/gipbench/configs/vi_abbrev.txt` (sha256 c30283ef4047), `/mnt/d/code/ZAIC/gipbench/configs/vi_loanwords.txt` (sha256 e415e2abf361); 920 từ.

Catalog: `configs/catalog.example.tsv`. 102 câu đã đọc, 60 câu có span tiếng Anh. Ngoài ra: 6 span viết tắt tiếng Việt, 4 span phiên âm Việt trong 4 câu (không tính vào recall). So span: gộp span kề nhau thành dải liền; không gộp thì P/R trên câu có từ Anh = 0.924/0.960.

| Tập | Span vàng | Span dự đoán | Precision | Precision (bỏ qua viết tắt VN) | Recall | F1 | Token P | Token R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Câu có từ Anh (60) | 101 | 103 | 0.961 | 0.990 | 0.980 | 0.971 | 0.980 | 0.986 |
| Mọi câu đã đọc (102) | 101 | 106 | 0.934 | 0.980 | 0.980 | 0.957 | 0.960 | 0.986 |

## Span vàng không khớp (2)

| id | span vàng | loại |
| --- | --- | --- |
| gg_tech_115499 | docker run | chồng lấn |
| gg_finance_115704 | nab | bỏ sót |

## Span dự đoán không khớp (7)

| id | span dự đoán | loại |
| --- | --- | --- |
| gg_pubadmin_114769 | abc | vi_abbr |
| gg_finance_113150 | bvd | vi_abbr |
| gg_tech_115499 | docker | chồng lấn |
| gg_edu_115934 | vn | vi_abbr |
| gg_edu_113618 | alo | thừa |
| gg_pubadmin_113554 | cm | vi_abbr |
| gg_pubadmin_114541 | vn | vi_abbr |
