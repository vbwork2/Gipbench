# Đánh giá auto_tag — reports/t02_gold.jsonl

Danh sách từ không gắn <en>: `/mnt/d/code/ZAIC/gipbench/configs/vi_abbrev.txt` (sha256 2f603fcdb3a8), `/mnt/d/code/ZAIC/gipbench/configs/vi_loanwords.txt` (sha256 e415e2abf361); 928 từ.

Catalog: `configs/catalog.example.tsv`. 161 câu đã đọc, 100 câu có span tiếng Anh. Ngoài ra: 13 span viết tắt tiếng Việt, 13 span phiên âm Việt trong 12 câu (không tính vào recall). So span: gộp span kề nhau thành dải liền; không gộp thì P/R trên câu có từ Anh = 0.943/0.959.

| Tập | Span vàng | Span dự đoán | Precision | Precision (bỏ qua viết tắt VN) | Recall | F1 | Token P | Token R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Câu có từ Anh (100) | 172 | 172 | 0.977 | 0.982 | 0.977 | 0.977 | 0.996 | 0.984 |
| Mọi câu đã đọc (161) | 172 | 175 | 0.960 | 0.982 | 0.977 | 0.968 | 0.984 | 0.984 |

## Span vàng không khớp (4)

| id | span vàng | loại |
| --- | --- | --- |
| gg_tech_114011 | run time | chồng lấn |
| gg_pubadmin_113391 | hi | bỏ sót |
| gg_finance_113633 | super government co contribution | chồng lấn |
| gg_finance_115061 | hi | bỏ sót |

## Span dự đoán không khớp (7)

| id | span dự đoán | loại |
| --- | --- | --- |
| gg_tech_114011 | time | chồng lấn |
| gg_edu_113283 | nhs | vi_abbr |
| gg_finance_113633 | super government | chồng lấn |
| gg_finance_113633 | contribution | chồng lấn |
| gg_pubadmin_113714 | bja | vi_abbr |
| gg_finance_114096 | tb | vi_abbr |
| gg_finance_113985 | qns | vi_abbr |
