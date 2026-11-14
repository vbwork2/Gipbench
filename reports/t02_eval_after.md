# Đánh giá auto_tag — reports/t02_gold.jsonl

Catalog: `configs/catalog.example.tsv`. 161 câu đã đọc, 100 câu có span tiếng Anh. Ngoài ra: 13 span viết tắt tiếng Việt, 13 span phiên âm Việt trong 12 câu (không tính vào recall). So span: gộp span kề nhau thành dải liền; không gộp thì P/R trên câu có từ Anh = 0.912/0.959.

| Tập | Span vàng | Span dự đoán | Precision | Precision (bỏ qua viết tắt VN) | Recall | F1 | Token P | Token R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Câu có từ Anh (100) | 172 | 178 | 0.944 | 0.982 | 0.977 | 0.960 | 0.973 | 0.984 |
| Mọi câu đã đọc (161) | 172 | 183 | 0.918 | 0.982 | 0.977 | 0.946 | 0.951 | 0.984 |

## Span vàng không khớp (4)

| id | span vàng | loại |
| --- | --- | --- |
| gg_tech_114011 | run time | chồng lấn |
| gg_pubadmin_113391 | hi | bỏ sót |
| gg_finance_113633 | super government co contribution | chồng lấn |
| gg_finance_115061 | hi | bỏ sót |

## Span dự đoán không khớp (15)

| id | span dự đoán | loại |
| --- | --- | --- |
| gg_tech_114011 | time | chồng lấn |
| gg_finance_115825 | vn | vi_abbr |
| gg_finance_113954 | plx | vi_abbr |
| gg_edu_113283 | tp hcm | vi_abbr |
| gg_edu_113283 | nhs | vi_abbr |
| gg_finance_113633 | super government | chồng lấn |
| gg_finance_113633 | contribution | chồng lấn |
| gg_pubadmin_114306 | gdt | vi_abbr |
| gg_pubadmin_114306 | vn | vi_abbr |
| gg_pubadmin_113714 | bja | vi_abbr |
| gg_pubadmin_115028 | vn | vi_abbr |
| gg_finance_114096 | tb | vi_abbr |
| gg_finance_113985 | qns | vi_abbr |
| gg_pubadmin_114974 | vn | vi_abbr |
| gg_finance_114729 | bidv | vi_abbr |
