# Đánh giá auto_tag — reports/t02_gold.jsonl

Catalog: `configs/catalog.example.tsv`. 161 câu đã đọc, 100 câu có span tiếng Anh. Ngoài ra: 13 span viết tắt tiếng Việt, 13 span phiên âm Việt trong 12 câu (không tính vào recall). So span: gộp span kề nhau thành dải liền; không gộp thì P/R trên câu có từ Anh = 0.892/0.913.

| Tập | Span vàng | Span dự đoán | Precision | Precision (bỏ qua viết tắt VN) | Recall | F1 | Token P | Token R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Câu có từ Anh (100) | 172 | 173 | 0.925 | 0.964 | 0.930 | 0.928 | 0.972 | 0.953 |
| Mọi câu đã đọc (161) | 172 | 178 | 0.899 | 0.964 | 0.930 | 0.914 | 0.949 | 0.953 |

## Span vàng không khớp (12)

| id | span vàng | loại |
| --- | --- | --- |
| gg_tech_114011 | run time | chồng lấn |
| gg_pubadmin_113391 | hi | bỏ sót |
| gg_tech_113606 | kit | bỏ sót |
| gg_edu_115507 | who tech | chồng lấn |
| gg_tech_115483 | decoin chain | chồng lấn |
| gg_finance_114842 | op bnb | chồng lấn |
| gg_edu_113171 | sing | bỏ sót |
| gg_finance_113633 | super government co contribution | chồng lấn |
| gg_pubadmin_113541 | chip | bỏ sót |
| gg_edu_113250 | sap | bỏ sót |
| gg_tech_116295 | mac | bỏ sót |
| gg_finance_115061 | hi | bỏ sót |

## Span dự đoán không khớp (18)

| id | span dự đoán | loại |
| --- | --- | --- |
| gg_tech_114011 | time | chồng lấn |
| gg_finance_115825 | vn | vi_abbr |
| gg_finance_113954 | plx | vi_abbr |
| gg_edu_115507 | who | chồng lấn |
| gg_edu_113283 | tp hcm | vi_abbr |
| gg_edu_113283 | nhs | vi_abbr |
| gg_tech_115483 | decoin | chồng lấn |
| gg_finance_114842 | bnb | chồng lấn |
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
