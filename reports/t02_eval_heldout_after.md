# Đánh giá auto_tag — reports/t02_gold_heldout.jsonl

Catalog: `configs/catalog.example.tsv`. 102 câu đã đọc, 60 câu có span tiếng Anh. Ngoài ra: 6 span viết tắt tiếng Việt, 4 span phiên âm Việt trong 4 câu (không tính vào recall). So span: gộp span kề nhau thành dải liền; không gộp thì P/R trên câu có từ Anh = 0.925/0.970.

| Tập | Span vàng | Span dự đoán | Precision | Precision (bỏ qua viết tắt VN) | Recall | F1 | Token P | Token R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Câu có từ Anh (60) | 101 | 104 | 0.962 | 0.990 | 0.990 | 0.976 | 0.980 | 0.993 |
| Mọi câu đã đọc (102) | 101 | 108 | 0.926 | 0.980 | 0.990 | 0.957 | 0.948 | 0.993 |

## Span vàng không khớp (1)

| id | span vàng | loại |
| --- | --- | --- |
| gg_tech_115499 | docker run | chồng lấn |

## Span dự đoán không khớp (8)

| id | span dự đoán | loại |
| --- | --- | --- |
| gg_pubadmin_114769 | abc | vi_abbr |
| gg_finance_115822 | vps | vi_abbr |
| gg_finance_113150 | bvs bvd | vi_abbr |
| gg_tech_115499 | docker | chồng lấn |
| gg_edu_115934 | vn | vi_abbr |
| gg_edu_113618 | alo | thừa |
| gg_pubadmin_113554 | cm | vi_abbr |
| gg_pubadmin_114541 | vn | vi_abbr |
