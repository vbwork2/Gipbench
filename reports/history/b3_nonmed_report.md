# Đề xuất reserved_b3 ngoài y khoa (CHƯA áp dụng)

Ứng viên đạt điều kiện: 45; chọn **45** (`selected = 1` trong `reports/b3_nonmed_terms.csv`), theo domain G-Group: tech 22, edu 15, pubadmin 8.

Lý do bỏ: bỏ: < 2 câu G-Group tech/edu/pubadmin 1292, bỏ: quá phổ biến (google-10000 top) hoặc <= 2 ký tự 159, bỏ: tổng số câu ngoài [3, 20] 79, bỏ: term_decisions = vi / drop_tag 59, bỏ: có trong consensus (giữ ở B2) 50, bỏ: có trong catalog 19

## Nếu áp dụng (đưa vào catalog với reserved_b3 = 1)

- vss_cs (`data/raw/vss_full/vss_cs_clean_dom.jsonl`): mất **56 câu, 0.20 h** / 4.15 h (4.8%).
- kho vss_cs_candidates (`data/raw/vss_full/vss_cs_candidates.jsonl`): 895 câu, 3.27 h chứa thuật ngữ được chọn (bị chặn ở bước lọc rẻ các lần sau).
- ViMedCSS train (cs_real): 9 câu chứa thuật ngữ được chọn (cũng phải loại khỏi train).
- G-Group: 159 câu B1 chứa thuật ngữ được chọn -> chuyển sang B3 (thuật ngữ không có trong train).

| Thuật ngữ | Domain | Câu G-Group | Câu vss_cs | Nguồn / người nói | Giờ vss_cs |
| --- | --- | --- | --- | --- | --- |
| logic | tech | 2 | 6 | 7 | 0.024 |
| edu | edu | 11 | 0 | 11 | 0.000 |
| visa | pubadmin | 9 | 0 | 9 | 0.000 |
| event | tech | 3 | 4 | 7 | 0.014 |
| singapore | edu | 3 | 7 | 9 | 0.027 |
| scan | pubadmin | 7 | 0 | 7 | 0.000 |
| blockchain | tech | 4 | 4 | 6 | 0.014 |
| background | edu | 3 | 4 | 7 | 0.015 |
| chip | pubadmin | 4 | 1 | 5 | 0.004 |
| demo | tech | 3 | 3 | 6 | 0.011 |
| harvard | edu | 2 | 4 | 5 | 0.014 |
| gmail | pubadmin | 4 | 0 | 4 | 0.000 |
| token | tech | 6 | 0 | 6 | 0.000 |
| master | edu | 2 | 4 | 5 | 0.014 |
| captcha | pubadmin | 3 | 0 | 3 | 0.000 |
| status | tech | 2 | 4 | 5 | 0.014 |
| apply | edu | 5 | 0 | 5 | 0.000 |
| file xml | pubadmin | 3 | 0 | 3 | 0.000 |
| build | tech | 5 | 0 | 5 | 0.000 |
| bachelor | edu | 5 | 0 | 5 | 0.000 |
| hanoi | pubadmin | 3 | 0 | 3 | 0.000 |
| https | tech | 5 | 0 | 5 | 0.000 |
| mba | edu | 5 | 0 | 5 | 0.000 |
| usb token | pubadmin | 3 | 0 | 3 | 0.000 |
| set up | tech | 2 | 3 | 5 | 0.010 |
| podcast | edu | 2 | 2 | 4 | 0.007 |
| admin | tech | 2 | 3 | 4 | 0.011 |
| diploma | edu | 3 | 0 | 3 | 0.000 |
| container | tech | 3 | 2 | 4 | 0.006 |
| hutech | edu | 3 | 0 | 3 | 0.000 |
| body | tech | 2 | 2 | 4 | 0.008 |
| official | edu | 2 | 1 | 3 | 0.003 |
| layer | tech | 2 | 2 | 4 | 0.006 |
| stem | edu | 3 | 0 | 3 | 0.000 |
| log | tech | 3 | 1 | 4 | 0.003 |
| toefl | edu | 3 | 0 | 3 | 0.000 |
| module | tech | 4 | 0 | 4 | 0.000 |
| topik | edu | 3 | 0 | 3 | 0.000 |
| smart contract | tech | 4 | 0 | 4 | 0.000 |
| clear | tech | 3 | 0 | 3 | 0.000 |
| developer | tech | 3 | 0 | 3 | 0.000 |
| docker | tech | 3 | 0 | 3 | 0.000 |
| iot | tech | 3 | 0 | 3 | 0.000 |
| password | tech | 2 | 1 | 3 | 0.003 |
| rule | tech | 3 | 0 | 3 | 0.000 |
