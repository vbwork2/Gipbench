# NHÁP v0-draft — E0 (Gipformer 1.5, modified_beam_search)

> **NHÁP.** Bench v0-draft, B1–B3 CHƯA được người duyệt (nhãn auto_tag / converter). Chỉ để xem xu hướng, không dùng để kết luận.

## e0_baseline

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 24.35 | [23.58, 25.13] | 70.49 | [68.00, 73.01] | 70.49 | 13.70 | [12.89, 14.41] | 39.87 | 71.67 | 26.39 |
| B2 | 218 | 16.66 | [15.31, 20.22] | 82.81 | [76.26, 92.31] | 82.81 | 11.17 | [9.63, 13.09] | 36.78 | 83.48 | 8.72 |
| B3 | 524 | 23.67 | [21.81, 25.54] | 106.81 | [102.42, 111.70] | 106.81 | 14.18 | [12.76, 15.66] | 9.32 | 53.80 | 15.08 |
| B5 | 2166 | 9.52 | [7.82, 10.51] | – | – | – | 9.52 | [7.82, 10.51] | – | 0.00 | 1.06 |
| main_B1+B2 | 1548 | 22.78 | [21.32, 24.43] | 71.66 | [69.05, 74.39] | 71.66 | 13.13 | [12.44, 13.99] | 39.55 | 72.67 | 23.90 |

Top 10 lỗi thuật ngữ (B1–B3, strict; 2462 span sai, 2283 cặp khác nhau)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `bilirubin` → `∅` | 15 | B3 15 |
| 2 | `bilirubin` → `billubin` | 11 | B3 11 |
| 3 | `bilirubin` → `billy rubin` | 9 | B3 9 |
| 4 | `code` → `cốt` | 8 | B1 8 |
| 5 | `listeria` → `listoria` | 7 | B3 7 |
| 6 | `enzyme` → `engine` | 7 | B3 7 |
| 7 | `aldosterone` → `andosterol` | 5 | B3 5 |
| 8 | `aldosterone` → `∅` | 5 | B3 5 |
| 9 | `bilirubin` → `bin` | 5 | B3 5 |
| 10 | `renin` → `renine` | 4 | B3 4 |

Top 10 lỗi thuật ngữ riêng B1+B2 (1563 span sai)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `code` → `cốt` | 8 | B1 8 |
| 2 | `diploma` → `deploma` | 3 | B1 3 |
| 3 | `log` → `lock` | 3 | B1 3 |
| 4 | `ielts` → `ielt` | 3 | B1 3 |
| 5 | `hanoi` → `hà nội` | 3 | B1 3 |
| 6 | `coin` → `coi` | 3 | B1 3 |
| 7 | `check` → `trách` | 3 | B1 1, B2 2 |
| 8 | `photo` → `phô tô` | 3 | B1 3 |
| 9 | `pizza` → `visa` | 3 | B2 3 |
| 10 | `book` → `bút` | 3 | B2 3 |

