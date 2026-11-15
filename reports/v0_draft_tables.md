# NHÁP v0-draft — E0 / E1 (hotwords 1.0 / 1.5 / 2.0) / A4 FT thử

> **NHÁP.** Bench v0-draft, B1–B3 CHƯA được người duyệt (nhãn auto_tag / converter). Chỉ để xem xu hướng, không dùng để kết luận.

## Leaderboard (delta so với baseline trong ngoặc)

| Hệ thống | CS-WER B1+B2 | CI95 | N-WER B1+B2 | Term R B1+B2 | Term P B1+B2 | CS-WER B3 | WER B5 | Chèn sai B5 /100 utt | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e0_baseline | 71.66 | [69.06, 74.36] | 13.13 | 39.55 | 72.67 | 106.81 | 9.52 | 1.06 | 0.059 |
| e1_hw1.0 | 61.81 (-9.84) | [58.96, 64.65] | 13.17 (+0.05) | 48.27 (+8.72) | 69.22 (-3.45) | 107.96 (+1.15) | 9.57 (+0.05) | 1.85 (+0.78) | 0.067 |
| e1_hw1.5 | 60.21 (-11.44) | [57.15, 62.94] | 13.13 (+0.00) | 49.96 (+10.41) | 68.60 (-4.08) | 109.03 (+2.22) | 9.63 (+0.11) | 2.31 (+1.25) | 0.060 |
| e1_hw2.0 | 59.71 (-11.95) | [56.96, 62.25] | 13.30 (+0.17) | 50.68 (+11.13) | 67.80 (-4.88) | 109.33 (+2.52) | 9.68 (+0.16) | 3.09 (+2.03) | 0.061 |
| a4_trial | 71.21 (-0.44) | [68.66, 74.09] | 15.94 (+2.82) | 35.61 (-3.94) | 74.89 (+2.22) | 101.15 (-5.66) | 11.41 (+1.90) | 0.83 (-0.23) | 0.063 |
| a4_trial_hw1.5 | 59.09 (-12.57) | [56.52, 62.02] | 16.20 (+3.07) | 46.82 (+7.27) | 69.43 (-3.25) | 102.60 (-4.21) | 11.51 (+2.00) | 2.08 (+1.02) | 0.060 |

## So sánh theo split: e0_baseline → e1_hw1.5

Hotwords của E1 = thuật ngữ B1/B2 (catalog nháp, bỏ reserved_b3): thuật ngữ B3 KHÔNG có trong hotwords, nên term recall B3 đo khả năng tổng quát và mức chèn nhầm.

| Split | Utt | Đơn vị bootstrap | WER | CS-WER strict | CS-WER lenient | N-WER | Term recall | Term precision | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 208 | 24.35 → 22.18 (-2.17) | 70.49 → 59.02 (-11.46) | 70.49 → 59.02 (-11.46) | 13.70 → 13.68 (-0.02) | 39.87 → 50.16 (+10.28) | 71.67 → 67.86 (-3.81) | 26.39 → 39.92 (+13.53) |
| B2 | 218 | 11 | 16.66 → 15.87 (-0.79) | 82.81 → 71.56 (-11.25) | 82.81 → 71.56 (-11.25) | 11.17 → 11.25 (+0.08) | 36.78 → 48.28 (+11.49) | 83.48 → 75.90 (-7.57) | 8.72 → 18.35 (+9.63) |
| B3 | 524 | 347 | 23.67 → 23.89 (+0.22) | 106.81 → 109.03 (+2.22) | 106.81 → 109.03 (+2.22) | 14.18 → 14.17 (-0.01) | 9.32 → 8.81 (-0.51) | 53.80 → 33.08 (-20.72) | 15.08 → 33.59 (+18.51) |
| B5 | 2166 | 101 | 9.52 → 9.63 (+0.11) | – | – | 9.52 → 9.63 (+0.11) | – | – | 1.06 → 2.31 (+1.25) |
| main_B1+B2 | 1548 | 219 | 22.78 → 20.89 (-1.89) | 71.66 → 60.21 (-11.44) | 71.66 → 60.21 (-11.44) | 13.13 → 13.13 (+0.00) | 39.55 → 49.96 (+10.41) | 72.67 → 68.60 (-4.08) | 23.90 → 36.89 (+12.98) |

## e0_baseline

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 24.35 | [23.63, 25.03] | 70.49 | [67.79, 73.04] | 70.49 | 13.70 | [12.86, 14.58] | 39.87 | 71.67 | 26.39 |
| B2 | 218 | 16.66 | [15.31, 20.22] | 82.81 | [76.26, 92.31] | 82.81 | 11.17 | [9.63, 13.09] | 36.78 | 83.48 | 8.72 |
| B3 | 524 | 23.67 | [21.81, 25.54] | 106.81 | [102.42, 111.70] | 106.81 | 14.18 | [12.76, 15.66] | 9.32 | 53.80 | 15.08 |
| B5 | 2166 | 9.52 | [7.68, 10.53] | – | – | – | 9.52 | [7.68, 10.53] | – | 0.00 | 1.06 |
| main_B1+B2 | 1548 | 22.78 | [20.90, 24.32] | 71.66 | [69.06, 74.36] | 71.66 | 13.13 | [12.22, 14.02] | 39.55 | 72.67 | 23.90 |

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

### Phân loại lỗi thuật ngữ — e0_baseline (B1–B3, strict)

| Loại | Span sai | % | B1 | B2 | B3 | 5 ví dụ hay gặp nhất (ref → hyp ×lần) |
| --- | --- | --- | --- | --- | --- | --- |
| (a) chuẩn hoá / biến thể | 34 | 1.4 | 25 | 2 | 7 | `sideway` → `sideways` ×2; `set up` → `setup` ×2; `bicarbonat` → `bi carbonat` ×2; `ip` → `i p` ×1; `talkshow` → `talk show` ×1 |
| (b) phiên âm Việt | 655 | 26.6 | 449 | 78 | 128 | `code` → `cốt` ×8; `hanoi` → `hà nội` ×3; `coin` → `coi` ×3; `check` → `trách` ×3; `photo` → `phô tô` ×3 |
| (c) xoá | 206 | 8.4 | 60 | 5 | 141 | `bilirubin` → `∅` ×15; `aldosterone` → `∅` ×5; `listeria` → `∅` ×3; `gov` → `∅` ×2; `tt` → `∅` ×2 |
| (d) sai từ khác | 1567 | 63.6 | 860 | 84 | 623 | `bilirubin` → `billubin` ×11; `bilirubin` → `billy rubin` ×9; `listeria` → `listoria` ×7; `enzyme` → `engine` ×7; `aldosterone` → `andosterol` ×5 |

Luật: (c) hyp rỗng ở span; (a) giống ref sau khi bỏ dấu cách / gạch nối / nháy, hoặc số ít / số nhiều (hyp thêm `s`; hoặc ref bỏ `s` khi từ cuối > 5 chữ và không tận cùng `ss`: `ielts` → `ielt` là lỗi thật), hoặc có chữ số; (b) hyp có chữ có dấu tiếng Việt, hoặc mọi âm tiết có trong Viet74K và không phải từ tiếng Anh thông dụng (google-10000); (d) còn lại (từ Anh khác, chính tả sai kiểu Anh).

#### ĐỀ XUẤT thêm `variants` vào catalog (loại a, CHƯA áp dụng; 31 thuật ngữ)

| term | variants đề xuất (hyp ×lần) |
| --- | --- |
| sideway | `sideways` ×2 |
| set up | `setup` ×2 |
| bicarbonat | `bi carbonat` ×2 |
| ip | `i p` ×1 |
| talkshow | `talk show` ×1 |
| total assets | `total asset` ×1 |
| vnindex | `vn index` ×1 |
| a s | `as` ×1 |
| vinhomes | `vinhome` ×1 |
| op bnb | `opbnb` ×1 |
| vivu | `vi vu` ×1 |
| breakout | `break out` ×1 |
| midcap | `mid cap` ×1 |
| qd ub | `qdub` ×1 |
| hp | `h p` ×1 |
| docx | `doc x` ×1 |
| reinvent | `re invents` ×1 |
| bit banker | `bitbanker` ×1 |
| start port number | `startport number` ×1 |
| intest | `in test` ×1 |
| api | `apis` ×1 |
| script hhtest | `script hh test` ×1 |
| comlist | `com list` ×1 |
| backend | `back end` ×1 |
| logistic | `logistics` ×1 |
| redwood analytics | `red wood analytics` ×1 |
| cacao | `ca cao` ×1 |
| somatosan | `somato san` ×1 |
| polysacarit | `polysa carit` ×1 |
| latinh | `la tinh` ×1 |
| bilirubin | `bi lirubin` ×1 |

## e1_hw1.0

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 22.48 (-1.87) | [21.66, 23.22] | 60.43 (-10.06) | [57.26, 63.11] | 60.43 | 13.72 (+0.02) | [12.92, 14.53] | 48.77 (+8.89) | 68.47 | 37.67 (+11.28) |
| B2 | 218 | 16.18 (-0.48) | [15.01, 19.07] | 75.00 (-7.81) | [68.95, 80.65] | 75.00 | 11.30 (+0.13) | [9.82, 13.09] | 44.06 (+7.28) | 77.18 | 15.60 (+6.88) |
| B3 | 524 | 23.77 (+0.10) | [21.95, 25.60] | 107.96 (+1.15) | [103.43, 113.22] | 107.96 | 14.16 (-0.02) | [12.72, 15.58] | 9.22 (-0.10) | 38.72 | 27.48 (+12.40) |
| B5 | 2166 | 9.57 (+0.05) | [7.74, 10.57] | – | – | – | 9.57 (+0.05) | [7.74, 10.57] | – | 0.00 | 1.85 (+0.78) |
| main_B1+B2 | 1548 | 21.20 (-1.58) | [19.61, 22.50] | 61.81 (-9.84) | [58.96, 64.65] | 61.81 | 13.17 (+0.05) | [12.29, 14.04] | 48.27 (+8.72) | 69.22 | 34.56 (+10.66) |

Top 10 lỗi thuật ngữ (B1–B3, strict; 2256 span sai, 2092 cặp khác nhau)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `bilirubin` → `∅` | 15 | B3 15 |
| 2 | `listeria` → `listoria` | 13 | B3 13 |
| 3 | `bilirubin` → `billubin` | 12 | B3 12 |
| 4 | `bilirubin` → `billy rubin` | 10 | B3 10 |
| 5 | `code` → `cốt` | 7 | B1 7 |
| 6 | `enzyme` → `engine` | 6 | B3 6 |
| 7 | `renin` → `renine` | 5 | B3 5 |
| 8 | `aldosterone` → `andosterol` | 5 | B3 5 |
| 9 | `mg` → `miligam` | 5 | B3 5 |
| 10 | `aldosterone` → `∅` | 4 | B3 4 |

Top 10 lỗi thuật ngữ riêng B1+B2 (1355 span sai)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `code` → `cốt` | 7 | B1 7 |
| 2 | `diploma` → `deploma` | 3 | B1 3 |
| 3 | `log` → `lock` | 3 | B1 3 |
| 4 | `hanoi` → `hà nội` | 3 | B1 3 |
| 5 | `pizza` → `visa` | 3 | B2 3 |
| 6 | `book` → `bút` | 3 | B2 3 |
| 7 | `eva` → `hê va` | 2 | B1 2 |
| 8 | `topik` → `topic` | 2 | B1 2 |
| 9 | `vip` → `pip` | 2 | B1 2 |
| 10 | `link acber` → `link aserber` | 2 | B1 2 |

## e1_hw1.5

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 22.18 (-2.17) | [21.28, 22.90] | 59.02 (-11.46) | [55.81, 61.82] | 59.02 | 13.68 (-0.02) | [12.89, 14.42] | 50.16 (+10.28) | 67.86 | 39.92 (+13.53) |
| B2 | 218 | 15.87 (-0.79) | [14.56, 18.60] | 71.56 (-11.25) | [65.85, 77.23] | 71.56 | 11.25 (+0.08) | [9.75, 12.88] | 48.28 (+11.49) | 75.90 | 18.35 (+9.63) |
| B3 | 524 | 23.89 (+0.22) | [22.07, 25.70] | 109.03 (+2.22) | [104.34, 114.58] | 109.03 | 14.17 (-0.01) | [12.72, 15.62] | 8.81 (-0.51) | 33.08 | 33.59 (+18.51) |
| B5 | 2166 | 9.63 (+0.11) | [7.74, 10.64] | – | – | – | 9.63 (+0.11) | [7.74, 10.64] | – | 0.00 | 2.31 (+1.25) |
| main_B1+B2 | 1548 | 20.89 (-1.89) | [19.29, 22.27] | 60.21 (-11.44) | [57.15, 62.94] | 60.21 | 13.13 (+0.00) | [12.22, 13.94] | 49.96 (+10.41) | 68.60 | 36.89 (+12.98) |

Top 10 lỗi thuật ngữ (B1–B3, strict; 2222 span sai, 2059 cặp khác nhau)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `bilirubin` → `∅` | 16 | B3 16 |
| 2 | `bilirubin` → `billubin` | 12 | B3 12 |
| 3 | `listeria` → `listoria` | 12 | B3 12 |
| 4 | `bilirubin` → `billy rubin` | 10 | B3 10 |
| 5 | `code` → `cốt` | 7 | B1 7 |
| 6 | `bilirubin` → `bill` | 6 | B3 6 |
| 7 | `enzyme` → `engine` | 6 | B3 6 |
| 8 | `aldosterone` → `andosterol` | 5 | B3 5 |
| 9 | `mg` → `miligam` | 5 | B3 5 |
| 10 | `renin` → `renine` | 4 | B3 4 |

Top 10 lỗi thuật ngữ riêng B1+B2 (1318 span sai)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `code` → `cốt` | 7 | B1 7 |
| 2 | `diploma` → `deploma` | 3 | B1 3 |
| 3 | `log` → `lock` | 3 | B1 3 |
| 4 | `hanoi` → `hà nội` | 3 | B1 3 |
| 5 | `pizza` → `visa` | 3 | B2 3 |
| 6 | `eva` → `hê va` | 2 | B1 2 |
| 7 | `topik` → `topic` | 2 | B1 2 |
| 8 | `vip` → `pip` | 2 | B1 2 |
| 9 | `topo` → `top pho` | 2 | B1 2 |
| 10 | `add` → `ad` | 2 | B1 2 |

## e1_hw2.0

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 22.24 (-2.11) | [21.32, 22.92] | 58.60 (-11.89) | [55.53, 61.01] | 58.60 | 13.84 (+0.14) | [13.01, 14.58] | 50.92 (+11.05) | 67.10 | 41.95 (+15.56) |
| B2 | 218 | 15.94 (-0.72) | [14.63, 18.56] | 70.31 (-12.50) | [64.79, 74.53] | 70.31 | 11.43 (+0.26) | [9.73, 13.05] | 48.66 (+11.88) | 74.71 | 19.72 (+11.01) |
| B3 | 524 | 23.92 (+0.25) | [22.13, 25.74] | 109.33 (+2.52) | [104.51, 114.93] | 109.33 | 14.17 (-0.01) | [12.76, 15.66] | 8.92 (-0.41) | 29.93 | 39.31 (+24.24) |
| B5 | 2166 | 9.68 (+0.16) | [7.82, 10.70] | – | – | – | 9.68 (+0.16) | [7.82, 10.70] | – | 0.00 | 3.09 (+2.03) |
| main_B1+B2 | 1548 | 20.95 (-1.83) | [19.34, 22.30] | 59.71 (-11.95) | [56.96, 62.25] | 59.71 | 13.30 (+0.17) | [12.35, 14.09] | 50.68 (+11.13) | 67.80 | 38.82 (+14.92) |

Top 10 lỗi thuật ngữ (B1–B3, strict; 2208 span sai, 2048 cặp khác nhau)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `bilirubin` → `∅` | 17 | B3 17 |
| 2 | `listeria` → `listoria` | 12 | B3 12 |
| 3 | `bilirubin` → `billubin` | 11 | B3 11 |
| 4 | `bilirubin` → `billy rubin` | 10 | B3 10 |
| 5 | `enzyme` → `engine` | 7 | B3 7 |
| 6 | `code` → `cốt` | 6 | B1 6 |
| 7 | `bilirubin` → `bill` | 6 | B3 6 |
| 8 | `aldosterone` → `andosterol` | 5 | B3 5 |
| 9 | `mg` → `miligam` | 5 | B3 5 |
| 10 | `aldosterone` → `∅` | 4 | B3 4 |

Top 10 lỗi thuật ngữ riêng B1+B2 (1306 span sai)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `code` → `cốt` | 6 | B1 6 |
| 2 | `diploma` → `deploma` | 3 | B1 3 |
| 3 | `log` → `lock` | 3 | B1 3 |
| 4 | `hanoi` → `hà nội` | 3 | B1 3 |
| 5 | `eva` → `hê va` | 2 | B1 2 |
| 6 | `topik` → `topic` | 2 | B1 2 |
| 7 | `vip` → `pip` | 2 | B1 2 |
| 8 | `topo` → `top pho` | 2 | B1 2 |
| 9 | `sideway` → `sideways` | 2 | B1 2 |
| 10 | `coin` → `altcoin` | 2 | B1 2 |

## a4_trial

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 26.67 (+2.32) | [25.67, 27.56] | 70.23 (-0.26) | [67.38, 72.98] | 70.23 | 16.62 (+2.92) | [15.45, 17.64] | 36.06 (-3.82) | 74.28 | 20.90 (-5.49) |
| B2 | 218 | 18.77 (+2.11) | [17.82, 20.65] | 80.62 (-2.19) | [75.22, 89.83] | 80.62 | 13.64 (+2.46) | [11.32, 14.60] | 31.80 (-4.98) | 81.37 | 8.72 (+0.00) |
| B3 | 524 | 24.90 (+1.23) | [23.06, 26.77] | 101.15 (-5.66) | [98.09, 104.85] | 101.15 | 16.20 (+2.02) | [14.65, 17.85] | 8.41 (-0.91) | 57.64 | 11.64 (-3.44) |
| B5 | 2166 | 11.41 (+1.90) | [9.10, 12.73] | – | – | – | 11.41 (+1.90) | [9.10, 12.73] | – | 0.00 | 0.83 (-0.23) |
| main_B1+B2 | 1548 | 25.06 (+2.28) | [23.19, 26.67] | 71.21 (-0.44) | [68.66, 74.09] | 71.21 | 15.94 (+2.82) | [14.80, 17.08] | 35.61 (-3.94) | 74.89 | 19.19 (-4.72) |

Top 10 lỗi thuật ngữ (B1–B3, strict; 2535 span sai, 2326 cặp khác nhau)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `bilirubin` → `∅` | 24 | B3 24 |
| 2 | `listeria` → `listoria` | 13 | B3 13 |
| 3 | `aldosterone` → `∅` | 11 | B3 11 |
| 4 | `bilirubin` → `billyrobin` | 8 | B3 8 |
| 5 | `check` → `trách` | 6 | B1 4, B2 2 |
| 6 | `bilirubin` → `billubin` | 6 | B3 6 |
| 7 | `bilirubin` → `billy rubin` | 6 | B3 6 |
| 8 | `ielts` → `ielt` | 5 | B1 5 |
| 9 | `gov` → `∅` | 5 | B1 5 |
| 10 | `aldosterone` → `anh` | 5 | B3 5 |

Top 10 lỗi thuật ngữ riêng B1+B2 (1630 span sai)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `check` → `trách` | 6 | B1 4, B2 2 |
| 2 | `ielts` → `ielt` | 5 | B1 5 |
| 3 | `gov` → `∅` | 5 | B1 5 |
| 4 | `edu` → `∅` | 4 | B1 4 |
| 5 | `coin` → `coi` | 4 | B1 4 |
| 6 | `photo` → `phô tô` | 4 | B1 4 |
| 7 | `check` → `∅` | 3 | B1 1, B2 2 |
| 8 | `user` → `∅` | 3 | B1 3 |
| 9 | `code` → `cốt` | 3 | B1 3 |
| 10 | `zalo` → `∅` | 3 | B2 3 |

## a4_trial_hw1.5

| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1330 | 24.59 (+0.23) | [23.60, 25.59] | 57.75 (-12.74) | [55.09, 60.38] | 57.75 | 16.93 (+3.23) | [15.76, 18.01] | 47.42 (+7.54) | 68.84 | 36.24 (+9.85) |
| B2 | 218 | 18.15 (+1.48) | [17.12, 20.60] | 71.88 (-10.94) | [67.15, 75.27] | 71.88 | 13.69 (+2.51) | [11.85, 15.35] | 41.76 (+4.98) | 75.69 | 16.06 (+7.34) |
| B3 | 524 | 24.79 (+1.12) | [22.94, 26.66] | 102.60 (-4.21) | [99.22, 106.99] | 102.60 | 15.91 (+1.73) | [14.36, 17.54] | 8.11 (-1.22) | 40.40 | 22.52 (+7.44) |
| B5 | 2166 | 11.51 (+2.00) | [9.14, 12.81] | – | – | – | 11.51 (+2.00) | [9.14, 12.81] | – | 0.00 | 2.08 (+1.02) |
| main_B1+B2 | 1548 | 23.27 (+0.49) | [21.72, 24.70] | 59.09 (-12.57) | [56.52, 62.02] | 59.09 | 16.20 (+3.07) | [15.05, 17.34] | 46.82 (+7.27) | 69.43 | 33.40 (+9.50) |

Top 10 lỗi thuật ngữ (B1–B3, strict; 2272 span sai, 2097 cặp khác nhau)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `bilirubin` → `∅` | 19 | B3 19 |
| 2 | `listeria` → `listoria` | 15 | B3 15 |
| 3 | `bilirubin` → `billyrobin` | 11 | B3 11 |
| 4 | `bilirubin` → `billubin` | 11 | B3 11 |
| 5 | `aldosterone` → `∅` | 10 | B3 10 |
| 6 | `bilirubin` → `billy rubin` | 8 | B3 8 |
| 7 | `enzyme` → `engine` | 6 | B3 6 |
| 8 | `mg` → `miligam` | 5 | B3 5 |
| 9 | `check` → `trách` | 4 | B1 2, B2 2 |
| 10 | `aldosterone` → `andosterol` | 4 | B3 4 |

Top 10 lỗi thuật ngữ riêng B1+B2 (1363 span sai)

| # | ref → hyp | Lần | Theo split |
| --- | --- | --- | --- |
| 1 | `check` → `trách` | 4 | B1 2, B2 2 |
| 2 | `add` → `ad` | 3 | B1 3 |
| 3 | `code` → `cốt` | 3 | B1 3 |
| 4 | `sms` → `∅` | 3 | B2 3 |
| 5 | `check` → `chắc` | 3 | B2 3 |
| 6 | `topik` → `topic` | 2 | B1 2 |
| 7 | `log` → `lock` | 2 | B1 2 |
| 8 | `coupon` → `kubon` | 2 | B1 2 |
| 9 | `wat` → `swap` | 2 | B1 2 |
| 10 | `dotcom` → `rot com` | 2 | B1 2 |

