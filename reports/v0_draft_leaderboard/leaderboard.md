# Leaderboard (baseline: e0_baseline)

Phương án tốt nhất hiện tại: e1_hw2.0 — CS-WER B1+B2 59.71% (-11.95 điểm so với baseline, CI95 [-13.43, -10.42]), B5 WER Δ 0.16.

| Hệ thống | CS-WER B1+B2 | Δ vs baseline [CI95] | CS-WER lenient | N-WER B1–B4 | CS-WER B3 | CS-WER B4 | B5 WER (Δ) | Term R / P | B5 chèn sai /100 utt | RTF | Kết luận |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e0_baseline | 71.66 | – | 71.66 | 13.55 | 106.81 | – | 9.52 (+0.00) | 31.0 / 70.6 | 1.06 | 0.059 | baseline |
| e1_hw2.0 | 59.71 | -11.95 [-13.43, -10.42] | 59.71 | 13.65 | 109.33 | – | 9.68 (+0.16) | 38.8 / 62.6 | 3.09 | 0.061 | #1 đủ điều kiện |
| e1_hw1.5 | 60.21 | -11.44 [-12.87, -9.96] | 60.21 | 13.55 | 109.03 | – | 9.63 (+0.11) | 38.3 / 64.1 | 2.31 | 0.060 | #2 đủ điều kiện |
| e1_hw1.0 | 61.81 | -9.84 [-11.18, -8.47] | 61.81 | 13.57 | 107.96 | – | 9.57 (+0.05) | 37.2 / 65.6 | 1.85 | 0.067 | #3 đủ điều kiện |
| a4_trial_hw1.5 | 59.09 | -12.57 [-14.70, -10.78] | 59.09 | 16.08 | 102.60 | – | 11.51 (+2.00) | 35.8 / 66.4 | 2.08 | 0.060 | loại: B5 WER +2.00; N-WER +2.53; B5 domain pubadmin +3.87 |
| a4_trial | 71.21 | -0.44 [-2.12, +1.12] | 71.21 | 16.05 | 101.15 | – | 11.41 (+1.90) | 27.9 / 73.0 | 0.83 | 0.063 | loại: B5 WER +1.90; N-WER +2.50; B5 domain pubadmin +3.82; ΔCS-WER chưa có ý nghĩa (CI chứa 0) |

Luật: loại nếu B5 WER tăng > 0.3, N-WER tăng > 0.3, RTF > 1.2× baseline, một domain B5 tăng > 0.5, hoặc CI95 của ΔCS-WER chứa 0. Bootstrap theo người nói, 1000 lần. CS-WER lenient − strict = lỗi chính tả (nghe đúng, viết sai) mà hậu xử lý có thể sửa.
