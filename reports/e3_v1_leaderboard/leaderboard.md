# Leaderboard (baseline: e0_baseline)

Phương án tốt nhất hiện tại: e1_hw1.5 — CS-WER B1+B2 60.21% (-11.44 điểm so với baseline, CI95 [-12.98, -10.22]), B5 WER Δ 0.11.

| Hệ thống | CS-WER B1+B2 | Δ vs baseline [CI95] | CS-WER lenient | N-WER B1–B4 | CS-WER B3 | CS-WER B4 | B5 WER (Δ) | Term R / P | B5 chèn sai /100 utt | RTF | Kết luận |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e0_baseline | 71.66 | – | 71.66 | 13.55 | 106.81 | – | 9.52 (+0.00) | 31.0 / 70.6 | 1.06 | 0.059 | baseline |
| e1_hw1.5 | 60.21 | -11.44 [-12.98, -10.22] | 60.21 | 13.55 | 109.03 | – | 9.63 (+0.11) | 38.3 / 64.1 | 2.31 | 0.060 | #1 đủ điều kiện |
| e7_v1 | 81.97 | +10.32 [+7.26, +12.45] | 81.97 | 17.57 | 102.07 | – | 12.99 (+3.47) | 21.6 / 57.4 | 1.11 | 0.078 | loại: B5 WER +3.47; N-WER +4.02; B5 domain pubadmin +10.65; RTF ×1.32; ΔCS-WER chưa có ý nghĩa (CI chứa 0) |
| e3_v1 | 92.44 | +20.78 [+17.97, +23.07] | 92.44 | 17.50 | 101.15 | – | 12.81 (+3.29) | 14.7 / 67.3 | 0.14 | 0.098 | loại: B5 WER +3.29; N-WER +3.95; B5 domain pubadmin +10.23; RTF ×1.65; ΔCS-WER chưa có ý nghĩa (CI chứa 0) |

Luật: loại nếu B5 WER tăng > 0.3, N-WER tăng > 0.3, RTF > 1.2× baseline, một domain B5 tăng > 0.5, hoặc CI95 của ΔCS-WER chứa 0. Bootstrap theo người nói, 1000 lần. CS-WER lenient − strict = lỗi chính tả (nghe đúng, viết sai) mà hậu xử lý có thể sửa.
