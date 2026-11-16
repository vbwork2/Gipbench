# Kế hoạch trộn dữ liệu train

Cấu hình: `/mnt/d/code/ZAIC/gipbench/configs/train_mix.tsv`. Tổng tỉ trọng = 1.00.

| Nhóm | Tỉ trọng | Datasets | Giờ có | Domain (giờ) |
| --- | --- | --- | --- | --- |
| cs_real | 0.35 | ViMedCSS | 24.30 | medical 24.30 |
| cs_pseudo | 0.10 | vss_cs | 0.83 | conversation 0.83 |
| synth | 0.05 | tts | 0.00 | — |
| replay_human | 0.25 | vivos, commonvoice | 0.00 | — |
| distill | 0.20 | distill | 0.00 | — |
| confusable | 0.05 | confusable | 0.00 | — |

Phân bố domain dự kiến trong phần CS của batch (cs_real + cs_pseudo), trần 35%:

| Domain | Tỉ lệ |
| --- | --- |
| medical | 77.8% |
| conversation | 22.2% |

Giả định (--assume): `ViMedCSS:medical:24.30`

**CẢNH BÁO:** domain medical (ViMedCSS) chiếm 77.8% phần CS (cs_real + cs_pseudo) > trần 35%
