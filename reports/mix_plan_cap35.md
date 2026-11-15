# Kế hoạch trộn dữ liệu train

Cấu hình: `/mnt/d/code/ZAIC/gipbench/configs/train_mix.tsv`. Tổng tỉ trọng = 1.00.

| Nhóm | Tỉ trọng | Datasets | Giờ có | Domain (giờ) |
| --- | --- | --- | --- | --- |
| cs_real | 0.35 | ViMedCSS | 24.30 | medical 24.30 |
| cs_pseudo | 0.10 | vss_cs | 4.15 | other 3.38, business 0.18, entertainment 0.16, tech 0.15, daily 0.10, edu 0.06, medical 0.06, finance 0.05, pubadmin 0.01 |
| synth | 0.05 | tts | 0.00 | — |
| replay_human | 0.25 | vivos, commonvoice | 14.92 | read 14.92 |
| distill | 0.20 | distill | 0.00 | — |
| confusable | 0.05 | confusable | 0.90 | confusable 0.90 |

Phân bố domain dự kiến trong phần CS của batch (cs_real + cs_pseudo), trần 35%:

| Domain | Tỉ lệ |
| --- | --- |
| medical | 78.1% |
| other | 18.1% |
| business | 1.0% |
| entertainment | 0.8% |
| tech | 0.8% |
| daily | 0.5% |
| edu | 0.3% |
| finance | 0.3% |
| pubadmin | 0.0% |

Giả định (--assume): `ViMedCSS:medical:24.30`

**CẢNH BÁO:** domain medical (ViMedCSS) chiếm 78.1% phần CS (cs_real + cs_pseudo) > trần 35%

## Phương án đạt trần 35% (--fit-cap)

Đạt trần. Domain bị cắt: medical; miễn trần: other.

| Domain | Giờ có (phần CS) | Tỉ lệ tự nhiên | Tỉ lệ mục tiêu | Hệ số lặp tương đối |
| --- | --- | --- | --- | --- |
| other | 3.38 | 18.1% | 53.7% | 4.52× |
| medical | 24.36 | 78.1% | 35.0% | 0.41× |
| business | 0.18 | 1.0% | 2.9% | 4.52× |
| entertainment | 0.16 | 0.8% | 2.5% | 4.52× |
| tech | 0.15 | 0.8% | 2.4% | 4.52× |
| daily | 0.10 | 0.5% | 1.6% | 4.52× |
| edu | 0.06 | 0.3% | 1.0% | 4.52× |
| finance | 0.05 | 0.3% | 0.8% | 4.52× |
| pubadmin | 0.01 | 0.0% | 0.1% | 4.52× |

Hệ số lặp tương đối = (tỉ lệ mục tiêu / giờ có của domain) chia (1 / tổng giờ phần CS): 1× = lấy mẫu đều theo giờ; số lớn = audio domain đó bị lặp nhiều (dễ overfit).
