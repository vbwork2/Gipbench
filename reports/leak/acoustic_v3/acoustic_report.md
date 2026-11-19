# Kiểm trùng âm thanh (dấu vân tay Haitsma–Kalker)

Train: `pool/vimedcss_train@v3`, `pool/vivos_train@v3`, `pool/confusable_train@v3`, `pool/cv_other@v3`, `pool/vss_cs@v3`, `pool/distill@v3`

Bench: `bench/general-v1.0`, `bench/code-switch-v1.0`, `bench/cs-ggroup-v1.0`, `bench/external-v1.0`, `bench/code-switch-v1.1`, `bench/external-v1.1`

- train 19336 / 19336 câu có dấu vân tay; bench 45130 câu đọc được (45130 audio khác nhau), 26252 câu không đọc được audio (bỏ qua).
- ngưỡng: BER <= 0.3, chồng >= 1.0 s (clip ngắn: >= 90% độ dài).
- **15 cặp khớp**; câu train cần loại: `pool/vimedcss_train@v3` 7, `pool/distill@v3` 1

| Bench | Câu test | Train | Câu train | BER | Chồng (s) |
| --- | --- | --- | --- | --- | --- |
| `bench/code-switch-v1.0` | `c0_vmc_Med_CS-484-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-4160-2` | 0.239 | 2.18 |
| `bench/code-switch-v1.0` | `c0_vmc_Med_CS-484-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-254-2` | 0.275 | 2.97 |
| `bench/code-switch-v1.0` | `c0_vmc_Med_CS-486-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-254-2` | 0.156 | 3.62 |
| `bench/code-switch-v1.1` | `c0_vmc_Med_CS-484-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-4160-2` | 0.239 | 2.18 |
| `bench/code-switch-v1.1` | `c0_vmc_Med_CS-484-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-254-2` | 0.275 | 2.97 |
| `bench/code-switch-v1.1` | `c0_vmc_Med_CS-486-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-254-2` | 0.156 | 3.62 |
| `bench/external-v1.1` | `e0_bud500_test-00001-of-00002-1b2cd3ea16a2c5b1_1_122` | `pool/distill@v3` | `vss_0037945` | 0.045 | 1.17 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-145-7` | `pool/vimedcss_train@v3` | `vmc_Med_CS-285-4` | 0.277 | 3.56 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-2749-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-2758-1` | 0.246 | 1.61 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-3655_2-27` | `pool/vimedcss_train@v3` | `vmc_Med_CS-3656_2-28` | 0.025 | 2.62 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-4254-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-423-1` | 0.233 | 4.62 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-426-2` | `pool/vimedcss_train@v3` | `vmc_Med_CS-254-2` | 0.22 | 3.62 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-426-2` | `pool/vimedcss_train@v3` | `vmc_Med_CS-1530-1` | 0.279 | 4.62 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-582-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-254-2` | 0.136 | 3.62 |
| `bench/external-v1.1` | `e0_vimedcss_test_Med_CS-582-1` | `pool/vimedcss_train@v3` | `vmc_Med_CS-1530-1` | 0.278 | 4.62 |
