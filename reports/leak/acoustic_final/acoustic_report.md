# Kiểm trùng âm thanh (dấu vân tay Haitsma–Kalker)

Train: `pool/vimedcss_train@v4`, `pool/vivos_train@v3`, `pool/confusable_train@v3`, `pool/cv_other@v3`, `pool/vss_cs@v3`, `pool/distill@v4`

Bench: `bench/general-v1.0`, `bench/code-switch-v1.0`, `bench/cs-ggroup-v1.0`, `bench/external-v1.0`, `bench/code-switch-v1.1`, `bench/external-v1.1`

- train 19301 / 19301 câu có dấu vân tay; bench 45130 câu đọc được (45130 audio khác nhau), 26252 câu không đọc được audio (bỏ qua).
- ngưỡng: BER <= 0.3, chồng >= 1.0 s (clip ngắn: >= 90% độ dài).
- **0 cặp khớp**; câu train cần loại: không

