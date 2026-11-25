# Trạng thái nghiên cứu VietNote

Mục tiêu giảm lỗi chuyển mã Việt–Anh với thuật ngữ chưa thấy và giữ độ chính xác tiếng Việt.

| Lượt | Trạng thái | Điều kiện trước |
| --- | --- | --- |
| V02P | Pilot Icefall đã chạy cục bộ. Chờ xác minh Kaggle. | — |
| V00 | Tích hợp và quick baseline có 10 lượt đã lưu. Full dev còn mở. | — |
| V01 | Đạt. | — |
| V02 | Một phần. Xem tiny-set và batch FP32 bên dưới. | — |
| V03 | Bị chặn bởi cổng dữ liệu hoặc model. | V00, V01, V02 |
| V04 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V04b | Chờ kết quả V03 và điều kiện trước. | V03 |
| V05 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V05b | Chờ kết quả V03 và điều kiện trước. | V03, V05 |
| V06 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V07 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V08 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V09 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V10 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V11 | Chờ kết quả V03 và điều kiện trước. | V03, V05 |
| V12 | Chờ kết quả V03 và điều kiện trước. | V03, V05 |
| V13 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V14 | Chờ kết quả V03 và điều kiện trước. | V03 |
| V15 | Chờ kết quả V03 và điều kiện trước. | V14 |

## Dữ liệu V03

| Nhóm | Giờ có sẵn | Giờ đủ bằng chứng | Mục tiêu | Giờ đủ chuẩn còn thiếu |
| --- | ---: | ---: | ---: | ---: |
| VI | 1.348 | 0.000 | 12 | 12.000 |
| CS | 5.952 | 0.000 | 8 | 8.000 |

Giờ có sẵn dùng audio tại chỗ, khử trùng theo SHA256 ghi trong manifest hoặc đường dẫn. Giờ đủ bằng chứng cần hồ sơ quyền sử dụng, kiểm nhãn, phân tách người nói và kiểm âm học của release.

| Release trong hỗn hợp | Số dòng | Giờ có sẵn | Giờ đủ bằng chứng |
| --- | ---: | ---: | ---: |
| pool/cv_other@v3 | 859 | 1.035 | 0.000 |
| pool/vietnote_fleurs_vi@v1 | 96 | 0.313 | 0.000 |
| pool/vietnote_vimedcss_cs@v1 | 3035 | 5.952 | 0.000 |

### Tiến độ lấy dữ liệu

| Nguồn | Trạng thái registry | Giờ ứng viên cục bộ | Giờ qua audit kỹ thuật | Giờ duyệt tại repo | Giờ đủ V03 |
| --- | --- | ---: | ---: | ---: | ---: |
| code_switch_om | quarantined_unverified_audio_rights_and_transcript_quality | — | — | 0.000 | 0.000 |
| vss | reuse_local_research_pools_pending_original_recording_rights | 4.010 | — | 0.601 | 0.000 |
| vimedcss | reuse_local_train | 7.529 | — | 0.000 | 0.000 |
| canvec | blocked_audio_by_author_request | — | — | 0.000 | 0.000 |
| vietspeech | gated | — | — | 0.000 | 0.000 |
| vietprism | pending_author_approval | — | — | 0.000 | 0.000 |
| vimd | blocked_for_pool_pending_nc_nd_review | — | — | 0.000 | 0.000 |
| vivoice | gated_research_only | — | — | 0.000 | 0.000 |
| lsvsc | pilot_quarantine_transcript_and_recording_rights | 0.204 | — | 0.000 | 0.000 |
| vivos | reuse_local_research_pool | 13.956 | — | 0.000 | 0.000 |
| fleurs_vi | pilot_eligible | 9.076 | 8.918 | 0.000 | 0.000 |
| commonvoice_vi | reuse_local_cv27 | 1.035 | — | 0.000 | 0.000 |
| l2_arctic | blocked_download_form | — | — | 0.000 | 0.000 |
| l2_arctic_mirror | gated_same_recordings_as_l2_arctic | — | — | 0.000 | 0.000 |
| librispeech | pilot_eligible | 3.000 | 2.595 | 0.000 | 0.000 |
| commonvoice_en | blocked_release_access_unverified | — | — | 0.000 | 0.000 |
| zerospeech | pilot_quarantine_text_rights_and_pronunciation_review | 0.732 | — | 0.000 | 0.000 |
| pool/vietnote_fleurs_vi@v2 | Release bổ trợ đã xác minh | — | 8.918 | 0.000 | 0.000 |
| pool/vietnote_librispeech_en@v2 | Release bổ trợ đã xác minh | — | 2.595 | 0.000 | 0.000 |

Audit dấu vân tay đối chiếu tới 49,146 đường dẫn audio đánh giá.
26,252 dòng E0 v1.0 khóa bằng metadata khớp ID, nguồn upstream và transcript với bản E0 v1.1 có audio đã được đưa vào audit.

### Kho bổ trợ theo loại dữ liệu

| Loại | Mục tiêu giờ | Giờ ứng viên đo được | Giờ duyệt tại repo | Giờ qua tiêu chí của nhóm | Giờ còn thiếu đến mức tối thiểu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Giọng thật tiếng Việt và CS trong kho mở rộng | 50–100 | 9.076 từ đợt mới | 0.000 đợt mới | 8.918 bổ trợ đã phát hành | 41.082 tính trên phần đã xác minh |
| Tiếng Anh giọng Việt | 2–5 | 0.000 | 0.000 | 0.000 | 2.000 |
| Tiếng Anh bản ngữ | — | 3.000 | 0.000 | 2.595 | — |
| CS tổng hợp bên ngoài | 5 | 0.732 | 0.000 | 0.000 | 5.000 |
| TTS tùy chỉnh | 1–2 | 0.0151 | 0.000 | 0.000 | 1.000 |

FLEURS là tiếng Việt đọc. LibriSpeech là tiếng Anh bản ngữ. Cột giờ phát hành của kho mở rộng hiện ghi phần đợt mới, tránh cộng trùng v1 với v2.
Kho train cục bộ sau gộp có 21,426 bản ghi duy nhất và 43.059 giờ theo SHA256 hoặc đường dẫn. Trong đó tiếng Anh bản ngữ 2.595 giờ, tiếng Anh giọng Việt 0.000 giờ, âm thanh tổng hợp 0.000 giờ và các bản ghi khác 40.464 giờ.

### Quyền truy cập và xét duyệt

- VietSpeech và viVoice cần chấp nhận điều khoản trên Hugging Face và xác minh quyền bản ghi theo subset. [VietSpeech](https://huggingface.co/datasets/NhutP/VietSpeech), [viVoice](https://huggingface.co/datasets/capleaf/viVoice).
- VietSuperSpeech, code_switch_OM, LSVSC và ViMedCSS cần hồ sơ quyền bản ghi và quyền chuyển lên Kaggle cho dữ liệu dùng từ xa. V02 có 163 câu VietSuperSpeech đã nghe duyệt, tương đương 0,601 giờ.
- Common Voice VI tái sử dụng bản cục bộ. Bản EN cần thông tin truy cập Mozilla Data Collective. L2-ARCTIC cần biểu mẫu của PSI Lab hoặc quyền mirror KoelLabs, chỉ chọn một bản. [Common Voice](https://commonvoice.mozilla.org/vi/datasets), [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/).
- CanVEC cần tác giả cấp audio. ZeroSpeech còn chờ quyền văn bản nguồn và nghe duyệt. VietPrism chờ yêu cầu truy cập riêng. [CanVEC](https://github.com/Bak3rLi/CanVEC).

Hàng đợi kiểm nguồn có 1076 câu VietSuperSpeech (3.969 giờ) và 119 câu LSVSC (0.203 giờ). Chưa bổ sung giờ đủ điều kiện V03.

### Dev CS độc lập

Y khoa còn 72 ứng viên qua lọc ban đầu, 0 người nói xác minh theo báo cáo ứng viên. Release độc lập cần 1500 câu, 4 domain và 100 người nói đã xác minh.
Quick CS (65 câu) dùng chẩn đoán kỹ thuật. Cấm dùng cho kết luận tổng quát hóa hoặc chấp nhận V03.

## Kết quả

Exporter tính softmax FP32 của `encoder.encoders.3.downsample.bias` trên PyTorch 2.15.0+cpu (win32) từ checkpoint gốc và ghi hệ số vào encoder trước khi lượng tử hóa. Bản phát hành và bản export được giải mã trên Windows-11-10.0.26200-SP0, Python 3.12.10, Sherpa-ONNX 2.0.1. SHA256 bộ câu `3cb69a25b1bf8453517ab32d597aa1adf3d82c6d10dd664e31ec065099045b45`.

| Dev nhanh | Model | WER | CS-WER | Recall Anh |
| --- | --- | ---: | ---: | ---: |
| interim_cs_quick | gip15_mbs4_open | 4.52% | 38.05% | 68.57% |
| interim_cs_quick | vietnote78_offline_int8 | 5.93% | 41.59% | 73.33% |
| interim_cs_quick | vietnote78_streaming_int8_pad660 | 6.45% | 46.90% | 68.57% |
| interim_cs_quick | vietnote78_offline_fp32_t4 | 5.82% | 41.59% | 72.38% |
| interim_cs_quick | vietnote78_streaming_fp32_t4_pad660 | 6.61% | 48.67% | 69.52% |
| vi_quick | gip15_mbs4_open | 9.06% | — | — |
| vi_quick | vietnote78_offline_int8 | 10.18% | — | — |
| vi_quick | vietnote78_streaming_int8_pad660 | 12.63% | — | — |
| vi_quick | vietnote78_offline_fp32_t4 | 9.22% | — | — |
| vi_quick | vietnote78_streaming_fp32_t4_pad660 | 11.51% | — | — |

- PyTorch offline đã lưu 12/12 transcript trùng FP32.
- GPU smoke một bước có loss 21.27697, CUDA allocated đỉnh 1.91 GiB, checkpoint reload True.
- offline_fp32 chính thức và re-export khớp 113/113 transcript. WER 6.67% → 6.67%, CS-WER 41.59% → 41.59%. Kết quả đạt.
- offline_int8 chính thức và re-export khớp 113/113 transcript. WER 7.07% → 7.07%, CS-WER 41.59% → 41.59%. Kết quả đạt.
- streaming_fp32 chính thức và re-export khớp 113/113 transcript. WER 7.97% → 7.97%, CS-WER 48.67% → 48.67%. Kết quả đạt.
  24 câu dài, mất 0 lượt phát sau nửa câu. Độ lệch lớn nhất ở thời điểm phát chữ đầu tiên là 0.0 giây.
- streaming_int8 chính thức và re-export khớp 113/113 transcript. WER 8.31% → 8.31%, CS-WER 46.90% → 46.90%. Kết quả đạt.
  24 câu dài, mất 0 lượt phát sau nửa câu. Độ lệch lớn nhất ở thời điểm phát chữ đầu tiên là 0.0 giây.
- Tensor FP32 offline có 0 phép ngoài dung sai trong 96 phép.
- Tensor FP32 streaming có 0 phép ngoài dung sai trong 69892 phép.
- Kiểm cấu trúc INT8 theo kênh đạt 12/12 graph.
- Tiny-set 163 câu, 489 bước, LR=0.0045. WER 7.19% → 7.89%, CS-WER 36.90% → 20.54%, recall Anh 80.94% → 81.94%. Đỉnh allocated 1.96 GiB. Kết quả không đạt.
- Lượt LR thấp dừng ở 0 bước. Windows thấy NVIDIA GeForce RTX 3050 Laptop GPU, còn PyTorch trong WSL thấy 0 thiết bị CUDA và NVML báo lỗi.
- eSpeak NG được lưu làm đối chứng âm tại data/work/vietnote/tts/espeak_control/two_family_review.jsonl.report.md.
- ZeroTTS sinh 12 câu, 0.008 giờ. Đã duyệt bởi người nghe 0 câu.
- TTS zerotts/gipformer qua ASR có WER 13.40%, CS-WER 76.47%, recall Anh 40.00%.
- TTS zerotts/vietnote qua ASR có WER 10.31%, CS-WER 58.82%, recall Anh 40.00%.
- TTS espeak/gipformer qua ASR có WER 67.01%, CS-WER 100.00%, recall Anh 0.00%.
- TTS espeak/vietnote qua ASR có WER 57.73%, CS-WER 88.24%, recall Anh 13.33%.
- Lọc tự động giữ 4/12 mẫu ZeroTTS và 0/12 mẫu eSpeak. Phiếu nghe mù đã sẵn sàng. Chờ người duyệt.
- VieNeu-TTS CPU sinh 12 câu (0.0074 giờ). Hai hệ ASR giữ 3 câu qua lọc.
  vieneu/gipformer có WER 13.40%, CS-WER 76.47% và recall Anh 40.00%.
  vieneu/vietnote có WER 10.31%, CS-WER 58.82% và recall Anh 33.33%.
- Phiếu nghe mù ở `data/work/vietnote/tts/vieneu_pilot/blind_review.csv`. Còn chờ phê duyệt của người nghe.
- Pilot Icefall cục bộ trên 24 câu dev có WER 8.29% → 8.62% và CS-WER 42.86% → 42.86%. Kết quả tổng không đạt.

### Đối chiếu INT8 cùng runtime trước khi sửa exporter

Cả bản chính thức và các biến thể đều được giải mã lại trên Windows-11-10.0.26200-SP0 với Sherpa-ONNX 2.0.1, beam 4 và 4 luồng CPU. Các graph lượng tử hóa lại dùng ONNX Runtime 1.32.0. Dùng đúng 113 câu đã khóa, SHA256 manifest `3cb69a25b1bf8453517ab32d597aa1adf3d82c6d10dd664e31ec065099045b45`.

Bản chính thức khớp kết quả tham chiếu lần trước ở 113/113 câu offline và 113/113 câu streaming.

| Chế độ | Biến thể | Khớp transcript | ΔWER (điểm %) | ΔCS-WER (điểm %) | Đạt |
| --- | --- | ---: | ---: | ---: | --- |
| offline | Lượng tử hóa lại FP32 chính thức | 113/113 | +0.00 | +0.00 | Có |
| offline | Thay encoder | 81/113 | +0.15 | +0.88 | Chưa |
| offline | Thay decoder và joiner | 113/113 | +0.00 | +0.00 | Có |
| offline | Re-export toàn bộ | 81/113 | +0.15 | +0.88 | Chưa |
| streaming | Lượng tử hóa lại FP32 chính thức | 113/113 | +0.00 | +0.00 | Có |
| streaming | Thay encoder | 95/113 | +0.00 | -0.88 | Chưa |
| streaming | Thay decoder và joiner | 113/113 | +0.00 | +0.00 | Có |
| streaming | Re-export toàn bộ | 95/113 | +0.00 | -0.88 | Chưa |

Các biến thể streaming được đo trên 24 câu từ 20 giây. Số câu mất lượt phát sau nửa câu nhiều nhất là 0.
Ở chế độ offline, biến thể chỉ thay encoder khớp 113/113 transcript với bản re-export toàn bộ.
Ở chế độ streaming, biến thể chỉ thay encoder khớp 113/113 transcript với bản re-export toàn bộ.
Cả hai encoder lượng tử hóa lại từ FP32 chính thức đều trùng SHA256 với encoder INT8 phát hành.
So graph offline, 580 initializer INT8 và 344 scale FP32 có cùng giá trị. Có 1 initializer FP32 và 6 nút Constant FP32 khác giá trị.
So graph streaming, 580 initializer INT8 và 344 scale FP32 có cùng giá trị. Có 1 initializer FP32 và 1 nút Constant FP32 khác giá trị.

### Thử thay từng nhóm hằng FP32 trong encoder INT8

Mỗi biến thể chỉ lấy giá trị FP32 đã chọn từ bản phát hành, rồi so trên 113 câu đã khóa. Giữ cố định decoder, joiner và cấu hình nhận dạng dùng trong phép hoán đổi encoder.

| Chế độ | Nhóm được khôi phục | Khớp bản phát hành | Khớp encoder re-export |
| --- | --- | ---: | ---: |
| offline | Hệ số nhân downsampling | 113/113 | 81/113 |
| offline | Bảng mã hóa vị trí | 81/113 | 113/113 |
| offline | Cả hai nhóm | 113/113 | 81/113 |
| streaming | Hệ số nhân downsampling | 113/113 | 95/113 |
| streaming | Bảng mã hóa vị trí | 95/113 | 113/113 |
| streaming | Cả hai nhóm | 113/113 | 95/113 |

Ở chế độ offline, khôi phục riêng nhóm hệ số nhân downsampling đưa toàn bộ transcript về bản phát hành. Biến thể chỉ khôi phục nhóm còn lại cho nguyên transcript re-export.
Ở chế độ streaming, khôi phục riêng nhóm hệ số nhân downsampling đưa toàn bộ transcript về bản phát hành. Biến thể chỉ khôi phục nhóm còn lại cho nguyên transcript re-export.
Khôi phục cả hai nhóm tạo encoder trùng SHA256 bản phát hành ở cả hai chế độ. Hai biến thể trùng byte dùng lại transcript đối chứng đã kiểm SHA256 trên cùng runtime.
Các biến thể trên dùng để xác định nguyên nhân lệch của exporter cũ.

Softmax FP32 của `encoder.encoders.3.downsample.bias` từ checkpoint gốc trên PyTorch 2.15.0+cpu khớp bit với hệ số phát hành. Bản re-export lệch cả 8 phần tử, với độ lệch tuyệt đối tối đa 2.98e-08.
Các hệ số đã khác nhau giữa bản phát hành và bản re-export trước khi lượng tử hóa ở cả chế độ offline và streaming.

## Kaggle

- CLI 2.2.5. Xác thực đã xác minh.

| Lượt | Trạng thái | Phiên bản từ xa | GPU đã xác minh |
| --- | --- | --- | --- |
| [gipbench-smoke-256230eae499a2ca](https://www.kaggle.com/code/vinhbui40/gipbench-smoke-256230eae499a2ca) | completed | 1 | Tesla T4 |
| [gipbench-synthetic-7c0d1eabd90c49c4](https://www.kaggle.com/code/vinhbui40/gipbench-synthetic-7c0d1eabd90c49c4) | completed | 1 | Tesla T4 |
| gipbench-v02-0d077f2409bb07d8 | prepared | — | — |

Đã gửi 2 lượt. Đã đồng bộ và xác minh 2 lượt. V02 cần quyền chuyển đúng 163 bản ghi và khóa dependency khớp GPU smoke.

- Smoke v1 đạt phép nhân ma trận CUDA trên Tesla T4. PyTorch thấy 1 GPU, VRAM trống 14.46/14.56 GiB. Giới hạn phiên 600 giây.
- Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0. k2 chưa cài, lhotse chưa cài.
- SHA256 manifest kết quả `a2edbe92f1c1137250a3d9b197e336aecd9394dc705e7e86dd7b0a00c2dfa694`.

- Lhotse 1.33.0 trích 100 khung × 80 hệ số log-Mel từ âm thanh tổng hợp 1 giây, 16 kHz.
- k2 1.24.4.dev20260625+cuda13.0.torch2.11.0 chạy RNN-T trên Tesla T4. Loss 165.70718, chuẩn L2 gradient của trọng số phép chiếu 600.76367. Giới hạn phiên 600 giây.
- SHA256 manifest kết quả `13cd1979103d3cee41e585b8c6f08afbff0c24265230189e8cd6de487af576b2`.

## Cổng chưa đạt

- pool/cv_other@v3: missing V03 rights or data-quality evidence
- pool/vietnote_fleurs_vi@v1: missing V03 rights or data-quality evidence
- pool/vietnote_vimedcss_cs@v1: missing V03 rights or data-quality evidence
- vi: 0.000/12 eligible unique hours
- cs: 0.000/8 eligible unique hours
- Thiếu artifact hoặc release đã đăng ký
- tiny · registered criteria failed
- tiny_followup · Thiếu artifact hoặc release đã đăng ký
- memory · Thiếu artifact hoặc release đã đăng ký

## Phụ thuộc theo chiến lược

| Nhánh | Chiến lược | Tiền đề |
| --- | --- | --- |
| A | conventional | Registered model and GPU validation, 12 h Vietnamese plus 8 h CS, and independent dev. |
| B | replay | V03 and a surplus of eligible unique Vietnamese and CS hours. |
| B | duration | V03 and eligible examples across duration bands. |
| B | confusables | V03 and a train-only confusable catalog. |
| B | mining | V03, V05 and verified labels drawn only from train. |
| B | distillation | V03, teacher comparison, rights and blinded verification of sampled labels. |
| B | accented | V03 and a rights-cleared speaker-disjoint accented-English release. |
| B | external_tts | V03 and cleared source-text and recording terms for ZeroSpeech. |
| B | custom_tts | V03, approved weights and voices, ASR triage and human listening pilot. |
| C | mechanisms | V03 and compatible VietNote instrumentation. |
| C | adaptation | V03 and a validated failure slice from V05. |
| D | weights | V03 and strict architecture, tokenizer, state-key, shape and shared-checkpoint compatibility. |
| E | combined | V03 and independent dev evidence for every selected change. |
| E | final | V14, locked benchmarks and a clean audit of training lineage. |

## Hành động nghiên cứu kế tiếp

Giữ 163 bản ghi V02 ở máy cho đến khi xác minh quyền huấn luyện và chuyển lên Kaggle. Ghim các gói phụ thuộc cho toàn bộ recipe Icefall trước khi chạy V02 với LR 0,0015.
