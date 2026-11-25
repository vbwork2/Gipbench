# Trạng thái nghiên cứu VietNote

Mục tiêu giảm lỗi chuyển mã Việt–Anh với thuật ngữ chưa thấy và giữ độ chính xác tiếng Việt.

| Lượt | Trạng thái | Điều kiện trước |
| --- | --- | --- |
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

| Nhóm | Giờ có sẵn | Giờ đủ bằng chứng | Mục tiêu | Thiếu trước lọc |
| --- | ---: | ---: | ---: | ---: |
| VI | 1.348 | 0.000 | 12 | 10.652 |
| CS | 5.952 | 0.000 | 8 | 2.048 |

Giờ có sẵn dùng audio tại chỗ, khử trùng theo SHA256 ghi trong manifest hoặc đường dẫn. Giờ đủ bằng chứng cần hồ sơ quyền sử dụng, kiểm nhãn, phân tách người nói và kiểm âm học của release.

| Release trong hỗn hợp | Số dòng | Giờ có sẵn | Giờ đủ bằng chứng |
| --- | ---: | ---: | ---: |
| pool/cv_other@v3 | 859 | 1.035 | 0.000 |
| pool/vietnote_fleurs_vi@v1 | 96 | 0.313 | 0.000 |
| pool/vietnote_vimedcss_cs@v1 | 3035 | 5.952 | 0.000 |

### Tiến độ lấy dữ liệu

| Nguồn | Trạng thái registry | Giờ cục bộ | Giờ đủ V03 |
| --- | --- | ---: | ---: |
| code_switch_om | quarantined_unverified_audio_rights_and_transcript_quality | — | 0.000 |
| vss | reuse_local_research_pools_pending_original_recording_rights | 4.010 | 0.000 |
| vimedcss | reuse_local_train | 7.529 | 0.000 |
| canvec | blocked_audio_by_author_request | — | 0.000 |
| vietspeech | gated | — | 0.000 |
| vietprism | pending_author_approval | — | 0.000 |
| vimd | blocked_for_pool_pending_nc_nd_review | — | 0.000 |
| vivoice | gated_research_only | — | 0.000 |
| lsvsc | pilot_quarantine_transcript_and_recording_rights | 0.204 | 0.000 |
| vivos | reuse_local_research_pool | 13.956 | 0.000 |
| fleurs_vi | pilot_eligible | 0.329 | 0.000 |
| commonvoice_vi | reuse_local_cv27 | 1.035 | 0.000 |
| l2_arctic | blocked_download_form | — | 0.000 |
| l2_arctic_mirror | gated_same_recordings_as_l2_arctic | — | 0.000 |
| librispeech | pilot_eligible | 0.425 | 0.000 |
| commonvoice_en | blocked_release_access_unverified | — | 0.000 |
| zerospeech | pilot_quarantine_text_rights_and_pronunciation_review | 0.365 | 0.000 |

Hàng đợi kiểm nguồn có 1076 câu VietSuperSpeech (3.969 giờ) và 119 câu LSVSC (0.203 giờ). Chưa bổ sung giờ đủ điều kiện V03.
Metadata ở revision đã khóa xác nhận VietSpeech yêu cầu mở quyền trên Hugging Face.

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
- Batch pooled 32 giây, 3 bước, allocated 2.08 GiB, reserved 2.67 GiB, clip dài nhất 16.00 giây, đạt.
- Batch pooled 64 giây, 3 bước, allocated 3.28 GiB, reserved 3.48 GiB, clip dài nhất 21.00 giây, thiếu 10% dự phòng.
- Batch pooled 200 giây, 3 bước, allocated 8.28 GiB, reserved 8.96 GiB, clip dài nhất 18.00 giây, thiếu 10% dự phòng.
- Batch long 37 giây, 3 bước, allocated 2.35 GiB, reserved 3.16 GiB, clip dài nhất 36.66 giây, thiếu 10% dự phòng.
- eSpeak NG được lưu làm đối chứng âm tại data/work/vietnote/tts/espeak_control/two_family_review.jsonl.report.md.
- ZeroTTS sinh 12 câu, 0.008 giờ. Đã duyệt bởi người nghe 0 câu.
- TTS zerotts/gipformer qua ASR có WER 13.40%, CS-WER 76.47%, recall Anh 40.00%.
- TTS zerotts/vietnote qua ASR có WER 10.31%, CS-WER 58.82%, recall Anh 40.00%.
- TTS espeak/gipformer qua ASR có WER 67.01%, CS-WER 100.00%, recall Anh 0.00%.
- TTS espeak/vietnote qua ASR có WER 57.73%, CS-WER 88.24%, recall Anh 13.33%.
- Lọc tự động giữ 4/12 mẫu ZeroTTS và 0/12 mẫu eSpeak. Phiếu nghe mù đã sẵn sàng. Chờ người duyệt.

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

## Cổng chưa đạt

- pool/cv_other@v3: missing V03 rights or data-quality evidence
- pool/vietnote_fleurs_vi@v1: missing V03 rights or data-quality evidence
- pool/vietnote_vimedcss_cs@v1: missing V03 rights or data-quality evidence
- vi: 0.000/12 eligible unique hours
- cs: 0.000/8 eligible unique hours
- Thiếu artifact hoặc release đã đăng ký
- tiny · registered criteria failed
- tiny_followup · Thiếu artifact hoặc release đã đăng ký
- memory · registered criteria failed

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

Tiếp tục lượt V02 giảm learning rate đã đăng ký khi WSL truy cập lại được GPU, giữ nguyên 163 câu và quy tắc chấm.
