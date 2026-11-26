# Trạng thái nghiên cứu VietNote

Mục tiêu giảm lỗi chuyển mã Việt–Anh với thuật ngữ chưa thấy và giữ độ chính xác tiếng Việt.

| Lượt | Trạng thái | Điều kiện trước |
| --- | --- | --- |
| V02P | CLI Icefall đã xác minh trên Kaggle. Pilot cục bộ có kết quả chấm riêng. | — |
| V00 | Có 10 lượt quick baseline và 2/2 bộ dev đầy đủ đã chấm. | — |
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
| vietspeech | pilot_quarantine_provenance_and_speaker_review | 0.267 | 0.267 | 0.000 | 0.000 |
| vietprism | pending_author_approval | — | — | 0.000 | 0.000 |
| vimd | blocked_for_pool_pending_nc_nd_review | — | — | 0.000 | 0.000 |
| vivoice | pilot_quarantine_research_only_channel_review | 0.225 | 0.224 | 0.000 | 0.000 |
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

- VietSpeech và viVoice có cổng điều khoản trên Hugging Face và cần xác minh quyền bản ghi theo subset. [VietSpeech](https://huggingface.co/datasets/NhutP/VietSpeech), [viVoice](https://huggingface.co/datasets/capleaf/viVoice).
- VietSuperSpeech, code_switch_OM, LSVSC và ViMedCSS cần hồ sơ quyền bản ghi và quyền chuyển lên Kaggle cho dữ liệu dùng từ xa. V02 có 163 câu VietSuperSpeech đã nghe duyệt, tương đương 0,601 giờ.
- Common Voice VI tái sử dụng bản cục bộ. Bản EN cần thông tin truy cập Mozilla Data Collective. L2-ARCTIC cần biểu mẫu của PSI Lab hoặc quyền mirror KoelLabs, chỉ chọn một bản. [Common Voice](https://commonvoice.mozilla.org/vi/datasets), [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/).
- CanVEC cần tác giả cấp audio. ZeroSpeech còn chờ quyền văn bản nguồn và nghe duyệt. VietPrism chờ yêu cầu truy cập riêng. [CanVEC](https://github.com/Bak3rLi/CanVEC).

Hàng đợi kiểm nguồn có 1076 câu VietSuperSpeech (3.969 giờ) và 119 câu LSVSC (0.203 giờ). Chưa bổ sung giờ đủ điều kiện V03.
Shard code_switch_OM đã có 1,000 clip duy nhất (2.388 giờ) và 24 video trong phiếu kiểm nguồn.
Metadata theo revision xác nhận VietSpeech yêu cầu mở quyền trên Hugging Face.
Tài khoản đã xác minh quyền đọc thực tế (HTTP 206 PAR1) với VietSpeech và viVoice. Thẻ VietSpeech chỉ liệt kê audio và transcription (không có ID người nói/nguồn ghi trong shard). Thẻ viVoice có kênh nguồn YouTube (gồm cả kênh sách nói/thuyết minh như `@VoizFM`, `@FonosVietnam` — không tính giọng đọc là hội thoại tự nhiên) và giới hạn nghiên cứu CC BY-NC-SA 4.0. Cần bằng chứng quyền bản ghi, tách người nói và duyệt thủ công trước khi đưa vào V03.
Footer shard train VietSpeech có 38,002 dòng và các trường `audio, transcription`. Cần hồ sơ ngoài shard để kiểm nguồn và tách người nói.
Đã tải và kiểm hash 8 cuộc hội thoại tiếng Việt DataHive (3.452 giờ). Nguồn công bố quyền huấn luyện và CC BY 4.0. Cần phiên âm và đối chiếu người nói trước khi phát hành. [Điều khoản và đồng thuận](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/blob/c29728a0c9792ba486807fa1ea9ca64e292cec01/README.md#consent-and-privacy).
Phiếu VSS ưu tiên 192 câu có domain cụ thể. Phiếu LSVSC có 119 câu và cần phân biệt hội thoại với giọng đọc.

### Dev CS độc lập

Y khoa còn 72 ứng viên qua lọc ban đầu, 0 người nói xác minh theo báo cáo ứng viên. Release độc lập cần 1500 câu, 4 domain và 100 người nói đã xác minh.
Quick CS (65 câu) dùng chẩn đoán kỹ thuật. Cấm dùng cho kết luận tổng quát hóa hoặc chấp nhận V03.
Đã dành nguyên 3 cuộc gọi DataHive cho ứng viên dev. Phiếu nghe có 18 đoạn chẩn đoán. Đã duyệt 0 đoạn và xác minh 0 người nói.
Đối chiếu dấu vân tay 18 đoạn với 23,611 tệp train/dev đã có cache, ghi nhận 0 cặp khớp. Chờ audit đầy đủ khi có transcript đã duyệt.

## Kết quả

| Dev đầy đủ | Số câu | WER | CER | CS-WER | RTF CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev/dev_v2_vi@1 | 1,691 | 9.20% | 6.94% | — | 0.0437 |
| dev/dev_interim_cs@1 | 323 | 6.55% | 4.42% | 39.09% | 0.0732 |

Baseline là VietNote offline FP32 gốc với beam 4, batch 1 và 4 luồng. Bộ CS tạm dùng chẩn đoán.

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
- Phiếu chung đã xáo trộn 24 mẫu ZeroTTS và VieNeu-TTS. Đã duyệt 0 mẫu.
- Tiếp tục ZeroSpeech từ cache. Có 235 ứng viên, 0.732 giờ và 0 byte tải mới.
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
| [gipbench-diag-ft-7ba9aab6431ab46b](https://www.kaggle.com/code/vinhbui40/gipbench-diag-ft-7ba9aab6431ab46b) | completed | 1 | Tesla T4 |
| [gipbench-diag-ft-feb9d119b2393aef](https://www.kaggle.com/code/vinhbui40/gipbench-diag-ft-feb9d119b2393aef) | completed | 1 | Tesla T4 |
| [gipbench-expl-ft-d477aaa0766addac](https://www.kaggle.com/code/vinhbui40/gipbench-expl-ft-d477aaa0766addac) | completed | 1 | Tesla T4 |
| [gipbench-icefall-08d6b53fd3e63175](https://www.kaggle.com/code/vinhbui40/gipbench-icefall-08d6b53fd3e63175) | completed | 1 | Tesla T4 |
| [gipbench-icefall-memory-e6be6ca486125e1d](https://www.kaggle.com/code/vinhbui40/gipbench-icefall-memory-e6be6ca486125e1d) | completed | 1 | Tesla T4 |
| gipbench-icefall-pilot-439acfcb3d5ae2fb | rejected | — | — |
| [gipbench-icefall-pilot-5346eabd383e4026](https://www.kaggle.com/code/vinhbui40/gipbench-icefall-pilot-5346eabd383e4026) | failed | 1 | — |
| [gipbench-icefall-pilot-94a84b5d2b072dcb](https://www.kaggle.com/code/vinhbui40/gipbench-icefall-pilot-94a84b5d2b072dcb) | completed | 1 | Tesla T4 |
| [gipbench-icefall-pilot-c1de6ad6eea86f97](https://www.kaggle.com/code/vinhbui40/gipbench-icefall-pilot-c1de6ad6eea86f97) | failed | 1 | — |
| [gipbench-smoke-256230eae499a2ca](https://www.kaggle.com/code/vinhbui40/gipbench-smoke-256230eae499a2ca) | completed | 1 | Tesla T4 |
| [gipbench-synthetic-7c0d1eabd90c49c4](https://www.kaggle.com/code/vinhbui40/gipbench-synthetic-7c0d1eabd90c49c4) | completed | 1 | Tesla T4 |
| [gipbench-tiny-diag-21f9a9afb2cc438c](https://www.kaggle.com/code/vinhbui40/gipbench-tiny-diag-21f9a9afb2cc438c) | completed | 1 | Tesla T4 |
| [gipbench-tiny-diag-45a956bac22a0e4e](https://www.kaggle.com/code/vinhbui40/gipbench-tiny-diag-45a956bac22a0e4e) | failed | 1 | — |
| [gipbench-tiny-diag-9428a945dd969784](https://www.kaggle.com/code/vinhbui40/gipbench-tiny-diag-9428a945dd969784) | completed | 1 | Tesla T4 |
| [gipbench-tiny-diag-c0614c47523dcda6](https://www.kaggle.com/code/vinhbui40/gipbench-tiny-diag-c0614c47523dcda6) | failed | 1 | — |
| gipbench-v02-e86021aba06caa64 | prepared | — | — |

Đã gửi 14 lượt. Đã đồng bộ và xác minh 10 lượt. V02 cần văn bản quyền huấn luyện và chuyển đúng 163 bản ghi.
- gipbench-icefall-pilot-5346eabd383e4026 tới bước export sau huấn luyện và resume. Exporter dừng vì thiếu ONNX Runtime. Lượt khôi phục ghim wheel theo SHA256.
- gipbench-icefall-pilot-c1de6ad6eea86f97 dừng trước huấn luyện. Kaggle đã giải nén archive thành `bundle/pilot.json`. Lượt khôi phục kiểm SHA256 payload trong thư mục mount.

- Recipe VietNote/Icefall `finetune.py` trên Tesla T4 đã chạy 1 bước với audio tổng hợp và 77,686,775 tham số. Train loss mỗi frame 0.21527.
- Đã nạp lại checkpoint vào model mới và đối chiếu trọng số, optimizer và scheduler. SHA256 checkpoint `e9229e4d5b0f42a5d42971976868b591ac62f105af00fbc5cf384b80347a2df5`. Các wheel cài thêm được ghim trong `requirements.txt`, phiên bản package của phiên chạy nằm trong `runtime-packages.json`.

- Pilot FLEURS dùng 4 câu train và 32 câu validation. Số bước trước và sau resume là 4 và 8. Loss validation theo epoch là 1.1110, 1.0830.
- Trạng thái trùng lượt chạy liên tục tại cuối epoch 2 gồm scheduler, grad_scaler. RNG vắng trong checkpoint. Kết quả kiểm khôi phục áp dụng cho ranh giới epoch của pilot.
- Checkpoint và ba graph ONNX FP32 offline đã đồng bộ qua manifest. SHA256 checkpoint `bab780f03b355305e51eadaef0c4ec648d4eef7794c17e1c81551be674eaf401`.
- Lượt resume lệch lượt liên tục ở model, optimizer. Kiểm phục hồi chính xác chưa đạt.
- Chấm offline FP32 trên 32 câu validation FLEURS cho WER 0.20% → 1.01% và CER 0.19% → 0.84%. ΔWER +0.81 điểm phần trăm, CI 95% ghép cặp [+0.00, +1.81]. Pilot tiếng Việt đọc chưa cải thiện.
- Log T4 ghi đỉnh allocated 2,122 MB với giới hạn 8 giây/batch. V03 cần probe 200 giây.

- Smoke v1 đạt phép nhân ma trận CUDA trên Tesla T4. PyTorch thấy 1 GPU, VRAM trống 14.46/14.56 GiB. Giới hạn phiên 600 giây.
- Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0. k2 chưa cài, lhotse chưa cài.
- SHA256 manifest kết quả `a2edbe92f1c1137250a3d9b197e336aecd9394dc705e7e86dd7b0a00c2dfa694`.

- Lhotse 1.33.0 trích 100 khung × 80 hệ số log-Mel từ âm thanh tổng hợp 1 giây, 16 kHz.
- k2 1.24.4.dev20260625+cuda13.0.torch2.11.0 chạy RNN-T trên Tesla T4. Loss 165.70718, chuẩn L2 gradient của trọng số phép chiếu 600.76367. Giới hạn phiên 600 giây.
- SHA256 manifest kết quả `13cd1979103d3cee41e585b8c6f08afbff0c24265230189e8cd6de487af576b2`.

### Probe bộ nhớ FP32 200 giây trên FLEURS

- Tesla T4, `pool/vietnote_fleurs_vi@v2`, câu dài nhất 33.24 giây. Hoàn tất 3/3 bước cập nhật.
- Đỉnh CUDA allocated 8.75 GiB và reserved 9.52 GiB. VRAM trống trước probe 14.41 GiB. Dư địa ước tính tại đỉnh reserved 4.90 GiB.

| Bước | Audio thật (giây) | Sau padding (giây) | Tensor đầu vào | Loss |
| ---: | ---: | ---: | --- | ---: |
| 1 | 169.14 | 199.44 | 6 × 3324 × 80 | 1350.52295 |
| 2 | 159.42 | 199.44 | 6 × 3324 × 80 | 1097.03662 |
| 3 | 153.18 | 199.44 | 6 × 3324 × 80 | 949.84155 |

Đạt ngưỡng kỹ thuật. Probe gọi trực tiếp bước huấn luyện để đo batch dài bằng FLEURS. Cổng V03 vẫn chưa đạt.
[Artifact phép đo](../data/work/vietnote/kaggle/gipbench-icefall-memory-e6be6ca486125e1d/output/engineering-memory.json)

## Khảo sát phân bố thời lượng

- Vị trí bộ lọc cứng trong recipe Icefall ghim: `data/work/vietnote/recipe/finetune.py:1302-1340` (`1.0s <= duration <= 20.0s` và `T >= len(tokens)`).
- Cấu hình `vietnote_v03.yaml` đặt `features.max_duration = 45s`, tạo sai lệch ngầm nếu không cấu hình tường minh qua `vietnote_logged_cli.py --min-utt-duration/--max-utt-duration` (mặc định vẫn giữ `1.0–20.0s` làm đối chứng upstream).
- Lưu ý augment: vi_datamodule.py:299 applies PerturbSpeed(factors=[0.9, 1.1], p=2/3) whenever --on-the-fly-feats=1, after remove_short_and_long_utt filters raw cut durations.
- Không kết luận sai lệch thời lượng gây ra thất bại X0 khi chưa có bằng chứng thực nghiệm đối chứng.

| Tập dữ liệu | Vai trò | Số câu | Giờ | P50 (s) | P90 (s) | Max (s) | >20s (câu / giờ) | Giữ lại (1–20s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pool/vietnote_vimedcss_cs@v1` | train_pool | 3,035 | 5.952 | 7.00 | 11.00 | 24.00 | 3 (0.018h) | 3,032 (99.7% giờ) |
| `pool/vimedcss_train@v4` | train_pool | 3,868 | 7.529 | 7.00 | 11.00 | 24.00 | 4 (0.024h) | 3,864 (99.7% giờ) |
| `pool/vss_cs@v5` | train_pool | 1,087 | 4.010 | 13.60 | 15.00 | 15.00 | 0 (0.000h) | 1,087 (100.0% giờ) |
| `pool/vivos_train@v3` | train_pool | 10,555 | 13.956 | 4.03 | 8.12 | 17.12 | 0 (0.000h) | 10,553 (100.0% giờ) |
| `pool/cv_other@v3` | train_pool | 859 | 1.035 | 4.07 | 6.23 | 13.72 | 0 (0.000h) | 859 (100.0% giờ) |
| `pool/vietnote_fleurs_vi@v2` | train_pool | 2,951 | 8.918 | 10.38 | 15.60 | 42.18 | 58 (0.379h) | 2,893 (95.8% giờ) |
| `pool/vietnote_librispeech_en@v2` | train_pool | 713 | 2.595 | 14.14 | 15.85 | 17.21 | 0 (0.000h) | 713 (100.0% giờ) |
| `pilot/vietspeech` | pilot | 250 | 0.267 | 3.51 | 6.19 | 7.97 | 0 (0.000h) | 250 (100.0% giờ) |
| `pilot/vivoice` | pilot | 194 | 0.225 | 3.41 | 8.00 | 14.76 | 0 (0.000h) | 194 (100.0% giờ) |
| `pilot/lsvsc` | pilot | 120 | 0.204 | 5.79 | 10.27 | 12.74 | 0 (0.000h) | 120 (100.0% giờ) |
| `pilot/zerospeech` | pilot | 235 | 0.732 | 9.92 | 20.69 | 31.04 | 26 (0.172h) | 209 (76.5% giờ) |
| `v02/tiny_163` | v02 | 163 | 0.601 | 13.80 | 15.00 | 15.00 | 0 (0.000h) | 163 (100.0% giờ) |
| `dev/dev_v2_vi@1` | dev | 1,691 | 9.060 | 19.47 | 28.05 | 31.83 | 780 (5.403h) | 911 (40.4% giờ) |
| `dev/dev_interim_cs@1` | dev | 323 | 1.183 | 13.30 | 15.00 | 15.00 | 0 (0.000h) | 323 (100.0% giờ) |
| `quick/vi_quick` | quick_dev | 36 | 0.156 | 15.18 | 26.44 | 30.60 | 12 (0.084h) | 24 (46.0% giờ) |
| `quick/interim_cs_quick` | quick_dev | 65 | 0.251 | 14.60 | 15.00 | 15.00 | 0 (0.000h) | 65 (100.0% giờ) |

## Chẩn đoán LR trên tập nhỏ được cấp phép (không phải V02)

Do 163 bản ghi VietSuperSpeech của V02 vẫn thiếu văn bản quyền bản ghi/chuyển cloud, thí nghiệm đối chứng LR độc lập này dùng 163 câu được cấp phép CC-BY-4.0 / CC0-1.0 (FLEURS VI + Common Voice VI + LibriSpeech EN, mỗi câu 8,1–16,0s, 489 bước FP32, seed 0 trên Tesla T4). Kết quả **không** thay thế cổng V02 đã đăng ký.

| Nhánh | LR | Bước | Tiếp xúc (giờ) | WER đầu → cuối | CS-WER đầu → cuối | Recall Anh đầu → cuối | S / D / I cuối | Loss đầu → cuối | Grad L2 (med / max) | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: |
| Arm A (`gipbench-tiny-diag-9428a945dd969784`) | 0.0045 | 489 | 1.593 | 4.86% → 9.75% | 0.72% → 14.16% | 76.67% → 6.67% | 182/133/155 | 80.9 → 38.7 | 690.1 / 13546.6 | 1.98 |
| Arm B (`gipbench-tiny-diag-21f9a9afb2cc438c`) | 0.0015 | 489 | 1.593 | 4.86% → 1.78% | 0.72% → 1.08% | 76.67% → 70.00% | 41/25/20 | 80.9 → 38.6 | 313.2 / 10772.9 | 1.98 |

## Fine-tune chẩn đoán đầu-cuối & đối chứng chính sách thời lượng (dữ liệu cấp phép)

Huấn luyện chính thức bằng Icefall `finetune.py` + `vietnote_logged_cli.py` trên hỗn hợp cấp phép CC-BY-4.0 / CC0-1.0 (FLEURS VI gồm 35 câu >20s + Common Voice VI + LibriSpeech EN, 3 epoch, FP32, LR 0,0015, seed 0) với tập dev tách hoàn toàn nguồn và người nói (71 câu, gồm 14 câu >20s). Thí nghiệm này chỉ mang tính chẩn đoán kỹ thuật, **không** dùng dữ liệu test khóa và **không** suy rộng cho V03.

| Chính sách | Lượt Kaggle | Câu giữ lại (>20s) | Bước / Giờ tiếp xúc | Epoch chọn | Dev WER (ep0 → chọn) | Dev >20s WER (ep0 → chọn) | Dev >20s D (ep0 → chọn) | Peak VRAM (MB) |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: |
| `control_20s` | `gipbench-diag-ft-7ba9aab6431ab46b` | 580/615 (0 câu >20s) | 171 / 4.602h (13 lượt >20s) | 1 | 8.49% → 8.21% | 12.55% → 11.82% | 27 → 20 | 6272 |
| `extended_45s` | `gipbench-diag-ft-feb9d119b2393aef` | 615/615 (35 câu >20s) | 203 / 5.234h (91 lượt >20s) | 1 | 8.49% → 8.44% | 12.55% → 12.06% | 27 → 21 | 10020 |

### Đánh giá xuất ONNX FP32 & INT8 cục bộ (Sherpa-ONNX Windows)

| Model | Tập chấm | Chế độ | WER | CS-WER | Recall Anh | >20s WER (D) | Mất phát chữ sau nửa câu (≥20s) |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `pretrained` | `authorized_dev` | `offline_fp32` | 8.07% | 0.75% | 81.82% | 11.70% (D=18) | — |
| `pretrained` | `authorized_dev` | `offline_int8` | 8.26% | 1.75% | 63.64% | 11.10% (D=21) | — |
| `pretrained` | `authorized_dev` | `streaming_fp32` | 8.85% | 1.00% | 72.73% | 11.94% (D=13) | 0/14 |
| `pretrained` | `authorized_dev` | `streaming_int8` | 9.67% | 2.00% | 63.64% | 12.55% (D=13) | 0/14 |
| `pretrained` | `vi_quick` | `offline_fp32` | 9.22% | — | — | 9.57% (D=32) | — |
| `pretrained` | `vi_quick` | `offline_int8` | 10.18% | — | — | 10.51% (D=47) | — |
| `pretrained` | `vi_quick` | `streaming_fp32` | 11.51% | — | — | 11.34% (D=35) | 0/12 |
| `pretrained` | `vi_quick` | `streaming_int8` | 12.63% | — | — | 13.42% (D=60) | 0/12 |
| `pretrained` | `interim_cs_quick` | `offline_fp32` | 5.82% | 41.59% | 72.38% | — | — |
| `pretrained` | `interim_cs_quick` | `offline_int8` | 5.93% | 41.59% | 73.33% | — | — |
| `pretrained` | `interim_cs_quick` | `streaming_fp32` | 6.61% | 48.67% | 69.52% | — | — |
| `pretrained` | `interim_cs_quick` | `streaming_int8` | 6.45% | 46.90% | 68.57% | — | — |
| `control_20s` | `authorized_dev` | `offline_fp32` | 7.62% | 0.75% | 81.82% | 11.10% (D=17) | — |
| `control_20s` | `authorized_dev` | `offline_int8` | 8.85% | 2.00% | 54.55% | 12.18% (D=24) | — |
| `control_20s` | `authorized_dev` | `streaming_fp32` | 9.12% | 1.00% | 72.73% | 12.06% (D=17) | 0/14 |
| `control_20s` | `authorized_dev` | `streaming_int8` | 9.95% | 3.24% | 45.45% | 12.18% (D=24) | 0/14 |
| `control_20s` | `vi_quick` | `offline_fp32` | 10.34% | — | — | 10.72% (D=45) | — |
| `control_20s` | `vi_quick` | `offline_int8` | 12.57% | — | — | 13.01% (D=75) | — |
| `control_20s` | `vi_quick` | `streaming_fp32` | 12.04% | — | — | 12.28% (D=48) | 0/12 |
| `control_20s` | `vi_quick` | `streaming_int8` | 15.45% | — | — | 16.23% (D=103) | 0/12 |
| `control_20s` | `interim_cs_quick` | `offline_fp32` | 5.64% | 30.97% | 79.05% | — | — |
| `control_20s` | `interim_cs_quick` | `offline_int8` | 5.80% | 32.74% | 78.10% | — | — |
| `control_20s` | `interim_cs_quick` | `streaming_fp32` | 6.01% | 40.71% | 74.29% | — | — |
| `control_20s` | `interim_cs_quick` | `streaming_int8` | 6.29% | 38.05% | 74.29% | — | — |
| `extended_45s` | `authorized_dev` | `offline_fp32` | 7.89% | 1.50% | 63.64% | 11.22% (D=17) | — |
| `extended_45s` | `authorized_dev` | `offline_int8` | 8.80% | 2.49% | 54.55% | 12.06% (D=27) | — |
| `extended_45s` | `authorized_dev` | `streaming_fp32` | 9.26% | 1.25% | 72.73% | 12.55% (D=19) | 0/14 |
| `extended_45s` | `authorized_dev` | `streaming_int8` | 9.72% | 2.49% | 54.55% | 12.67% (D=25) | 0/14 |
| `extended_45s` | `vi_quick` | `offline_fp32` | 9.91% | — | — | 10.72% (D=42) | — |
| `extended_45s` | `vi_quick` | `offline_int8` | 11.24% | — | — | 11.86% (D=62) | — |
| `extended_45s` | `vi_quick` | `streaming_fp32` | 11.99% | — | — | 12.17% (D=46) | 0/12 |
| `extended_45s` | `vi_quick` | `streaming_int8` | 13.48% | — | — | 13.53% (D=73) | 0/12 |
| `extended_45s` | `interim_cs_quick` | `offline_fp32` | 5.85% | 31.86% | 79.05% | — | — |
| `extended_45s` | `interim_cs_quick` | `offline_int8` | 5.95% | 33.63% | 77.14% | — | — |
| `extended_45s` | `interim_cs_quick` | `streaming_fp32` | 6.24% | 38.05% | 74.29% | — | — |
| `extended_45s` | `interim_cs_quick` | `streaming_int8` | 6.84% | 39.82% | 71.43% | — | — |

## Tuyển lọc dữ liệu tự động theo nguồn gốc

Phân tách 5 nhóm nguồn gốc (`human_labeled`, `existing_supervised`, `pseudo_labeled`, `unlabeled_conversational`, `synthetic_speech`) theo `configs/dataset/vietnote_curation_policy.yaml`. Số giờ/dòng đã duyệt thủ công giữ nguyên `0` nếu chưa nghe kiểm; **không** xem đồng thuận giữa các teacher ASR là nhãn chuẩn (ground truth).

| Tập dữ liệu | Nguồn gốc | Chuyển cloud | Tổng (câu / giờ) | Giữ lại (câu / giờ) | WER teacher | Rà soát người |
| --- | --- | --- | --- | --- | --- | ---: |
| `pool/vietnote_fleurs_vi@v2` | existing_supervised | Có (CC-BY-4.0) | 2,951 / 8.918h | spot 150 câu (0 fail, 3 ngoại lệ) | VN med 1.75% | 0 |
| `pool/cv_other@v3` | existing_supervised | Có (CC0-1.0) | 859 / 1.035h | 615 / 0.747h (loại 244: 42 leak) | pair med 14.29% | 0 |
| `pool/vietnote_librispeech_en@v2` | existing_supervised | Có (CC-BY-4.0) | 713 / 2.595h | spot 100 câu (0 fail, 0 ngoại lệ) | VN med 0.00% | 0 |
| `pool/vivos_train@v3` | existing_supervised | Nội bộ (NC-SA) | 10,555 / 13.956h | spot 150 câu (0 fail, 0 ngoại lệ) | pair med 0.00% | 0 |
| `pool/vietnote_vimedcss_cs@v1` | existing_supervised | Chờ quyền YT | 3,035 / 5.952h | spot 150 câu (0 fail, 7 ngoại lệ) | pair med 10.58% | 0 |
| `pool/vss_cs@v5` | pseudo_labeled | Chờ quyền ghi/cloud | 1,087 / 4.010h | Qwen=1087, FW=317 (1084 span ok) | pair med 12.90% | 0 |
| `datahive_vi` (5 cuộc train) | unlabeled_conversational -> pseudo | Có (CC-BY-4.0) | 826 VAD / 1.040h (từ 2.024h thô) | 315 / 0.485h (gồm 68 câu `<en>` = 0.115h; loại 8 thuật ngữ khóa) | pair med 10.00% (gốc 28.39%) | 0 |

## Nhánh nghiên cứu thăm dò (V02A & VE01 — tách biệt V03)

- **V02A** (`configs/model/experiments/vietnote_v02a_validation.yaml`): giao thức kiểm định huấn luyện thay thế trên dữ liệu có quyền hợp lệ (tổng hợp `tiny-diag` Arm A LR 0,0045 vs Arm B LR 0,0015, `icefall-memory` 200s FP32 và `diag-ft`), độc lập về mặt khoa học và **không** thay thế hay xóa bỏ V02 lịch sử (163 câu VSS) lẫn cổng V03.
- Các thí nghiệm `VE01` chỉ dùng kết luận thăm dò (`allow_generalization_claims: false`), giữ nguyên quyền bản ghi, chuyển cloud và chống rò rỉ benchmark.

### Thí nghiệm thăm dò `VE01` trên Kaggle T4 (`gipbench-expl-ft-d477aaa0766addac`)

- Cấu hình: LR=0.0015, FP32, `max_duration=160`, `utt_duration=[1, 35]s`, 3 epoch, seed 0.
- Dữ liệu vào: 1950 câu (4.753h: fleurs_vi=2.454h, cv_other_curated=0.663h, datahive_vi_curated=0.485h, librispeech_en=1.151h); giữ lại 1949 câu (4.741h, 42 câu >20s); 393 bước, tiếp xúc 14.333h (134 lượt >20s); đỉnh VRAM 9239 MB; val loss `[0.2411, 0.2463, 0.258]`; chọn epoch 1.

| Epoch (Kaggle GPU greedy) | Bước | WER | CS-WER | Recall Anh | VI-only WER | S / D / I | >20s WER (D) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0 | 8.49% | 0.75% | 81.82% | 13.46% | 99/45/42 | 12.55% (D=27) |
| 1 * | 131 | 8.99% | 2.24% | 45.45% | 13.31% | 118/37/42 | 12.55% (D=19) |
| 2 | 262 | 10.77% | 7.48% | 27.27% | 13.62% | 144/53/39 | 13.63% (D=25) |
| 3 | 393 | 12.50% | 13.97% | 18.18% | 14.52% | 146/83/45 | 13.75% (D=29) |

### Đánh giá Sherpa-ONNX cục bộ (Windows, beam 4): `VE01` vs `pretrained`

| Model | Tập chấm | Chế độ | WER | CS-WER | Recall Anh | VI-only WER | S / D / I | >20s WER (D) | Mất phát chữ >=20s |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `pretrained` | `authorized_dev` | `offline_fp32` | 8.07% | 0.75% | 81.82% | 12.86% | 103/31/43 | 11.70% (D=18) | — |
| `ve01` | `authorized_dev` | `offline_fp32` | 8.94% | 1.75% | 54.55% | 13.62% | 115/38/43 | 12.18% (D=24) | — |
| `pretrained` | `authorized_dev` | `offline_int8` | 8.26% | 1.75% | 63.64% | 12.71% | 103/38/40 | 11.10% (D=21) | — |
| `ve01` | `authorized_dev` | `offline_int8` | 9.72% | 3.49% | 36.36% | 13.77% | 123/43/47 | 13.15% (D=24) | — |
| `pretrained` | `authorized_dev` | `streaming_fp32` | 8.85% | 1.00% | 72.73% | 14.67% | 115/33/46 | 11.94% (D=13) | 0/14 |
| `ve01` | `authorized_dev` | `streaming_fp32` | 9.49% | 2.24% | 54.55% | 14.22% | 134/30/44 | 13.15% (D=13) | 0/14 |
| `pretrained` | `authorized_dev` | `streaming_int8` | 9.67% | 2.00% | 63.64% | 15.28% | 126/39/47 | 12.55% (D=13) | 0/14 |
| `ve01` | `authorized_dev` | `streaming_int8` | 10.68% | 4.24% | 27.27% | 15.73% | 139/52/43 | 13.99% (D=33) | 0/14 |
| `pretrained` | `vi_quick` | `offline_fp32` | 9.22% | — | — | 9.22% | 57/60/56 | 9.57% (D=32) | — |
| `ve01` | `vi_quick` | `offline_fp32` | 9.75% | — | — | 9.75% | 62/63/58 | 10.41% (D=34) | — |
| `pretrained` | `vi_quick` | `offline_int8` | 10.18% | — | — | 10.18% | 58/93/40 | 10.51% (D=47) | — |
| `ve01` | `vi_quick` | `offline_int8` | 10.82% | — | — | 10.82% | 63/89/51 | 10.61% (D=38) | — |
| `pretrained` | `vi_quick` | `streaming_fp32` | 11.51% | — | — | 11.51% | 88/70/58 | 11.34% (D=35) | 0/12 |
| `ve01` | `vi_quick` | `streaming_fp32` | 11.13% | — | — | 11.13% | 79/68/62 | 11.45% (D=35) | 0/12 |
| `pretrained` | `vi_quick` | `streaming_int8` | 12.63% | — | — | 12.63% | 86/108/43 | 13.42% (D=60) | 0/12 |
| `ve01` | `vi_quick` | `streaming_int8` | 12.68% | — | — | 12.68% | 75/111/52 | 13.01% (D=57) | 0/12 |
| `pretrained` | `interim_cs_quick` | `offline_fp32` | 5.82% | 41.59% | 72.38% | — | 90/20/113 | — | — |
| `ve01` | `interim_cs_quick` | `offline_fp32` | 6.27% | 31.86% | 77.14% | — | 98/29/113 | — | — |
| `pretrained` | `interim_cs_quick` | `offline_int8` | 5.93% | 41.59% | 73.33% | — | 96/25/106 | — | — |
| `ve01` | `interim_cs_quick` | `offline_int8` | 6.84% | 37.17% | 74.29% | — | 104/44/114 | — | — |
| `pretrained` | `interim_cs_quick` | `streaming_fp32` | 6.61% | 48.67% | 69.52% | — | 109/19/125 | — | — |
| `ve01` | `interim_cs_quick` | `streaming_fp32` | 7.00% | 41.59% | 73.33% | — | 107/30/131 | — | — |
| `pretrained` | `interim_cs_quick` | `streaming_int8` | 6.45% | 46.90% | 68.57% | — | 111/21/115 | — | — |
| `ve01` | `interim_cs_quick` | `streaming_int8` | 7.99% | 46.90% | 62.86% | — | 121/56/129 | — | — |
| `pretrained` | `dev_interim_cs` | `offline_fp32` | 6.55% | 39.09% | 80.48% | — | 422/90/638 | — | — |
| `ve01` | `dev_interim_cs` | `offline_fp32` | 6.91% | 37.82% | 80.48% | — | 453/125/636 | — | — |
| `pretrained` | `dev_interim_cs` | `offline_int8` | 6.79% | 39.45% | 79.68% | — | 440/130/622 | — | — |
| `ve01` | `dev_interim_cs` | `offline_int8` | 7.47% | 39.09% | 78.49% | — | 478/208/627 | — | — |
| `pretrained` | `dev_v2_vi_strat300` | `offline_fp32` | 9.58% | — | — | 9.58% | 616/427/546 | 8.86% (D=222) | — |
| `ve01` | `dev_v2_vi_strat300` | `offline_fp32` | 10.26% | — | — | 10.26% | 689/424/590 | 9.29% (D=205) | — |
| `pretrained` | `dev_v2_vi_strat300` | `offline_int8` | 10.42% | — | — | 10.42% | 596/690/442 | 9.86% (D=361) | — |
| `ve01` | `dev_v2_vi_strat300` | `offline_int8` | 11.16% | — | — | 11.16% | 674/646/532 | 10.19% (D=313) | — |

#### Bootstrap ghép cặp 95% CI (`VE01` − `pretrained`)

| Tập chấm | Chế độ | Metric | Pretrained | VE01 | Δ | CI 95% |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `dev_interim_cs` | `offline_fp32` | `wer` | 6.55% | 6.91% | +0.36 | [+0.12, +0.64] |
| `dev_interim_cs` | `offline_fp32` | `cs_wer` | 39.09% | 37.82% | -1.27 | [-5.98, +3.67] |
| `dev_interim_cs` | `offline_fp32` | `term_recall` | 80.48% | 80.48% | +0.00 | [-3.15, +2.57] |
| `dev_interim_cs` | `offline_fp32` | `n_wer` | 5.49% | 5.91% | +0.42 | [+0.20, +0.65] |
| `dev_interim_cs` | `offline_fp32` | `en_hallucination_per100` | — | — | — | — |
| `dev_interim_cs` | `offline_int8` | `wer` | 6.79% | 7.47% | +0.69 | [+0.30, +1.22] |
| `dev_interim_cs` | `offline_int8` | `cs_wer` | 39.45% | 39.09% | -0.36 | [-5.29, +4.36] |
| `dev_interim_cs` | `offline_int8` | `term_recall` | 79.68% | 78.49% | -1.20 | [-4.97, +2.23] |
| `dev_interim_cs` | `offline_int8` | `n_wer` | 5.73% | 6.45% | +0.72 | [+0.35, +1.17] |
| `dev_interim_cs` | `offline_int8` | `en_hallucination_per100` | — | — | — | — |
| `dev_v2_vi_strat300` | `offline_fp32` | `wer` | 9.58% | 10.26% | +0.69 | [+0.42, +0.98] |
| `dev_v2_vi_strat300` | `offline_fp32` | `cer` | 7.23% | 7.67% | +0.44 | [+0.25, +0.64] |
| `dev_v2_vi_strat300` | `offline_fp32` | `en_hallucination_per100` | 0.67% | 3.00% | +2.33 | [+0.33, +5.08] |
| `dev_v2_vi_strat300` | `offline_int8` | `wer` | 10.42% | 11.16% | +0.75 | [+0.30, +1.21] |
| `dev_v2_vi_strat300` | `offline_int8` | `cer` | 8.04% | 8.65% | +0.61 | [+0.20, +1.01] |
| `dev_v2_vi_strat300` | `offline_int8` | `en_hallucination_per100` | 1.00% | 2.67% | +1.67 | [-0.67, +4.42] |

Khóa offline gồm 941 phiên bản distribution từ phiên Icefall v1 và năm wheel được kiểm SHA256. Khóa yêu cầu ảnh Kaggle có Python, CUDA, PyTorch và các gói dựng sẵn đúng phiên bản đã ghi.
Khóa huấn luyện và export FLEURS ghi 942 distribution và sáu wheel có SHA256, gồm ONNX Runtime 1.32.0.

Gói V02 `gipbench-v02-e86021aba06caa64` đã chuẩn bị cục bộ với LR 0,0015, seed 0 và 163 câu. SHA256 manifest `f3ff60b1e7739fcc78dbf1eb99e92342113a3ff6958dfca157b26f4d3831b46a`.

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
- Probe bộ nhớ V03 cần các pool đã đăng ký và ba bước FP32 ở mức 200 giây/batch có câu từ 25 giây. Recipe CLI Icefall ghim hiện loại câu train trên 20 giây. Phép đo kỹ thuật FLEURS dùng trực tiếp bước huấn luyện.

## Việc cần người dùng

- Cung cấp văn bản quyền huấn luyện và chuyển lên Kaggle cho danh sách V02. [Phạm vi 163 bản ghi](../data/work/vietnote/v02/rights-scope.csv).
- Xin hồ sơ nguồn và người nói từ đơn vị phát hành VietSpeech. VietPrism chờ tác giả phê duyệt. [VietSpeech](https://huggingface.co/datasets/NhutP/VietSpeech).
- Phân công người nghe duyệt các phiếu và xác minh người nói. [Hướng dẫn duyệt và hồ sơ quyền](../docs/VIETNOTE_KAGGLE.md#phiếu-nghe-và-quyền-nguồn).

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

Sau khi có quyền cho đúng 163 bản ghi, tái lập arm đối chứng LR 0,0045 trên Kaggle T4 với seed 0, cùng môi trường và 489 bước. Giữ nguyên quy tắc chấm đã đăng ký.
