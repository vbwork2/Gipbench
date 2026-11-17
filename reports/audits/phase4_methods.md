# Phần 4 · Chẩn đoán phương pháp

## Luật chọn checkpoint đăng ký trước cho E4

Dùng `select.rule: cs_wer` trên `dev_v2_cs` độc lập, chấm theo `text_tagged`. Trong các ứng viên đủ điều kiện, chọn CS-WER thấp nhất. Nếu bằng điểm, ưu tiên WER dev VI thấp hơn rồi đến checkpoint sớm hơn. Giữ baseline nếu không có ứng viên đủ điều kiện.

Điều kiện đủ của Phần 5 là cận trên CI95 của ΔWER trên `dev_v2_vi` ≤ +0.5 điểm phần trăm, số token Anh chèn trên 100 câu thuần Việt không tăng, và ΔCS-WER của từng domain dev CS ≤ +1.0 điểm phần trăm. Tính CI theo so cặp và bootstrap theo người nói, seed 0 với 1.000 lần lấy mẫu. X0 dùng thêm điều kiện ΔWER điểm ước lượng ≤ +0.3. Các ngưỡng này cần được áp dụng đầy đủ trong cấu hình và selector E4 ở Phần 5.

Đăng ký ngày 17/11/2026 trước khi chạy chẩn đoán Phần 4. G0, C0 và C1 chỉ dùng để đo lỗi, đối chiếu thứ hạng lịch sử và so chế độ decode. E4 dùng duy nhất dev độc lập để chọn checkpoint.

## BPE Gipformer trên C1

Trên 1.260 câu, tokenizer BPE 2.000 piece biểu diễn đủ 1.443 thuật ngữ duy nhất, không có `<unk>` và giải mã ngược đúng toàn bộ sau khi đổi sang chữ hoa. Trung bình **6,32 token mỗi thuật ngữ**, trung vị **5**, với **1.093/1.443 (75,74%)** thuật ngữ dài từ 4 token. Riêng nhóm một từ có **554/896 (61,83%)** dài từ 4 token. Tính theo 2.105 lượt span, trung bình 5,46 token và 65,89% dài từ 4 token.

Giữ tokenizer và backbone ở các đối chứng E4 đầu tiên. Phân mảnh đủ lớn để theo dõi sai số theo độ dài token. Phép đo xác nhận khả năng biểu diễn và số token. Số liệu được lưu trong `results/phase4/tokenizer/{summary.json,en_spans.csv}`.

## Đối chiếu ứng viên E3 v1

Lát lịch sử 787 câu là 2.047 câu G-Group của G0 trừ 1.260 nguồn xuất hiện ở C1. Trên lát này, scorer 1.1.0 đánh dấu 686 câu đạt tiêu chí `vi_only_wer` sau khi kiểm token giống tiếng Anh. Báo WER trên đủ 787 câu để đối chiếu mốc audit 18,50 và báo thêm `vi_only_wer` trên 686 câu.

So các bản ONNX INT8 sẵn có bằng modified beam 4 và batch 1. `wise_1` trùng các hash ONNX và `tokens.txt` với `ck4132`, nên dùng chung kết quả. Dev lịch sử dùng batch 32 và giữ nguyên số đã lưu. Các so cặp mới dùng `gipbench_strict_v1`, 1.000 bootstrap theo speaker, seed 0. G-Group dùng id câu làm speaker thay thế.

Trên đủ 1.260 câu C1, `ck4132` có CS-WER **91,88** so với **70,25** của model gốc (**+21,62** điểm, CI95 **[+19,14; +23,95]**). Dev CS lịch sử xếp ngược chiều, **74,11** so với **97,20** (**−23,09** điểm). Chỉ số C1 sau khi bỏ lỗi chèn cũng tăng từ **56,37** lên **80,77** (Δ **+24,40**, CI95 **[+22,50; +26,02]**).

Trên 787 câu G-Group, WER tăng từ **18,50** lên **29,64** (Δ **+11,14**, CI95 **[+9,93; +12,37]**). Trên 686 câu thỏa tiêu chí thuần Việt của scorer 1.1.0, WER tăng từ **18,30** lên **29,54**. Như vậy `ck4132` làm tăng lỗi ở cả lát CS và lát chủ yếu tiếng Việt. Các ứng viên còn lại được chấm lần lượt trên CPU.

Trình chạy `scripts/diagnostics/e3_checkpoint_check.py` tiếp tục theo id và chấm riêng từng lát hoàn tất trong `results/phase4/checkpoints/`. Dùng `--report-only` để làm mới bảng `comparison.md` từ các lượt đã chấm.

## Ma trận giải mã

Mỗi cặp dùng cùng checkpoint và cùng câu trong G0, C0 và C1. Giữ batch 1 cho Gipformer, batch 8 cho VieGem. VieGem beam và LM dùng chung CTC prefix beam 32 và pruning của upstream. Beam đặt trọng số LM và điểm chèn từ bằng 0 rồi bỏ danh sách unigram. LM bật cấu hình phát hành α=0,35, β=1,5 và unigram. Hiệu giữa hai hàng VieGem đo tác động của việc bật tổ hợp điểm và ràng buộc từ vựng của LM.

Slot Gipformer + LM cần một LM tương thích chính xác với token ID của Gipformer 1.5 và có nguồn văn bản độc lập với bench. Bản phát hành Gipformer không có artifact đó. Sherpa-ONNX hỗ trợ RNN LM tương thích và có tùy chọn LODR riêng. [Bản phát hành Gipformer](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/tree/main), [ví dụ Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-decode-files.py).

Khi có LM phù hợp, sao chép cấu hình `gip15_mbs4_open.yaml`, đặt `system: gip15_mbs4_lm`, `inference_mode: lm`, thêm `decode.lm` trỏ tới artifact và khai báo `decode.lm_scale` đã chốt. Truyền YAML này qua `decode_matrix.py gipformer --gipformer-lm-config <yaml>`.

Kết quả ở `results/phase4/decode/` được so với greedy VieGem và beam Gipformer đã có ở `results/general_asr/`. Hai hệ tham chiếu ở 1d là Whisper-large-v3 và Qwen3-ASR. Tệp `results/phase4/decode/matrix.md` tổng hợp các lượt đã chấm.

| Hệ | G0 WER | C0 CS-WER | C1 CS-WER |
| --- | ---: | ---: | ---: |
| Gipformer greedy | Đang chạy | Đang chạy | 70,90 |
| Gipformer beam 4 | 15,21 | 107,12 | 70,25 |
| Gipformer beam 4 + LM | Chờ LM | Chờ LM | Chờ LM |
| VieGem greedy | 22,27 | 84,54 | 82,39 |
| VieGem beam 32 | Trong hàng đợi GPU | Trong hàng đợi GPU | Trong hàng đợi GPU |
| VieGem beam 32 + LM | Trong hàng đợi GPU | Trong hàng đợi GPU | Trong hàng đợi GPU |
| Whisper-large-v3 (1d) | 27,55 | — | 53,91 |
| Qwen3-ASR-0.6B (1d) | 28,49 | — | 74,71 |

Trên C1, đổi Gipformer từ beam 4 sang greedy làm CS-WER tăng **0,65** điểm (CI95 **[−0,29; +1,69]**). CS-WER chỉ tính thay và xoá tăng **1,06** điểm (CI95 **[+0,35; +1,81]**). Chênh lệch do giải mã quan sát được giữa hai cấu hình này nhỏ hơn mức tăng **21,62** điểm của E3 `ck4132` với cùng beam 4.

## Kiểm tra số học CTC

Dùng nguyên trọng số Gipformer 1.5 có head CTC, recipe icefall `f8c8578` và dữ liệu train A4 đã lưu. Giữ seed 42, CTC loss scale 0,2, ScaledAdam và LR 0,001 với Eden. Tính đặc trưng trực tiếp từ audio. Ghi id cut mỗi bước để xác nhận cùng đầu vào. Tắt TF32 ở đối chứng fp32. BF16 dùng CUDA autocast, với `ctc_loss` thuộc nhóm phép toán PyTorch chạy ở fp32 khi autocast.

Mốc A4 lịch sử dùng fp16 và budget 150 giây. Log cho thấy grad scale từ 2 xuống 2,38e-7 ở batch 50 rồi dừng ở 1,46e-11. Phép thử fp32 ở budget 60 giây hoàn thành 127 bước hữu hạn rồi kẹt ở bước 128 khi mức sử dụng VRAM xấp xỉ 4 GB. CTC loss/frame trung vị tăng từ **0,099** (bước 1–25) lên **1,368** (bước 101–125). Mức tăng xuất hiện trong cả lô chỉ có CS và lô chỉ có replay Việt. Kiểm từ duration trên 3.147 cut train cho thấy tất cả đều đủ số frame ước tính cho nhãn CTC, kể cả token lặp liền kề.

Lần đối chứng dùng budget 40 giây cho cả hai precision, một seed và 200 bước mỗi lượt. Trọng số thử nghiệm chỉ tồn tại trong tiến trình. Log được lưu trong `results/phase4/ctc/`. Log `fp32.jsonl` thuộc lần 60 giây. Log đối chứng có đuôi `_40s.jsonl`.

X0 tiếp tục dùng `--drop-ctc` trong khi chờ phép so ở budget 40 giây. Số đo hiện có cho thấy loss fp32 tăng trước khi lượt 60 giây bị kẹt.

## Môi trường thu trên G0

Tất cả 4.400 câu G0 có metadata `channel=wideband`. Lát đọc gồm VIVOS 760 câu và Common Voice 27 1.593 câu. Lát nói tự nhiên gồm 2.047 câu G-Group chia theo bốn bộ. Metadata không có lát điện thoại. Vì môi trường thu đi cùng bộ dữ liệu, khác biệt WER còn bao gồm domain, lời nói và nhãn.

| Lát G0 | Số câu | WER Gipformer beam 4 | WER VieGem greedy | Δ WER VieGem − Gipformer [CI95] |
| --- | ---: | ---: | ---: | ---: |
| VIVOS đọc | 760 | 4,51 | 6,68 | +2,17 [1,46; 3,11] |
| CV27 đọc cộng đồng | 1.593 | 7,47 | 9,50 | +2,03 [1,44; 2,64] |
| G-Group tự nhiên | 2.047 | 22,37 | 33,43 | +11,06 [10,32; 11,78] |
| G-Group edu | 495 | 23,52 | 33,70 | +10,17 [8,67; 11,60] |
| G-Group finance | 529 | 23,11 | 34,60 | +11,49 [10,08; 12,94] |
| G-Group pubadmin | 515 | 15,28 | 29,02 | +13,74 [12,21; 15,32] |
| G-Group tech | 508 | 27,70 | 36,46 | +8,76 [7,23; 10,23] |

Giữ cấu hình augment của recipe cho X0. Dùng phép so có lát điện thoại hoặc dữ liệu cùng nội dung qua nhiều kênh để kiểm tra codec 8 kHz và MUSAN trong D5. Kết quả chấm được lưu trong `results/phase4/g0_recording/`.

## Tái lập từ WSL

Chạy tại thư mục gốc repo. Các phép chấm lấy manifest đã khóa và ghi tệp mới dưới `results/phase4/`.

```bash
~/.venvs/gipbench/bin/python -m gipbench.model.systems.tokenizer_check \
  --model-dir models/gip15 --manifest data/benchmarks/cs_ggroup_v1/manifest.jsonl \
  --out results/phase4/tokenizer
~/.venvs/gipbench/bin/python scripts/diagnostics/g0_recording_check.py
~/.venvs/gipbench/bin/python scripts/diagnostics/e3_checkpoint_check.py --threads 2
~/.venvs/gipbench/bin/python scripts/diagnostics/decode_matrix.py gipformer

for precision in fp32 bf16; do
  PYTHONPATH="$HOME/icefall:$HOME/icefall/egs/librispeech/ASR/zipformer" \
    ~/.venvs/gipbench-ft/bin/python scripts/diagnostics/ctc_precision_check.py \
    --precision "$precision" --max-duration 40 --out "results/phase4/ctc/${precision}_40s.jsonl"
done

uv pip install --only-binary :all: --target results/phase4/kenlm_wheel camel-kenlm==2026.2.7
PYTHONPATH="$PWD/src:$PWD/results/phase4/kenlm_wheel" \
  ~/.venvs/gipbench-viegem/bin/python scripts/diagnostics/decode_matrix.py viegem
```

Wheel CAMeL KenLM được kiểm trực tiếp với file 4-gram phát hành kèm VieGem. Venv VieGem dùng Python 3.11. Chạy lần lượt hai precision và VieGem trên RTX 3050 4 GB. Chọn tên log mới nếu lặp CTC để lưu riêng từng lần chạy. [Wheel CAMeL KenLM](https://pypi.org/project/camel-kenlm/2026.2.7/).

Các lượt còn lại chạy nền trong WSL. Xem `results/phase4/gpu_queue.log` và `results/phase4/continuation.log` để theo dõi. Bộ kiểm thử repo đã đạt 21/21.
