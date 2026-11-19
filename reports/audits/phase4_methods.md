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

Trên 787 câu G-Group, WER tăng từ **18,50** lên **29,64** (Δ **+11,14**, CI95 **[+9,93, +12,37]**). Trên 686 câu thỏa tiêu chí thuần Việt của scorer 1.1.0, WER tăng từ **18,30** lên **29,54**.

Tất cả checkpoint đều có đủ đúng 1.260 ID C1 và 787 ID của lát G-Group. So với model gốc, CS-WER trên C1 và WER trên lát G-Group đều tăng ở từng checkpoint dù CS-WER trên dev lịch sử giảm.

| Ứng viên | CS-WER dev cũ | CS-WER C1 | WER lát G-Group |
| --- | ---: | ---: | ---: |
| Model gốc | 97,20 | 70,25 | 18,50 |
| ck4132 | 74,11 | 91,88 | 29,64 |
| ck8264 | 70,92 | 96,06 | 32,27 |
| ck12396 | 69,95 | 100,41 | 34,95 |
| ck16528 | 69,28 | 98,63 | 35,93 |
| wise_0.3 | 91,11 | 73,65 | 19,43 |
| wise_0.5 | 82,90 | 77,04 | 20,40 |
| wise_0.7 | 76,62 | 84,00 | 21,96 |

Trình chạy `scripts/diagnostics/e3_checkpoint_check.py` tiếp tục theo id và chấm riêng từng lát hoàn tất trong `results/phase4/checkpoints/`. Dùng `--report-only` để làm mới bảng `comparison.md` từ các lượt đã chấm.

## Ma trận giải mã

Mỗi cặp dùng cùng checkpoint và cùng câu trong G0, C0 và C1. Giữ batch 1 cho Gipformer, batch 8 cho VieGem. VieGem beam và LM dùng chung CTC prefix beam 32 và pruning của upstream. Beam đặt trọng số LM và điểm chèn từ bằng 0 rồi bỏ danh sách unigram. LM bật cấu hình phát hành α=0,35, β=1,5 và unigram. Hiệu giữa hai hàng VieGem đo tác động của việc bật tổ hợp điểm và ràng buộc từ vựng của LM.

Slot Gipformer + LM cần một LM tương thích chính xác với token ID của Gipformer 1.5 và có nguồn văn bản độc lập với bench. Bản phát hành Gipformer không có artifact đó. Sherpa-ONNX hỗ trợ RNN LM tương thích và có tùy chọn LODR riêng. [Bản phát hành Gipformer](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/tree/main), [ví dụ Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-decode-files.py).

### RNN LM cho Gipformer (18/11/2026)

LM là RNN LM token-level của icefall (BPE 2000 của Gipformer, LSTM 2 lớp 512 chiều, tie weights), xuất ONNX không state cho sherpa-onnx (`results/phase4/lm/gip15_rnnlm/gip15_rnnlm_e2.onnx`).

- **Text** (`scripts/diagnostics/gip15_lm_text.py`): `pool/vimedcss_train@v2`, `pool/vivos_train@v2`, `pool/cv_other@v2`, `pool/vss_cs@e3_v1`, chuẩn hoá bằng `model_case_text` (IN HOA, catalog `train-v1`). Lần kiểm đầu, `leak_audit --text-files` báo **219 câu VIVOS trùng hẳn G0**. Với pool, đây chỉ là mức review vì audio khác, nhưng với LM thì chữ chính là thứ được học. Bản cuối bỏ mọi câu trùng hẳn hoặc gần trùng với 6 bench khoá (178 VIVOS + 41 CV trùng hẳn, 5 câu gần trùng) → **14 751 câu train / 814 câu valid (311k từ)**. Kiểm lại G0 / C0 v1.1 / C1 / E0 v1.1: **0 critical**, 0 thuật ngữ `reserved_terms_v1`.
- **Train** (`scripts/diagnostics/train_gip15_rnnlm.sh`): ppl valid thấp nhất ở epoch 2 (**112,8**), rồi overfit (epoch 19: 8 947). Bản 256 chiều tốt nhất 127,4. Chỉ chọn trên valid tách từ text train, không nhìn bench (`valid_ppl.txt`).
- **Giải mã**: beam 4, mọi tham số khác giống `gip15_mbs4_open`. Lưới `lm_scale` 0,1 / 0,3 / 0,5 (`configs/bench/asr_models/gip15_mbs4_lm_s0{1,3,5}.yaml`, `decode_matrix.py gipformer --gipformer-lm-config <yaml…>`). Cả lưới được báo cáo, không chọn hệ số trên bench nên không hàng nào rankable.

Δ so với Gipformer beam 4 trên cùng câu (CI95 bootstrap cặp):

| `lm_scale` | G0 WER | C0 CS-WER | C1 CS-WER | C1 `cs_wer_sd` | C1 term recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0,1 | +0,02 [−0,05; +0,10] | −0,31 [−0,89; +0,26] | −0,31 [−0,87; +0,24] | −0,14 [−0,54; +0,27] | 39,76 |
| 0,3 | +0,06 [−0,05; +0,17] | **−1,30 [−2,14; −0,46]** | −0,51 [−1,45; +0,41] | +0,27 [−0,35; +0,89] | 39,38 |
| 0,5 | **+0,18 [+0,04; +0,31]** | **−1,07 [−1,96; −0,17]** | −0,75 [−1,82; +0,31] | +0,69 [−0,07; +1,45] | 39,29 |

Beam 4 không LM có C1 term recall **39,57**. G0 `vi_only_wer` lần lượt 9,82 / 9,82 / 9,94 (beam 4: 9,79).

LM chỉ giúp C0, nơi text LM cùng domain y khoa với ViMedCSS train. Trên C1, CS-WER giảm trong nhiễu, nhưng `cs_wer_sd` và term recall xấu đi khi tăng hệ số: LM bớt chèn từ chứ không nhận đúng thuật ngữ hơn. G0 xấu đi có ý nghĩa ở 0,5. Cùng chiều với VieGem + LM (mục 8d datasheet): LM có sẵn không thay được dữ liệu CS ngoài y khoa. Với X6 / T2, chỉ đáng thử lại khi có text CS ngoài y khoa lớn hơn nhiều (độc lập với bench).

Kết quả ở `results/phase4/decode/` được so với greedy VieGem và beam Gipformer đã có ở `results/general_asr/`. Hai hệ tham chiếu ở 1d là Whisper-large-v3 và Qwen3-ASR. Tệp `results/phase4/decode/matrix.md` tổng hợp các lượt đã chấm.

| Hệ | G0 WER | C0 CS-WER | C1 CS-WER |
| --- | ---: | ---: | ---: |
| Gipformer greedy | 15,33 | 106,89 | 70,90 |
| Gipformer beam 4 | 15,21 | 107,12 | 70,25 |
| Gipformer beam 4 + RNN LM 0,1 | 15,23 | 106,81 | 69,95 |
| Gipformer beam 4 + RNN LM 0,3 | 15,27 | 105,81 | 69,74 |
| Gipformer beam 4 + RNN LM 0,5 | 15,38 | 106,04 | 69,50 |
| VieGem greedy | 22,27 | 84,54 | 82,39 |
| VieGem beam 32 | 22,19 | 84,54 | 83,07 |
| VieGem beam 32 + LM | 20,72 | 82,71 | 85,13 |
| Whisper-large-v3 (1d) | 27,55 | — | 53,91 |
| Qwen3-ASR-0.6B (1d) | 28,49 | — | 74,71 |

Trên C1, đổi Gipformer từ beam 4 sang greedy làm CS-WER tăng **0,65** điểm (CI95 **[−0,29; +1,69]**). CS-WER chỉ tính thay và xoá tăng **1,06** điểm (CI95 **[+0,35; +1,81]**). Chênh lệch do giải mã quan sát được giữa hai cấu hình này nhỏ hơn mức tăng **21,62** điểm của E3 `ck4132` với cùng beam 4.

## Kiểm tra số học CTC

Dùng nguyên trọng số Gipformer 1.5 có head CTC, recipe icefall `f8c8578` và dữ liệu train A4 đã lưu. Giữ seed 42, CTC loss scale 0,2, ScaledAdam và LR 0,001 với Eden. Tính đặc trưng trực tiếp từ audio. Ghi id cut mỗi bước để xác nhận cùng đầu vào. Tắt TF32 ở đối chứng fp32. BF16 dùng CUDA autocast, với `ctc_loss` thuộc nhóm phép toán PyTorch chạy ở fp32 khi autocast.

Mốc A4 lịch sử dùng fp16 và budget 150 giây. Log cho thấy grad scale từ 2 xuống 2,38e-7 ở batch 50 rồi dừng ở 1,46e-11. Phép thử fp32 ở budget 60 giây hoàn thành 127 bước hữu hạn rồi kẹt ở bước 128 khi mức sử dụng VRAM xấp xỉ 4 GB. CTC loss/frame trung vị tăng từ **0,099** (bước 1–25) lên **1,368** (bước 101–125). Mức tăng xuất hiện trong cả lô chỉ có CS và lô chỉ có replay Việt. Kiểm từ duration trên 3.147 cut train cho thấy tất cả đều đủ số frame ước tính cho nhãn CTC, kể cả token lặp liền kề.

Lần đối chứng dùng budget 40 giây cho cả hai precision, một seed và 200 bước mỗi lượt. Trọng số thử nghiệm chỉ tồn tại trong tiến trình. Log được lưu trong `results/phase4/ctc/`. Log `fp32.jsonl` thuộc lần 60 giây. Log đối chứng có đuôi `_40s.jsonl`.

Cả hai lượt 40 giây hoàn thành 200 bước với cùng ID cut ở từng bước. Loss và gradient đều hữu hạn trong log. CTC loss/frame trung vị ở bước 1–25 và 176–200 lần lượt là **0,111 → 3,329** (fp32) và **0,116 → 1,801** (bf16). VRAM đỉnh lần lượt là **2.176 MiB** và **1.895 MiB**.

BF16 dùng ít VRAM hơn và có loss thấp hơn ở cuối phép thử. Loss tăng ở cả hai lượt và kết quả chỉ có một seed, nên X0 vẫn dùng `--drop-ctc`.

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

Các lượt decode và CTC đã hoàn thành được lưu dưới `results/phase4/`. LM Gipformer: `results/phase4/lm/gip15_rnnlm/` (text + `meta.json`, báo cáo rò rỉ, log train, ONNX), log decode ở `results/phase4/lm/logs/`.
