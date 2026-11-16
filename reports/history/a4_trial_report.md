# A4: FT thử nhỏ trên GPU (NHÁP, 16/11/2026)

> **NHÁP.** Mục đích là chạy thông pipeline FT và đo tài nguyên, không phải chọn mô hình. Chấm trên bench `v0-draft`
> (B1–B3 chưa duyệt). Dữ liệu CS là nhãn giả (vss_cs), không có ViMedCSS train đầy đủ trên máy.

## Cấu hình

| | Giá trị |
| --- | --- |
| Model gốc | Gipformer 1.5 `model.pt` (tải từ HF `g-group-ai-lab/gipformer1.5-68M-rnnt`), 68.6 M tham số sau khi bỏ head CTC |
| Dữ liệu train | CS: 293 câu, 1.07 h từ `vss_cs_clean` (60 video) · replay: 791 câu, 1.00 h VIVOS train (còn 756 sau khi bỏ 34 câu trùng text bench và 1 câu quá dài). Speed perturb 0.9/1.0/1.1, mô phỏng 8 kHz 30% |
| Dev | 47 câu, 0.17 h `vss_cs_clean`, 14 video không có trong train |
| Lệnh | `data/ft_trial/`: `bash ft/run_e3.sh 0 5` với `EPOCHS=2 AVG=1 USE_AVG_MODEL=0 MAXDUR=150 LR=0.001 CS_W=VI_W=0.5` |
| Số step | 194 (khoảng 97 / epoch; `CutSet.mux` dừng khi một tập hết) |
| GPU | RTX 3050 Laptop 4 GB, fp16 |

## Đo tài nguyên

| | Giá trị |
| --- | --- |
| VRAM đỉnh | **3 912 / 4 096 MiB** ở `MAXDUR=150` (3 914 MiB ở `MAXDUR=200`): 4 GB là trần, không tăng batch được |
| Tốc độ | **0.66 step/s** (97 step / 146 s mỗi epoch, `MAXDUR=150`); khoảng **93 giờ audio / GPU-giờ** (đã tính bản speed perturb) |
| Thời gian | stage 1–2 (lhotse + fbank, CPU): khoảng 4 phút · FT 2 epoch: 5 phút 11 s · export ONNX + int8: khoảng 1 phút · decode bench 4 238 câu: khoảng 20 phút / lần trên CPU 16 luồng |

**Ước tính GPU-giờ cho FT thật.** Dữ liệu khoảng 48 h thô (ViMedCSS 24.3 + vss_cs 6.1 + replay VIVOS/CV 17.2 + confusable 0.9).
Speed perturb ×3 cho khoảng 145 h mỗi epoch; 8 epoch là khoảng 1 160 h audio.

| GPU | Ước tính | Cơ sở |
| --- | --- | --- |
| RTX 3050 4 GB (đo) | khoảng 12–13 GPU-h | 93 h audio / GPU-h |
| GPU 16–24 GB (`MAXDUR` 600–1 000) | **khoảng 2–4 GPU-h** | GIẢ ĐỊNH thông lượng 3–6× (batch lớn hơn, GPU nhanh hơn), **chưa đo** |

## Kết quả (bench NHÁP)

| Hệ thống | CS-WER B1+B2 | Δ [CI95 ghép cặp] | N-WER B1+B2 | CS-WER B3 | WER B5 | Chèn sai B5 /100 câu | Leaderboard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e0_baseline | 71.66 | – | 13.13 | 106.81 | 9.52 | 1.06 | baseline |
| e1_hw1.5 | 60.21 | −11.44 [−12.87, −9.96] | 13.13 | 109.03 | 9.63 | 2.31 | đạt |
| a4_trial | 71.21 | −0.44 [−2.12, +1.12] | 15.94 (+2.82) | 101.15 | 11.41 (+1.90) | 0.83 | **loại**: B5 +1.90, N-WER +2.50 |
| a4_trial_hw1.5 | 59.09 | −12.57 [−14.70, −10.78] | 16.20 (+3.07) | 102.60 | 11.51 (+2.00) | 2.08 | **loại**: B5 +2.00, N-WER +2.53 |

Đọc nhanh:
- 194 step trên 1 h nhãn giả **chưa giúp được CS** (Δ −0.44, CI chứa 0) mà **làm tiếng Việt quên rõ**: N-WER +2.8, B5 +1.9.
  Đúng kiểu lỗi leaderboard được thiết kế để chặn.
- B3 giảm 5.7 điểm. Có thể vì vss_cs có ít câu medical, cũng có thể vì model chèn ít hơn. Với 1 lần chạy thì chưa kết luận được.
- Hướng cho FT thật: cần cs_real (ViMedCSS train đầy đủ), replay nhiều hơn (tỉ lệ replay 0.25 + distill 0.20 theo
  `configs/train_mix.tsv`), và chọn LR / số step trên dev chứ không trên bench.

## Lỗi pipeline đã sửa

| Lỗi | Sửa |
| --- | --- |
| `Unexpected key(s) ctc_output.1.*` khi nạp checkpoint | Gipformer 1.5 có head CTC; `run_e3.sh` có thêm `MODEL_ARGS` |
| FT phân kỳ (`grad_scale is too small`) ở khoảng step 55, cả LR 0.0045 lẫn 0.001; gradient tập trung ở `ctc_output.1.bias` | CTC loss của icefall dùng `reduction="sum"`, không `zero_infinity`; fp16 + batch nhỏ. Thêm `wrap_ckpt.py --drop-ctc`; `run_e3.sh` mặc định `--use-ctc 0` (decode sherpa-onnx chỉ dùng transducer). Log hai lần lỗi: `data/ft_trial/failed/` |
| `FileNotFoundError data/fbank/...lca` khi FT | `compute_fbank.py` ghi `storage_path` tương đối, nhưng `finetune_cs.py` chạy với cwd là thư mục icefall. Giờ ghi path tuyệt đối |
| `AssertionError: 0` khi export `--avg 1 --use-averaged-model 1` | `run_e3.sh` có thêm `USE_AVG_MODEL` (đặt 0 khi `AVG=1`) |
| lhotse Resample lỗi `No module named scipy` | `tools/setup_ft_env.sh` cài thêm `scipy` |
| Audio vss_cs bị xoá sau ASR | Tải lại 339 wav từ HF về `data/dl/VietSuperSpeech` (đúng path trong manifest) |

Kết quả: `results/v0_draft/a4_trial*`, checkpoint / ONNX: `exp/a4_trial/` (gitignore).
