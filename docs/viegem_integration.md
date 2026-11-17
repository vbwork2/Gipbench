# Tích hợp VieGem-ASR

Adapter: `src/gipbench/infer/adapters/viegem.py`; config: `configs/bench/asr_models/viegem_{greedy,lm}.yaml`.
Kết quả smoke test và số đo: `reports/general_asr_progress_vi.md`.

## 1. Thông tin đã kiểm (từ model card + code upstream, revision `95eae70b21fb0ef640e3ac4ea8cfa2380872ed20`)

| Mục | Giá trị |
| --- | --- |
| Kiến trúc | CTC 359M: audio encoder EmbeddingGemma 2 (12 conformer, 25 frame/s, top 4 lớp FT) + reader 6 block transformer (d=768, chunked attention) + head 21 031 token Gemma |
| Checkpoint | `viegem_asr.pt` (fp16, 311 MB): CHỈ phần đã train + head. Encoder gốc nạp từ `google/embeddinggemma-2` lúc khởi tạo `CTCModel` |
| Encoder gốc | `google/embeddinggemma-2` @ `f9972ac22e696a105c482202646979e33e4b5a1e`, 1.49 GB, Apache-2.0 trên HF, **không gated**; model card VieGem yêu cầu tuân Gemma Terms of Use |
| Thư viện | transformers có `embedding_gemma2` (config ghi 5.18.0.dev0). venv `gipbench-asr` (4.57.6) **không đủ**. Đã tạo `~/.venvs/gipbench-viegem`: Python 3.11, torch 2.15.0+cu132, transformers 5.23.0 |
| Audio | 16 kHz mono, mỗi lần ≤ 30 s (adapter cắt đoạn 30 s liên tiếp nếu dài hơn, ghi `n_chunked_utts`) |
| Output | chữ thường, không dấu câu, số viết bằng chữ |
| Giải mã | greedy CTC (mặc định); KenLM 4-gram đi kèm (`lm/viegem_lm.bin`, alpha 0.35, beta 1.5, beam 32), cần `pip install kenlm`; `LMDecoder` có `bias_words` (biasing ngữ cảnh) |
| License | CC-BY-NC-SA-4.0 (chỉ nghiên cứu phi thương mại) + Gemma Terms of Use |
| Dữ liệu train (model card) | ~5 800 h: PhoAudiobook, InfoRe, VietSpeech, viVoice, GigaSpeech2 (một phần), **VietSuperSpeech**, Bud500, VLSP2020, LSVSC, ViMD, FPT FOSD, FLEURS, **Common Voice 17**, LibriSpeech-100, **ViMedCSS (~23 h)**. "Đã lọc theo mọi câu test và dev" (theo câu, không theo người nói / video) |

Hệ quả cho so sánh: ViMedCSS (C0) và Common Voice (G0) là `known_possible` với VieGem. VIVOS và G-Group không có
trong danh sách train của VieGem (`not_listed`, không phải bằng chứng là sạch). G-Group là `known_possible` với
Gipformer (cùng tổ chức phát hành). Không bộ nào trong G0 / C0 là "độc lập" với **cả hai** model.

## 2. Không được giả định

- Không train VieGem bằng recipe RNN-T của Gipformer (icefall / k2). Nếu FT: dùng chính `CTCModel` upstream
  (`encoder_mode: frozen | top_k | lora`, `head_mode`), loss CTC, `Units.encode`. Repo upstream **không** có code train.
- KenLM của VieGem train trên transcript train của VieGem: được coi là `lm` (text không thuộc test của gipbench), nhưng
  có thể chứa câu của các bộ train trùng nguồn với test (VietSuperSpeech, CV17).

## 3. Cài đặt (đã chạy 16/11/2026)

```bash
uv venv -p 3.11 ~/.venvs/gipbench-viegem
uv pip install --python ~/.venvs/gipbench-viegem/bin/python "torch==2.15.0" --index-url https://download.pytorch.org/whl/cu132
uv pip install --python ~/.venvs/gipbench-viegem/bin/python "transformers==5.23.0" soundfile soxr numpy pyyaml sentencepiece huggingface-hub pyarrow
# tải theo revision cố định (HF_HUB_DISABLE_XET=1: bản Xet bị treo khi ghi vào /mnt/d)
export HF_HUB_DISABLE_XET=1 M=/mnt/d/code/ZAIC/gipbench/models
hf download 5CD-AI/VieGem-ASR viegem_asr.pt README.md asr/__init__.py asr/features.py asr/lm_decode.py asr/model.py asr/text.py \
   lm/viegem_lm.bin lm/viegem_lm.unigrams.txt --revision 95eae70b21fb0ef640e3ac4ea8cfa2380872ed20 --local-dir $M/viegem
hf download google/embeddinggemma-2 --revision f9972ac22e696a105c482202646979e33e4b5a1e --local-dir $M/embeddinggemma-2
```

## 4. Chạy

```bash
export PYTHONPATH=$PWD/src GIPBENCH_DATA=/mnt/d/code/ZAIC/gipbench/data GIPBENCH_MODELS=/mnt/d/code/ZAIC/gipbench/models
~/.venvs/gipbench-viegem/bin/python -m gipbench.bench.transcribe --model configs/bench/asr_models/viegem_greedy.yaml \
    --benchmark general_v1_draft --per-dataset 20 --out /tmp/viegem_smoke          # smoke test 120 câu
~/.venvs/gipbench-viegem/bin/python -m gipbench.bench.transcribe --model configs/bench/asr_models/viegem_greedy.yaml \
    --benchmark general_v1_draft --out results/general_asr/general_v1_draft/viegem_greedy
```

Adapter nạp package `asr/` của snapshot dưới tên riêng (`viegem_upstream_asr`, không sửa `sys.path`), nạp checkpoint
bằng `torch.load(weights_only=True)`, dừng nếu checkpoint thiếu khoá ngoài phần encoder gốc hoặc có khoá thừa.
