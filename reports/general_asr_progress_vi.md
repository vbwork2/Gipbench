# Báo cáo tiến độ: General ASR + code-switching, Gipformer 1.5 vs VieGem-ASR (16/11/2026)

Nhánh `feat/general-asr` (worktree `D:/code/ZAIC/gipbench-general-asr`), tách từ `refactor/pipelines` @ `8a5d20c`.
**Mọi benchmark ở đây là `draft`**: nhãn do bên phát hành gõ, dự án chưa duyệt lại. Số liệu dùng để chẩn đoán, chưa
phải kết quả cuối. Audit: [benchmark_audit_v1.md](benchmark_audit_v1.md). Giao thức:
[../docs/benchmark_protocol_v1.md](../docs/benchmark_protocol_v1.md).

## 1. Kết quả G0 (tiếng Việt phổ thông), hồ sơ `gipbench_strict_v1`, open_vocab

4 400 câu / 4.47 h, manifest_sha256 `781fe7a0…`. WER = space_token_v1 (token ≈ âm tiết). CI95 bootstrap theo cụm
(speaker; G-Group không có id người nói nên theo câu). Δ = VieGem − Gipformer, ghép cặp trên cùng id câu.

| Bộ | Câu | Gipformer 1.5 (mbs4, int8, CPU) | VieGem greedy (bf16, GPU) | Δ WER [CI95] | Trùng pretraining Gip / VieGem |
| --- | --- | --- | --- | --- | --- |
| VIVOS test | 760 | **4.51** [3.23, 6.21] | 6.68 [4.72, 9.23] | +2.18 [+1.46, +3.11] | unknown / not_listed |
| Common Voice vi 27 test | 1 593 | **7.47** [6.23, 8.85] | 9.50 [7.85, 11.25] | +2.03 [+1.44, +2.64] | unknown / known_possible |
| G-Group tech | 508 | **27.70** [26.41, 29.02] | 36.46 [34.96, 37.91] | +8.76 [+7.23, +10.23] | known_possible / not_listed |
| G-Group edu | 495 | **23.52** [22.21, 24.87] | 33.70 [32.33, 35.13] | +10.17 [+8.67, +11.60] | known_possible / not_listed |
| G-Group finance | 529 | **23.11** [21.75, 24.37] | 34.60 [33.18, 35.89] | +11.49 [+10.08, +12.94] | known_possible / not_listed |
| G-Group pubadmin | 515 | **15.28** [14.06, 16.56] | 29.02 [27.40, 30.74] | +13.74 [+12.21, +15.32] | known_possible / not_listed |
| **Macro theo bộ** (6 bộ, không CI) | | **16.93** | 24.99 | | |
| VieGem + KenLM | | | N/A (chưa chạy: bị chặn, mục 5) | | |
| Whisper large-v3 / Qwen3-ASR-0.6B | | N/A (chưa chạy) | | | |

CER (char_v1): Gipformer 9.44, VieGem 15.91 (toàn G0). Hồ sơ `legacy_v0` cho cùng thứ tự, lệch ≤ 0.06 điểm.

**Kiểm tính đúng của tích hợp:** VieGem trên VIVOS đo được **6.68**; model card VieGem công bố **6.63** (greedy). Gipformer
đo được VIVOS 4.51 / tech 27.70 / edu 23.52 / finance 23.11 / pubadmin 15.28; model card Gipformer công bố 4.25 / 27.49 /
23.32 / 22.34 / 14.82. Hai bộ số gần nhau, dù chuẩn hoá của model card không có code nên không thể khẳng định trùng khớp.

**Chẩn đoán khoảng cách trên G-Group:** VieGem xoá (D) ~10–11% token trên G-Group, Gipformer 2.6–4.4%. VieGem mất token
cuối ở 10–14.5% số câu (Gipformer 2–6%), khớp hạn chế model card tự nêu ("clip cắt sát mất từ cuối"). Ablation chẩn
đoán đệm 0.3 s im lặng hai đầu (`viegem_greedy_pad0.3_diag`) chỉ giảm WER G-Group 0.3–1.7 điểm (finance −1.72
[−2.56, −0.84]); VIVOS / CV không đổi. Biên clip giải thích một phần nhỏ, **không** giải thích khoảng cách 9–14 điểm.
Lưu ý: G-Group do chính tổ chức phát hành Gipformer công bố (known_possible), nên lợi thế của Gipformer ở đây có thể một
phần do trùng phân phối / dữ liệu.

## 2. Kết quả C0 (code-switching y khoa, ViMedCSS hard), open_vocab

658 câu / 1.29 h, catalog ViMedCSS gốc (không chứa span lấy từ bench).

| Hệ | WER | CS-WER (strict) | Term recall | Δ ghép cặp (VieGem − Gip) |
| --- | --- | --- | --- | --- |
| Gipformer 1.5 | 21.92 [19.96, 23.98] | 107.12 | 9.52 | |
| VieGem greedy | 21.05 [19.43, 22.78] | **84.54** | **17.33** | WER −0.86 [−1.73, +0.02]; CS-WER −22.57 [−28.93, −15.55] |

Không kết luận khả năng tổng quát từ bảng này: VieGem đã train trên ViMedCSS train (~23 h), và **308 / 658 câu hard đến
từ 201 video có cả trong ViMedCSS train** (ViMedCSS không tách theo video). Model card VieGem báo hard WER 21.08
(chuẩn hoá riêng); đo được 21.05.

## 3. Kết quả v0-draft (B1–B5, code-switching tiếng Việt – Anh), open_vocab

4 238 câu, hồ sơ `legacy_v0` (cùng định nghĩa với số lịch sử). Baseline Gipformer = **hyps E0 lịch sử** (lô 32, đúng
thứ tự manifest); VieGem chạy mới. Δ = VieGem − E0, ghép cặp trên cùng id, bootstrap theo `speaker_cluster` / speaker.
E1 (hotwords) là `oracle_bias` nên **không** đưa vào so sánh open-vocabulary này.

| Nhóm | Metric | Gipformer E0 | VieGem greedy | Δ [CI95] |
| --- | --- | --- | --- | --- |
| B1+B2 (G-Group CS + consensus) | CS-WER | **71.66** | 83.61 | +11.95 [+9.09, +14.16] |
| B1+B2 | WER | **22.78** | 32.81 | +10.03 [+9.25, +10.58] |
| B1+B2 | term_acc_macro | **27.98** | 14.80 | (chưa tính CI cặp) |
| B3 (ViMedCSS hard, 524 câu) | CS-WER | 106.81 | **84.47** | −22.34 [−28.56, −14.40] |
| B3 | WER | 23.67 | **21.67** | −2.01 [−3.13, −0.95] |
| B5 (tiếng Việt: G-Group pubadmin + VIVOS + consensus) | WER | **9.52** | 17.13 | +7.61 [+5.53, +8.80] |

Theo bộ: VieGem kém hơn trên mọi bộ G-Group (CS-WER +9 tới +24 điểm) và trên consensus (WER +9.20; CS-WER +0.31, CI
chứa 0). VieGem tốt hơn duy nhất trên ViMedCSS, bộ mà nó đã train trên split khác (cùng video, mục 2).

**Trả lời câu hỏi chính (với dữ liệu nháp hiện có):** trên tiếng Việt phổ thông (G0) và code-switching ngoài y khoa
(B1 + B2), Gipformer 1.5 tốt hơn VieGem-ASR (greedy, open_vocab) với CI không chứa 0. Lợi thế G-Group có thể một phần do
G-Group thuộc cùng tổ chức với Gipformer; trên VIVOS / CV (không gắn với Gipformer) khoảng cách nhỏ hơn (~2 điểm) nhưng
vẫn nghiêng về Gipformer. VieGem tốt hơn rõ trên code-switching y khoa (ViMedCSS), nhưng đó là dữ liệu nó đã train.
Chưa có bộ nào độc lập với cả hai model; chưa đo VieGem + LM; nhãn chưa duyệt.

## 4. Đã làm (code, test, commit)

| Commit | Nội dung |
| --- | --- |
| `0b069d3` | `core/normprofiles.py` (legacy_v0 / strict_v1 / lenient_diagnostic_v1), `core/evalstats.py` (kiểm hyps, CER, bootstrap cụm + cặp), `core/decontam.py`, `bench/registry.py` + `configs/bench/{datasets.yaml,benchmarks/}`, `bench/evaluate.py` (final mode), 4 file test |
| `b849434`, `2e7cf50` | `infer/adapters/` (base, sherpa_transducer, viegem, hf_multilingual), `bench/transcribe.py`, `dataset/release/build_general.py`, `dataset/release/leak_audit.py`, `configs/bench/asr_models/`, docs protocol / metrics / registry |
| `912532b` | `bench/compare.py`, `core/langspans.py`, sửa lỗi evaluate với `context_vocab: null`, docs VieGem / ma trận FT, audit |
| `486ab41` | VieGem `pad_sec` (ablation chẩn đoán) |

Không sửa: `bench/score.py`, `bench/leaderboard.py`, `infer/decode.py`, `core/textnorm.py`, `core/scoring.py`,
`core/leak.py`, dữ liệu / kết quả cũ. Kết quả mới ghi vào `results/general_asr/` (ngoài git).

**Test** (`bash tests/run_tests.sh`, WSL, CPU): 13 / 13 nhóm ĐẠT (7 nhóm cũ: boundaries, selftest, span_metrics,
dev_tools, converters, tts, hard_mining; 6 nhóm mới: scoring_v1 30 ca, normalization 15 ví dụ + bất biến, leakage 13 ca,
langspans 8 ca, adapters 19 ca, registry 12 ca). `test_boundaries`: 0 vi phạm ranh giới.

**Tương thích ngược:** `evaluate --benchmark v0_draft` với `legacy_v0` cho đúng số lịch sử (E0 CS-WER 71.657…,
N-WER 13.128…, B5 9.518…; E1 60.213…; E3 92.439…). Adapter Gipformer khớp từng byte với `decode.py` trên cùng đầu vào
(140 / 140 câu).

## 5. Bị chặn / chưa làm

| Việc | Lý do | Cách gỡ |
| --- | --- | --- |
| VieGem + KenLM (`viegem_lm`) | `pip install kenlm` cần build C++; WSL không có gcc / g++ / make | `sudo apt install build-essential cmake` (người dùng chạy), rồi `uv pip install --python ~/.venvs/gipbench-viegem/bin/python kenlm` |
| FLEURS vi, ViMedCSS test (1 614), ViMD, GigaSpeech2, Bud500 | chưa tải; GigaSpeech2 / Bud500 gated | xem `docs/dataset_registry.md` §2 |
| Whisper / Qwen3 trên G0 / C0 | không bắt buộc; adapter đã có | `transcribe --model configs/bench/asr_models/whisper_large_v3.yaml` (venv gipbench-asr) |
| Z0 zero-shot sau FT | chưa có audio thu mới, chưa có reserved_terms | `configs/bench/benchmarks/zero_shot_v1.yaml` (planned) |
| Fine-tune (Phase E) | cổng chưa đạt: benchmark draft, rò rỉ ViMedCSS / VIVOS, VieGem không FT được trên 4 GB | `docs/experiment_matrix.md`: M0 đã có, M1–M6 chưa chạy |
| Duyệt nhãn B1 / B3, khoá bench v1 | việc của người dùng (≈ 230 + 90 phút) | sau đó đặt `evidence_level: human_reviewed` + `manifest_sha256` |

## 6. Chi phí đã dùng

- Tải: VieGem 381 MB (`viegem_asr.pt` + LM + code), embeddinggemma-2 1.53 GB, venv `gipbench-viegem` (torch cu132
  lấy từ cache uv). Không tải dataset nào.
- Gipformer: CPU 16 luồng, lô 1, RTF 0.107 (G0) / 0.091 (C0). VieGem: RTX 3050 Laptop 4 GB, bf16, lô 8, RTF 0.020 (G0) /
  0.010 (C0), VRAM đỉnh 2 950 MiB, nạp model 28 s. Phần cứng khác nhau: không so tốc độ trực tiếp.

## 7. Lệnh tái lập

```bash
cd /mnt/d/code/ZAIC/gipbench-general-asr
export PYTHONPATH=$PWD/src GIPBENCH_DATA=/mnt/d/code/ZAIC/gipbench/data GIPBENCH_MODELS=/mnt/d/code/ZAIC/gipbench/models
R=/mnt/d/code/ZAIC/gipbench/results/general_asr
python -m gipbench.dataset.release.build_general --which general && python -m gipbench.dataset.release.build_general --which code_switch
~/.venvs/gipbench/bin/python -m gipbench.bench.transcribe --model configs/bench/asr_models/gip15_mbs4_open.yaml --benchmark general_v1_draft --out $R/general_v1_draft/gip15_mbs4_open
~/.venvs/gipbench-viegem/bin/python -m gipbench.bench.transcribe --model configs/bench/asr_models/viegem_greedy.yaml --benchmark general_v1_draft --out $R/general_v1_draft/viegem_greedy
~/.venvs/gipbench/bin/python -m gipbench.bench.evaluate --benchmark general_v1_draft --hyps $R/general_v1_draft/viegem_greedy/hyps.jsonl \
    --baseline-hyps $R/general_v1_draft/gip15_mbs4_open/hyps.jsonl
~/.venvs/gipbench/bin/python -m gipbench.bench.compare --benchmark general_v1_draft --runs $R/general_v1_draft/gip15_mbs4_open \
    $R/general_v1_draft/viegem_greedy --expect viegem_lm whisper_large_v3 qwen3_asr_0.6b
bash tests/run_tests.sh /tmp/gab
```

## 8. Cập nhật 17/11/2026: bản share, fetch, duyệt thuật ngữ

- `fetch` + `package` + datasheet (`404dc1c`). Kiểm thật: tải lại từ nguồn gốc G0 4 400 / 4 400 câu và C0 658 / 658 câu,
  trùng sha256 từng byte (VIVOS tải từ HF; G-Group từ parquet đã khoá; Common Voice từ bản tải tay; ViMedCSS tải từ HF
  và resample cũng trùng byte).
- Sự cố: lượt tải đầu làm đầy ổ D: (pagefile 19.6 GB + ổ ảo WSL phình) và làm WSL sập; người dùng đã dọn ổ. Lượt sau
  có monitor canh dung lượng.
- Người dùng duyệt 430 thuật ngữ B1 qua trang `term_review` (`7c35771`). Chạy lại converter G-Group vào
  `data/raw/ggroup_v1`: B1 còn 1 261 câu code-switching (trước 1 330; câu chỉ có từ người dùng chọn `vi` rời track).
  `build_bench --accept trusted-text` nhận 1 248 câu; còn chờ 13 câu: `ai` (tuỳ câu, 9 câu, phiếu duyệt từng câu) và 4
  thuật ngữ mới lộ ra sau khi gắn lại tag (`alt`, `chat`, `hip`, `fr`, mỗi từ 1 câu).

## 9. Cập nhật 17/11/2026 (chiều): track C1 G-Group code-switching

- Người dùng chốt 4 thuật ngữ cuối (`alt` vi, `chat` en, `fr` vi, `hip` en) và duyệt 9 câu `ai` (4 ok, 5 fix; bản sao
  `reports/term_review_ai_2026-11-17.csv`). Chạy lại converter: B1 còn 1 260 câu; `build_bench` nhận cả 1 260, không câu
  nào chờ.
- Benchmark mới `cs_ggroup_v1_draft` (track C1, 1 260 câu, 1.31 h, manifest_sha256 `764e2295…`), dựng bằng
  `build_general --which cs_ggroup`. Cùng audio với G-Group trong G0; chấm không catalog. `fetch` tái dựng 1 260 / 1 260
  câu trùng sha256.
- Kết quả (open_vocab, strict_v1): Gipformer WER 24.66 / CS-WER 70.25 / term recall 39.57; VieGem 34.38 / 82.39 / 23.52.
  Δ ghép cặp VieGem − Gipformer: CS-WER +12.13 [9.24, 14.91]. Macro CS-WER theo bộ: 66.40 / 80.34.
- Hyps Gipformer trên C1 trùng hyps G0 1 260 / 1 260 (batch 1). VieGem (batch 8, bf16) trùng 1 177 / 1 260 vì ghép
  batch khác; WER lệch 0.03, CS-WER 0.24 điểm. Đã ghi vào datasheet (giới hạn 9).
- Sửa: `compare` ghi nhãn macro theo đúng primary metric (trước luôn ghi "WER"); `test_converters` / `eval_autotag` không
  còn phụ thuộc `term_decisions.tsv` thật (hỏng từ khi người dùng chốt `gov` = vi); `package` đóng kèm
  `term_decisions.tsv`.
- Gói thử `gipbench-bench-v1.0-draft` (G0 + C0 + C1) ở `data/share_test/pkg3`: SHA256SUMS khớp, không có audio / dòng
  test_only. Cả ba benchmark vẫn `evidence_level: draft`, chưa khoá `manifest_sha256`.

## 10. Khoá và đóng gói v1.0 (17/11/2026, theo yêu cầu người dùng)

- Benchmark khoá mới: `general_v1` (`e908c75a…`), `code_switch_v1` (`744d4773…` + catalog `a21c590e…`), `cs_ggroup_v1`
  (`e3c3cc1b…`); `evidence_level: locked`. Bản `*_v1_draft` giữ nguyên. Cùng câu / nhãn với draft.
- Sửa trước khi khoá: `package` từng tự thêm `audio_derivation` nên manifest share có hash khác bản khoá (người nhận sẽ bị
  `evaluate` từ chối); nay `build_general` ghi `audio_derivation` vào manifest, `package` kiểm hash share = bản khoá (và
  catalog = `catalog_sha256`) trước khi ghi. `evaluate` báo lỗi khi catalog khác `catalog_sha256`.
- `evaluate --mode final` trên cả ba bản khoá: đạt, số trùng bản draft (Gipformer G0 WER 15.21, C0 CS-WER 107.12, C1
  70.25; VieGem 22.27 / 84.54 / 82.39). Chạy thử phía người nhận (yaml + manifest trong gói) cho cùng số.
- Gói `data/share/gipbench-bench-v1.0` (4.3 MB, 6 318 dòng manifest, không audio, không dòng test_only); SHA256SUMS khớp.
