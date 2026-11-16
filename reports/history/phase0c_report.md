# Phase 0c: SEA, lọc / gán lại `vss_cs`, pilot 1 000 câu

**Trạng thái: XONG, đang DỪNG chờ anh/chị duyệt pilot** (mục 3d). Em chưa chạy thêm câu nào ngoài 1 000 câu pilot.

Tất cả nằm trong commit chứa báo cáo này. Hai bộ test (`tools/test_converters.py`, `tools/selftest.sh`) đều đạt, kể cả test mới cho gán lại span và lọc rẻ.

## 1. Bảng task

| Mục | Trạng thái | File chính |
| --- | --- | --- |
| 0. Bỏ SEA | Xong | `README.md`, `SCHEMA.md`, `docs/research/01, 02, 04`, `configs/train_mix.tsv`, `reports/decisions.md` (mục 11), docstring `build_bench_v0.py`, comment `ft/run_e3.sh` / `requirements-asr.txt` |
| 1. `--relabel-spans` | Xong (mặc định tắt) | `lib/agreement.py` (`relabel_spans`, `span_segments`), `tools/agreement_filter.py`, `reports/vss_relabel_listen.csv` |
| 2. `--max-en-frac 0.5` trước ASR | Xong | `tools/vss_cs_prefilter.py` |
| 3a. Lọc rẻ trước ASR | Xong, chạy trên cả 31 949 câu | `tools/vss_cs_prefilter.py` |
| 3b. Chạy tầng | Xong | `tools/vss_cs_pilot.sh` |
| 3c. Chunk, resume, xoá audio tạm, dừng khi D: < 3 GB | Xong | `tools/asr_hyps.py` (`--chunk-size`, `--hf-repo`, `--tmp-dir`, `--min-free-gb`, `--ids`, log `.runs.jsonl`) |
| 3d. Pilot 1 000 câu | Xong, **dừng chờ duyệt** | `data/raw/vss_full/vss_cs_pilot_*` (không commit) |
| 4. `vi_abbrev` / `vi_loanwords` | Giữ nguyên, chờ duyệt | Câu hỏi ở mục 6 |

## 2. Mục 0: bỏ SEA

Các tài liệu giờ ghi rõ:

- **B1** = G-Group + dữ liệu tự thu.
- **Nguồn CS thật cho train** = ViMedCSS train + `vss_cs` + dữ liệu tự có.
- **SEA** = tùy chọn, không dùng.

Đã bỏ SEA khỏi các ví dụ lệnh mặc định: runbook bước 0, lệnh `build_bench_v0`, `huggingface-cli login`. Trong `configs/train_mix.tsv`, nhóm `cs_real` không còn `sea_speech`.

`conv_sea.py` và test của nó em giữ nguyên, để "tùy chọn" vẫn chạy được khi cần.

Có một chỗ cần anh/chị xem. Tài liệu ghi `vss_cs` là nguồn CS thật (vì âm thanh là CS thật), nhưng `train_mix.tsv` vẫn để `vss_cs` ở nhóm `synth_pseudo` (vì nhãn là nhãn giả). Đây là quyết định của Phase 0b và có test giữ nó. Spec đã ghi chú điểm này. Có chuyển `vss_cs` sang `cs_real` không thì anh/chị quyết (mục 7).

## 3. Mục 1: `--relabel-spans`

**Luật** (`lib/agreement.relabel_spans`). Chỉ xét những câu không qua lọc, và trong câu đó chỉ xét **span chưa được xác nhận**. Span được thay khi đủ cả bốn điều kiện:

1. có ít nhất 2 file hyps và câu có hyp ở mọi file;
2. đoạn hyp ứng với span **giống hệt nhau ở mọi ASR** sau chuẩn hoá (strict, catalog), không rỗng, và khác nhãn cũ;
3. **WER phần tiếng Việt** của nhãn so với **từng** ASR ≤ `--relabel-max-vi-wer` (0.10);
4. sau khi thay, câu đạt `--min-span-frac`.

Cách tính:

- Căn chỉnh theo `lib/align`, cùng quy ước với `score.py`.
- Token hyp thuộc span khi được căn (C/S) với token của span, hoặc là token chèn kề span (ưu tiên bên trái).
- Lỗi tiếng Việt = S/D trên token tiếng Việt + chèn không quy được về span nào.

Không gán lại cả câu: phần tiếng Việt và các span khác giữ nguyên chữ gốc.

Câu gán lại được ghi:

- `relabeled: true`;
- `text_orig` = `text_tagged` gốc;
- `text` / `text_tagged` mới;
- `relabel: {changed: [[span, cũ, mới]], vi_wer}`.

Sau đó câu được **kiểm tra rò rỉ lại**: text trùng bench, `reserved_b3`, speaker, source. Test có ca một span gán lại thành `kubernetes` (reserved_b3) và câu đó bị loại.

**Kết quả trên pilot:** gán lại được 104 câu, 120 span. Phân bố kết quả của 721 câu chưa qua lọc:

| Kết quả | Câu |
| --- | --- |
| WER tiếng Việt > 0.10 | 382 |
| Hai ASR không cho cùng chữ | 189 |
| **Gán lại** | **104** |
| Gán lại xong vẫn còn span chưa xác nhận | 46 |

Ví dụ tốt:

- `fimment center` → fulfillment center
- `ivlik school` → ivy league school
- `postca` → podcast
- `troma` → trauma
- `concep` → concept
- `cremsen` → crimson

Lỗi em thấy khi đọc text (chưa nghe audio):

- **4/120 span hút luôn chữ Việt bên cạnh:** `expe` → "expectation cái", `st id` → "sti đi", `rocket` → "rap việt", `vietnam innovator` → "việt nam innovator". Nguyên nhân là token chèn kề span được quy về span.
- **Vài span đổi hẳn sang một từ khác:** `bre` → ruby, `united` → initiative, `mood disort` → mood, `free` → fruity. Cả hai ASR cùng nghe như vậy, nhưng chưa chắc đúng.

**Đề xuất (chưa làm, cần anh/chị quyết):** thêm luật "span mới không được chứa token có dấu tiếng Việt". Luật này loại được 4/120 trường hợp kể trên mà không làm mất ca tốt nào em đã xem.

**Phiếu nghe:** `reports/vss_relabel_listen.csv` có 30 câu gán lại, chọn ngẫu nhiên với seed 0. Các cột:

- `audio_url`: link HF, mở trên trình duyệt là nghe được;
- `span_cu_moi`, `text_orig`, `text_tagged_moi`;
- hyp của hai ASR, `vi_wer`;
- `decision` (ok / sai) và `note` để anh/chị điền.

## 4. Mục 2 và 3a: lọc rẻ trước ASR

Chạy trên toàn bộ `vss_cs_candidates`. Chặn rò rỉ dùng **bench tạm** (`data/raw/_proxy`: 3 633 câu B1 + B3, 260 thuật ngữ `reserved_b3`), vì bench v0 thật chưa dựng.

| Bước | Câu còn lại | Giờ còn lại |
| --- | --- | --- |
| `vss_cs_candidates` | 31 949 | 117.32 |
| Độ dài 1–20 s | 31 949 | 117.32 |
| `--max-en-frac 0.5` | 20 228 | 74.41 |
| Rò rỉ B1–B3 + reserved_b3 | 20 228 | 74.41 |
| `--term-cap 50` (seed 0) | **15 325** | **56.39** |

- **Độ dài:** VSS đã cắt sẵn trong khoảng 1–20 s, nên bước này không loại câu nào.
- **Rò rỉ:** không loại thêm, vì converter đã chặn ở Phase 0b.
- **`--max-en-frac`:** tỉ lệ token tiếng Anh có hai đỉnh rõ. 18 145 câu dưới 0.1, nhưng 11 540 câu trên 0.7; đó là các câu gần như toàn tiếng Anh, nhãn vô nghĩa.
- **Trần thuật ngữ:** có 13 229 thuật ngữ khác nhau, 69 thuật ngữ vượt trần 50 (`ok` 1 744 câu, `marketing` 365, `podcast` 306…).

## 5. Mục 3b–3d: chạy tầng và pilot 1 000 câu

**Cách chạy:**

1. Qwen3-ASR-0.6B chạy trên mọi câu.
2. `agreement_filter --write-unconfirmed-ids` lấy ra các câu còn span Qwen chưa xác nhận.
3. faster-whisper large-v3 int8 chỉ chạy trên các câu đó.

Khi bật `--relabel-spans`, faster-whisper vẫn chạy đúng tập này. Câu mà Qwen đã xác nhận hết span thì không còn span nào để gán lại, nên không cần faster-whisper.

**Chunk, resume, dung lượng:**

- Mỗi chunk 200 câu. Audio thiếu được tải từ `thanhnew2001/VietSuperSpeech` vào `data/tmp/vss_audio` (NTFS trên D:, xoá là giải phóng chỗ thật) và bị xoá sau mỗi chunk.
- Trước mỗi chunk, script kiểm tra dung lượng D:; còn dưới 3 GB thì dừng với exit 3.
- Câu đã có trong hyps được bỏ qua, nên chạy lại cùng lệnh là chạy tiếp từ chỗ dừng.
- `HF_HOME` giữ ở `~/.cache/huggingface` trong WSL: model đã có sẵn ở đó, và file ổ đĩa ảo của WSL nằm trên D:.
- Trong suốt pilot, D: luôn còn 12.1 GB.

**Mẫu pilot:** 1 000 câu chọn ngẫu nhiên (seed 0) từ 15 325 câu sau lọc rẻ, tổng 3.66 h.

| Tầng | Câu | Audio | GPU (ASR) | Nạp model | Tải audio | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-ASR-0.6B, bf16 | 1 000 | 3.66 h | 0.84 h | 17 s | 3.9 phút | 0.23 |
| faster-whisper large-v3, int8 | 800 | 2.93 h | 0.82 h | 15 s | 2.4 phút | 0.28 |
| **Tổng** | | | **1.67 GPU-h** | | | |

Tổng thời gian chạy là 1 h 47 phút (12:07–13:53). Nhờ chạy theo tầng, faster-whisper không phải chạy 200 câu (0.73 h audio), tiết kiệm khoảng 0.2 GPU-h, tức 11%.

**Tỉ lệ giữ:**

| | Câu | Giờ | % câu | % giờ |
| --- | --- | --- | --- | --- |
| Chỉ Qwen (sau tầng 1) | 200 | 0.73 | 20.0% | 19.9% |
| **Chỉ lọc** (hai ASR, `--min-agree 1`) | **279** | **1.02** | **27.9%** | 27.8% |
| **Có gán lại** | **383** | **1.40** | **38.3%** | 38.2% |

Mẫu 1 000 câu cho khoảng tin cậy 95% xấp xỉ ±3 điểm: chỉ lọc khoảng 25–31%, có gán lại khoảng 35–41%.

**Ước tính:** khoảng 0.46 GPU-h cho mỗi giờ audio đầu vào; trung bình 13.25 s mỗi câu.

| Nếu chạy | Audio | GPU-h | Giờ sạch (chỉ lọc) | Giờ sạch (có gán lại) |
| --- | --- | --- | --- | --- |
| ~4 900 câu | 18 h | ~8 | ~5 | ~7 |
| **~10 000 câu** (9 000 câu thêm) | 36.8 h | ~17 (thêm ~15) | **~10** | ~14 |
| Cả 15 325 câu | 56.4 h | ~26 | ~16 | ~22 |

Tải audio thêm khoảng 0.4 s mỗi câu, tức khoảng 1 h cho 9 000 câu. Mỗi chunk tốn tối đa khoảng 80 MB đĩa tạm.

### Phát hiện từ pilot

1. **Trần thuật ngữ ưu tiên thuật ngữ hiếm, nhưng thuật ngữ hiếm phần lớn là span nhận dạng sai.**

   | Câu | Số câu | Chỉ lọc | Có gán lại |
   | --- | --- | --- | --- |
   | Mọi thuật ngữ chỉ có trong 1 câu (trong cả 31 949 câu) | 235 | **7.7%** | 20.4% |
   | Thuật ngữ phổ biến nhất có 2–9 câu | 249 | 25.3% | 39.4% |
   | Có thuật ngữ ≥ 10 câu | 516 | 38.4% | 45.9% |

   **Đề xuất:** trước ASR, bỏ các câu mà mọi thuật ngữ chỉ xuất hiện đúng 1 lần.
   - Trên pilot, cách này bớt 23.5% số câu phải chạy ASR, nhưng chỉ mất 18/279 câu giữ được (6.5%).
   - Tỉ lệ giữ khi chỉ lọc tăng từ 27.9% lên khoảng 34%.
   - Để có cùng số giờ sạch thì cần ít hơn khoảng 18% GPU-h.
   - Cách khác: không ưu tiên thuật ngữ hiếm nữa mà lấy ngẫu nhiên dưới trần.
2. **Phần tiếng Việt của câu "chỉ lọc" chưa được kiểm.** Bộ lọc chỉ xét span tiếng Anh. Ở 279 câu giữ lại, WER tiếng Việt so với ASR khớp hơn có trung vị 6.4%; 81 câu vượt 0.10 và 15 câu vượt 0.20. Câu gán lại thì đều ≤ 0.10, vì đó là điều kiện để được gán lại.
   - **Đề xuất:** thêm ngưỡng WER tiếng Việt, ví dụ 0.20, cho cả nhánh chỉ lọc. Ở ngưỡng 0.20, pilot mất 15/279 câu.
3. **WER tiếng Việt là nút thắt của việc gán lại:** 382/721 câu bị chặn vì vượt 0.10. Nếu nới ngưỡng lên 0.15 thì thêm tối đa 122 câu được xét, nhưng chất lượng phải nghe mới biết.

**Mục tiêu 5–10 h sạch:** chỉ lọc cần khoảng 5 000–10 000 câu, tương ứng 8–17 GPU-h. Có gán lại thì cần ít câu hơn khoảng 27%. Nếu áp dụng đề xuất 1, cần ít hơn thêm khoảng 18% nữa.

## 6. Mục 4: `vi_abbrev`, `vi_loanwords`, vẫn chờ duyệt

Hai file `configs/vi_abbrev.txt` và `configs/vi_loanwords.txt` (`same_surface`) giữ nguyên. Các câu hỏi mục 2.2 của `phase0b_report.md`, chép nguyên văn:

> **Cần anh/chị quyết:**
>
> 1. Có thêm `vn` không, và nếu thêm thì dựa vào nguồn nào?
> 2. Có bật lại `tp` không?
> 3. Có thêm sàn UPCOM không (thêm khoảng 800 mã)?
> 4. Nhóm cơ quan còn mỏng: chưa có `ubnd`, `bhxh`, `cccd`, `cmnd`. Nếu anh/chị chỉ ra một văn bản chính thức, em sẽ lấy theo văn bản đó.

Cùng mục 2.2 còn một dòng:

> - Cần anh/chị xem riêng `cdn`: đây là mã chứng khoán, nhưng trong bộ tech có lẽ là "CDN" đọc kiểu Anh.

## 7. Quyết định em tự đưa ra, cần anh/chị xem lại

1. Trong `train_mix.tsv`, `vss_cs` vẫn ở nhóm `synth_pseudo`, dù tài liệu gọi nó là nguồn CS thật (mục 2).
2. Câu được giữ khi tỉ lệ span xác nhận ≥ `--min-span-frac` (1.0). Một câu chỉ gán lại được một phần span thì bị loại (46 câu).
3. Span mới được ghi ở dạng chuẩn hoá: chữ thường, số đọc thành chữ, biến thể catalog quy về dạng chuẩn. Cách này khớp với `text` của VSS vốn đã viết thường.
4. `text_orig` giữ `text_tagged` gốc (còn tag), không phải bản text không tag.
5. Chặn rò rỉ (lọc rẻ và sau gán lại) đang dùng **bench tạm**. **Khi có bench v0 thật thì phải chạy lại** `vss_cs_prefilter.py` và `agreement_filter.py --relabel-spans --bench data/bench_v0.jsonl`.
6. Phiếu nghe chỉ có link HF, không tải audio về máy.
7. Audio tạm đặt trên D: qua `/mnt/d`, không đặt trong ổ đĩa ảo WSL, vì file ổ đĩa ảo không co lại sau khi xoá.

## 8. Lệnh chạy lại

```bash
# 3a: lọc rẻ + mẫu pilot (venv chính)
python tools/vss_cs_prefilter.py --manifest data/raw/vss_full/vss_cs_candidates.jsonl \
    --catalog data/raw/_proxy/catalog_proxy.tsv --bench data/raw/_proxy/bench_proxy.jsonl \
    --out data/raw/vss_full/vss_cs_pre.jsonl --sample 1000 --sample-out data/raw/vss_full/vss_cs_pilot.jsonl --seed 0
# 3b–3d: tầng ASR + lọc + gán lại + phiếu nghe (WSL, cần ~/.venvs/gipbench và ~/.venvs/gipbench-asr)
M=data/raw/vss_full/vss_cs_pilot.jsonl bash tools/vss_cs_pilot.sh
# chạy tiếp sau khi duyệt: lấy mẫu lớn hơn (seed khác hoặc --sample N), rồi chạy lại script với M mới;
# hyps cũ được dùng lại (resume theo id)
```

Kết quả nằm ở (không commit, thư mục `data/` và `results/` bị bỏ qua):

- `data/raw/vss_full/vss_cs_pilot_agreed.jsonl` (chỉ lọc);
- `data/raw/vss_full/vss_cs_pilot_agreed_relabel.jsonl` (có gán lại);
- báo cáo `.report.md` của từng file;
- `results/asr_hyps/{qwen3_asr_0.6b,fw_large_v3_int8}/vss_cs_pilot.jsonl` và `.runs.jsonl`.

## 9. Chờ anh/chị

1. **Duyệt pilot và quyết bước tiếp theo:** chạy tới khoảng bao nhiêu câu, chỉ lọc hay có gán lại.
2. **Nghe `reports/vss_relabel_listen.csv`** (30 câu) và điền `decision`.
3. Ba đề xuất ở mục 3 và 5:
   - (a) cấm span mới chứa chữ có dấu tiếng Việt;
   - (b) bỏ câu mà mọi thuật ngữ chỉ xuất hiện 1 lần, trước ASR;
   - (c) thêm ngưỡng WER tiếng Việt cho nhánh chỉ lọc.
4. Có chuyển `vss_cs` sang nhóm `cs_real` trong `train_mix.tsv` không.
5. `vi_abbrev`, `vi_loanwords` và các câu hỏi ở mục 6.
