# Phase 0b: các việc nhỏ trước Phase 1

**Trạng thái: XONG, trừ SEA (chờ quyền truy cập).**

Lúc 01:30 ngày 15/11, ổ C: đầy và WSL dừng giữa lúc đang chạy ASR. Sau đó em đã:

- chuyển swap của WSL sang D: (`.wslconfig`);
- chuyển pagefile của Windows sang D: (trên C: chỉ còn 2 GB cố định);
- xoá các `swap.vhdx` cũ.

Khởi động lại máy xong, C: trống 28 GB. Em đã chạy lần lượt từng model ASR trên 20 câu, chạy bộ lọc đồng thuận, kiểm tra resample ViMedCSS và chạy lại hai bộ test. Kết quả ở 2.4 và 2.5.

Commit của phase này:

| Commit | Nội dung |
| --- | --- |
| `7d647ce` | T1 |
| `95d8060` | T2 |
| `ac53aa8` | T3 |
| `c117cc3` | T4 (chưa có phần chạy ASR). Ghi chú: commit `ac53aa8` (T3) ghi nhầm "vi_spelling: 307 mục"; số đúng là 306 |
| (commit chứa báo cáo này) | T5 + báo cáo Phase 0b + `converter_check_cv.md` |

## 1. Bảng task

| Mục | Trạng thái | File chính | Đã kiểm thử | Chưa chạy trên dữ liệu / model thật |
| --- | --- | --- | --- | --- |
| 1. Quy ước theo cách đọc | Xong | `SCHEMA.md`, `docs/huong_dan_duyet_b1.md`, `common.py` (cột `reading_note`, cờ `abbr_reading`, `vi_word`), `build_bench_v0.py` | test: `reading_note` đi vào bench; cờ `abbr_reading` | Chưa có người duyệt dùng phiếu mới |
| 2. `configs/vi_abbrev.txt` | Xong, **chờ duyệt** | `tools/build_wordlists.py`, `configs/vi_abbrev.txt`, `auto_tag(vi_words=)`, `eval_autotag --vi-words` | test: luật từng nhóm và luật loại trên nguồn giả; catalog thắng danh sách | Danh sách chưa được người duyệt |
| 3. ViMedCSS hai nhãn, `vi_loanwords.txt`, `--label-field` | Xong, danh sách **chờ duyệt** | `conv_vimedcss.py`, `configs/vi_loanwords.txt`, `score.py` | test: hai nhãn, cờ `loanword_untagged`, dựng danh sách, `--label-field` cho ra số khác nhau. Chạy thật `--limit 20` | Chưa chấm B3 thật (chưa có hyps) |
| 4. VSS `vss_cs`, bộ lọc đồng thuận, cấu hình trộn | Xong | `conv_local.py`, `lib/leak.py`, `lib/agreement.py`, `tools/agreement_filter.py`, `tools/asr_hyps.py`, `configs/train_mix.tsv` | test: chặn rò rỉ, `agree()`, CLI lọc, cấu hình trộn. Chạy thật trên toàn bộ text VSS; ASR (faster-whisper large-v3 + Qwen3-ASR-0.6B) và bộ lọc chạy thật trên 20 câu | Chưa chạy ASR trên toàn bộ 31 949 câu; Qwen3-ASR mới dùng bản 0.6B |
| 5. Sửa tài liệu, resample 16 kHz | Xong | `common.save_audio(target_sr)`, `conv_vimedcss --target-sr`, spec, runbook, `requirements.txt` (+ `soxr`) | test: audio 44.1 kHz stereo -> WAV 16 kHz mono, duration đúng. Chạy thật `conv_vimedcss --limit 20`: 80/80 WAV 16 kHz mono PCM16. Hai bộ test đạt trên toàn bộ working tree | — |
| 6. SEA | Chờ anh/chị | — | — | Vẫn `gated: manual`. README đọc công khai được (en-vi: 10 306 clip, 45.3 h, license `other`); file dữ liệu cần quyền |
| Thêm: Common Voice `--limit 20` (khép T0.1) | Xong | `reports/converter_check_cv.md` (chưa commit) | `conv_local commonvoice --limit 20` | — |

## 2. Kết quả từng mục

### 2.1 Quy ước gán nhãn

Quy ước gắn `<en>` theo cách đọc được ghi vào `SCHEMA.md`, đúng bốn ý anh/chị đưa ra. Hướng dẫn duyệt nằm ở `docs/huong_dan_duyet_b1.md`.

- Phiếu duyệt có thêm cột `reading_note`.
- Có thêm hai cờ cho người duyệt:
  - `abbr_reading:<từ>`: viết tắt 2–4 chữ do heuristic đoán, chưa biết đọc kiểu Anh hay kiểu Việt;
  - `vi_word:<từ>`: từ chữ Latin không bị gắn `<en>` vì nằm trong một trong hai danh sách.
- Câu B5 có cờ `vi_word` không còn được nhận thẳng mà phải qua người duyệt.

### 2.2 `configs/vi_abbrev.txt`

Danh sách có 614 mục được dùng và 91 mục bị loại tự động (ghi lại thành dòng `#-` kèm lý do). Mỗi dòng ghi nguồn.

| Nhóm | Số mục | Nguồn và luật |
| --- | --- | --- |
| Mã chứng khoán | 579 | Mã cổ phiếu sàn HSX + HNX, chỉ gồm chữ cái (API symbols của Vietcap, lấy ngày 15/11) |
| Ngân hàng | 31 (21 mục trùng với mã) | vi.wikipedia "Danh sách ngân hàng tại Việt Nam", cột tên giao dịch. Chỉ lấy dạng chữ cái đầu (tối đa 1 nguyên âm, không chứa "bank"); brand Latin như `vietcombank` là EN theo quy ước; bỏ ngân hàng nước ngoài |
| Tỉnh thành | 7 | Chữ cái đầu tên 34 tỉnh thành (vi.wikipedia). Bỏ tên có chữ Đ, vì chữ Đ đã có dấu |
| Viết tắt | 21 | vi.wiktionary, Thể loại "Từ / Chữ viết tắt tiếng Việt": chỉ lấy mục có dạng đầy đủ toàn tiếng Việt |
| **Bị loại** | 91 | Trùng catalog (`api`) hoặc trùng google-10000 (`gas`, `bid`, `vic`, `tp`, `pt`, `cm`…) |

Kết quả trên tập vàng **chính** (100 câu có từ Anh):

| | P | P bỏ qua viết tắt VN | R |
| --- | --- | --- | --- |
| Sau Phase 0 | 0.944 | 0.982 | 0.977 |
| + `vi_abbrev` | **0.955** | 0.982 | 0.977 |

Recall không giảm. Em không thêm từ nào dựa trên tập vàng. Các FP viết tắt Việt còn lại trên tập chính:

- `vn` (4 lần): không có trong nguồn nào;
- `tp`: bị loại vì trùng google-10000;
- `qns`: mã UPCOM, nằm ngoài HOSE/HNX;
- `nhs`, `bja`, `tb`: không có nguồn.

Ảnh hưởng trên toàn bộ G-Group (2047 câu):

- 64 span bị bỏ tag;
- 28 câu (23 finance, 5 edu) không còn span nào, nên trở thành ứng viên B5. Các câu này mang cờ `vi_word`, nên phải qua người duyệt.
- Cần anh/chị xem riêng `cdn`: đây là mã chứng khoán, nhưng trong bộ tech có lẽ là "CDN" đọc kiểu Anh.

**Cần anh/chị quyết:**

1. Có thêm `vn` không, và nếu thêm thì dựa vào nguồn nào?
2. Có bật lại `tp` không?
3. Có thêm sàn UPCOM không (thêm khoảng 800 mã)?
4. Nhóm cơ quan còn mỏng: chưa có `ubnd`, `bhxh`, `cccd`, `cmnd`. Nếu anh/chị chỉ ra một văn bản chính thức, em sẽ lấy theo văn bản đó.

### 2.3 ViMedCSS và `configs/vi_loanwords.txt`

ViMedCSS giờ có hai nhãn:

- `text_tagged` theo Spec: bỏ tag từ mượn có trong từ điển tiếng Việt, cộng catalog và heuristic;
- `text_tagged_orig`: chỉ gắn các thuật ngữ trong `cs_terms_list` của dataset.

Chạy thật `--limit 20` × 4 split: 44/80 câu có hai nhãn khác nhau. Phần lớn chênh lệch là do Spec thêm span heuristic (Super, HDAC); còn lại là bỏ tag lipid ×10, aceton ×3, gen, virus, gram.

`score.py --label-field text_tagged_orig` chấm số phụ theo nhãn gốc. Câu không có trường này thì dùng `text_tagged`.

`vi_loanwords.txt` (nguồn: từ điển Viet74K, words_alpha, google-10000):

| kind | Số mục | Luật | Phạm vi |
| --- | --- | --- | --- |
| `vi_spelling` | 306 | Từ một chữ trong Viet74K, không phải một âm tiết, không có trong words_alpha, không trùng google-10000 (`axit`, `canxi`, `vaccin`, `tivi`, `hecta`) | Mọi dataset |
| `same_surface` | 29 | Thuật ngữ ViMedCSS (`cs_terms_list`) có trong Viet74K và trùng mặt chữ tiếng Anh (`vitamin`, `protein`, `insulin`, `virus`, `gen`, `test`, `co`…). **Chờ duyệt**, mặc định không EN | **Chỉ ViMedCSS** |
| Bị loại | 3 | `mega`, `nano` (trùng google-10000); `iot` (loại tay: là IoT trong vi-asr-tech) | — |

**Chỗ em làm khác với yêu cầu, cần anh/chị xem:** `same_surface` chỉ áp trong ViMedCSS, không áp cho mọi bộ.

- Nếu áp cho mọi bộ, G-Group mất tag `test` ×8 (cộng thêm "unit test", "test case"), và `co` làm vỡ span "super government co contribution".
- Recall tập chính khi đó giảm từ 0.977 xuống 0.971.

Kết quả với cả hai danh sách:

| | Tập chính (100 câu) | Tập kiểm tra 60 câu, chấm **đúng 1 lần** |
| --- | --- | --- |
| Trước Phase 0b | P 0.944 / R 0.977 | P 0.962 / R 0.990 |
| Sau | P **0.955** (P bỏ qua viết tắt VN 0.982) / R 0.977 | P **0.961** (P bỏ qua viết tắt VN 0.990) / R **0.980** |

Trên tập kiểm tra, recall mất một span là `nab` (mã Nam A Bank) trong câu "nhiều đó là [[nab]] là một cái sự vượt trội". Tập vàng do em gắn ở Phase 0 coi `nab` là EN, trái với quy ước mới. Em không sửa tập vàng sau khi đã xem kết quả; câu này cần nghe audio để quyết.

File kết quả: `reports/t0b2_eval_main_abbrev.*`, `reports/t0b3_eval_main_final.*`, `reports/t0b3_eval_heldout_once.*`. Trong mỗi file `.md` có ghi sha256 của danh sách đã dùng.

### 2.4 VietSuperSpeech

Em chạy toàn bộ text VSS (`--trust-duration`, không mở audio). Bộ chặn rò rỉ đối chiếu với một bench tạm: `b1_candidates` của G-Group (1361 câu) cộng các split hard/test của ViMedCSS (từ CSV meta). Catalog tạm gồm `catalog.example` cộng 257 thuật ngữ `reserved_b3`, tính theo cùng luật của converter. Bench v0 thật chưa dựng, nên **cần chạy lại bước này khi có bench**.

| | Câu | Giờ |
| --- | --- | --- |
| `train_vss` (không có từ Anh) | 28 551 | 102.94 |
| `dev_vss` (không có từ Anh) | 3 050 | 10.99 |
| **`vss_cs_candidates`** (có từ Anh, phần train) | **31 949** | **117.32** |
| Bị loại vì có `reserved_b3` | 156 | |
| Bị loại vì trùng text với bench | 0 | |
| Bị loại vì là câu có từ Anh ở phần dev | 3 699 | |

Top 30 thuật ngữ theo span tự gắn (`data/raw/vss_full/vss_cs_terms.tsv`):

ok ×1951, marketing ×678, web ×585, app ×545, facebook ×390, podcast ×346, team ×330, video ×324, ruby ×282, youtube ×263, album ×240, stress ×234, website ×232, comment ×231, game ×220, covid ×216, inside ×210, sho ×193, show ×175, startup ×171, online ×166, tiktok ×163, brand ×154, top ×153, business ×138, bar ×135, and ×134, blockchain ×131, mv ×123, ekip ×122.

Phát hiện:

1. **Nhiều câu tiếng Anh dài.** Đếm theo từ đơn, các từ đứng đầu là `the` ×20 781, `of`, `to`, `and`, `you`. Trung bình mỗi câu có 2.45 span. Như vậy `vss_cs` có nhiều câu hoặc cụm tiếng Anh dài (lời bài hát, câu nhận dạng sai), mà câu hoàn toàn tiếng Anh nằm ngoài phạm vi Spec. Bộ lọc đồng thuận là bắt buộc.
   - Đề xuất: thêm tham số `--max-en-frac` (tỉ lệ token tiếng Anh tối đa trong câu). Em chưa đo được tỉ lệ này vì WSL dừng giữa lúc đang chạy.
2. **Train và dev của chính VSS dùng chung 561 video.** Vì vậy chia train/dev của dataset rò rỉ theo `source`. Nếu dùng `dev_vss` làm dev thì cần chia lại theo `source`. Ngoài ra `vss_cs` (lấy từ train) cũng chung video với `dev_vss`.

Bộ lọc đồng thuận (dùng chung với SEA): `lib/agreement.py` và `tools/agreement_filter.py` chỉ đọc hyps, còn `tools/asr_hyps.py` ghi hyps.

- Một span được xác nhận khi nó có mặt (sau `normalize_hyp`, strict, tính cả biến thể catalog) trong ít nhất `--min-agree` (1) file hyps.
- Câu được giữ khi tỉ lệ span được xác nhận ≥ `--min-span-frac`. Giá trị mặc định 1.0 nghĩa là mọi span; tham số này em tự thêm.

**Chạy thật trên 20 câu** (`data/raw/vss_full/vss_cs_20.jsonl`, 267 s audio). Em chạy lần lượt từng model trên GPU 4 GB:

| Model | Nạp model | RTF | Hyps |
| --- | --- | --- | --- |
| faster-whisper large-v3, int8, cuda | 8.4 s | 0.32 | `results/asr_hyps/fw_large_v3_int8/vss_cs_20.jsonl` |
| Qwen3-ASR-0.6B, bf16, cuda | 59.3 s | 0.24 | `results/asr_hyps/qwen3_asr_0.6b/vss_cs_20.jsonl` |

Chạy cả 31 949 câu (117 h) với hai model này mất khoảng 37 h + 28 h GPU.

Nguyên nhân lỗi lần trước và cách sửa:

- Lần OOM trước là do bộ nhớ bị phân mảnh. Khởi động lại máy xong, model nạp lên GPU bình thường.
- Lỗi thật là CTranslate2 cần `libcublas.so.12`, trong khi torch trong venv ASR dùng CUDA 13. Em cài `nvidia-cublas-cu12` vào venv ASR và đặt `LD_LIBRARY_PATH` trong wrapper `gb-asr`.

Kết quả bộ lọc đồng thuận:

| `--min-agree` | `--min-span-frac` | Câu giữ |
| --- | --- | --- |
| 1 | 1.0 (mặc định) | **5/20** |
| 2 | 1.0 | 3/20 |
| 1 | 0.5 | 7/20 |

Báo cáo của bộ lọc: `data/raw/vss_full/vss_cs_20_agreed_*.report.md`.

Đọc từng câu thì thấy **bộ lọc loại đúng**. Các span bị loại đều là **nhãn VSS sai**, còn cả hai ASR nghe đúng:

| Nhãn VSS | Hai ASR nghe được |
| --- | --- |
| `edward` | advice / ad vai |
| `netword` | network |
| `poca` | vodka |
| `lico tele` | cocktail |
| `dubai` | bar |
| `matthe make meliance` | make millions |
| `hennye`, `eurocham the bodle` | tên riêng H'Hen Niê, Eurocham, Gabor Fluit |

Ngoài ra 2/20 câu là **tiếng Anh nguyên câu**, nhãn vô nghĩa ("user factory not oliver to your fire"). Năm câu được giữ đều đúng.

Hệ quả:

1. Nhãn `<en>` của VSS rất nhiễu: 15/20 câu có ít nhất một span mà cả hai ASR đều không xác nhận, và các span xem được ở trên đều là nhãn sai. Nếu tỉ lệ giữ 25% đúng cho toàn bộ dữ liệu thì còn khoảng 29 h `vss_cs`. Mẫu 20 câu còn nhỏ, nên con số này chỉ là ước lượng thô.
2. **Đề xuất (cần anh/chị quyết):**
   - Thay vì chỉ lọc, có thể **gán lại nhãn** những câu mà hai ASR khớp nhau (ví dụ dùng hyp của faster-whisper và gắn `<en>` bằng `auto_tag`), để giữ được nhiều câu hơn. Cách này là nhãn giả, vẫn thuộc nhóm `synth_pseudo`.
   - Thêm `--max-en-frac` để bỏ câu tiếng Anh nguyên câu trước khi chạy ASR, cho đỡ tốn GPU.
3. Qwen3-ASR-1.7B không vừa GPU 4 GB (riêng trọng số bf16 đã khoảng 3.4 GB). Chạy thử bản 1.7B cần GPU lớn hơn.

**Cấu hình trộn** (`configs/train_mix.tsv`): `vss_cs` thuộc `synth_pseudo` (0.10), cùng nhóm với TTS, không thuộc `cs_real`. Biến môi trường để ghi đè: `MIX_CS_REAL`, `MIX_SYNTH_PSEUDO`, `MIX_REPLAY_HUMAN`, `MIX_DISTILL`, `MIX_CONFUSABLE`.

### 2.5 Tài liệu và resample (chưa commit)

- Spec: VSS là 245 h (60 656 + 6 749 câu); bổ sung ghi chú về `vss_cs`. G-Group là MP3 16 kHz. ViMedCSS là WAV 44.1 kHz, converter resample về 16 kHz; `duration_seconds` làm tròn theo giây.
- Runbook: cập nhật bảng converter.
- Docstring của `conv_ggroup` và `conv_vimedcss`.
- `save_audio(target_sr)` dùng soxr HQ, gộp mono và ghi WAV PCM16. `conv_vimedcss --target-sr 16000` là mặc định.
- Chạy thật `conv_vimedcss --limit 20` × 4 split: cả 80/80 WAV đều 16 kHz, mono, PCM_16; độ dài file khớp trường `duration`.
- Hai bộ test (`tools/test_converters.py`, `tools/selftest.sh`) đều đạt trên toàn bộ working tree.

### 2.6 Common Voice (`cv-corpus-27.0-2026-09-11/vi`)

- Cột khớp code (`client_id`, `path`, `sentence`…). Có 1593 câu test và 1848 câu train.
- **Audio là MP3 32/48 kHz**, không phải 16 kHz. Converter không copy audio, nên bước decode và tính fbank phải resample. Cần kiểm lại ở Phase 1/3 xem `to_lhotse` / `compute_fbank` có resample không.
- Text viết hoa đầu câu (2929/3441 câu) và có dấu câu; bước chuẩn hoá đã xử lý.
- 66/3441 câu có span heuristic (`kumanthong`, `alo`, `chấm net`), nên bị loại khỏi B5.
- Dataset có cột `gender` và `accents` ("Miền nam"…). Đề xuất dùng hai cột này để điền `region` / `gender`.

## 3. Quyết định em tự đưa ra, cần anh/chị xem lại

1. `same_surface` chỉ áp trong ViMedCSS (lý do ở 2.3).
2. `iot` bị loại tay khỏi `vi_loanwords` (`--exclude`), có ghi lý do trong file.
3. `vss_cs` được đánh dấu bằng `dataset: vss_cs` + `train_only: true`, còn `source` vẫn giữ `vss:<video>`. Anh/chị viết "source=vss_cs", nhưng nếu ghi đè `source` thì mọi video VSS sẽ gộp thành một source. Khi đó bộ chặn rò rỉ và việc chia dev theo source (Spec mục 5) đều hỏng.
4. `vss_cs` chỉ lấy từ phần train của VSS; câu có từ Anh ở phần dev bị bỏ.
5. Thêm `--min-span-frac` (mặc định 1.0: mọi span phải được xác nhận).
6. Các công cụ ASR chạy trong venv riêng `~/.venvs/gipbench-asr` (wrapper `~/.local/bin/gb-asr`), vì `qwen-asr` kéo theo `transformers` 4.57 và hạ `huggingface-hub` làm hỏng `datasets`. Venv chính đã được khôi phục. Còn sót `scipy` và `soxr` trong venv chính; `soxr` giờ là phụ thuộc thật.
7. Bench tạm để chặn rò rỉ `vss_cs`, lập bằng script trong scratchpad (2.4).
8. Cài `nvidia-cublas-cu12` vào venv ASR (không đụng venv chính) và đặt `LD_LIBRARY_PATH` trong `~/.local/bin/gb-asr`, để faster-whisper chạy được trên GPU.
9. Cấu hình máy: tạo `%USERPROFILE%\.wslconfig` (`swapFile=D:\\WSL\\swap.vhdx`, `swap=16GB`); pagefile của Windows chuyển sang D: (system managed), trên C: giữ 2 GB cố định. D: giờ còn khoảng 14 GB trống, cần để ý khi tải thêm dữ liệu.

## 4. Lệnh chạy lại

```bash
# nguồn cho hai danh sách (vào data/dl/lexicons, không commit)
curl -sSfL -o data/dl/lexicons/Viet74K.txt https://raw.githubusercontent.com/duyet/vietnamese-wordlist/master/Viet74K.txt
curl -sSfL -o data/dl/lexicons/words_alpha.txt https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
curl -sSfL -o data/dl/lexicons/google-10000-english-no-swears.txt https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt
curl -sS -o data/dl/lexicons/vietcap_symbols.json https://trading.vietcap.com.vn/api/price/symbols/getAll
#   wiki_*.txt: https://vi.wikipedia.org/w/index.php?title=<tiêu đề>&action=raw
#   wikt_viettat_pages.json: các trang trong Thể loại:Từ viết tắt tiếng Việt và Chữ viết tắt tiếng Việt (vi.wiktionary)
python tools/build_wordlists.py --src data/dl/lexicons --catalog configs/catalog.example.tsv \
    --vimedcss-meta data/dl/vimedcss_meta --out-abbrev configs/vi_abbrev.txt --out-loan configs/vi_loanwords.txt \
    --exclude "iot:IoT là viết tắt tiếng Anh trong vi-asr-tech (3 câu); i-ốt viết có dấu"
python tools/eval_autotag.py --gold reports/t02_gold.jsonl --catalog configs/catalog.example.tsv \
    --texts "data/dl/ggroup/*.parquet" --out reports/t0b3_eval_main_final
python converters/conv_vimedcss.py --out data/raw/vimedcss --catalog configs/catalog.example.tsv --limit 20
python converters/conv_local.py vietsuperspeech --root data/dl/VietSuperSpeech --out data/raw/vss_full \
    --catalog <catalog có reserved_b3> --bench <bench B1–B3> --trust-duration
python converters/conv_local.py commonvoice --root data/dl/cv-corpus-27.0-2026-09-11/vi --out data/raw/cv --limit 20
# bộ lọc đồng thuận (venv gipbench-asr)
python tools/asr_hyps.py --engine faster-whisper --model large-v3 --compute-type int8 --device cuda \
    --manifest data/raw/vss_full/vss_cs_20.jsonl --out results/asr_hyps/fw_large_v3_int8/vss_cs_20.jsonl
python tools/asr_hyps.py --engine qwen3-asr --model Qwen/Qwen3-ASR-0.6B \
    --manifest data/raw/vss_full/vss_cs_20.jsonl --out results/asr_hyps/qwen3_asr_0.6b/vss_cs_20.jsonl
python tools/agreement_filter.py --manifest data/raw/vss_full/vss_cs_20.jsonl \
    --hyps results/asr_hyps/fw_large_v3_int8/vss_cs_20.jsonl results/asr_hyps/qwen3_asr_0.6b/vss_cs_20.jsonl \
    --catalog configs/catalog.example.tsv --min-agree 1 --min-span-frac 1.0 --out data/raw/vss_full/vss_cs_20_agreed_a1_f1.0.jsonl
```

## 5. Chờ anh/chị

1. **`vss_cs`: lọc hay gán lại nhãn** (2.4)? Có thêm `--max-en-frac` không? Có chạy ASR trên cả 31 949 câu không (khoảng 65 h GPU cho hai model)?
2. Thư mục tạm `D:\tmp_gitbash` (em tạo để git ghi commit được lúc C: đầy) giờ xoá được.
3. Duyệt `configs/vi_abbrev.txt`, `configs/vi_loanwords.txt` (`same_surface`) và các câu hỏi ở 2.2.
4. Các quyết định ở mục 3.
5. SEA: quyền truy cập + `HF_TOKEN`.
