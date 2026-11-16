# Báo cáo Phase 0 — Xác minh converter trên dữ liệu thật

Ngày 14/11/2026. Commit: `b22cc45` (T0.1), `89e8498` (T0.2 a), `09e80dc` (T0.2 b). Quyết định trước phase: [decisions.md](decisions.md).
Chi tiết T0.1: [converter_check.md](converter_check.md). Chi tiết T0.2: [t02_eval_after.md](t02_eval_after.md), [t02_eval_heldout_after.md](t02_eval_heldout_after.md).

## 1. Bảng task

| Task | Trạng thái | File đã đổi | Kiểm thử đã chạy | Chưa chạy trên dữ liệu / model thật |
| --- | --- | --- | --- | --- |
| Môi trường | Xong | `.gitignore`, `requirements.txt` (+urllib3) | `selftest.sh` và `test_converters.py` đạt trước khi sửa và sau mỗi task; `py_compile` toàn bộ `.py` bằng Python 3.10 | — |
| T0.1 G-Group | Xong | `conv_ggroup.py`, `common.py` (`iter_hf_parquet`) | `--limit 20` đọc từ HF; chạy đủ 2 047 câu từ parquet | — |
| T0.1 ViMedCSS | Xong (chỉ `--limit 20`) | `conv_vimedcss.py`, `common.py` | `--limit 20` × 4 split từ HF; thống kê text trên toàn bộ CSV metadata | Chưa chạy converter đầy đủ (~18 GB audio) |
| T0.1 VIVOS | Xong (chỉ `--limit 20`) | — | `--limit 20` trên 40 wav trích ra; thống kê text trên toàn bộ 12 420 prompt | Chưa tải toàn bộ audio |
| T0.1 VietSuperSpeech | Xong (chỉ `--limit 20`) | `conv_local.py` (`--limit` theo phần) | `--limit 20` train + dev; thống kê text trên toàn bộ 67 405 câu | Chưa tải toàn bộ audio (~245 h) |
| T0.1 SEA | **Chưa làm** | — | — | Gated, API trả 401. `english_spans()` chưa được sửa |
| T0.1 Common Voice | **Chưa làm** | — | — | Người dùng lo phần tải |
| T0.1 báo cáo | Xong | `tools/converter_check.py`, `reports/converter_check.md` | Có test trong `test_converters.py` | — |
| T0.2 đo auto_tag | Xong | `tools/eval_autotag.py`, `reports/t02_gold*.jsonl`, `reports/t02_eval_*` | Test với số liệu tính tay; kiểm tra file vàng khớp text gốc | Vàng do Claude gắn, **người chưa duyệt** (phiếu 30 câu: `reports/t02_spotcheck.csv`) |
| T0.2 sửa heuristic | Xong (commit riêng `09e80dc`) | `converters/common.py` (`is_vietnamese_token`) | Test luật âm tiết (14 token không phải tiếng Việt, 18 token tiếng Việt); so cũ/mới trên VIVOS, VSS, pubadmin | Text không dấu (luật giả định text có dấu) |

### Kết quả T0.2 (vàng do Claude gắn, người duyệt 30 câu)

Span được so theo dải tiếng Anh liền nhau (gộp span kề nhau ở cả vàng lẫn dự đoán), dùng catalog ví dụ.

| Tập | Câu có từ Anh / câu đã đọc | Span vàng | Recall trước → sau | Precision trước → sau | Precision bỏ qua viết tắt VN, trước → sau |
| --- | --- | --- | --- | --- | --- |
| Chính (dùng để tìm lỗi) | 100 / 161 | 172 | 0.930 → **0.977** | 0.925 → **0.944** | 0.964 → 0.982 |
| Held-out (không dùng để chỉnh) | 60 / 102 | 101 | 0.950 → **0.990** | 0.960 → **0.962** | 0.990 → 0.990 |

Nếu so từng span đúng như ghi, không gộp: tập chính 0.892 / 0.913 (P / R) trước khi sửa.

Những lỗi còn lại:
- **Bỏ sót:** `hi` ×2 (chào), `run time`, `docker run` (`run` hợp lệ âm tiết Việt), `super government co contribution` (`co`).
- **Thừa:** phần lớn là viết tắt tiếng Việt (`vn`, `gdt`, `tp hcm`, `bidv`, mã chứng khoán), cùng `alo`.
- **Không đo được:** 13 + 3 span từ Anh bị viết theo phiên âm Việt (`chai đinh`, `nét lích`). Text không có dấu hiệu tiếng Anh, nên chỉ người nghe audio mới sửa được.

## 2. Schema thật so với code cũ

Đầy đủ ở [converter_check.md](converter_check.md#schema-thật-so-với-giả-định-trong-code). Tóm tắt:

- **G-Group:** cột khớp (`id, audio, text, duration`). Audio là MP3. Không có thông tin người nói.
- **ViMedCSS:**
  - Cột khớp. Audio là WAV **44.1 kHz**, cắt tròn theo giây.
  - `duration_seconds` (int) lớn hơn thời lượng audio thật khoảng 1 s.
  - Mỗi shard ~400 MB, tổng ~18 GB, nên trước đây `--limit` vẫn tải cả split.
- **VIVOS (HF `AILAB-VNUHCM/vivos`):** một tar.gz 1.47 GB, cấu trúc như converter giả định. ID có hai dạng.
- **VietSuperSpeech (HF `thanhnew2001/VietSuperSpeech`):** `train.json` / `dev.json` khớp. Quy mô thật **245 h** (README ghi 103 h, tài liệu research ghi 267 h). `--limit` trước đây bỏ qua dev.
- **Nhãn gốc VIVOS và VSS:** 100% IN HOA, 100% NFC; sau `basic_tokens` 100% chữ thường.
- **SEA:** chưa đọc được.

## 3. Quyết định em tự đưa ra, nên xem lại

1. **Ubuntu 26.04 thay vì 22.04.** WSL có sẵn bản 26.04 nên em dùng luôn. Python lấy từ uv nên không phụ thuộc bản hệ thống. Chưa biết icefall/k2 có cài được không, tới Phase 3 mới rõ. venv đặt ở `~/.venvs/gipbench` trong ổ Linux.
2. **Tải đủ parquet G-Group (40 MB)**, vượt mức "chỉ `--limit 20`", vì T0.2 phải lấy mẫu ngẫu nhiên trên cả bộ. ViMedCSS, VIVOS, VSS vẫn chỉ lấy 20 câu mỗi phần.
3. **VIVOS:** `prompts.txt` trong `data/dl/vivos` đã bị lọc về 20 câu có wav; bản đầy đủ là `prompts.full.txt`. Khi tải đủ VIVOS thì thay lại.
4. **`--limit` đọc parquet trực tiếp bằng pyarrow** thay cho `datasets` streaming, vì lỗi crash lúc thoát. Đường không có `--limit` vẫn dùng `datasets` như cũ.
5. **Quy ước của file vàng T0.2:**
   - Brand Việt viết chữ Latin có gốc Anh (`vietcombank`, `techcombank`, `sacombank`, `vinschool`, `doji`, `vietstock`, `cafef`, `azota`) được tính là `<en>`.
   - Viết tắt tiếng Việt và mã chứng khoán Việt (`vn`, `gdt`, `tp hcm`, `bidv`, `plx`, `vps`, `cm`, `abc`) **không** tính là tiếng Anh.
   - `alo` không tag. `mana` (không rõ nghĩa) tính là `<en>`.
   - Chữ số đứng cạnh (`web 3`) để ngoài span.
   - Mọi quy ước này cần anh/chị chốt, vì chúng sẽ thành quy ước gán nhãn B1.
6. **Chấm theo dải liền** (gộp span kề nhau) làm metric chính, vì `auto_tag` cố ý tách span theo thuật ngữ catalog. Số theo từng span vẫn ghi trong báo cáo.
7. **Đã sửa heuristic dù recall trước khi sửa (0.930) đã vượt ngưỡng 0.9.** Lý do:
   - Cận dưới CI95 khoảng 0.89.
   - Có từ Anh (`chip`, `net`) lọt vào B5.
   - Luật sửa là luật chính tả tiếng Việt, không phải danh sách từ chỉnh theo mẫu, và đã kiểm trên held-out.
   
   Đây là commit riêng (`09e80dc`); nếu không muốn giữ thì `git revert 09e80dc`.
8. **Commit T0.1 (`b22cc45`) đã có test cho `eval_autotag.py`**, trong khi tool này chỉ vào ở commit kế tiếp. Riêng tại `b22cc45`, đoạn cuối của `test_converters.py` sẽ báo thiếu file. Từ `89e8498` trở đi đều đạt.

## 4. Đề xuất cho các phase sau (chưa tự làm)

1. **Danh sách viết tắt tiếng Việt** (`configs/vi_abbr.tsv`: `vn`, `tp`, `hcm`, `bidv`, `gdt`, mã chứng khoán…) để `auto_tag` không gắn `<en>`. Trên tập chính, đổi này đưa precision từ 0.944 lên khoảng 0.98. Đây là danh sách dài và cần người quản lý, nên em chưa làm.
2. **VietSuperSpeech:** 52.5% số câu có từ Anh viết chính tả Anh. Nếu giữ mặc định "bỏ câu có từ Anh" thì replay/distill (T1.2) chỉ còn khoảng 115 h. Cần quyết trước T1.2: giữ nguyên, hay giữ câu mà span nằm trong catalog.
3. **ViMedCSS gắn tag cả từ mượn đã Việt hoá** (`axit`, `canxi`, `vitamin`, `oxy`), trái với Spec mục 1. Cần chốt trước khi duyệt B3: bỏ tag hay giữ theo dataset.
4. **SEA:** khi có quyền, cần đọc schema `spans` thật để sửa `english_spans()`. Converter đọc tuần tự mọi cặp ngôn ngữ (en-vi chỉ chiếm 45 h trong 474 h), nên đường streaming có thể gặp lại lỗi crash lúc thoát. Cũng nên chuyển sang pyarrow, lọc `language_pair` theo từng row group.
5. **G-Group không có người nói**, nên mỗi câu là một "người nói" khi bootstrap. CI của ΔCS-WER trên phần này sẽ hẹp hơn thực tế. Đây là giới hạn đã ghi trong Spec mục 2; nhắc lại vì G-Group đang là nguồn B1 duy nhất chạy được.

## 5. Lệnh chạy lại Phase 0

Chạy trong WSL, venv `~/.venvs/gipbench`, tại thư mục gốc dự án.

```bash
uv venv --python 3.11 ~/.venvs/gipbench && source ~/.venvs/gipbench/bin/activate
uv pip install -r requirements.txt
bash tools/selftest.sh /tmp/gipbench_selftest && python tools/test_converters.py /tmp/gipbench_convtest

# dữ liệu (G-Group đầy đủ; các bộ khác 20 câu)
mkdir -p data/dl/ggroup && for n in tech edu finance pubadmin; do
  curl -sfL -o data/dl/ggroup/$n.parquet https://huggingface.co/datasets/g-group-ai-lab/vi-asr-$n-test/resolve/main/data/test-00000-of-00001.parquet; done
#   VIVOS: trích 20 wav mỗi phần + prompts từ AILAB-VNUHCM/vivos (data/vivos.tar.gz) vào data/dl/vivos/{test,train}
#   VietSuperSpeech: train.json, dev.json + 20 audio đầu mỗi phần từ thanhnew2001/VietSuperSpeech vào data/dl/VietSuperSpeech

CAT=configs/catalog.example.tsv
python converters/conv_ggroup.py --out data/raw/ggroup_l20 --catalog $CAT --limit 20
python converters/conv_ggroup.py --out data/raw/ggroup --catalog $CAT --parquet "data/dl/ggroup/{name}.parquet"
python converters/conv_vimedcss.py --out data/raw/vimedcss --catalog $CAT --limit 20
python converters/conv_local.py vivos --root data/dl/vivos --out data/raw/vivos --catalog $CAT --limit 20
python converters/conv_local.py vietsuperspeech --root data/dl/VietSuperSpeech --out data/raw/vss --catalog $CAT --limit 20
python tools/converter_check.py --manifests data/raw/*/*.jsonl --samples 5 --out /tmp/converter_check_tables.md

# T0.2
python tools/eval_autotag.py --gold reports/t02_gold.jsonl --catalog $CAT --texts "data/dl/ggroup/*.parquet" --out reports/t02_eval_after
python tools/eval_autotag.py --gold reports/t02_gold_heldout.jsonl --catalog $CAT --texts "data/dl/ggroup/*.parquet" --out reports/t02_eval_heldout_after
```

Bước trích một phần VIVOS và tải 20 audio VSS dùng script tạm ngoài repo, chưa thành CLI trong `tools/`. Nếu cần chạy lại thường xuyên, em sẽ đưa vào `tools/`.

## Chờ anh/chị duyệt

1. Phiếu `reports/t02_spotcheck.csv`: 30 câu vàng, điền `dong_y` = y/n và `sua_lai` nếu sai.
2. Các quy ước ở mục 3.5 và các đề xuất 4.1–4.3.
3. Giữ hay revert `09e80dc`.
4. Quyền truy cập SEA và dữ liệu Common Voice, để hoàn tất T0.1.
