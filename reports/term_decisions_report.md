# Mục C: chốt tag theo thuật ngữ (G-Group B1, consensus agreed)

Ngày 15/11/2026. Quyết định: `decisions.md` §12.3–12.4. Mọi số liệu dưới đây chạy trên CPU (`gbcpu`, 2 thread) khi job ASR nền đang chạy.

## Anh/chị cần làm: duyệt MỘT file

**`reports/term_decisions_ask.csv`**: 523 dòng, mỗi dòng một thuật ngữ. Chỉ sửa cột `decision` ở những dòng sai.

- Giá trị: `en` / `vi` / `drop_tag`; để trống = chưa chốt.
- Đề xuất của em: **375 `en`**, **147 `vi`**, **1 để trống** (`ai`: "ai" tiếng Việt hay AI tuỳ câu, 19 câu).
- Các dòng đã được sắp theo số câu, giảm dần. Cột `note` ghi lý do; chữ "đoán" nghĩa là em không chắc.

| Loại cờ | Dòng | Ghi chú |
| --- | --- | --- |
| `abbr_reading` (từ / viết tắt 2–4 chữ, một mình một span) | 385 | Phần lớn là từ Anh thường (`link`, `file`, `user`); viết tắt thì em đoán theo cách đọc |
| `top_review` (gộp từ `reports/top_review.csv`) | 51 | Top 30 của `vi_abbrev` / `vi_loanwords`. Ở các dòng này, `vi` = giữ trong danh sách, `en` = coi là EN ở mọi bộ |
| `ambiguous_joined` (cụm có `to`, `the`, `in`…) | 45 | Đề xuất `en`; nên kiểm tra ranh giới span |
| `vi_word` (danh sách coi là tiếng Việt) | 20 | vd `vn` 49 câu, `test` 14 câu (xem lưu ý dưới) |
| `name` (tên người nước ngoài) | 19 | `en`, theo luật tên người đã duyệt |
| `suspect` / `maybe_name` | 3 | `ai` để trống; `sam kim` đề xuất `en` |

Có 3 dòng ghi "vòng 2" (`alt`, `chat`, `hip`). Chúng chỉ xuất hiện sau khi áp các đề xuất vòng 1, vì nhãn của câu thay đổi; số câu của chúng tính trên nhãn giả định đó.

**Không cần duyệt:** `reports/term_auto_en.csv` gồm 1 139 span heuristic không có cờ (`token`, `photo`, `click`, `vietcombank`, `smart contract`…). Các span này được coi là EN rõ ràng: text do người gõ, có dấu đầy đủ, mà token không đọc được như âm tiết Việt. File này chỉ để tham khảo.

**Lưu ý khi duyệt:**
- `test` (14 câu, chủ yếu `vi-asr-tech`, vd "unit test"): theo §12 thì `same_surface` = không EN, nên em đề xuất `vi`. Nếu anh/chị muốn "unit test" là EN thì đổi thành `en`. Quyết định này áp cho MỌI bộ, kể cả ViMedCSS.
- Quyết định theo thuật ngữ áp ở mọi nơi gọi `auto_tag` (G-Group, consensus, VSS, ViMedCSS), không riêng bench. Nhãn gốc của ViMedCSS (`text_tagged_orig`) thì không bị ảnh hưởng.
- `gov` (24 câu, vd "gov vn"): em đoán `en`. `edu` (11 câu): `en`. `pdf`, `gdp`, `km`, `st`, `rv`: em đoán `vi` (đánh vần kiểu Việt).

## Sau khi anh/chị sửa xong

```bash
python tools/term_decisions.py import --ask reports/term_decisions_ask.csv      # -> configs/term_decisions.tsv
python converters/conv_ggroup.py --names tech edu finance pubadmin --out data/raw/ggroup \
    --catalog data/raw/_proxy/catalog_proxy.tsv --parquet "data/dl/ggroup/{name}.parquet"
python converters/conv_consensus.py --root data/dl/zac --out data/raw/consensus \
    --catalog data/raw/_proxy/catalog_proxy.tsv --max-per-term 10                 # ~7 phút (speaker embedding)
python tools/term_decisions.py ask ...   # nếu còn thuật ngữ mới lộ ra (vòng 2), lặp lại 1 lần
python converters/build_bench_v0.py ... --accept trusted-text --pending-out data/tmp/pending.tsv
```

Phải chạy lại converter, vì `build_bench_v0` không nhận câu mà nhãn còn theo quyết định cũ (báo "chờ: nhãn cũ, chạy lại converter").

## Số câu / giờ dự kiến (B1, B2)

| | B1 (G-Group) | B2 (consensus) | B5 (G-Group pubadmin + consensus product) |
| --- | --- | --- | --- |
| Ứng viên (trusted_text) | 1 330 câu, 1.38 h | 218 câu, 0.40 h (sau trần 10 / thuật ngữ) | 1 406 câu, 1.63 h |
| **Hiện tại** (chỉ 12 thuật ngữ đã chốt) | **786 câu, 0.80 h** | **135 câu, 0.24 h** | 1 384 câu, 1.60 h |
| **Nếu chấp nhận mọi đề xuất** (2 vòng) | **1 265 câu, 1.32 h** | **204 câu, 0.36 h** | 1 431 câu, 1.67 h |
| Vẫn chờ sau khi chấp nhận | 9 câu | 2 câu | 8 câu |

- Câu còn chờ sau khi chấp nhận đề xuất đều chỉ vì `ai`.
- B1 theo bộ, nếu chấp nhận đề xuất: tech 460 câu (0.46 h), edu 342 (0.39 h), finance 294 (0.30 h), pubadmin 169 (0.18 h).
- Khi chấp nhận đề xuất, B1 giảm từ 1 330 xuống 1 274 ứng viên, vì có câu mà mọi span đều được chốt `vi`. Câu pubadmin loại này chuyển sang B5 (325 → 346); câu của bộ khác thì bị bỏ, như luật cũ của `conv_ggroup`.
- B2 tính sau trần `--max-per-term 10`, nên con số có thể đổi nhẹ khi quyết định thay đổi.
- Lệnh và log của phần ước tính: `data/tmp/run_c.log`. Quyết định giả định: `data/tmp/est/td2.tsv`.

## Thay đổi so với manifest trước (Phase 0 / consensus)

| | Trước | Sau | Lý do |
| --- | --- | --- | --- |
| G-Group B1 ứng viên | 1 361 | 1 330 | `same_surface` = không EN mọi bộ; danh sách từ mới; `sms`/`ok`… = en |
| G-Group B5 (pubadmin) | 318 | 325 | câu mất hết span thì chuyển sang B5 |
| consensus B2 (sau trần) | 231 | 218 | `sjc`, `tv`, `sctv` = vi: câu chỉ có các từ này chuyển sang B5 |
| consensus overcap | 59 | 37 | `sjc` không còn bị cắt |
| consensus B5 product | 1 046 | 1 081 | như trên |
| Cụm speaker | 16 | 16 | tất định, cùng ngưỡng |

## C.5: luật loại mã chứng khoán trong `vi_abbrev`

`tools/build_wordlists.py --nonfin-text <VSS train.json dev.json, VIVOS prompts train/test> --nonfin-min 20`. Luật chỉ xét mã **chỉ** thuộc nhóm "mã chứng khoán" và không được thêm tay. Vì vậy `vtv` (đã chốt `vi`), `acb` và `shb` (còn là ngân hàng) không bị loại.

**11 mã bị loại** (số lần gặp): tot 362, fir 348, hap 120, fid 56, ame 39, ats 34, asm 30, fpt 26, ast 25, lig 24, bic 20. Danh sách nằm ở cuối `configs/vi_abbrev.txt` (dòng `#-`, lý do "gặp N lần trong text không phải tài chính").

`fpt` và `bic` là tên công ty thật. Chúng có trong phiếu với đề xuất `vi` (FPT 6 câu G-Group).

`vi_abbrev.txt`: 622 → 614 mục. Thêm `sjc`, `tv`, `sctv` (quyết định 15/11); `vtv`, `htv` giờ mang nguồn "quyết định 15/11".

```bash
python tools/build_wordlists.py --src data/dl/lexicons --catalog configs/catalog.example.tsv --out-abbrev configs/vi_abbrev.txt \
    --nonfin-text data/dl/VietSuperSpeech/train.json data/dl/VietSuperSpeech/dev.json \
                  data/dl/vivos/train/prompts.full.txt data/dl/vivos/test/prompts.full.txt
python tools/wordlist_top_review.py --out reports/top_review.csv    # đã gộp vào term_decisions_ask.csv
```

## Cách hoạt động (code)

- `lib/term_decisions.py` có các hàm sau:
  - đọc và ghi `configs/term_decisions.tsv`;
  - `needed_terms`: thuật ngữ còn cờ của một câu. Bỏ qua `vi_word` của âm tiết Việt hợp lệ như `co`, `di`, `va`, vì thiếu danh sách thì các từ này cũng không bao giờ bị gắn tag;
  - `term_status`: trả về pending / blocking / stale.
- `auto_tag(term_dec=)`:
  - `vi` / `drop_tag` thắng catalog;
  - `en` thắng `vi_abbrev` / `vi_loanwords`, và được ghép với từ Anh liền kề (`stop loss`, `new york`) chứ không tách thành span riêng;
  - mặc định đọc `configs/term_decisions.tsv`; `--term-decisions ''` để tắt.
- `conv_ggroup`: mọi dòng có `trusted_text: true`. `conv_consensus`: chỉ dòng `agreed` có `trusted_text: true`; quyết định theo thuật ngữ áp cả cho cụm tên người.
- `build_bench_v0 --accept trusted-text` nhận câu có `trusted_text` khi đủ 3 điều kiện:
  - mọi thuật ngữ còn cờ đã chốt;
  - không có cờ chặn (vd `numbers_disagree`, `email_or_url`; mấy câu này vẫn phải duyệt từng câu);
  - nhãn không cũ.
- `build_bench_v0` có thêm `--pending-out` (thuật ngữ còn chờ, kèm số câu và giờ) và `--draft-splits` (dùng cho bench nháp A2).
- `tools/term_decisions.py ask | import`.

Kiểm thử (`tools/test_converters.py`, hàm `test_term_decisions` và luật C.5) đạt; `tools/selftest.sh` đạt.
