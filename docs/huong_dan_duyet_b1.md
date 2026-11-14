# Hướng dẫn duyệt phiếu B1 (và B3, B4)

Phiếu duyệt là các file `*_review.csv` do converter ghi ra. Mở bằng Excel hoặc Google Sheets, nghe audio từng câu (cột `audio`), rồi điền các cột dưới đây. `build_bench_v0.py` chỉ nhận câu có `decision` = `ok` hoặc `fix`.

## Các cột cần điền

| Cột | Điền gì |
| --- | --- |
| `decision` | `ok`: `text_tagged` đúng như đang ghi · `fix`: có sửa, ghi câu đúng vào `text_tagged_fixed` · `drop`: không dùng được (ồn, cắt cụt, câu toàn tiếng Anh, có `<unk>` trong span Anh) |
| `text_tagged_fixed` | Câu đúng (chỉ khi `fix`), giữ quy ước `<en>…</en>` |
| `viet_levels` | Mức Việt hoá của từng span theo thứ tự, ngăn bằng dấu phẩy: `strong` ("phây búc"), `mid`, `native` (gần bản ngữ). Ví dụ `strong,mid` |
| `reading_note` | Bắt buộc khi câu có cờ `abbr_reading:…` hoặc khi phải tự quyết một từ mơ hồ. Ghi từ, cách đọc nghe được và quyết định, ví dụ `pc: pê xê -> không EN` hoặc `ai: ây ai -> EN` |
| `region`, `gender` | Nếu nghe ra: `north` / `central` / `south` / `other`, `f` / `m` |
| `note` | Ghi chú khác |

## Quy ước gắn `<en>`: theo cách đọc

Một từ Latin được gắn `<en>` khi người nói **đọc nó như tiếng Anh**. Bảng đầy đủ ở `SCHEMA.md`, mục "Quy ước gán `<en>`". Tóm tắt:

1. **Từ tiếng Anh**, kể cả khi phát âm kiểu Việt: gắn `<en>`, viết chính tả Anh.
2. **Brand / tên riêng chữ Latin**, kể cả brand Việt (`vietcombank`, `zalo`): gắn `<en>`.
3. **Viết tắt tiếng Anh đọc kiểu Anh** (`ai` đọc "ây ai", `ceo`, `kpi`): gắn `<en>`.
4. **Viết tắt / mã đọc kiểu Việt** (`vn` đọc "vê en", `bidv`, `plx`, `ubnd`, mã chứng khoán): **không** gắn.
5. **Từ mượn có trong từ điển tiếng Việt** (`axit`, `canxi`, `vitamin`): **không** gắn.
6. **Mơ hồ**: nghe audio. Cùng một chữ có thể được đọc theo hai cách ở hai câu khác nhau, nên quyết theo từng câu và ghi lại vào `reading_note`.

## Các cờ trong cột `flags`

| Cờ | Ý nghĩa | Cần làm |
| --- | --- | --- |
| `heuristic_span` | Có span được đoán từ cấu trúc âm tiết, không nằm trong catalog | Kiểm tra span đó có đúng là tiếng Anh không |
| `ambiguous_joined` | Một từ vừa là tiếng Việt vừa là tiếng Anh (the, to, top…) bị ghép vào span bên cạnh | Kiểm tra chỗ cắt span |
| `suspect:<từ>` | Từ đứng riêng có thể là tiếng Anh (chat, top, pay…) nhưng chưa gắn | Nghe xem có phải tiếng Anh không |
| `abbr_reading:<từ>` | Viết tắt ngắn (2–4 chữ) chưa có trong catalog hay `vi_abbrev.txt`, đang được gắn `<en>` | Nghe cách đọc, điền `reading_note`, sửa tag nếu đọc kiểu Việt |
| `vi_word:<từ>` | Từ chữ Latin không bị gắn `<en>` vì nằm trong `vi_abbrev.txt` / `vi_loanwords.txt` (mã chứng khoán, viết tắt, từ mượn) | Nghe xem có đúng là đọc kiểu Việt không. Nếu người nói đọc kiểu Anh thì `fix` và gắn `<en>` |
| `term_not_found` | (ViMedCSS) Không tìm thấy thuật ngữ của dataset trong câu | Kiểm tra câu hoặc `drop` |

Nếu thấy một viết tắt lặp lại nhiều lần và luôn được đọc cùng một kiểu, ghi nó vào `note`. Viết tắt đó nên được thêm vào `configs/vi_abbrev.txt` (đọc kiểu Việt) hoặc vào catalog (đọc kiểu Anh) ở phiên bản danh sách sau.
