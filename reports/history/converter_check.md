# T0.1 — Kiểm tra converter trên dữ liệu thật

Ngày 14/11/2026. Môi trường: WSL2 Ubuntu 26.04, Python 3.11.18 (uv), datasets 5.2.0, pyarrow 26.0.1, lhotse 1.33.0,
huggingface_hub 2.4.0. Catalog dùng để gắn tag: `configs/catalog.example.tsv` (chưa có catalog thật).

## Đã chạy gì

| Bộ | Lệnh | Dữ liệu đã tải | Kết quả |
| --- | --- | --- | --- |
| G-Group tech/edu/finance/pubadmin | `conv_ggroup.py --limit 20` (đọc từ HF), rồi chạy đủ từ parquet tải về | 4 parquet test, tổng ~40 MB (toàn bộ, vì T0.2 cần lấy mẫu trên cả bộ) | Không lỗi, mã thoát 0 |
| ViMedCSS | `conv_vimedcss.py --limit 20` (cả 4 split) | 20 câu đầu mỗi split, đọc trực tiếp row group từ HF; 4 file CSV metadata (5 MB) để thống kê text | Không lỗi sau khi sửa (xem dưới) |
| VIVOS | `conv_local.py vivos --limit 20` | 20 wav test + 20 wav train trích dạng stream từ `AILAB-VNUHCM/vivos` (tar.gz 1.47 GB, không lưu cả bộ); `prompts.txt` đầy đủ lưu thành `prompts.full.txt`, `prompts.txt` lọc về 20 câu có wav | Không lỗi |
| VietSuperSpeech | `conv_local.py vietsuperspeech --limit 20` | `train.json`, `dev.json`, `manifest.json` (36 MB) + 20 audio đầu mỗi phần từ `thanhnew2001/VietSuperSpeech` | Không lỗi sau khi sửa (xem dưới) |
| SEA (`gijs/sea_speech`) | **Không chạy** | — | Gated, API trả HTTP 401; chưa có quyền. `english_spans()` **chưa** được sửa vì chưa thấy dữ liệu thật |
| Common Voice | **Không chạy** | — | Người dùng lo phần tải |

## Schema thật so với giả định trong code

| Bộ | Schema thật (đọc từ parquet/json thật) | Khác giả định cũ |
| --- | --- | --- |
| G-Group (4 bộ) | `id` string, `audio` Audio 16 kHz, `text` string, `duration` float32; chỉ split `test`; 508 / 495 / 529 / 515 câu | Khớp. Audio là **MP3** (không phải WAV); `duration` đo được khớp cột gốc. Text thường, không dấu câu, số đã viết thành chữ. License CC-BY-NC-4.0 |
| ViMedCSS | `segment_id`, `audio` (Audio, decode=false), `duration_seconds` **int64**, `segment_text`, `cs_terms_list` ("a; b"), `cs_terms_count`, `topic`, `original_video_link`, `original_video_title`, `start_time`, `end_time`; split train / validation / test / hard; 30 + 5 + 5 + 2 shard parquet, ~400 MB mỗi shard (~18 GB) | Cột khớp. Audio là **WAV 44.1 kHz** (không phải 16 kHz); to_lhotse đã tự resample. Audio cắt tròn theo giây (5.0, 8.0 s), còn `duration_seconds` trong CSV lớn hơn thời lượng audio thật 1 s (vd 6 so với 5.0) — converter dùng thời lượng audio thật nên đúng, nhưng số giờ tính từ CSV bị thổi lên. Text có dấu câu, chữ hoa, chữ số. File CSV metadata của split test ghi cột `Topic` (chữ hoa), parquet thì ghi `topic` |
| VIVOS (HF) | `data/vivos.tar.gz` chứa `vivos/{train,test}/prompts.txt` + `waves/<SPK>/<ID>.wav`; 11 660 + 760 câu | Khớp cấu trúc thư mục. ID có hai dạng (`VIVOSSPK01_R001` và `VIVOSDEV05_004`), converter dùng ID nguyên văn nên không lệch. License CC-BY-NC-SA-4.0 |
| VietSuperSpeech | `train.json` / `dev.json` là mảng JSON, mỗi phần tử `{audio, text, duration, source}`; `audio` là đường dẫn tương đối | Khớp. Quy mô thật: **60 656 + 6 749 câu, 220.84 + 24.57 h** (theo `manifest.json`; README ghi 103 h là bản cũ; tài liệu research ghi 267 h). Nhãn do Zipformer-30M-RNNT-6000h sinh. License MIT |
| SEA | Không đọc được (gated) | Thẻ dataset chỉ ghi `spans` là "per-language text + start/end", chưa đủ để biết tên khoá |

## Kiểu chữ của nhãn gốc (yêu cầu T0.1)

| Bộ | Nhãn gốc IN HOA | Đã ở dạng NFC | Sau `basic_tokens` toàn chữ thường |
| --- | --- | --- | --- |
| VIVOS (12 420 câu) | 100% | 100% | 100% |
| VietSuperSpeech train (60 656 câu) | 100% | 100% | 100% |

## Sửa trong T0.1

1. **`--limit` không còn tải cả split.** `conv_vimedcss.py` và `conv_ggroup.py` trước đây tải toàn bộ split ngay cả khi có `--limit` (riêng ViMedCSS train ~14 GB). Hàm mới `common.iter_hf_parquet` đọc đúng N dòng đầu từ các row group đầu của file parquet trên HF.
2. **Crash lúc thoát khi đọc stream.** Dùng `datasets` streaming rồi dừng giữa một shard lớn thì Python crash lúc thoát (`Fatal Python error: PyGILState_Release`, mã thoát 134). Em đã thử `close()`, `del` + `gc.collect()`, `take()`, tắt hf-xet; cách nào cũng vẫn crash. Đây là lỗi của datasets 5.2 + pyarrow 26, nhưng đủ làm hỏng mọi script có `set -e`. `iter_hf_parquet` tự kết thúc sau N dòng nên không gặp lỗi này. **Chưa sửa** đường streaming của `conv_sea.py`, vì chưa chạy được SEA.
3. **`conv_local.py vietsuperspeech --limit`** trước đây tính trên train và dev gộp lại, nên dev không bao giờ được đọc. Giờ `--limit` áp cho từng phần, giống vivos và commonvoice.
4. `requirements.txt` thêm `urllib3`: lhotse 1.33 import gói này mà không khai báo phụ thuộc.

## Phân bố `auto_tag.confidence` trên toàn bộ text

Tính bằng `auto_tag` y như converter gọi (ViMedCSS có truyền `cs_terms_list`), trên toàn bộ text của từng bộ, không cần audio. G-Group lấy từ lần chạy converter đầy đủ. Catalog chỉ là bản ví dụ, nên tỉ lệ `catalog` sẽ tăng khi có catalog thật.

| Bộ | Câu | Giờ | auto_tag.confidence | term_not_found |
| --- | --- | --- | --- | --- |
| G-Group → `b1_candidates` (tech 464, edu 354, finance 337, pubadmin 192) | 1 347 | 1.40 | heuristic 1 309 (97.2%), catalog 38 (2.8%) | – |
| G-Group → `b5_ggroup` (pubadmin không có span) | 323 | 0.27 | none 323 (100%) | – |
| ViMedCSS hard (CSV) | 658 | 1.38* | catalog 439 (66.7%), heuristic 219 (33.3%) | 13 |
| ViMedCSS test (CSV) | 1 614 | 3.39* | catalog 1 214 (75.2%), heuristic 393 (24.3%), none 7 (0.4%) | 37 |
| ViMedCSS train (CSV) | 11 832 | 24.30* | catalog 9 100 (76.9%), heuristic 2 648 (22.4%), none 84 (0.7%) | 281 |
| ViMedCSS valid (CSV) | 1 714 | 3.57* | catalog 1 299 (75.8%), heuristic 407 (23.7%), none 8 (0.5%) | 53 |
| VietSuperSpeech train | 60 656 | 220.84 | heuristic 31 296 (51.6%), none 28 836 (47.5%), catalog 524 (0.9%) | – |
| VietSuperSpeech dev | 6 749 | 24.57 | heuristic 3 594 (53.3%), none 3 097 (45.9%), catalog 58 (0.9%) | – |
| VIVOS test | 760 | – | none 760 (100%) | – |
| VIVOS train | 11 660 | – | none 11 659, heuristic 1 (`a b`) | – |

\* Giờ ViMedCSS tính từ `duration_seconds` của CSV, cao hơn thời lượng audio thật khoảng 1 s mỗi câu.

## Phát hiện cần anh/chị xem

1. **B5 G-Group có từ Anh lọt vào.** `b5_ggroup.jsonl` có 3 câu chứa `chip` ("căn cước công dân gắn chip") và 1 câu chứa `net`, chỉ được gắn cờ `suspect:*`. Lý do: `chip`, `net` trùng cấu trúc âm tiết Việt trong regex cũ. Phần sửa ở T0.2 sẽ chuyển các câu này sang ứng viên B1.
2. **VietSuperSpeech: 52.5% số câu có từ Anh**, và nhãn máy **viết từ Anh bằng chính tả Anh** (ok, marketing, web, app, facebook, podcast, team…). Ngoài ra có những câu tiếng Anh vô nghĩa ("AND IN TON AND TRANTY TO NIGHT PERCENT…"). Mặc định `conv_local` bỏ mọi câu có từ Anh, tức mất khoảng một nửa trong 221 h. Phase 1 cần quyết: giữ nguyên mặc định, hay giữ câu chỉ có từ Anh nằm trong catalog cho replay hoặc distill.
3. **ViMedCSS gắn tag cả từ mượn đã Việt hoá** (`axit`, `canxi`, `magie`, `oxy`, `vitamin`, `peptit`) vì chúng nằm trong `cs_terms_list` của dataset. Theo Spec mục 1, từ mượn có trong từ điển tiếng Việt thì không tag. Người duyệt B3 cần bỏ các tag này, hoặc chốt lại quy ước cho ViMedCSS.
4. **Bản phiên âm Việt trong G-Group:** 12/161 câu đọc ở T0.2 viết từ Anh theo kiểu phiên âm (`chai đinh` = trading, `nét lích` = Netflix…). Text không cho thấy đó là từ Anh, nên `auto_tag` không bắt được. Người duyệt B1 phải nghe audio để sửa.
5. **Viết tắt tiếng Việt bị gắn `<en>`:** `tp hcm`, `gdt`, `vn`, `bidv`, mã chứng khoán (`plx`, `qns`). Đây là chỗ precision thấp nhất (xem T0.2).
6. **Đường dẫn audio tuyệt đối:** với VIVOS và VSS, manifest ghi đường dẫn tuyệt đối (`/mnt/d/...`), vì audio không được copy vào thư mục `--out`. Chạy được trên máy này, nhưng không mang sang máy khác được.

## Câu mẫu

Mỗi manifest 5 câu ngẫu nhiên (seed 0), sinh bằng
`python tools/converter_check.py --manifests data/raw/ggroup_l20/*.jsonl data/raw/vimedcss/*.jsonl data/raw/vivos/*.jsonl data/raw/vss/*.jsonl data/raw/ggroup/*.jsonl --samples 5 --out reports/converter_check_tables.md`.
`ggroup_l20` là lần chạy `--limit 20`; `ggroup` là lần chạy đủ.

| Manifest | Câu | Giờ | Người nói | Source | Domain | auto_tag.confidence |
| --- | --- | --- | --- | --- | --- | --- |
| `data/raw/ggroup_l20/b1_candidates.jsonl` | 50 | 0.052 | 50 | 50 | tech 18, finance 13, edu 11, pubadmin 8 | heuristic 50 (100%) |
| `data/raw/ggroup_l20/b5_ggroup.jsonl` | 12 | 0.009 | 12 | 12 | pubadmin 12 | none 12 (100%) |
| `data/raw/vimedcss/b3_candidates.jsonl` | 20 | 0.043 | 11 | 11 | medical 20 | catalog 10 (50%), heuristic 10 (50%) |
| `data/raw/vimedcss/dev_vimedcss.jsonl` | 20 | 0.042 | 16 | 16 | medical 20 | catalog 11 (55%), heuristic 9 (45%) |
| `data/raw/vimedcss/ext_vimedcss_test.jsonl` | 20 | 0.043 | 14 | 14 | medical 20 | catalog 10 (50%), heuristic 10 (50%) |
| `data/raw/vimedcss/train_vimedcss.jsonl` | 20 | 0.043 | 3 | 3 | medical 20 | heuristic 13 (65%), catalog 7 (35%) |
| `data/raw/vivos/b5_vivos.jsonl` | 20 | 0.017 | 1 | 1 | read 20 | none 20 (100%) |
| `data/raw/vivos/train_vivos.jsonl` | 20 | 0.049 | 1 | 1 | read 20 | (không có) 20 (100%) |
| `data/raw/vss/dev_vss.jsonl` | 7 | 0.027 | 7 | 7 | conversation 7 | (không có) 7 (100%) |
| `data/raw/vss/train_vss.jsonl` | 7 | 0.027 | 2 | 2 | conversation 7 | (không có) 7 (100%) |
| `data/raw/ggroup/b1_candidates.jsonl` | 1347 | 1.401 | 1347 | 1347 | tech 464, edu 354, finance 337, pubadmin 192 | heuristic 1309 (97%), catalog 38 (3%) |
| `data/raw/ggroup/b5_ggroup.jsonl` | 323 | 0.266 | 323 | 323 | pubadmin 323 | none 323 (100%) |

### `data/raw/ggroup_l20/b1_candidates.jsonl`

Cờ: `heuristic_span` 50, `ambiguous_joined` 3, `suspect:pin` 1, `suspect:bot` 1

5 câu mẫu (`text_tagged`):

- `gg_edu_113048`: thầy huy nhìn vào hoạt động đầu tiên nhé <en>complete sentences</en>
- `gg_pubadmin_113067`: và trên trang nộp thuế <en>gdt gov vn</en>
- `gg_edu_113085`: nãy giờ mình đang trao đổi ví dụ như <en>engineering</en> nè <en>business</en> nè kiến trúc thiết kế hay <en>media</en>
- `gg_tech_112994`: và mình sẽ bật luôn <en>snap to grid</en>
- `gg_tech_113173`: để có thể cài đặt được <en>module qr</en> <en>code</en> về máy của mình

### `data/raw/ggroup_l20/b5_ggroup.jsonl`

5 câu mẫu (`text_tagged`):

- `gg_pubadmin_113065`: người dân phải làm thế nào để được cấp hộ chiếu vắc xin
- `gg_pubadmin_113057`: tiền hay chưa thế thì khi tôi xuất hóa đơn tôi đã phát sinh nghĩa vụ
- `gg_pubadmin_113054`: trình tự thủ tục giải quyết khiếu nại
- `gg_pubadmin_113037`: thì nó sẽ cập nhật trên trang tư này
- `gg_pubadmin_113084`: và chị đã giảm cho em về quy trình tình trạng thai nhé

### `data/raw/vimedcss/b3_candidates.jsonl`

Cờ: `heuristic_span` 10

5 câu mẫu (`text_tagged`):

- `vmc_Med_CS-1167-1`: Thưa các anh các chị và các bạn đồng nghiệp, chúng tôi đã trình bày với các anh các chị và các bạn đồng nghiệp về hoạt động của hệ <en>renin</en>, <en>Angiotensin</en>, <en>aldosterone</en>.
- `vmc_Med_CS-1168-3`: Như vậy, nếu chúng ta dùng cái lợi tiểu mà kháng <en>aldosterone</en> á thì sao các bạn?
- `vmc_Med_CS-1105-2`: Nguyên nhân là do đột biến gen <en>HGD</en>, dẫn đến cơ thể không thể phân giải hoàn toàn <en>homogentisic</en> <en>acid</en>.
- `vmc_Med_CS-1167-7`: Các thuốc tác động lên hệ <en>renin</en>, <en>Angiotensin</en>, <en>aldosterone</en> ra đời, trở thành nền tảng trong điều trị suy tim, tăng huyết áp, cả tiên phát và thứ phát.
- `vmc_Med_CS-1057-10`: Sản phẩm được tạo ra từ quá trình này được gọi là <en>Hb</en> bị <en>glycosyl</en> hóa. Thông thường để biểu thị chỉ số <en>Hb</en> đã bị <en>glycosyl</en> hóa,

### `data/raw/vimedcss/dev_vimedcss.jsonl`

Cờ: `heuristic_span` 9

5 câu mẫu (`text_tagged`):

- `vmc_Med_CS-101_1-5`: Ngoài ra, trong cây còn có nhiều <en>flavonoid</en> cùng các chất chống <en>oxy</en> hóa khác.
- `vmc_Med_CS-1006-9`: Thông thường, <en>hormone</en> này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp. <en>Estradiol E2</en>, các chuyên gia đánh giá <en>Estradiol</en> là loại <en>estrogen</en> mạnh nhất.
- `vmc_Med_CS-1006-35`: Dễ dàng mệt mỏi, tâm trạng lâng lâng, thường xuyên cảm thấy chán nản và lo lắng, da khô, ngược lại, nếu hàm lượng <en>estrogen</en> ở nam giới quá cao có thể dẫn đến.
- `vmc_Med_CS-1016-9`: À cường giáp, đặc biệt là bệnh <en>Basedow</en> rất là thường gặp ở những người phụ nữ từ 20 đến 50 tuổi, tức là phụ nữ ở tuổi tuổi sinh sản.
- `vmc_Med_CS-1061-8`: Thì trong quả kỷ tử này nó chứa rất là nhiều các <en>axit</en> <en>amin</en> và một trong các thành phần quan trọng đó chính là tiền <en>vitamin</en> A.

### `data/raw/vimedcss/ext_vimedcss_test.jsonl`

Cờ: `heuristic_span` 10

5 câu mẫu (`text_tagged`):

- `vmc_Med_CS-1006-7`: Sau đó, hàm lượng <en>hormone</en> sinh dục nữ này sẽ từ từ hạ dần về mức bình thường, các loại <en>hormone</en> <en>estrogen</en>, thực tế.
- `vmc_Med_CS-1000-3`: Và chúng ta thấy là đối với người bình thường thì <en>methionin</en> lại rất là tốt.
- `vmc_Med_CS-100-13`: Rồi về chất khoáng ấy thì <en>Alpha</en> <en>lipid</en> thì có ba loại khoáng chất chính, đó là sắt, đó là <en>magie</en> và đó là <en>canxi</en>.
- `vmc_Med_CS-1008-3`: Tại viện <en>Wistar</en> năm 1958, <en>Hayflick</en> bắt đầu nghiên cứu xem liệu <en>virus</en> có thể gây ung thư ở người hay không.
- `vmc_Med_CS-104-3`: Bạn có thể biết đến nó là thành phần chính trong nước tẩy sơn móng tay. Vậy chúng ta tiếp xúc với <en>aceton</en> ở đâu?

### `data/raw/vimedcss/train_vimedcss.jsonl`

Cờ: `heuristic_span` 13, `ambiguous_joined` 1

5 câu mẫu (`text_tagged`):

- `vmc_Med_CS-100-16`: <en>BioLactol</en> thì ngoài cái công nghệ <en>Alpha</en> <en>lipid</en> bảo vệ thì còn được hấp thụ tăng thêm những cái <en>lactoferrin</en> và cũng như <en>axit</en> mật ngắn để giúp cho cái quá trình bảo vệ hạt sữa non được được tối ưu hơn.
- `vmc_Med_CS-0-12`: Chẩn đoán thiếu 5 <en>alpha</en> <en>reductase</en> thường dựa vào những nghi ngờ có cơ quan sinh dục lưỡng giới ở trẻ sơ sinh.
- `vmc_Med_CS-10-3`: Khi mà các chị em có triệu chứng là sưng đau hoặc khó chịu mà nghĩ đến cái dấu hiệu của viêm tuyến <en>Bartholin</en> thì các chị em thường đến bác sĩ để được khám và tư vấn.
- `vmc_Med_CS-10-5`: Vì vậy, khi chị em có cái triệu chứng, có dấu hiệu của viêm tuyến <en>Bartholin</en>, đừng ngại ngần.
- `vmc_Med_CS-10-2`: Vậy thì cái điều trị mà viêm tuyến <en>Bartholin</en> nó có đau không? Thì hãy cùng bác sĩ Hương tìm hiểu.

### `data/raw/vivos/b5_vivos.jsonl`

5 câu mẫu (`text_tagged`):

- `vivos_VIVOSDEV05_286`: thế nhưng khi giá phôi thép thế giới cao dần
- `vivos_VIVOSDEV05_109`: doanh nghiệp tuyển dụng lao động nước ngoài
- `vivos_VIVOSDEV05_249`: tôi nghĩ thủ tướng sẽ không né tránh bất cứ câu hỏi nào
- `vivos_VIVOSDEV05_212`: bốn chiếc trống đại phèng la
- `vivos_VIVOSDEV05_209`: tên gọi tại địa phương là trắc thối

### `data/raw/vivos/train_vivos.jsonl`

5 câu mẫu (`text`):

- `vivos_VIVOSSPK27_178`: tôi có thể kiểm tra ở đâu để biết chính xác số ruộng này đang thuộc quyền sở hữu của ai
- `vivos_VIVOSSPK27_078`: hạnh phúc đã mỉm cười đối với gia đình chúng tôi mặc dù tiền đề của hạnh phúc đó không phải là tiền đâu
- `vivos_VIVOSSPK27_022`: đừng vội vàng ảo tưởng về một mối quan hệ hoàn toàn đẹp và tuyệt vời như trong mộng của hai người
- `vivos_VIVOSSPK27_018`: xây dựng bắc trung bộ và duyên hải trung bộ tiến kịp các vùng khác
- `vivos_VIVOSSPK27_023`: mấu chốt của vấn đề nằm ở chỗ cơ quan nào đang là nhạc trưởng phụ trách việc liên kết và phát triển vùng

### `data/raw/vss/dev_vss.jsonl`

5 câu mẫu (`text`):

- `vss_0060674`: thì rất nhiều người cũng hỏi là tại sao bởi vì trong lĩnh vực đó thì mình cũng có kinh nghiệm nhất định và mình cũng đạt được một số thành tựu nhất định vâng thế nhưng mình nghĩ là thế giới rộng lớn hơn mình muốn thử những thứ khác
- `vss_0060670`: vâng ạ và thêm cả cái tập hai chứ tại vì trong vòng mười năm qua thì gọi là tích lũy chia sẻ và phát ngôn thì nhiều lắm có cái sự cập nhật
- `vss_0060664`: chứ còn đối với công nghệ thì mình cũng không cần và cũng không nên rằng là tự phát triển mọi thứ thì do đó thì lực lượng công nghệ là một mảng và cái thứ hai là việc mà đầu tư chi phí để có thể là hợp tác với các đối tác công nghệ khác
- `vss_0060675`: đôi khi mọi thứ xung quanh hơi hạn hẹp lại là cái việc là rất là kiên định và biết rất là rõ mình muốn gì hay không dì nghĩ mỗi thế hệ nó có một cái đặc điểm và nó có những cái điều kiện mà nó hình thành nên tính cách
- `vss_0060668`: cái là họ sẽ ghi lại cái đó trong cuốn sổ tay của quán đó để chi để lần sau ví dụ như tới mà không có cái bạn lần trước pha á thì họ sẽ nói ngày nay em hôm bữa là các bạn đã pha cái món này thì những cái bạn mà đi làm ngày hôm đó sẽ pha lại được cái món đó trời ơi anh ơi anh nói bây giờ em mới biết luôn á thì ra em quá dễ

### `data/raw/vss/train_vss.jsonl`

5 câu mẫu (`text`):

- `vss_0000009`: em em nghĩ nó lại có tức là nói trải đời thì không đúng nhưng mà chúng ta đều có những cái trải nghiệm khác nhau trải nghiệm và trải đời thì em nghĩ thế là hai bạn chùa khác nhau nhưng thật ra em là người khá là nhiều trải nghiệm tại vì em em thấy là
- `vss_0000013`: thế minh thích cà phê nói chung và mê tất cả các loại cà phê nhưng mà sau này mình phát hiện ra là mình sẽ không bao giờ uống những thứ mà nó có nhiều các cái vị mà những vị này lại có thể giả vờ được
- `vss_0000010`: chị nghĩ là rất là cũng là một rất là may mắn đấy bởi vì là mình đúng là nếu mà làm nghệ thuật và mình phải nghĩ quá nhiều nhìn trước ngó sau xem là mọi người cảm thấy thế nào thì nó thiếu cái sự tự do nhưng vừa rồi em có nhắc đến cái việc là
- `vss_0000008`: mình cũng làm việc trong nhóm người úc mà người úc họ cười những cái câu chuyện mà đúng là chỉ có họ biết ví dụ như mình nhưng mà mình cũng vậy mình cũng chỉ cười những cái câu chuyện mà mình nhắc đến ví dụ ông câu ông táo thì người nước ngoài họ không có thể ngay lập tức họ ghét được đấy thì
- `vss_0000011`: mình càng thử nhiều thứ mình càng trải nghiệm nhiều thứ thì cái tấm thảm đấy nó càng nhiều màu mình cũng thấy có những cái tấm thảm nó đơn sắc có những tấm thảm nó rất là rực rỡ bản thân cái câu hỏi thì minh hỏi cái câu mà mọi

### `data/raw/ggroup/b1_candidates.jsonl`

Cờ: `heuristic_span` 1309, `ambiguous_joined` 65, `suspect:ai` 9, `suspect:top` 5, `suspect:bot` 3, `suspect:hot` 2, `suspect:pin` 1, `suspect:chip` 1, `suspect:mac` 1

5 câu mẫu (`text_tagged`):

- `gg_pubadmin_113082`: là bản <en>photo</en> hay là căn cước công dân <en>photo</en>
- `gg_tech_116457`: <en>self service password reset</en> nó cho phép người dùng có thể thay đổi <en>password</en> và đồng bộ một số tài khoản
- `gg_edu_113209`: trường đại học bách khoa <en>tp hcm</en> cho hay
- `gg_tech_114865`: viên sẽ chọn là cái <en>root</en> này chọn cái <en>default</en> a <en>default</en> b
- `gg_finance_115637`: đó và cái quan trọng không kém nữa là gì ạ là nếu sai những cái <en>stop loss</en>

### `data/raw/ggroup/b5_ggroup.jsonl`

Cờ: `suspect:chip` 3, `suspect:net` 1

5 câu mẫu (`text_tagged`):

- `gg_pubadmin_114644`: chúng tôi cũng đề nghị hội đồng xét xử được công nhận thỏa thuận hai bên đều đồng ý
- `gg_pubadmin_113348`: mười sáu để xác định thửa đất số bốn trăm chín mươi sáu
- `gg_pubadmin_113321`: chúng tôi không có ý kiến hay bổ sung gì mời hội đồng xét xử tiếp
- `gg_pubadmin_114205`: luật thuế giá trị gia tăng năm hai ngàn không trăm lẻ tám
- `gg_pubadmin_114923`: điều một trăm tám mưới chín vốn

