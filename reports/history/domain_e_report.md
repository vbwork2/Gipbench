# Mục E: domain (decisions §10, §12.2, §12.6)

Ngày 15/11/2026. Mọi bước chạy trên CPU. Số liệu của `vss_cs` là ở mốc **4.15 h** (snapshot `vss_cs_clean.jsonl` lúc 20:18). Job đang chạy tiếp tới 6 h; khi job xong, `tools/rebuild_vss_cs.sh` sẽ tự chạy lại bước suy domain, sau đó cần chạy lại các báo cáo dưới đây.

## E1. Suy domain từ text

- Code: `lib/domain.py` + `tools/infer_domain.py` + `configs/domain_keywords.tsv`.
  - Từ khoá: 9 domain, khoảng 200 mục, gồm cả tiếng Việt và thuật ngữ Anh hay gặp.
  - Cộng thêm category catalog của các span `<en>`.
- Luật:
  - Mỗi keyword được 1 điểm, tối đa 3 lần một keyword; thêm 1 điểm cho mỗi thuật ngữ catalog có category trùng domain.
  - Domain thắng khi đạt ≥ 2 điểm và hơn domain thứ hai ≥ 1 điểm; không thì `other`.
  - `--min-score` và `--margin` là tham số, **chưa kiểm chứng**.
- Chỉ suy cho dòng có domain `conversation` / `other` / rỗng (domain mặc định của converter). Dòng được suy ghi `domain_source: inferred`, giữ domain cũ ở `domain_orig`, kèm điểm từng domain ở `domain_scores`.
- Đã áp:
  - `vss_cs` → `data/raw/vss_full/vss_cs_clean_dom.jsonl`;
  - `train_vss` (replay / chưng cất, 28 551 câu) → `data/raw/vss_full/train_vss_dom.jsonl`;
  - `rebuild_vss_cs.sh` có thêm bước này ở cuối.
  - Các bộ train khác đã có domain thật (VIVOS `read`, ViMedCSS `medical`, confusable), nên không suy.

| Domain suy ra | vss_cs (câu / giờ) | train_vss (câu / giờ) |
| --- | --- | --- |
| other | 924 / 3.38 | 25 136 / 90.42 |
| business | 50 / 0.18 | 496 / 1.83 |
| entertainment | 42 / 0.16 | 684 / 2.49 |
| tech | 40 / 0.15 | 397 / 1.46 |
| daily | 27 / 0.10 | 666 / 2.43 |
| edu | 18 / 0.06 | 479 / 1.75 |
| medical | 15 / 0.06 | 197 / 0.72 |
| **finance** | **15 / 0.05** | **400 / 1.50** |
| pubadmin | 2 / 0.01 | 96 / 0.35 |

**20 câu mẫu được suy ra là finance** (để anh/chị kiểm tra trước khi dùng `--holdout-domain finance`):
- `reports/domain_infer_train_vss.md`: đủ 20 câu, lấy trên 400 câu;
- `reports/domain_infer_vss_cs.md`: chỉ có 15 câu vì `vss_cs` chỉ có 15 câu finance.

Nhìn qua các câu mẫu: đa số đúng chủ đề đầu tư / chứng khoán. Có vài câu sai vì từ đa nghĩa, vd "đầu tư công sức", "thị trường" theo nghĩa chung.

`--holdout-domain finance` sẽ loại 15 câu (0.05 h) `vss_cs` và 400 câu (1.50 h) `train_vss`.

## E2. Phân bố domain theo nhóm

Bảng đầy đủ (theo domain: câu, giờ, số thuật ngữ, top 5) nằm ở `reports/domain_report.md`, sinh bằng `tools/domain_report.py`.

| Nhóm | Nguồn | Câu | Giờ | Thuật ngữ khác nhau | Domain | Top thuật ngữ |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | G-Group (ứng viên, trusted_text) | 1 330 | 1.38 | 1 518 | tech 0.46 h, edu, finance, pubadmin | website 30, ok 25, gov 24, code 18, online 15 |
| B2 | consensus (sau trần) | 218 | 0.40 | 134 | product 100% | zalo 10, sms 10, ok 10, check 9, email 9 |
| B3 | ViMedCSS hard (meta) | 658 | 1.38 | 582 | medical 100% | bilirubin 92, aldosterone 39, listeria 28 |
| cs_real | ViMedCSS train (meta) | 11 832 | 24.30 | 2 117 | medical 100% | colagen 289, glucose 251, hormone 221 |
| cs_pseudo | vss_cs 4.15 h (suy domain) | 1 133 | 4.15 | 665 | other 81%, business 4%, entertainment 4%, tech 4% | amazon 14, happy 13, mindset 13, victor 13 |

## E3. Đề xuất reserved_b3 ngoài y khoa: CHƯA áp dụng

File: `reports/b3_nonmed_terms.csv` (cột `selected`), sinh bằng `tools/b3_nonmed_terms.py`. Báo cáo: `reports/b3_nonmed_report.md`.

Em đề xuất **45 thuật ngữ**: tech 22, edu 15, pubadmin 8.
- Nguồn: G-Group tech / edu / pubadmin (không lấy finance) và `vss_cs`. Không lấy consensus: 50 thuật ngữ có trong consensus bị loại để giữ ở B2.
- Điều kiện:
  - tổng 3–20 câu, trong đó ≥ 2 câu G-Group;
  - ≥ 3 nguồn / người nói;
  - không nằm trong 500 từ Anh phổ biến nhất (google-10000), không có trong catalog, không được chốt hay đề xuất `vi` / `drop_tag`.
- Ngưỡng "phổ biến" chọn 500 để đạt 40–80 thuật ngữ: 2000 → 32 thuật ngữ, 1000 → 38, 500 → 45.

Danh sách: logic, edu, visa, event, singapore, scan, blockchain, background, chip, demo, harvard, gmail, token, master, captcha, status, apply, file xml, build, bachelor, hanoi, https, mba, usb token, set up, podcast, admin, diploma, container, hutech, body, official, layer, stem, log, toefl, module, topik, smart contract, clear, developer, docker, iot, password, rule.

Nên xem lại vài mục: `edu`, `hanoi`, `body`, `clear` khá chung chung; `edu` thường đi trong "gov edu vn".

**Nếu áp dụng** (thêm vào catalog với `reserved_b3 = 1`):

| | Bị ảnh hưởng |
| --- | --- |
| vss_cs 4.15 h | mất **56 câu, 0.20 h (4.8%)** |
| Kho `vss_cs_candidates` (117 h) | 895 câu, 3.27 h bị chặn ở bước lọc rẻ các lần sau |
| ViMedCSS train (cs_real) | 9 câu phải loại |
| G-Group B1 | 159 câu chuyển sang B3 (thuật ngữ không có trong train) |

## E4. mix_plan: 2 phương án trần domain (2 biến thể E3)

Lệnh: `tools/mix_plan.py --fit-cap`. Đầu vào: `vss_cs` 4.15 h đã suy domain, ViMedCSS 24.30 h (giả định từ meta), VIVOS train, confusable train.
- Báo cáo: `reports/mix_plan_cap35.md` / `reports/mix_plan_cap50.md`.
- Trọng số theo (nhóm, domain): `reports/mix_plan_cap35.json` / `reports/mix_plan_cap50.json`.

Cách tính:
- Domain vượt trần bị cắt về đúng trần, phần dư chia cho các domain khác theo tỉ lệ tự nhiên.
- `other` được miễn trần vì đó không phải một domain thật.
- Tỉ lệ tự nhiên: medical **78.1%**.

| | Trần 35% | Trần 50% |
| --- | --- | --- |
| medical (ViMedCSS) | 35.0% (lặp 0.41×) | 50.0% (lặp 0.58×) |
| other (vss_cs) | 53.7% | 41.3% |
| các domain vss_cs còn lại | 11.3% | 8.7% |
| Hệ số lặp của audio vss_cs | **4.52×** | **3.48×** |

Hệ quả:
- Muốn medical ≤ 35% thì mỗi giờ audio `vss_cs` (nhãn giả) phải được lấy mẫu gấp 4.5 lần trung bình, nên dễ overfit vào nhãn giả.
- Với 6 h `vss_cs`, hệ số này giảm còn khoảng 3.2× (trần 35%) và 2.5× (trần 50%), theo ước tính tuyến tính.
- Pipeline train (Phase 3) chưa đọc trọng số theo domain; file JSON để dùng sau.
