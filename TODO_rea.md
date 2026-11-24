# TODO_rea: phân tích cơ chế Gipformer 1.5 (hướng A)

Cập nhật: 24/11/2026 (sau audit) · Nhánh `feture/rea` · Code ở `src/gipformer_mech`, script ở `scripts/0x_*.py` và `scripts/diagnostics/fpt_*.py`, kết quả ở `reports/phase*.md` · Cách chạy ở [IMPLEMENTATION.md](IMPLEMENTATION.md).

Danh sách của hướng B (fine-tune đáng tin) ở [TODO.md](TODO.md), không trộn vào đây. Cách dùng: làm từ trên xuống. Mục gắn **👤** là việc chỉ bạn quyết định hoặc làm được.

Dữ liệu và manifest đã khoá nằm ở worktree `.worktrees/feture-rea/` (không vào Git). Luật chung: heldout không dùng để thiết kế hay chọn, mọi ngưỡng đăng ký trước khi chạy, mọi con số ghi kèm cỡ mẫu.

---

## Câu hỏi chính: model đang có bug hay gap, ở đâu?

Phạm vi: lỗi từ Anh (CS) trên FPT, pool dev, 365 span trong 279 câu, greedy INT8, context là prefix greedy đã phát. Nhóm lấy theo `logit_lens.classify`, ngưỡng top-10. Đây là nhãn vận hành theo rank, không phải chứng minh nhân quả. ([phase2_pool_near_miss.md](reports/phase2_pool_near_miss.md))

| Nhóm | Nghĩa (theo code) | Tỉ lệ | Thuộc loại | Đã thử sửa tĩnh |
|---|---|---:|---|---|
| Near miss | Mọi piece đã ở top-10 tầng cuối nhưng vẫn sai | 50,1% | **bug sau encoder**: Joiner chấm blank cao hơn piece (117 span blank đứng đầu) | Hotword, blank penalty toàn cục và có điều kiện, tắt đơn vị Joiner: đều trượt |
| Encoder bug | Piece lên top-10 ở stack giữa rồi mất ở tầng cuối | 1,4% | bug trong encoder | Cổng quét encoder (25%) đóng |
| Predictor override | Top-10 theo âm học nhưng mất khi thêm context decoder | 1,1% | bug ở decoder | Chưa cần |
| True gap | Có `<unk>` hoặc không ở top-10 ở bất kỳ đâu | 2,2% | **gap** | Chỉ sửa được bằng dữ liệu hoặc tokenizer |
| Inconclusive · dây chuyền (3.2) | Một piece trước piece thiếu đã thua frame, lỗi lan sang context | 30,1% (110) | **bug sau encoder**, dạng dây chuyền | Chưa |
| Inconclusive · START + LOCAL (3.2) | Piece thiếu dù context đúng, hoặc là piece đầu của từ | 15,1% (55) | **gần gap**: từ hiếm, tên riêng (3.3) | Chưa |

**Kết luận tạm (dev, FPT):** lỗi chính là **bug sau encoder** (Joiner calibration), không phải gap và không nằm trong encoder. Bug này phân tán. Mười đơn vị Joiner chiếm phần lớn margin, nhưng tắt chúng làm đổi argmax ở 8–30% frame thường (pha 4), nên không sửa được bằng chỉnh tĩnh. Cách hợp lý nhất là train lại decoder và Joiner, tức **X2 của hướng B**.

**Điều còn chặn kết luận:** 45% inconclusive (165 span). Đã khớp theo từng span ở mục 3.1 ([phase2_inconclusive_breakdown.md](reports/phase2_inconclusive_breakdown.md)): nhóm này đúng là 157 span "một phần piece trong top-10" và 8 span "không piece nào"; 0 span có `<unk>`. Nó **không phải gap rõ ràng**:

- 90 span chỉ thiếu **một** piece, 43 thiếu hai. Piece thiếu thường ở rank 11–100: 35 span có mọi piece trong top-20, 67 trong top-50, chỉ 68 span có piece xấu nhất dưới rank 100.
- 72 / 165 (44%) có mọi piece vào top-10 **nếu predictor được cho prefix đúng**, và 70 trong 72 span đó piece đầu tiên bị mất không phải piece đầu của từ. Dạng này khớp với lỗi dây chuyền: một emission sai làm đổi context cho các piece sau. Đây là context oracle, nên nó đo "có sẵn" chứ không đo "sửa được".

Mục 3.2 ([phase2_inconclusive_cascade.md](reports/phase2_inconclusive_cascade.md)) kiểm giả thuyết dây chuyền theo luật đăng ký trước, và nó được xác nhận: 110 / 165 span inconclusive (67%) có một piece **trước** piece thiếu đầu tiên đã thua frame (rank 2–10), và 67 / 72 span đạt được với prefix đúng là loại này (93%, so với 46% ở các span còn lại). Chỉ 55 / 165 (33%, tức 15% của 365 span) mà piece thiếu đầu tiên thiếu dù context đúng hoặc nó chính là piece đầu của từ. Đây là phần gần với "gap" nhất, và là phần duy nhất mà X2 có thể không sửa được.

Hệ quả: tỉ lệ lỗi sau encoder là 51,2% theo nhãn `classify`, 60,8% nếu nới top-10 thành top-20, 71,0% nếu tính cả các span đạt được với prefix đúng. Ba con số này là ba cách đọc, không phải ba lần đo. Phần gap thật (true gap + piece xa dưới rank 100) nhỏ hơn nhiều so với dự đoán ban đầu. Ghi chú: gốc của các chuỗi lỗi không chỉ là blank. Ở 110 span dây chuyền, piece đầu tiên thua frame bị blank dẫn đầu ở 48 span và bị một piece khác dẫn đầu ở 62 span. Ở near miss là 117 / 183 span blank dẫn đầu (64%). Nên bug gồm cả "blank át piece" lẫn "piece sai át piece đúng", đều nằm ở Joiner và decoder.

**Quy tắc chọn A hay B** (👤 chốt trước, vì cách đọc nào được tính là "sau encoder" quyết định kết quả nằm trên hay dưới 70%: 60,8% và 71,0% nằm hai phía ngưỡng, và ngưỡng 70% do mình đề xuất trước khi có các con số này):

- Nếu lỗi sau encoder chiếm phần lớn (≥ ~70% theo cách đọc bạn chọn): đi **X2** (chỉ decoder + Joiner) trước. Hướng A chỉ còn trigger học được (mục 2.2), và chỉ khi đủ sự kiện.
- Nếu không: X2 không đủ. Cần **X3** (FT với CS thật) và có thể `tokenizer_check` / O1 nếu piece thiếu tập trung ở BPE hiếm. Hướng A dừng ở chẩn đoán.
- Mình nghiêng về cách đọc "top-10 cộng prefix đúng" thành **bằng chứng ủng hộ X2** nhưng không dùng nó để tuyên bố vượt ngưỡng. Phép thử thật là mục 4.1 (X2 có sửa được các span này không).
- Dự đoán đăng ký trước cho X2: X2 phải sửa được một phần near miss trên danh sách span đã khoá (mục 4.1). X2 cải thiện WER mà không sửa near miss nghĩa là cơ chế giả định sai.

**Kiểm chứng chéo ngoài FPT** (24/11). Cùng luật, dự đoán đăng ký và commit trước khi chẩn đoán (`scripts/diagnostics/cs_dev_near_miss.py`); không dùng bench khoá:

| Tập | Span | Near miss | Dây chuyền | Encoder + predictor | Gần gap | H1 / H2 / H3 |
|---|---:|---:|---:|---:|---:|---|
| FPT dev (đọc) | 365 | 50,1% | 30,1% | 2,5% | 17,3% | (gốc) |
| FPT pool mới | 1 487 | 51,3% | 29,9% | 4,0% | 14,8% | đạt / đạt / đạt |
| `dev_interim_cs` (nói tự nhiên, 8 domain) | 174 | 71,3% | 20,7% | 6,9% | 1,1%* | đạt / đạt / đạt |
| ViMedCSS train (y khoa, nhãn gốc) | 2 113 (mẫu 1 200 / 3 099 câu) | 42,5% | 31,6% | 3,2% | 22,6% | đạt / đạt / đạt |

\* `dev_interim_cs` chỉ giữ câu mà mọi span tiếng Anh đã được một ASR nghe ra, nên phần gap bị ép thấp từ lúc dựng tập và không so được. Báo cáo: [phase2_dev_interim_near_miss.md](reports/phase2_dev_interim_near_miss.md), [phase2_vimedcss_train_near_miss.md](reports/phase2_vimedcss_train_near_miss.md).

Đọc: bug sau encoder (near miss cộng dây chuyền) chiếm 74–92% ở cả bốn tập; encoder bug tối đa 4,0% ở mọi tập (ngưỡng mở quét encoder là 25%); dự đoán dây chuyền giữ ở cả bốn. Phần gap phụ thuộc domain: cao nhất ở y khoa (22,6%, tên thuốc, bệnh, giải phẫu), nên X3 (dữ liệu CS) quan trọng hơn ở domain có nhiều thuật ngữ hiếm.

Giới hạn: chỉ greedy, chưa có kết luận người nghe cho nhãn FPT, heldout chưa dùng cho chẩn đoán. Hai tập ngoài FPT đều là YouTube; chưa kiểm trên bench khoá (cố ý, để giữ chúng cho hướng B).

---

## Đang ở đâu (tóm tắt 1 phút)

- **Đã bị loại trên dev:** hotword (FAIL trên heldout), blank penalty toàn cục, blank penalty có điều kiện (word-continuation), tắt đơn vị Joiner tĩnh. Báo cáo pha 3, 3b, 3c, 4.
- **Chưa thử:** gate theo top-10, trigger học được, đo near miss được sửa theo checkpoint FT.
- **Nút thắt:** số sự kiện near miss. Pha 4 chỉ có 65 sự kiện tái lập. Pool mới đã chấm xong (1.1): 2 172 span lỗi CS trong 1 541 câu. Heldout mới đã khoá (1.2): 685 span heldout, 1 487 span để chẩn đoán. **Đã gỡ** (1.3, 1.4): pool mới có 654 sự kiện tái lập, pool dev 144. Bước kế tiếp là 👤 chọn ngưỡng sự kiện tối thiểu cho 2.2 và quyết thứ tự 2.1 / 2.2.

---

## 1 · Mở rộng tập near miss

Mục đích: đủ sự kiện để thử trigger mà không bị in-sample. **Thứ tự bắt buộc:** chấm xong, quyết định heldout, khoá heldout, rồi mới chẩn đoán. Chẩn đoán trước thì heldout mới bị nhiễm.

- [x] **1.1 Chấm baseline 20 316 dòng chưa lấy mẫu** — xong 24/11 11:37. `extra_summary.json`: 19 725 câu chấm (22,8 giờ), 591 dòng bị loại (285 trùng văn bản heldout, 39 trùng audio heldout, 264 trùng audio, 3 audio lỗi). 2 117 câu có token CS, 1 541 câu có lỗi CS, **2 172 span lỗi CS** (gấp ~6 lần pool dev). Greedy: WER 8,1%, CS-WER 81,0%. Sha256 `extra_scored` `344b7e51…6ad3`. (`scripts/diagnostics/fpt_extra_pool.py`, 13 shard × 1 600 dòng, khoảng 4–5 phút mỗi shard, greedy INT8 batch 16 như pool gốc). Kết quả ở `data/manifests_extra/`, log ở `artifacts/extra_pool.log`. Chạy lại script thì tự bỏ qua shard đã xong.
  - Xong khi: có `extra_summary.json` và 13 shard.
  - `shard_000`: 1 566 câu chấm, 34 dòng bị loại (33 trùng văn bản heldout), 74 span lỗi CS trong 61 câu.
  - Lúc audit (9 shard, 13 994 câu): 895 câu có lỗi CS. Sha256 của pool gốc và heldout vẫn khớp `heldout_lock.json`. Heldout chỉ được đọc để lấy hash, id và văn bản làm danh sách loại trừ.
  - ⚠ **Không ngoại suy từ shard dở.** Dataset có thứ tự: trong mẫu gốc, câu có CS chiếm 7,2% ở chỉ số ≤16 289 và 16,3% ở chỉ số cao hơn. Ước lượng "khoảng 128 sự kiện" là tạm, cập nhật khi có `extra_summary.json`.
- [x] **1.2 Heldout mới** (24/11, bạn chọn phương án B: 30% theo văn bản, cân bằng theo Set). Script `scripts/diagnostics/fpt_extra_split.py` (chạy một lần, từ chối ghi đè), test `tests/mech/test_extra_split.py`, lock `reports/phase1_extra_split_lock.json`. Dữ liệu ở `data/manifests_extra/extra_heldout.jsonl` và `extra_diagnostic.jsonl` trong worktree.
  - Luật: đơn vị là nhóm văn bản chuẩn hoá (`tokens(reference, "upper")`); tầng là Set × có `cs_tokens` trong nhãn (mình thêm tầng CS để hai phía có cùng mật độ CS; chỉ dùng nhãn, không dùng lỗi baseline); trong mỗi tầng sắp nhóm theo `sha256(salt + văn bản)`, salt `fpt-extra-heldout-v1`, lấy đến khi ≥ 30% số dòng. 1 056 dòng trùng văn bản pool dev bị **ép sang phía chẩn đoán**, không bỏ.
  - Kết quả: heldout 5 603 dòng, 471 câu có lỗi CS, **685 span** (sha `8ce672ce…7f70`); chẩn đoán 14 122 dòng, 1 070 câu, **1 487 span**, gồm 26 span của dòng trùng dev (sha `e193ae5d…10ee`). Tỉ lệ heldout ở bốn tầng 30,0–30,1%.
  - Mục đích dùng heldout đã đăng ký trong docstring, mỗi thứ kiểm một lần: bản tốt nhất của 2.1, trigger tốt nhất của 2.2, preset tên riêng, có thể thêm một lần lặp P1/P2 cho X2 (đăng ký riêng). 1.3, 1.4 và mọi bước thiết kế chỉ đọc `extra_diagnostic.jsonl`.
  - Giới hạn: không có nhãn speaker nên một người nói có thể ở cả hai phía; Set chỉ là đại diện cho đợt thu.
- [x] **1.3** (24/11) `fpt_pool_near_miss.py --extra` chẩn đoán 1 070 câu, 1 487 span của `extra_diagnostic.jsonl` (sha kiểm với lock tách). Report `reports/phase2_extra_near_miss.md` do `scripts/diagnostics/fpt_extra_report.py` viết, script tự kiểm rằng cột dev tái tạo đúng số đã công bố. **Tái lập pool dev:** near miss 51,3% (dev 50,1%), dây chuyền 29,9% (30,1%), encoder + predictor 4,0% (2,5%), gần gap 14,8% (17,3%). Dự đoán dây chuyền của 3.2 **giữ lại** trên pool mới: CHAIN 95,1% ở nhóm đạt được, 47,0% ở nhóm còn lại.
- [x] **1.4** (24/11) `scripts/diagnostics/fpt_event_replay.py` mã hoá lại audio bằng encoder ONNX và replay theo luật của script 08. Report `reports/phase4_event_replay.md`. **Sự kiện tái lập: pool mới 654 / 857 (76,3%, 389 câu, 360 từ), pool dev 144 / 204 (70,6%)**, so với 65 ở pha 4. Kiểm harness: trên cache encoder của pha 4, bug-mining ra lại đúng 65 / 104; trên encoder tính mới ra 66, vì encoder INT8 nhạy với sai số filterbank cỡ 6e-5 (lật 1 / 104 sự kiện). Ước lượng 62,5% cũ là thấp: câu không bị chọn tái lập 76–78%.

## 2 · Thử can thiệp còn mở

Đăng ký luật chọn trước khi chạy, ghi vào docstring script như các pha trước.

- [x] **2.1 Gate theo top-10: đóng bằng lập luận, không chạy** (24/11). Gate này trùng hệt blank penalty toàn cục mà pha 3b đã loại:
  - Greedy chọn argmax từng frame. Phạt blank chỉ đổi kết quả ở frame mà blank đang là argmax.
  - Ở những frame đó, piece không-blank tốt nhất đứng hạng 2, nên luôn nằm trong top-10. Gate vì vậy luôn bật đúng ở mọi frame mà penalty có tác dụng, và output giống hệt `uniform`.
  - Gate theo margin của chính frame (blank − piece tốt nhất ≤ δ) cũng quy về `uniform` với penalty min(λ, δ).
  - Vậy mọi gate theo rank hay margin trên logit của chính frame đều quy về `phase3b_blank_penalty.md`. Gate theo xác suất tuyệt đối không trùng hệt, chỉ gần, nên không gộp vào kết luận này.
  - Một gate khác được `uniform` phải dùng thông tin ngoài logit của frame: trạng thái ẩn của Joiner, hoặc context. Đó chính là 2.2.
- [ ] **2.2 Trigger học được trên Joiner** (gần với hook trong bài Qwen nhất). Chia fold theo bản ghi, theo từ **và theo văn bản chuẩn hoá**, để cùng một câu không nằm ở cả hai phía.
  - 👤 Chọn ngưỡng số sự kiện tái lập tối thiểu trước khi chạy. Dưới ngưỡng thì dừng, không thử.
  - Xong khi: tỉ lệ sửa so với random-direction cùng layer, cộng số câu clean bị hỏng. Bài Qwen báo "8/8 sửa, 0/8 hỏng, ngẫu nhiên 1,6/8 và 2,2/8".
- [ ] 👤 Quyết định có mở cổng quét encoder không. Mình không khuyến nghị: encoder bug chỉ 1,4% (ngưỡng 25%), và bằng chứng nói lỗi nằm sau encoder. Mở là bỏ một cổng đã đăng ký.

## 3 · Độ tin cậy của chẩn đoán (quyết định A hay B phụ thuộc mục này)

- [x] **3.1 Xác nhận inconclusive hiện nay là nhóm nào** (24/11). Khớp theo từng span trên bug-mining: 85 "all" = đúng 85 near miss, 81 "some" + 5 "none" = đúng 86 inconclusive, không span nào lệch. Pool còn lại (79 span) cho cùng bức tranh. Script `scripts/diagnostics/fpt_inconclusive_breakdown.py`, kết quả `reports/phase2_inconclusive_breakdown.md` (chưa commit). Dùng rank đã lưu, không cần model.
- [x] **3.2 Kiểm giả thuyết dây chuyền trên inconclusive** (24/11). Dự đoán đăng ký trước (dây chuyền chiếm hơn nửa số span đạt được với prefix đúng và nhiều hơn ở các span còn lại): **xác nhận**. 110 / 165 span là dây chuyền (67 / 72 ở nhóm đạt được, 43 / 93 ở nhóm còn lại), 33 là START (piece đầu của từ đã ngoài top-10), 22 là LOCAL (piece thiếu đầu tiên nằm sau các piece rank 1). Gốc dây chuyền: 48 blank dẫn đầu, 62 piece khác dẫn đầu. Script `scripts/diagnostics/fpt_inconclusive_cascade.py`, kết quả `reports/phase2_inconclusive_cascade.md` (chưa commit). Giới hạn: rank 1 là rank tốt nhất trong cửa sổ, không chứng minh piece đã được phát; prefix đúng là context oracle. Sau 1.3 chạy lại trên pool mới.
- [x] **3.3 Hồ sơ của phần gần "gap"** (24/11, **thăm dò**, không đăng ký trước vì các đặc trưng được chọn sau khi đọc 14 span). Kết quả ở `reports/phase2_gap_profile.md` (chưa commit), script `scripts/diagnostics/fpt_gap_profile.py`. So 5 nhóm: near miss (183), dây chuyền đạt được (67), dây chuyền không đạt được (43), START + LOCAL (55), true gap (8).
  - **Không có dấu hiệu tokenizer.** Piece yếu nhất có tối đa 2 chữ cái ở 98–100% span ở mọi nhóm kể cả near miss, và số piece mỗi từ (trung vị 4–5) không khác. Từ Anh bị tách thành chữ cái là chuyện chung, không đặc thù của gap. Nên **chưa có lý do làm `tokenizer_check` / O1** từ bằng chứng này.
  - **Đặc trưng khác biệt** (so với near miss): từ chỉ xuất hiện đúng 1 dòng trong dev (91% ở START + LOCAL, 69% ở near miss), rank âm học của piece yếu nhất dưới 100 (69% so với 38%), và blank dẫn đầu ở piece yếu nhất (71% so với 49%). Rank âm học xa dần theo thứ tự near miss, chain đạt được, START + LOCAL, chain không đạt được, true gap.
  - **Đọc từ cụ thể** (62 từ của nhóm gap_like, đọc tay): khoảng một nửa là tên riêng, địa danh, thương hiệu (AOYAGI, BALTIMORE, JUVENTUS, KHRUSHCHEV, PEUGEOT, GUARDIOLA, EDINBURGH…), phần còn lại là từ Anh thông dụng (AIR, DEAR, DID, MAKE, GIRL, THANKS…) và viết tắt (DC, MM, K+). Chưa phân loại có hệ thống.
  - **Điều chưa biết:** các từ này có hiếm trong dữ liệu train của Gipformer không, vì chỉ có tần suất trong dev. Nên chỉ nói được: phần gap-like gồm từ hiếm và tên riêng nghe một lần, nhiều khả năng là độ phủ từ vựng và âm học chứ không phải lỗi cơ chế. Cần X3 hoặc dữ liệu CS để biết, và X2 có thể không sửa được.
  - Việc có thể làm sau: gán nhãn tên riêng / từ thông dụng / viết tắt cho 62 từ, 👤 nghe một mẫu để xác nhận nhãn tham chiếu của chúng (nhãn sai cũng cho ra "gap" giả).
- [ ] 👤 Nghe xác nhận nhãn 24 bản ghi trong [phase2_inconclusive_review.md](reports/phase2_inconclusive_review.md) (ASR chỉ giúp phân loại, chưa có kết luận người nghe).
- [ ] Chạy lại bug-mining trong môi trường hiện tại để đối chiếu với rank đã lưu. Lượt đếm toàn pool lấy nhóm bug-mining từ rank lưu, nên chưa có phép kiểm môi trường độc lập ngoài sự khớp giữa hai nhóm.

## 4 · Nối với hướng B

- [ ] **4.1** Khi X2 (chỉ train decoder + Joiner) có checkpoint, báo **số near miss được sửa** bên cạnh WER, trên cùng bộ span đã khoá của pool dev.
  - [x] **Danh sách span đã khoá** (24/11): `reports/phase2_span_lock.jsonl` (365 span, 279 bản ghi) và `reports/phase2_span_lock.json` (sha256 của danh sách là `5e988226…3453`, pool dev `6d8d6986…ef5`). Nhóm: near_miss 183, chain 110, other_post_encoder 9, gap_like 63. Nhóm "mechanism" = near_miss + chain = 293, nhóm đối chiếu = gap_like = 63. Script `scripts/diagnostics/fpt_span_lock.py` chỉ chạy một lần (từ chối ghi đè).
  - [x] **Dự đoán và định nghĩa ghi trong docstring của script**: span "sửa được" khi giả thuyết của checkpoint không còn lỗi CS tại đúng `word_index`; tỉ lệ sửa theo nhóm kèm khoảng tin cậy bootstrap **ghép cặp** theo bản ghi; điều khiển lặp bằng model gốc trong cùng harness, **luôn chạy**; **đối chiếu tiếng Việt** (lỗi từ không-CS của baseline trong cùng 279 bản ghi); **BROKEN** (từ CS baseline đúng mà checkpoint làm sai) luôn báo kèm. P1a (mechanism cao hơn điều khiển lặp), P1b (mechanism cao hơn đối chiếu tiếng Việt, thay cho ngưỡng độ lớn), P2 (mechanism cao hơn gap_like). BROKEN ≥ số span mechanism được sửa thì báo là churn, không tính là ủng hộ. Báo mô tả riêng near_miss, chain đạt được (67) / không đạt được (43), other_post_encoder. Điều kiện: dữ liệu train của checkpoint không chứa bản ghi hoặc văn bản của pool dev và heldout; ghi thành phần dữ liệu train (có văn bản CS không, từ gap_like nào xuất hiện), nếu có thì báo thêm P2 trên các span gap_like có từ không xuất hiện trong train. (Sửa 24/11 theo review.)
  - [x] **Đã duyệt và commit** (24/11, commit `5bd1e23`, trước khi có checkpoint nào được giải mã). Commit gồm `fpt_span_lock.py`, `fpt_inconclusive_cascade.py` (script khoá import nó), `phase2_span_lock.jsonl` và `phase2_span_lock.json`. Từ đây không sửa định nghĩa hay dự đoán; muốn đổi thì đăng ký lại trong một commit mới, ghi rõ lý do.
  - [ ] Chưa có mã chấm (đếm span sửa được, bootstrap, điều khiển lặp). Viết khi X2 gần có checkpoint, kèm test, và không đổi định nghĩa ở trên.
  - [ ] Kiểm rò: văn bản và bản ghi của pool dev FPT không nằm trong dữ liệu train X2 (cổng leak hiện chỉ kiểm bench).
- [ ] Preset tên riêng (ANDY MURRAY, CHAPLIN, CROWELL, SMITH): chỉ còn nghĩa nếu có heldout mới. Giữ ngoài `configs/bench/`.

## 5 · Dọn kỹ thuật

- [x] **Commit** (24/11) theo ba commit: danh sách khoá `5bd1e23`, pha 4 `d29266e`, rồi các script và report pool cùng `IMPLEMENTATION.md` và `TODO_rea.md`. `.worktrees/` không commit. Trước khi commit, 14 hàm test của `tests/mech` đều đạt trong WSL (`~/.venvs/gipbench-ft`, PYTHONPATH `src:~/jiwer_target`). Không venv nào có pytest, nên các hàm test được gọi trực tiếp bằng một runner tạm.
- [ ] **Thiếu test** cho hai script mới. Cần ít nhất:
  - pool bù đúng bằng phần bù của mẫu gốc;
  - lọc trùng audio và văn bản heldout;
  - merge giữ thứ tự và bỏ trùng giữa các shard;
  - nếu thêm luật ở 1.2: lọc trùng văn bản dev và gom văn bản lặp khi tách;
  - resume khi `shard_*.dropped.jsonl` bị ghi dở. Hiện có file đó là shard được coi là xong, nên một lần ngắt giữa lúc ghi sẽ để lại shard thiếu dòng. Nên ghi ra file tạm rồi đổi tên.
- [x] Ghi vào IMPLEMENTATION.md cách chạy `fpt_extra_pool.py` và `fpt_extra_split.py` cùng kết quả (24/11).
- [ ] Môi trường WSL `gipbench-ft` thiếu `jiwer` và `kaldi-native-fbank`. Hiện cài ở `~/jiwer_target` và thêm vào PYTHONPATH khi chạy. Python Windows gốc không có numpy và pytest, nên test `tests/mech` phải chạy trong WSL (4/4 file đạt lúc audit). Cần ghi cách cài chính thức hoặc thêm vào extra `mechanistic`.
- [ ] Hypothesis greedy phụ thuộc thành phần batch (đã biết từ pha 3c). Mọi lượt chấm lại phải giữ batch 16 và thứ tự chỉ số.

---

## Đã đóng, không làm lại

| Hướng | Kết quả | Báo cáo |
|---|---|---|
| Chẩn đoán bug hay gap | 47,5% near miss ở bug-mining, 50,1% ở toàn pool | `phase2_diagnosis.md`, `phase2_pool_near_miss.md` (chưa commit) |
| Hotword biasing | Dev tốt hơn, heldout FAIL (CS-WER −0,66, CI [−1,79; +3,41]) | `phase3_hotwords.md`, `final_acceptance.md` |
| Blank penalty toàn cục | Không penalty nào qua luật; sửa tối đa 6/83 near miss | `phase3b_blank_penalty.md` |
| Blank penalty có điều kiện | Sửa tối đa 4/85, vẫn làm hỏng câu clean | `phase3c_conditional_blank.md` |
| Tắt đơn vị Joiner tĩnh | 10 đơn vị mang margin thật nhưng đổi argmax ở 8% đến 30% frame | `phase4_joiner_units.md` (chưa commit) |
| LLM viết weight (weight-space programming) | Chưa làm, khả năng thấp theo kết quả pha 4 | |
