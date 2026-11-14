# Spec: dữ liệu, metric & độ tin cậy

Nov 14, 2026 · @God Goose

Một phương án fine-tune chỉ được coi là tốt hơn khi nó thắng trên cùng một bench đã đóng băng, được chấm bằng cùng một định nghĩa metric, với khoảng tin cậy không chứa 0 và không vi phạm ràng buộc nào; tab này đặc tả từng thành phần của câu đó.

## 1. Phạm vi & định nghĩa

Mục tiêu duy nhất của bench là đo Gipformer 1.5 và các bản cải tiến nhận từ tiếng Anh trong lời nói tiếng Việt tốt đến đâu, mà không làm hỏng tiếng Việt thuần.

| Thuật ngữ | Định nghĩa trong dự án |
| --- | --- |
| Code-switching (CS) | Utterance tiếng Việt có chèn từ hoặc cụm tiếng Anh (chèn trong câu). Câu hoàn toàn tiếng Anh nằm ngoài phạm vi |
| Span tiếng Anh | Một từ hoặc cụm liền nhau được đánh dấu `<en>…</en>` trong reference |
| Thuật ngữ (term) | Dạng chuẩn của một span, lấy từ catalog (vd `facebook`, `pull request`) |
| Mức Việt hoá | Cách người nói phát âm span: `strong` (“phây búc”), `mid`, `native` (gần bản ngữ) |
| Từ mượn đã Việt hoá | Có trong từ điển tiếng Việt (cà phê, xăng, ga): coi là tiếng Việt, không tag |
| Strict / lenient | Strict: hypothesis phải ra đúng chính tả Anh. Lenient: chấp nhận thêm phiên âm Việt trong catalog (chỉ để chẩn đoán) |
| Hệ thống (system) | Một cấu hình model + giải mã + hậu xử lý cụ thể, tên `eN_mô-tả` |

Ngoài phạm vi: dấu câu, viết hoa, diarization, streaming. Model xuất chữ in hoa không dấu câu, và bộ chấm chuẩn hoá các yếu tố này trước khi so.

## 2. Dữ liệu bench

Bench gồm 5 split, mỗi split trả lời một câu hỏi. Bản v0 ghép từ dataset công khai để chạy baseline ngay; bản v1 bổ sung dữ liệu tự thu cho B2 và B4 và là bản dùng để ra quyết định.

| Split | Câu hỏi | Nguồn v0 (công khai) | Nguồn v1 (bổ sung) | Tiêu chí chấp nhận v1 |
| --- | --- | --- | --- | --- |
| B1 Real-CS General | Nhận từ Anh trong lời nói tự nhiên? | Câu có từ Anh trong vi-asr-tech/edu/finance-test; 1–2 h từ SEA en-vi đã sửa nhãn tay | Thêm nguồn ngoài domain của G-Group | ≥ 3 h, ≥ 150 người nói, người nói lớn nhất ≤ 10% số giờ, mỗi miền ≥ 20% |
| B2 Real-CS Domain | Tốt trên đúng sản phẩm? | Không có nguồn công khai thay thế | Audio sản phẩm, đã ẩn PII, gõ transcript từ đầu | ≥ 2 h; ≥ 20% mẫu lấy ngẫu nhiên (không chọn theo model) |
| B3 Unseen Terms | Tổng quát sang thuật ngữ chưa thấy? | ViMedCSS split hard (658 câu, 1.38 h) | Câu chứa thuật ngữ `reserved_b3` từ B1/B4 | ≥ 1.5 h; 100% thuật ngữ B3 xuất hiện 0 lần trong mọi dữ liệu train |
| B4 Read-CS Controlled | Lỗi nằm ở loại từ, miền, mức Việt hoá nào? | Không có | 40–60 người đọc kịch bản \~600 thuật ngữ, 2 loại mic | ≥ 2.5 h; mỗi miền ≥ 20%; 100% span có mức Việt hoá |
| B5 Mono-VI Regression | FT có làm hỏng tiếng Việt? | vivos test, Common Voice vi test, vi-asr-pubadmin-test | 30 phút call center của bạn | ≥ 2 h; không có span `<en>` |

Tiêu chí toàn bộ B1–B4: ≥ 4.000 span tiếng Anh, ≥ 1.000 thuật ngữ khác nhau, ≤ 5% span ngoài catalog. eda.py kiểm tra tự động tất cả các tiêu chí này.

Ngoài 5 split, báo cáo thêm **ViMedCSS test** (1.615 câu) làm bench ngoài để so với số liệu đã công bố trong paper ViMedCSS; không dùng nó cho luật chọn.

Giới hạn của v0: thiếu B2 và B4; nguồn G-Group không có trường người nói (bootstrap phải theo `source` hoặc theo câu); nhãn ViMedCSS do Gemini tạo (κ = 0.65); các bộ vi-asr-\* do chính nhóm làm Gipformer tạo ra. Vì vậy kết luận trên v0 chỉ là định hướng; quyết định cuối dựa trên v1.

## 3. Manifest, catalog, transcript & chuẩn hoá

Ba hợp đồng này được version cùng bench; đổi một trong ba là đổi phiên bản bench.

**Manifest** (JSONL, một utterance mỗi dòng; bản đầy đủ ở SCHEMA.md trong gipbench):

| Trường | Bắt buộc | Ràng buộc |
| --- | --- | --- |
| `id` | có | Duy nhất toàn bench |
| `split` | có | B1–B5 (bench); tuỳ ý (train/dev) |
| `audio`, `duration` | có | Mono, ≥ 8 kHz; duration khớp audio ± 0.1 s; 1–30 s |
| `text_tagged` | có (bench) | Theo quy ước transcript bên dưới; span Anh trong `<en>…</en>`, không lồng, không rỗng |
| `text` | có (train) | Như trên nhưng không cần tag |
| `speaker`, `source` | có | ID ổn định trên toàn bộ dữ liệu, kể cả train |
| `region`, `gender`, `channel`, `domain` | nên có | Giá trị trong danh sách cố định; thiếu ghi `unk` |
| `en_meta[i].viet_level` | nên có | `strong` / `mid` / `native`, theo thứ tự span |

**Bộ giá trị `domain` chung** (mọi converter phải map về đây): tech, finance, edu, medical, pubadmin, business, daily, entertainment, product, callcenter, conversation, read, confusable, other. Dữ liệu train không có nhãn domain (SEA, VietSuperSpeech, crawl) được suy ra từ text: từ khoá theo domain trong `configs/domain_keywords.tsv` cộng category của thuật ngữ catalog trong câu; không chắc thì ghi other, kèm `domain_source: inferred`.

**Catalog** (TSV): `term`, `category`, `variants` (biến thể chính tả được coi là đúng), `phonetic` (phiên âm Việt, chỉ dùng cho lenient), `reserved_b3`. Không đưa vào `variants` một từ tiếng Việt có nghĩa riêng (vd “tim” cho team) — nó sẽ sửa nhầm cả câu không liên quan.

**Quy ước transcript**:

| Tình huống | Ghi |
| --- | --- |
| Từ Anh, dù phát âm kiểu Việt | Chính tả Anh, tag `<en>`: `<en>meeting</en>` |
| Từ mượn đã có trong từ điển Việt | Chính tả Việt, không tag |
| Viết tắt đánh vần | Viết liền: `<en>api</en>`, `<en>kpi</en>` |
| Số | Dạng đọc, chốt một cách: mốt, lăm, bốn, linh |
| Số trong tên sản phẩm | `<en>iphone</en> mười lăm` |
| Không nghe rõ | `<unk>`; câu có `<unk>` trong span Anh bị loại khỏi bench |

**Chuẩn hoá trước khi chấm** (áp giống hệt cho reference và hypothesis, theo thứ tự):

1. Unicode NFC, chữ thường.
2. Số thành chữ tiếng Việt: `25.990.000` → hai mươi lăm triệu chín trăm chín mươi nghìn; `3,5` → ba phẩy năm; `%` → phần trăm; số bắt đầu bằng 0 đọc từng chữ số.
3. Tách chữ dính số (`iphone15` → iphone mười lăm); gạch nối thành khoảng trắng; bỏ dấu câu (giữ dấu nháy trong từ).
4. Áp catalog: khớp n-gram dài nhất → dạng chuẩn (strict dùng `variants`; lenient dùng thêm `phonetic`).
5. Tách token theo khoảng trắng (mỗi âm tiết tiếng Việt là một token).

## 4. Gán nhãn & QA

Mọi utterance vào bench đều phải do người gán hoặc người sửa rồi người khác duyệt; nhãn máy (kể cả nhãn có sẵn trong dataset công khai) chỉ là bản nháp.

1. **Nháp**: transcript từ 2–3 model khác nhau; tự động đề xuất tag `<en>` theo catalog. B2 là ngoại lệ: gõ từ đầu, không dùng nháp, để nhãn không thiên về model đang được đo.
2. **Vòng 1**: người gán sửa transcript theo quy ước, sửa tag, chọn mức Việt hoá cho từng span, đánh dấu câu không dùng được.
3. **Vòng 2**: người thứ hai duyệt; bất đồng do trưởng nhóm gán nhãn quyết và ghi vào sổ quy ước.
4. **Đo đồng thuận**: 10% mẫu (phân tầng theo split) do hai người gán độc lập.

| Chỉ số QA | Ngưỡng | Nếu không đạt |
| --- | --- | --- |
| Đồng thuận trên span tiếng Anh (F1 khớp thuật ngữ) | ≥ 0.8 | Sửa hướng dẫn, gán lại lô đó |
| WER giữa hai người gán, phần tiếng Việt | ≤ 3% | Như trên |
| Đồng thuận mức Việt hoá | ≥ 0.6 (κ) | Giảm còn 2 mức (strong / khác) |
| Tỉ lệ lỗi nhãn ước lượng (duyệt lại 200 câu ngẫu nhiên sau đóng băng) | Ghi vào báo cáo | Dùng làm sàn nhiễu nhãn (mục 8) |

Mỗi utterance ghi `annotator` và `reviewer` để truy vết khi cần sửa.

## 5. Dữ liệu train & chống rò rỉ

Dữ liệu train được phép nhiễu hơn bench, nhưng tuyệt đối không được chạm vào bench: cùng người nói, cùng video, cùng audio, cùng câu hoặc thuật ngữ B3 đều bị loại.

| Nguồn | Vai trò | Nhãn | Xử lý trước khi dùng |
| --- | --- | --- | --- |
| ViMedCSS train (24.3 h) | CS thật, y tế | Gemini, kiểm tra mẫu | Dùng trực tiếp; split hard đã loại sẵn thuật ngữ của nó khỏi train |
| ViMedCSS validation | Dev CS | Như trên | Chỉ dùng để chọn siêu tham số |
| SEA Code-Switching en-vi (45.3 h) | CS thật, đa domain | Máy, chưa kiểm tra | Giữ câu khi span Anh của nhãn gốc được ít nhất 1 trong 2 model đa ngôn ngữ (faster-whisper large-v3, Qwen3-ASR) xác nhận, tức 2 nguồn đồng thuận (tham số --min-agree); không dùng Gipformer để xác nhận vì nó hiếm khi ra chính tả Anh; bỏ câu có video trùng bench |
| VietSuperSpeech (267 h) | Replay tiếng Việt hội thoại | Pseudo-label Zipformer-30M | Kiểm tra cách viết từ Anh trước; nếu bị phiên âm thì chỉ dùng câu không có từ Anh |
| vivos / Common Voice train | Replay tiếng Việt đọc | Người | Không dùng phần test (đã thuộc B5) |
| TTS tổng hợp (E4) | CS có kiểm soát, thuật ngữ hiếm | Chính xác theo kịch bản | Cấm thuật ngữ `reserved_b3`; speaker = ID giọng TTS |

Quy tắc chống rò rỉ (validate.py kiểm tra, to\_lhotse.py tự loại khi tạo dữ liệu train):

1. `source` dùng một quy ước chung cho mọi dataset: `youtube:<video_id>`, `call:<id>`, `read:<phiên>`. Nhờ đó một video xuất hiện ở cả SEA và ViMedCSS vẫn bị phát hiện.
2. Không người nói, video, hash audio hay câu (sau chuẩn hoá) nào của bench có trong train/dev.
3. Thuật ngữ `reserved_b3` không có trong mọi text train, kể cả text dùng cho LM (E2) và câu TTS (E4).
4. Dev dùng chọn siêu tham số tách theo `source`, không trùng bench.
5. Text train viết theo cùng quy ước và cùng kiểu chữ với tokenizer (Gipformer: IN HOA).

**Replay tự chưng cất** (mục 10): audio tiếng Việt không nhãn được Gipformer gốc giải mã hai lần (greedy và beam search, bằng decode.py); `make_distill.py` đọc hai file hyps.jsonl và chỉ giữ câu hai lần giống hệt nhau. Nguồn audio theo thứ tự ưu tiên: audio call center của bạn, rồi audio VietSuperSpeech (bỏ nhãn cũ, gán lại bằng Gipformer gốc); loại mọi source có trong bench.

## 6. Metric

Metric chính là CS-WER strict trên B1+B2; mọi metric gộp theo kiểu micro (tổng lỗi chia tổng token), không lấy trung bình theo câu.

Với S, D, I là số thay thế, xóa, chèn sau căn chỉnh Levenshtein theo token; N là số token reference:

```latex
\mathrm{WER} = \frac{S + D + I}{N}
\qquad
\mathrm{CS\text{-}WER} = \frac{S_{en} + D_{en} + I_{en}}{N_{en}}
\qquad
\mathrm{N\text{-}WER} = \frac{S_{vi} + D_{vi} + I_{vi}}{N_{vi}}
```

S và D thuộc phần của token reference bị lỗi. Một lỗi chèn thuộc phần tiếng Anh nếu một trong hai token reference kề bên là tiếng Anh (ưu tiên bên trái). Vì vậy “meeting” nhận thành “mít tinh” tính 2 lỗi cho span đó, và CS-WER có thể vượt 100%.

| Metric | Tính trên | Vai trò trong quyết định |
| --- | --- | --- |
| CS-WER strict | B1+B2 | Xếp hạng, kiểm định cải thiện |
| CS-WER lenient | B1+B2 | Chẩn đoán: lenient − strict = lỗi chính tả |
| N-WER | B1–B4 | Ràng buộc: tăng ≤ 0.3 điểm |
| WER | B5 | Ràng buộc: tăng ≤ 0.3 điểm |
| CS-WER | B3 | Phá hoà khi xếp hạng; đo tổng quát |
| CS-WER | B4, theo lát | Chẩn đoán theo miền, mức Việt hoá, category |
| Term recall / precision | B1–B4 | Chẩn đoán bias |
| Chèn sai thuật ngữ / 100 câu | B5 | Chẩn đoán over-biasing |
| RTF (CPU, int8, cùng số luồng) | toàn bench | Ràng buộc: ≤ 1.2× baseline |

```latex
\mathrm{Recall}_{term} = \frac{\sum_{u}\sum_{t}\min\big(c^{ref}_{u}(t),\,c^{hyp}_{u}(t)\big)}{\sum_{u}\sum_{t} c^{ref}_{u}(t)}
\qquad
\mathrm{Precision}_{term} = \frac{\sum_{u}\sum_{t}\min\big(c^{ref}_{u}(t),\,c^{hyp}_{u}(t)\big)}{\sum_{u}\sum_{t} c^{hyp}_{u}(t)}
```

c(t) là số lần thuật ngữ t xuất hiện trong câu u; phía hypothesis đếm mọi thuật ngữ catalog mà nó đưa ra, nên precision thấp nghĩa là model chèn thuật ngữ không có trong lời nói.

**Khoảng tin cậy**: bootstrap cụm theo người nói, 1.000 lần, lấy phân vị 2.5% và 97.5%. Khi so hai hệ thống, cùng một mẫu người nói được dùng cho cả hai (bootstrap cặp), rồi lấy CI của hiệu số. Lấy mẫu theo người nói chứ không theo câu, vì các câu của cùng một người không độc lập với nhau.

## 7. Phương pháp & giao thức so sánh

Mỗi phương pháp là một giả thuyết về nguồn gốc lỗi và phải chỉ ra metric nào sẽ thay đổi nếu giả thuyết đúng; chỉ đổi một thành phần so với hệ thống mà nó so sánh.

| ID | Giả thuyết | Thành phần thay đổi | Dữ liệu cần | Chi phí | Metric kỳ vọng thay đổi | Rủi ro cần canh |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | Mốc so sánh | — | — | Thấp | — | — |
| E1 | Model nghe được nhưng ưu tiên từ tiếng Việt | Giải mã (hotwords) | Catalog | Thấp | Term recall ↑, CS-WER B3 ↓ | Chèn sai trên B5 ↑ |
| E2 | Thiếu ngữ cảnh ngôn ngữ CS | LM ngoài | Text CS | Trung bình | CS-WER B1 ↓ | N-WER ↑, RTF ↑ |
| E3 | Thiếu dữ liệu âm thanh CS | Trọng số model (FT) | CS thật + replay tiếng Việt | Trung bình | CS-WER B1+B2 ↓ mạnh | B5 ↑ (quên tiếng Việt) |
| E4 | Thiếu độ phủ thuật ngữ hiếm | Dữ liệu FT (+ TTS) | Câu sinh + TTS | Trung bình | CS-WER B3, B4 ↓ | Overfit giọng TTS: B2 không đổi |
| E5 | Như E4, rẻ hơn | Dữ liệu FT (ghép audio) | Audio thuật ngữ + forced alignment | Trung bình | Như E4 | Tiếng nối ghép |
| E6 | Vocab xé vụn / thiếu chữ F, J, W, Z | Tokenizer + lớp ra | Như E3, nhiều hơn | Cao | CS-WER strict ↓, lenient − strict ↓ | B5 ↑, cần FT lâu |
| E7 | FT và bias bổ sung cho nhau | FT + hotwords | Như E3 | Thấp (sau E3) | B1+B2 và B3 cùng ↓ | Như E1 |
| E8 | Lỗi còn lại là chính tả | Hậu xử lý | Cặp hyp → ref | Trung bình | Lenient − strict → 0 | Sửa nhầm: N-WER ↑ |
| E9 | Cần bias học được, theo ngữ âm | Module bias (TCPGen) | Như E4 | Cao | CS-WER B3 ↓ | Độ phức tạp deploy |

Giao thức để so sánh công bằng:

1. Cùng phiên bản bench, catalog và bộ chuẩn hoá; baseline E0 chạy lại mỗi khi bench đổi phiên bản.
2. Cùng đường chạy production: sherpa-onnx int8, modified beam search beam 4, cùng số luồng và cùng máy khi đo RTF.
3. Chọn siêu tham số (hotwords-score, LR, tỉ lệ trộn, số epoch) trên dev; mỗi hướng chỉ chạy bench cho cấu hình cuối, và ghi mọi lần chạy bench vào nhật ký, kể cả lần thua.
4. Phương án FT dẫn đầu và á quân chạy ≥ 2 seed; báo cáo trung bình và khoảng giữa các seed.
5. Ghi chi phí: số giờ dữ liệu, giờ GPU; khi so E3 với E4/E5 thì giữ cùng số bước huấn luyện.
6. Thứ tự thử: E0 → E1 → E3 → E4/E5 → E7 → E8; E6 chỉ khi kiểm tra tokenizer ra đỏ hoặc E3 chững lại; E2, E9 là nhánh nghiên cứu.

Mỗi lần chạy ghi vào nhật ký thí nghiệm: tên system, commit gipbench và icefall, phiên bản + sha256 bench, sha256 model, danh sách manifest train và số giờ, siêu tham số, seed, cấu hình giải mã, phần cứng đo RTF. decode.py đã lưu cấu hình giải mã vào decode\_summary.json; phần còn lại ghi tay hoặc bằng MLflow.

## 8. Độ tin cậy của đánh giá

Một con số trên leaderboard chỉ đáng tin khi mọi rủi ro dưới đây đã được kiểm soát, và mức cải thiện lớn hơn cả khoảng tin cậy lẫn sàn nhiễu nhãn.

| Rủi ro | Hậu quả | Kiểm soát | Rủi ro còn lại |
| --- | --- | --- | --- |
| Nhiễu nhãn (nhất là nhãn máy của dataset công khai) | CS-WER bị thổi phồng hoặc che lỗi | Gán 2 vòng, đo đồng thuận, ước lượng tỉ lệ lỗi nhãn | Chênh lệch nhỏ hơn tỉ lệ lỗi nhãn span coi là không phân biệt được |
| Rò rỉ train ↔ bench | Điểm ảo cho phương án FT | validate.py, to\_lhotse.py, quy ước `source` chung | Video trùng mà ID khác (re-upload) |
| Model gốc đã thấy dữ liệu công khai | Baseline được lợi thế, hoặc FT trông tệ hơn thực tế | Quyết định dựa trên v1 (B2, B4 tự thu); báo cáo v0 và v1 riêng | Không biết dữ liệu train của Gipformer |
| Overfit vào bench | Chọn nhầm phương án | Chỉnh siêu tham số trên dev, ghi mọi lần chạy bench | Phụ thuộc kỷ luật nhóm |
| Cỡ mẫu nhỏ, đuôi dài thuật ngữ | Khoảng tin cậy rộng, kết luận ngẫu nhiên | Ngưỡng cỡ mẫu (bảng dưới), bootstrap theo người nói | Lát nhỏ chỉ để tham khảo |
| Mất cân bằng người nói | Một người quyết định kết quả | Trần 10% số giờ mỗi người, bootstrap cụm | — |
| Chuẩn hoá lệch giữa hệ thống | Model ra chữ số hoặc chữ hoa bị phạt oan | Một bộ chuẩn hoá cho mọi hypothesis | Cách đọc số khác nhau (tư / bốn) |
| Đường chạy khác production | Số đẹp trên PyTorch fp32 nhưng tệ khi deploy | Chấm bản int8 qua sherpa-onnx | — |
| Artefact của dữ liệu TTS | Thắng trên dữ liệu giống TTS, không thắng trên giọng thật | Bench toàn giọng thật; xem riêng B2 | — |
| Bench khác phân phối production | Thắng trên bench, không thắng trên sản phẩm | B2 nằm trong metric chính và được báo cáo riêng | B2 chỉ 2 h |

**Cỡ mẫu**: độ rộng nửa khoảng tin cậy 95% của CS-WER, ước lượng với CS-WER khoảng 30% và hệ số cụm theo người nói bằng 2 (giả định; số thật lấy từ bootstrap). So cặp thường cho khoảng hẹp hơn.

| Token tiếng Anh trong lát | ± nửa CI95 (điểm CS-WER) | Cách dùng |
| --- | --- | --- |
| 500 | 5.7 | Chỉ tham khảo |
| 1.000 | 4.0 | Ngưỡng tối thiểu để báo cáo một lát |
| 2.000 | 2.8 | Đủ cho từng split |
| 4.000 | 2.0 | Mục tiêu B1–B4 |
| 8.000 | 1.4 | Phân biệt các phương án sát nhau |

Check-list trước khi tin một kết luận:

- [ ] validate.py ĐẠT với mọi manifest train/dev của hệ thống đó
- [ ] Phiên bản và sha256 bench ghi trong kết quả; baseline chạy trên cùng phiên bản
- [ ] Cùng cấu hình giải mã production
- [ ] CI95 của ΔCS-WER nằm dưới 0 và |Δ| lớn hơn tỉ lệ lỗi nhãn ước lượng
- [ ] Ràng buộc B5, N-WER, RTF đều đạt
- [ ] Kết quả giữ nguyên hướng trên B2 riêng (production)
- [ ] Kết luận được xác nhận trên v1, không chỉ v0
- [ ] Chạy lại được từ nhật ký thí nghiệm

## 9. Báo cáo & phiên bản

Mọi kết quả gắn với một phiên bản bench; kết quả của hai phiên bản khác nhau không được đặt chung một bảng.

| Thay đổi | Phiên bản | Bắt buộc chạy lại |
| --- | --- | --- |
| Thêm/bỏ split, đổi nguồn, đổi quy ước transcript, đổi bộ chuẩn hoá | v(N+1).0 | Mọi hệ thống cần so sánh |
| Sửa nhãn, thêm/bỏ câu, bổ sung catalog | vN.(M+1) | Baseline và các hệ thống còn tranh chấp |
| Đổi ngưỡng luật chọn | Ghi trong báo cáo | Chạy lại leaderboard.py (không cần giải mã lại) |

Cấu trúc báo cáo đánh giá (mỗi lần ra quyết định):

1. **Kết luận** (1 câu): phương án chọn, ΔCS-WER kèm CI95, phương án đã loại và lý do — dòng đầu của leaderboard.md.
2. **Bench**: phiên bản, sha256, bảng quy mô và bảng mục tiêu từ eda\_summary.md, tỉ lệ lỗi nhãn ước lượng.
3. **Leaderboard**: bảng đầy đủ + biểu đồ CS-WER có CI từ leaderboard.html.
4. **Phương án thắng còn yếu ở đâu**: các lát từ slices.md (chỉ lát ≥ 1.000 token tiếng Anh), 25 thuật ngữ sai nhiều nhất.
5. **Chi phí**: giờ dữ liệu, giờ GPU, RTF.
6. **Rủi ro còn lại**: đối chiếu check-list ở mục 8, mục nào chưa đạt.
7. **Phụ lục**: nhật ký thí nghiệm, đường dẫn results/.

## 10. Thiết kế dữ liệu để FT tốt mà không overfit

Đề xuất của Claude, chưa kiểm chứng bằng thí nghiệm: coi fine-tune là bài toán hai mục tiêu (CS-WER giảm, chất lượng tổng quát giữ nguyên) và thiết kế cả dữ liệu train lẫn bench để nhìn thấy được từng kiểu overfit, thay vì chỉ dồn thật nhiều dữ liệu CS.

Model có thể overfit theo 5 kiểu, mỗi kiểu cần một phép đo riêng và một cách chặn riêng:

| Kiểu overfit | Biểu hiện | Phép đo trong bench | Chặn ở dữ liệu train |
| --- | --- | --- | --- |
| Thuộc lòng thuật ngữ | Tốt trên từ đã thấy, kém trên từ mới | CS-WER B3 so với B1+B2 | Trần số lần mỗi thuật ngữ (đề xuất ≤ 50 câu, loại phần vượt); báo cáo thuật ngữ có < 5 ngữ cảnh (cặp từ trước/sau) hoặc < 3 người nói để bổ sung dữ liệu, không loại |
| Lệch domain | Tốt trên y tế/tech, kém ở domain khác | CS-WER theo domain trong slices.md; giữ riêng một domain không đưa vào train (vd finance) | Không domain nào quá 35% số giờ CS; ViMedCSS (24 h y tế) phải bị giảm tỉ trọng khi trộn |
| Quên tiếng Việt | WER tiếng Việt thuần tăng, nhất là call center | B5 chia theo domain; luật mới: không domain B5 nào tăng > 0.5 điểm | Replay tiếng Việt ≥ 50% mỗi batch, phân tầng theo domain và kênh |
| Chèn bừa từ Anh | “tim” thành team, “mít tinh” thành meeting dù người nói dùng từ Việt | Chèn sai thuật ngữ trên B5; thêm nhóm câu “dễ nhầm” vào B5 | Câu tiếng Việt chứa từ gần âm với thuật ngữ (tim, sao, ba, ga, sốp) làm ví dụ âm |
| Học giọng thay vì học từ | Tốt trên giọng TTS/người nói đã thấy | Bench toàn giọng thật, tách theo người nói; so B2 với B4 | TTS ≤ 20% số giờ CS, nhiều giọng; không người nói nào > 2% dữ liệu CS |

**Công thức trộn dữ liệu train đề xuất** (điểm khởi đầu, chỉnh trên dev):

| Thành phần | Tỉ trọng trong batch | Lý do |
| --- | --- | --- |
| CS thật đa domain (SEA đã lọc, ViMedCSS giảm tỉ trọng, dữ liệu sản phẩm nếu có) | 35–40% | Nguồn tín hiệu chính |
| CS tổng hợp (TTS, ghép) cho thuật ngữ hiếm | ≤ 10% | Phủ đuôi dài mà không để giọng TTS chi phối |
| Replay tiếng Việt có nhãn người (vivos, Common Voice, dữ liệu call center của bạn) | 25–30% | Giữ năng lực gốc |
| Replay tự chưng cất: audio tiếng Việt không nhãn, nhãn do chính Gipformer gốc sinh ra, chỉ giữ câu mà greedy và beam search cho kết quả giống hệt | 20–25% | Dữ liệu train gốc của Gipformer không công bố; dạy model FT lặp lại đúng hành vi cũ trên tiếng Việt là cách thay thế trực tiếp nhất |
| Câu “dễ nhầm” (tiếng Việt gần âm thuật ngữ) | \~5% | Chống chèn bừa |

**Bổ sung cho bench v1** để đo được “vẫn tốt nhất trên general”:

1. Mở rộng B5 thành bộ kiểm tra tổng quát nhiều domain, gồm đúng các tập mà Gipformer công bố số liệu (vivos, Common Voice, VLSP-T1, VietMed, FLEURS, vi-asr-pubadmin; MultiMed và LSVSC tuỳ chọn, lọc trùng giữa các bộ bằng hash audio và text sau chuẩn hoá) và call center của bạn, mỗi domain ≥ 500 token; ghi `domain` cho từng câu. leaderboard.py đã có luật `--max-b5-domain-delta 0.5`.
2. Thêm nhóm “dễ nhầm” vào B5 (domain = `confusable`): 200–300 câu tiếng Việt thuần chứa từ gần âm thuật ngữ.
3. Giữ một domain CS hoàn toàn ngoài train (đề xuất finance của G-Group) và báo cáo CS-WER riêng cho nó. Câu train không có nhãn domain được suy ra từ text (mục 3) để loại đúng domain này.
4. Cân bằng thuật ngữ trong bench: mỗi thuật ngữ tối đa 10 lần, để CS-WER không bị vài brand phổ biến quyết định.

**Chống overfit ở khâu huấn luyện**, theo thứ tự nên thử:

| Kỹ thuật | Cách làm | Chi phí |
| --- | --- | --- |
| Trộn trọng số gốc và FT (WiSE-FT) | `ft/interpolate_ckpt.py --alphas 0.3 0.5 0.7`, export từng alpha, chọn trên dev điểm có CS-WER thấp nhất mà WER tiếng Việt không tăng | Không train thêm |
| Dừng sớm theo dev kép | Theo dõi cả dev CS lẫn dev tiếng Việt (truyền 2 file vào `--valid-cuts`); dừng khi dev tiếng Việt bắt đầu xấu đi | Không |
| FT ngắn, LR thấp | 3–5 epoch, thử LR 0.001–0.0045 | Thấp |
| Đóng băng phần dưới của encoder | Chỉ cập nhật các stack trên, decoder và joiner; cần sửa finetune\_cs.py | Trung bình |
| Adapter / LoRA | Thêm tham số nhỏ, giữ nguyên model gốc; ViMedCSS cho thấy LoRA hiệu quả trên Whisper, chưa có sẵn cho Zipformer | Cao |

Nguyên tắc chọn cuối cùng: vẽ từng hệ thống theo hai trục, CS-WER (B1+B2) và WER tổng quát (B5), rồi chọn điểm có CS-WER thấp nhất trong số các điểm không làm B5 xấu đi. Các ngưỡng trong mục này (50 câu, 35%, 20%, tỉ trọng batch) là điểm khởi đầu theo suy luận, cần được xác nhận bằng chính các thí nghiệm E3–E5.
