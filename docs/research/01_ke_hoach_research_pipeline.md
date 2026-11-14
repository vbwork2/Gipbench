# Gipformer 1.5 – Code-switching Việt-Anh: Benchmark & Pipeline

Nov 13, 2026 · @God Goose

Việc cần làm ngay là dựng một bộ benchmark code-switching (CS) có tag span tiếng Anh, khoảng 10–12 giờ chia 5 split, rồi mới so sánh các hướng fine-tune; không có bench thì không chọn được phương án.

Trạng thái 14/11: dataset chưa thu; bộ công cụ chấm điểm và hướng dẫn chạy từng bước ở Instruction: Runbook bench & FT.

## Hiện trạng Gipformer 1.5

Gipformer 1.5 (68M, Zipformer-RNNT, MIT) mạnh ở call center và đọc chuẩn, nhưng các test domain nhiều thuật ngữ tiếng Anh vẫn có WER 22–28%, cao gấp 5–6 lần vivos.

| Test set (WER %, báo cáo của G-Group) | Gipformer 1.5 | Ghi chú |
| --- | --- | --- |
| vivos (đọc chuẩn) | 4.25 | gần như bão hoà |
| vi-asr-pubadmin | 14.82 | ít từ ngoại |
| vi-asr-finance | 22.34 | nhiều tên sàn, crypto, ticker |
| vi-asr-edu | 23.32 | du học, IELTS, tên trường |
| vi-asr-tech | 27.49 | review đồ công nghệ, lập trình, IT |

Những điểm kỹ thuật quyết định hướng làm:

- Model xuất chữ IN HOA, không dấu câu, chạy bằng sherpa-onnx; hotwords chỉ dùng được với modified beam search trên transducer; script infer\_onnx.py chính thức chưa có tuỳ chọn hotwords, phải gọi sherpa-onnx trực tiếp.
- Fine-tune được bằng icefall: một đội ZaloAI đã FT Gipformer 1.5 trên dữ liệu domain của họ (speed/noise/SpecAugment) và hạ WER từ 15.2% xuống 7.7% — FT domain hiệu quả. Repo GitHub xác nhận có model.pt, bpe.model, tokens.txt và dùng recipe egs/librispeech/ASR/zipformer chuẩn của icefall với tham số mặc định, nên dùng thẳng finetune.py của recipe này.
- Cần kiểm tra ngay `tokens.txt` / `bpe.model`: nếu BPE được train chỉ trên chữ tiếng Việt thì từ tiếng Anh bị xé vụn thành nhiều token ngắn; đây là câu hỏi mở quyết định có nên mở rộng vocab hay không. Chạy check\_gipformer\_tokenizer.py để có kết luận xanh / vàng / đỏ.
- Model Việt-hóa thường tốt phần tiếng Việt nhưng yếu ở đoạn tiếng Anh chèn vào; trên ViMedCSS, VietASR-Zipformer (tiền thân của Gipformer) có WER tổng tốt nhất (27.56) nhưng CS-WER tới 58.38, kém Whisper-Large-v3 (46.69).

## Bộ benchmark: 5 split, \~11 giờ

Mỗi split trả lời một câu hỏi riêng; chỉ cần B2 và B4 là đã bắt đầu chạy baseline được. Toàn bộ bench là held-out: không một utterance, người nói hay video nào được lọt vào dữ liệu train.

| Split | Trả lời câu hỏi | Nguồn | Mục tiêu | Ưu tiên |
| --- | --- | --- | --- | --- |
| B1 Real-CS-General | Model có nhận đúng từ Anh trong lời nói tự nhiên không? | YouTube/podcast tech, finance, edu, vlog; lọc thêm CS từ VietPrism và cặp en-vi của SEA Code-Switching | 3 h, ≥ 150 người nói | Cao |
| B2 Real-CS-Domain | Có tốt trên đúng sản phẩm của mình không? | Audio thật của bạn (call, meeting, app) có từ Anh | 2 h | Cao nhất |
| B3 Unseen-Terms (Hard) | Có tổng quát sang thuật ngữ chưa thấy khi FT không? | Tách từ B1/B4 các thuật ngữ xuất hiện ≤ 2 lần, cấm mọi lần xuất hiện trong train | 1.5 h | Cao |
| B4 Read-CS Controlled | Lỗi nằm ở loại từ, kiểu phát âm, vùng miền nào? | Câu kịch bản chứa \~600 thuật ngữ, 40–60 người đọc, 3 miền, 2 loại mic | 2.5 h | Cao |
| B5 Mono-VI Regression | FT có làm hỏng tiếng Việt thuần không? | Mẫu cố định từ vivos, VLSP-T1, Common Voice, vi-asr-pubadmin + 30 phút call center của bạn | 2 h | Trung bình |

Tổng mục tiêu cho B1–B4: ≥ 4.000 span tiếng Anh và ≥ 1.000 thuật ngữ khác nhau, đủ để khoảng tin cậy của CS-WER hẹp hơn các mức cải thiện cần so sánh.

Mỗi utterance mang metadata để cắt lát kết quả: miền (Bắc/Trung/Nam), giới tính, kênh (8 kHz điện thoại / wideband), loại thuật ngữ (brand, tech, finance, edu, tên riêng, viết tắt đánh vần như API, KPI), mức Việt hoá cách phát âm (mạnh / trung gian / gần bản ngữ), và tần suất thuật ngữ trong tập train.

### Dữ liệu công khai dùng được ngay

| Dataset | Quy mô | Dùng làm gì | Lưu ý |
| --- | --- | --- | --- |
| [VietPrism](https://arxiv.org/pdf/2609.30005) | 993 h, 1.262 người nói, CS gần nửa thời lượng, 5 nhóm phương ngữ | Nguồn lớn nhất cho B1 và cho train | Preprint 2026; kiểm tra license, tách theo video/người nói để không rò rỉ |
| [SEA Code-Switching (gijs/sea\_speech)](https://huggingface.co/datasets/gijs/sea_speech) | cặp en-vi: 10.306 clip, 45.3 h | B1, có sẵn span + vị trí switch | Gated; transcript tự động, phải kiểm tra lại trước khi làm test |
| [ViMedCSS](https://huggingface.co/datasets/tensorxt/ViMedCSS) | 34.6 h; test 3.38 h, hard 1.38 h | Bench ngoài để so với paper; mẫu tham khảo cho B3 | CC-BY-4.0; chỉ y tế |
| [g-group vi-asr-tech/edu/finance](https://huggingface.co/datasets/g-group-ai-lab/vi-asr-tech-test) | test domain của chính Gipformer | Lọc utterance có từ Anh → bổ sung B1 | Không dùng để train |
| [CS-FLEURS](https://huggingface.co/datasets/byan/cs-fleurs) | 113 cặp ngôn ngữ, 52 ngôn ngữ | Không dùng được: không có cặp Việt–Anh (kiểm tra 14/11) | Phần lớn audio tổng hợp hoặc đọc |
| [CanVEC](https://preview.aclanthology.org/fix_video/2020.lrec-1.507.pdf) | 10 h hội thoại tự nhiên | Kiểm tra giọng Việt kiều | Người Việt ở Úc, khác với người dùng trong nước |

Câu hỏi mở: dữ liệu train của Gipformer 1.5 không được công bố, nên bộ công khai có thể đã bị nhìn thấy; quyết định cuối cùng nên dựa trên B2 và B4 do bạn tự thu.

## Quy trình thu thập & gán nhãn

Chạy 3 luồng song song, chung một danh mục thuật ngữ; nhãn tự động chỉ để lọc, mọi utterance vào test đều phải do người sửa và duyệt. ViMedCSS dùng Gemini để gán nhãn và chỉ đạt độ đồng thuận κ = 0.65 khi kiểm định, nên nhãn máy không đủ cho test.

**Bước 0 — Danh mục thuật ngữ (dùng cho cả 3 luồng)**

1. Gom 1.500–2.000 từ/cụm Anh hay dùng trong tiếng Việt: log sản phẩm, từ Anh có trong transcript vi-asr-tech/edu/finance, danh sách brand/app phổ biến ở VN, thuật ngữ IT/tài chính/giáo dục.
2. Gắn cho mỗi từ: loại, dạng viết chuẩn, biến thể (Zalo/zalo, Facebook/FB/phây búc), có đánh vần hay không (API, KPI, AI).
3. Chia sẵn 15–20% thuật ngữ vào nhóm “chỉ dùng cho B3” — cấm xuất hiện trong mọi dữ liệu train sau này.

**Luồng A — Lời nói tự nhiên (B1, B3)**

1. Dùng thuật ngữ làm query tìm video/podcast tiếng Việt; bỏ video chỉ có nhạc, giới hạn ≤ 3 clip/kênh để đa dạng người nói.
2. VAD (Silero) cắt đoạn 3–20 s.
3. Gán nhãn nháp bằng 2–3 model khác nhau (Gipformer 1.5, Whisper-large-v3 hoặc Qwen3-ASR, một LLM đa phương thức); một LLM đánh dấu span tiếng Anh.
4. Giữ đoạn có ≥ 1 thuật ngữ trong danh mục; ưu tiên đoạn các model bất đồng chỗ từ Anh (đó là chỗ khó).
5. Người sửa transcript theo quy ước ở phần sau, tag span `<en>`, chọn mức Việt hoá cách phát âm; người thứ hai duyệt lại.

**Luồng B — Đọc kịch bản có kiểm soát (B4)**

1. LLM sinh câu tiếng Việt tự nhiên chứa 1–3 thuật ngữ; mỗi thuật ngữ xuất hiện trong 2 câu khác ngữ cảnh; người sửa cho giống văn nói.
2. Tuyển 40–60 người đọc, cân bằng Bắc/Trung/Nam và nam/nữ, có cả người ít dùng tiếng Anh.
3. Thu qua web app hoặc bot, nói tự nhiên như bình thường (không cố “tây”); một phần thu qua điện thoại 8 kHz.
4. Kiểm tra âm lượng, cắt đầu/cuối, người nghe xác nhận đọc đúng kịch bản và gắn mức Việt hoá.

**Luồng C — Dữ liệu sản phẩm (B2)**

1. Lấy mẫu audio thật đã có sự đồng ý, ẩn PII (số điện thoại, CCCD, tên khách).
2. Chạy Gipformer + LLM tìm đoạn có khả năng chứa từ Anh; thêm 20% mẫu ngẫu nhiên để không chỉ đo đoạn model đã “nghi ngờ”.
3. Người gõ transcript từ đầu (không sửa từ output model) để tránh thiên lệch về phía model.

**Kiểm soát chất lượng**

- Hai người cùng gán 10% mẫu; mục tiêu độ đồng thuận trên span tiếng Anh ≥ 0.8, dưới mức này thì sửa hướng dẫn rồi gán lại.
- Tách split theo người nói và theo video, không theo utterance.
- Đóng băng bench thành phiên bản (v1.0), lưu hash file; mọi thay đổi tăng version và chạy lại toàn bộ baseline.
- Định dạng: Lhotse manifest (cuts + supervisions) để đưa thẳng vào icefall; mỗi supervision có `text`, `text_tagged` (có `<en>`), `en_spans`, và metadata ở phần trên.

## Quy ước transcript & chuẩn hoá

Thống nhất: từ tiếng Anh viết bằng chính tả tiếng Anh, dù người nói phát âm kiểu Việt; chốt quy ước này trước khi gán nhãn dòng đầu tiên, vì nó quyết định cả bench lẫn dữ liệu train.

| Tình huống | Người nói | Transcript chuẩn |
| --- | --- | --- |
| Từ Anh chèn vào | “mi-ting”, “đét-lai” | meeting, deadline |
| Từ mượn đã Việt hoá, có trong từ điển | cà phê, ga, xăng | giữ chính tả Việt |
| Viết tắt đánh vần | “a-pi-ai”, “ca-pi-ai” | api, kpi |
| Brand viết cách điệu | TikTok, ChatGPT | tiktok, chatgpt (viết liền theo bảng chuẩn) |
| Số trong tên sản phẩm | “ai phone mười lăm” | iphone mười lăm (số → dạng đọc, giống chuẩn của Gipformer) |
| Không nghe rõ từ Anh nào | — | `<unk>`, loại khỏi tính CS-WER |

Bảng chuẩn (`canonical_map.tsv`): mỗi dòng là biến thể → dạng chuẩn (fb, phây búc, face book → facebook). Bảng này được áp cho cả reference và hypothesis trước khi tính điểm, và được version cùng với bench.

Chấm hai chế độ:

- **Strict** (mục tiêu sản phẩm): hypothesis phải ra đúng chính tả Anh sau chuẩn hoá.
- **Lenient** (chẩn đoán): chấp nhận cả phiên âm Việt có trong bảng biến thể. Chênh lệch Strict − Lenient cho biết model nghe đúng âm nhưng viết sai — loại lỗi mà hậu xử lý (AdaCS, LLM) có thể sửa được.

Chuẩn hoá trước khi chấm: lowercase, bỏ dấu câu, số → dạng đọc, Unicode NFC, áp `canonical_map.tsv`.

## Metrics & luật chọn phương án

Phương án thắng là phương án có CS-WER (strict) thấp nhất trên B1+B2, với điều kiện không làm tệ tiếng Việt thuần và không chậm hơn đáng kể. Tách WER thành CS-WER và N-WER theo cách của AdaCS và ViMedCSS, vì WER tổng che mất lỗi ở đoạn tiếng Anh.

| Metric | Tính trên | Vai trò |
| --- | --- | --- |
| CS-WER strict | token trong span `<en>`, B1+B2 | Mục tiêu chính |
| N-WER | token ngoài span, B1–B4 | Không được tăng |
| WER tổng | mọi split | Báo cáo, so với paper |
| Term recall / precision | thuật ngữ trong danh mục | Recall đo bắt được từ; precision bắt lỗi “chèn bậy” do hotwords |
| CS-WER trên B3 | thuật ngữ chưa thấy | Đo khả năng tổng quát, quan trọng với biasing |
| ΔWER trên B5 | tiếng Việt thuần | Ràng buộc: tăng ≤ 0.3 điểm tuyệt đối |
| Tỉ lệ chèn sai thuật ngữ trên B5 | B5 + danh sách hotwords | Kiểm tra over-biasing |
| RTF, RAM (CPU, int8) | 100 file cố định | Ràng buộc: ≤ 1.2× baseline |

Cách đọc kết quả:

1. Mọi con số kèm khoảng tin cậy 95% bằng bootstrap theo người nói (1.000 lần); chỉ coi là cải thiện khi khoảng tin cậy của hiệu số không chứa 0.
2. Loại phương án vi phạm ràng buộc B5 hoặc RTF.
3. Xếp hạng phần còn lại theo CS-WER strict trên B1+B2; hòa thì chọn cái tốt hơn trên B3.
4. Cắt lát theo miền, kênh, loại thuật ngữ, mức Việt hoá để biết phương án thắng ở đâu và còn yếu ở đâu.

Một script duy nhất `bench.py --model X --decode Y` xuất một dòng JSON/kết quả vào bảng leaderboard, kèm commit hash và version bench.

## Research paper liên quan

Bằng chứng mạnh nhất cho tiếng Việt (ViMedCSS) cho thấy fine-tune trên dữ liệu CS thật cải thiện nhiều hơn hẳn so với chỉ bias lúc decode; hậu xử lý và hotwords là lớp bổ sung.

| Paper | Ý chính & kết quả | Áp dụng cho Gipformer |
| --- | --- | --- |
| [ViMedCSS](https://arxiv.org/html/2602.12911v1) (LREC 2026) | 34.6 h CS y tế, có split Hard. Trên PhoWhisper-small: Attention-Guide hạ CS-WER 62.55 → 19.50, LoRA → 30.26; bias lúc decode (DV, RS) gần như không đổi | Mẫu thiết kế bench (CS-WER, N-WER, Hard). Attention-Guide dành cho Whisper; với Zipformer, tương đương là FT trên dữ liệu CS |
| [AdaCS](https://arxiv.org/abs/2501.07102) (ICASSP 2025, VinBrain) | Module chuẩn hoá sau ASR có bias attention theo danh sách từ; giảm WER 56.2% và 36.8% tương đối trên 2 test CS tiếng Việt | Sửa “phây búc” → facebook; đúng loại lỗi Strict − Lenient. Cần canh N-WER vì có thể sửa nhầm |
| [TTS augmentation cho CS](https://arxiv.org/abs/2601.00935v1) (A\*STAR/NTU, 2026) | FT CosyVoice2 trên SEAME rồi sinh dữ liệu CS; MER 12.1 → 10.1 và 17.8 → 16.0; yếu tố quyết định: đa dạng giọng, đủ lượng, tỉ lệ thật/tổng hợp | Sinh câu Việt chứa từ Anh bằng TTS được FT trên CS tiếng Việt để có đúng giọng Anh kiểu Việt |
| [Phrase-mixed synthetic CS](https://www.isca-archive.org/interspeech_2025/nguyen25_interspeech.pdf) (Interspeech 2025) | Ghép cụm từ audio đơn ngữ để tạo CS tự nhiên hơn ghép theo từ | Augmentation rẻ: chèn audio thuật ngữ (thu hoặc TTS) vào câu tiếng Việt |
| [AsyncSwitch](https://papers.cool/arxiv/2506.14190) (2025) | Cho decoder học text CS trước rồi mới học speech-text; giảm 9.02% WER tương đối (Malay-English) | Với RNN-T: tận dụng text CS qua LM ngoài (n-gram/NN, LODR) hoặc pretrain predictor trên text |
| [TCPGen](https://arxiv.org/pdf/2205.09058v2.pdf) + [phoneme-aware TCPGen](https://arxiv.org/html/2312.09582v1) | Bias bằng cây tiền tố cho RNN-T, thêm thông tin phát âm cho từ khó đoán từ chính tả | Module bias học được, đóng băng ASR; bản phoneme-aware hợp với tiếng Anh giọng Việt |
| [Keyword-aware loss](https://arxiv.org/abs/2509.09197) (Interspeech 2025) | Train bias trên dữ liệu TTS dễ overfit; loss tập trung vào keyword. 10 h tổng hợp: WER 29.71 → 11.81 | Dùng khi FT bằng dữ liệu TTS của bước trên |
| [Aho-Corasick hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html) + [Single Trie Pass](https://arxiv.org/html/2409.13514v2) (TSD 2025) | Hotwords có sẵn trong sherpa-onnx/icefall; kết hợp keyword + n-gram LM: nhận thực thể tốt hơn tới 32%, WER giảm tới 12% tương đối | Baseline không cần train, chạy được ngay |
| [GER cho CS](https://arxiv.org/html/2310.13013v1) + [GER từ hiếm có ngữ âm](https://arxiv.org/abs/2505.17410v1) | LLM đọc N-best để sinh transcript đúng; bản 2025 thêm phonetic context để tránh sửa quá tay | Hậu xử lý tùy chọn cho server, không hợp chạy edge |
| [VietASR](https://arxiv.org/pdf/2505.21527) (2025) | Zipformer + SSL trên \~70k h, FT 50 h | Tiền thân của Gipformer; nếu có nhiều audio CS chưa nhãn thì có thể SSL tiếp |
| [Pipeline VLSP 2025](https://preview.aclanthology.org/author-page-lei-gao-usc/2025.vlsp-1.4.pdf) | Lọc/augment ưu tiên token OOV hiếm, vote nhiều model để làm sạch nhãn | Cách làm pseudo-label cho dữ liệu train CS |

## Pipeline cải thiện model & ma trận thí nghiệm

Đi từ rẻ đến đắt: hotwords (không train) → FT trên CS thật → thêm dữ liệu tổng hợp → hậu xử lý; mỗi bước chỉ giữ lại nếu thắng theo luật ở phần Metrics.

| ID | Thí nghiệm | Cần train? | Câu hỏi nó trả lời |
| --- | --- | --- | --- |
| E0 | Baseline: greedy và modified beam search, int8 và fp32 | Không | Mốc so sánh |
| E1 | Hotwords Aho-Corasick với danh mục thuật ngữ; quét hotwords-score 1.0–3.0 | Không | Bias lúc decode được bao nhiêu, chèn sai bao nhiêu trên B5? |
| E2 | E1 + LM n-gram trên text CS (shallow fusion/LODR của icefall) | Chỉ LM | Text CS rẻ có thay được audio không? |
| E3 | FT trên CS thật (VietPrism CS, SEA en-vi, crawl đã gán nhãn, trừ bench) trộn \~50% dữ liệu tiếng Việt để chống quên | Có | Phương án chính, đúng kết luận ViMedCSS |
| E4 | E3 + speech TTS tổng hợp; quét tỉ lệ tổng hợp 0 / 20 / 50% | Có | TTS có cứu được thuật ngữ hiếm (B3) không? |
| E5 | E3 + ghép cụm/thuật ngữ vào câu tiếng Việt | Có | Augmentation rẻ có bằng TTS không? |
| E6 | Mở rộng BPE bằng subword tiếng Anh (chỉ khi E3 chững lại và token Anh bị xé vụn) | Có, lớn | Vocab có phải nút thắt không? |
| E7 | Model tốt nhất E3–E6 + hotwords | Không thêm | FT và bias có cộng dồn không? |
| E8 | E7 + hậu xử lý kiểu AdaCS hoặc LLM đọc N-best | Module riêng | Còn sửa được bao nhiêu lỗi Strict − Lenient? |
| E9 | Bias học được kiểu TCPGen phoneme-aware | Có, nghiên cứu | Đáng làm nếu E7 vẫn yếu trên B3 |

Khi fine-tune (E3–E6):

- Dùng script finetune của icefall cho Zipformer, khởi tạo từ checkpoint Gipformer 1.5; learning rate thấp hơn train gốc nhiều lần, vài epoch, lấy trung bình checkpoint.
- Augment: speed perturb 0.9/1.0/1.1, nhiễu MUSAN, SpecAugment, mô phỏng kênh thoại 8 kHz (codec) vì thế mạnh của Gipformer là call center.
- Giữ dữ liệu tiếng Việt thuần trong mỗi batch; theo dõi B5 sau mỗi epoch để dừng sớm khi bắt đầu quên.
- Export lại ONNX int8 và chấm bằng đúng đường sherpa-onnx sẽ chạy production — không chỉ bản PyTorch.

## Lộ trình 4 tuần

Tuần 1–2 dành hết cho dataset bench; FT chỉ bắt đầu khi bench v1.0 đã đóng băng.

**Tuần 1 — Quy ước & thu thập**

- [ ] Kiểm tra `tokens.txt`/`bpe.model` của Gipformer 1.5: đếm số token/từ trên 200 thuật ngữ tiếng Anh
- [ ] Chốt quy ước transcript và bản đầu `canonical_map.tsv`
- [ ] Dựng danh mục 1.500–2.000 thuật ngữ, tách sẵn nhóm dành riêng cho B3
- [ ] Xin quyền truy cập VietPrism, SEA Code-Switching; tải ViMedCSS và test vi-asr-\*
- [ ] Viết kịch bản B4, tuyển người đọc, bắt đầu thu âm
- [ ] Chạy luồng A (crawl → VAD → nhãn nháp) và lấy mẫu luồng C

**Tuần 2 — Gán nhãn & đóng băng bench**

- [ ] Gán nhãn 2 vòng B1–B4, đo độ đồng thuận trên 10% mẫu
- [ ] Tách split theo người nói/video, tạo Lhotse manifest, đóng băng bench v1.0
- [ ] Viết `bench.py` (CS-WER, N-WER, term P/R, bootstrap CI, cắt lát metadata)
- [ ] Chạy E0, E1 trên Gipformer 1.5 và vài model tham chiếu (Whisper-large-v3, Qwen3-ASR, PhoWhisper)

**Tuần 3 — Dữ liệu train & FT đầu tiên**

- [ ] Gom dữ liệu train CS (loại mọi video/người nói/thuật ngữ B3 có trong bench)
- [ ] E2 (LM text CS), E3 (FT CS thật + replay tiếng Việt)
- [ ] Sinh 50–100 h CS bằng TTS cho E4

**Tuần 4 — So sánh & chọn**

- [ ] E4, E5, E7; E6/E8 nếu còn thời gian
- [ ] Bảng leaderboard + cắt lát; chọn phương án theo luật ở phần Metrics
- [ ] Export ONNX int8, chấm lại RTF trên CPU mục tiêu

## Nguồn

- [Gipformer 1.5 model card (Hugging Face)](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt)
- [Gipformer GitHub](https://github.com/ggroup-ai-lab/gipformer)
- [Gipformer captions plugin — định dạng output, hotwords](https://github.com/vpeopleonatank/gipformer-captions-bashcut-plugin)
- [ZaloAI LifeMate — FT Gipformer 1.5](https://github.com/Datyth/ZaloAI-Challenge-lifemate)
- [ViMedCSS (arXiv 2602.12911)](https://arxiv.org/html/2602.12911v1)
- [AdaCS (arXiv 2501.07102)](https://arxiv.org/abs/2501.07102)
- [VietPrism (arXiv 2609.30005)](https://arxiv.org/pdf/2609.30005)
- [SEA Code-Switching dataset](https://huggingface.co/datasets/gijs/sea_speech)
- [CS-FLEURS (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/yan25c_interspeech.html)
- [CanVEC (LREC 2020)](https://preview.aclanthology.org/fix_video/2020.lrec-1.507.pdf)
- [Improving CS ASR with TTS augmentation (arXiv 2601.00935)](https://arxiv.org/abs/2601.00935v1)
- [Train CS-ASR without real CS data (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/nguyen25_interspeech.pdf)
- [AsyncSwitch (arXiv 2506.14190)](https://papers.cool/arxiv/2506.14190)
- [TCPGen (arXiv 2205.09058)](https://arxiv.org/pdf/2205.09058v2.pdf)
- [Phoneme-aware TCPGen (arXiv 2312.09582)](https://arxiv.org/html/2312.09582v1)
- [Keyword-aware loss for synthetic biasing data (arXiv 2509.09197)](https://arxiv.org/abs/2509.09197)
- [sherpa-onnx hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- [Unifying Global and Near-Context Biasing (arXiv 2409.13514)](https://arxiv.org/html/2409.13514v2)
- [GER for CS ASR (arXiv 2310.13013)](https://arxiv.org/html/2310.13013v1)
- [LLM GER for rare words (arXiv 2505.17410)](https://arxiv.org/abs/2505.17410v1)
- [VietASR (arXiv 2505.21527)](https://arxiv.org/pdf/2505.21527)
- [VLSP 2025 Vietnamese ASR pipeline](https://preview.aclanthology.org/author-page-lei-gao-usc/2025.vlsp-1.4.pdf)
