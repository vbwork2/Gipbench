# Metric đánh giá v1

Áp cho `gipbench.bench.evaluate` (scorer `1.0.0`). `gipbench.bench.score` (v0) giữ nguyên định nghĩa cũ; hồ sơ
`legacy_v0` của evaluate cho ra đúng số của score.py (đã kiểm trên E0 / E1 / E3: CS-WER, N-WER, B5 WER, term recall
trùng tới mọi chữ số).

## 1. Hồ sơ chuẩn hoá (`src/gipbench/core/normprofiles.py`)

Mọi hồ sơ áp **đối xứng** cho reference và hypothesis, rồi đi qua pipeline cũ của `core.textnorm`: NFC → lowercase →
số thành chữ tiếng Việt → bỏ dấu câu (`-`, `_` thành khoảng trắng) → tách token theo khoảng trắng → bảng catalog.

| Hồ sơ | Thêm gì | Xếp hạng được |
| --- | --- | --- |
| `legacy_v0` | không gì (= score.py) | có (số lịch sử) |
| `gipbench_strict_v1` | `canon_v1`: dấu thanh vần mở oa/oe/uy về kiểu cũ (hoà = hòa, thuỷ = thủy), i/y **có dấu** sau phụ âm (kỹ = kĩ, lý = lí; "y" không dấu giữ nguyên để không đụng từ Anh), viết tắt có chấm (A.I. = ai), email / tên miền đọc kiểu Việt (`@` = "a còng", `.` = "chấm"), bỏ dấu nháy (don't = dont), bảng chính tả Anh-Anh → Anh-Mỹ cố định (colour = color; 29 mục, soạn tay, không lấy từ lỗi model) | có (hồ sơ chính của G0 / C0) |
| `gipbench_lenient_diagnostic_v1` | strict_v1 + cột `phonetic` của catalog (phiên âm Việt coi là đúng) | **không** (chỉ chẩn đoán; final mode từ chối) |

Không có hồ sơ `reference_compatible` chung: không bộ nào trong registry công bố code scorer chính thức
(`official_scorer: none_published`). Model card Gipformer và VieGem chỉ mô tả bằng lời ("lowercase, bỏ dấu câu, số thành
chữ"), gần với `legacy_v0` khi không có catalog nhưng không chứng minh trùng khớp.

Hiệu ứng đo được của strict_v1 so với legacy_v0 trên v0-draft (E0): B5 WER 9.52 → 9.38, CS-WER B1+B2 71.66 → 71.63.

## 2. Metric

| Tên | Định nghĩa |
| --- | --- |
| `wer` (**space_token_v1**) | (S + D + I) / N, token tách theo khoảng trắng sau chuẩn hoá. Tiếng Việt viết cách theo âm tiết nên token ≈ âm tiết, **không** phải "từ" theo tách từ. WER có tách từ, nếu thêm sau này, phải mang tên và phiên bản riêng. |
| `cer` (**char_v1**) | Levenshtein ký tự giữa chuỗi token đã chuẩn hoá nối bằng 1 dấu cách / số ký tự ref. |
| `cs_wer` | Lỗi trên token thuộc span `<en>` / số token tiếng Anh của ref. Chèn (I) quy về span Anh nếu một token ref kề bên (ưu tiên trái) là tiếng Anh. Có thể > 100% (1 từ Anh → 2 token chèn). |
| `n_wer` | Lỗi trên token tiếng Việt / số token tiếng Việt. |
| `term_recall` | Số lần thuật ngữ ref xuất hiện đúng trong hyp / số span ref. |
| `span_acc`, `term_acc_macro`, `span_cer` | Như score.py: span exact (kể cả ghép "set up" = "setup"), trung bình đều theo thuật ngữ, CER theo span. |

Hồ sơ lenient: CS-WER và term recall lấy theo bảng phonetic (`err_en_len`, `matched_terms_len`), WER tính lại theo cùng
bảng.

### 2b. Metric thêm từ scorer 1.1.0 (17/11/2026, audit R08 / IP-04)

Chỉ **thêm** trường; định nghĩa và giá trị các metric ở bảng trên không đổi (kiểm: chấm lại 6 kết quả v1 + cặp C1,
3 013 giá trị cũ gồm CI và Δ cặp khớp tới 1e-9). Cài ở `core.scoring.cs_breakdown`, tỉ lệ ở `core.evalstats.RATIOS`.

| Tên | Định nghĩa | Câu hỏi nó trả lời |
| --- | --- | --- |
| `cs_errors` | `S_en`, `D_en` (thay / xoá token span Anh); `I_en_left` (chèn, token ref trái là Anh); `I_en_right_only` (chèn, trái là Việt hoặc đầu câu, phải là Anh); `boundary_ins` (chèn có hai token ref kề bên khác ngôn ngữ). Bất biến: `err_en` = `S_en + D_en + I_en_left + I_en_right_only` | lỗi nằm ở chính từ Anh hay ở ranh giới chuyển ngữ? |
| `cs_wer_sd` | (`S_en` + `D_en`) / N_en | model có nhận ra đúng từ Anh không, bỏ qua lỗi chèn |
| `boundary_ins_rate` | `boundary_ins` / N_en | mức lỗi chèn ở ranh giới Việt \| Anh |
| `vi_only_wer` | lỗi / token ref trên câu **thuần Việt**: không có span `<en>` và không có token ref "giống từ Anh" | FT có làm hỏng tiếng Việt không? |
| `en_hallucination_per100` | 100 · token hyp "giống từ Anh" không có trong ref / số câu thuần Việt | FT có làm model chèn từ Anh vào câu Việt không? |

"Giống từ Anh" (`core.scoring.en_like`): chỉ gồm chữ Latin không dấu, dài ≥ 2, và **không** phải một âm tiết tiếng Việt
hợp lệ khi viết không dấu (phụ âm đầu + vần + âm cuối, vd `anh`, `cho`, `ban` là âm tiết Việt; `ok`, `team`, `file` thì
không). Kiểm trên ref VIVOS: 0 token bị nhận nhầm. Từ Anh trùng chính tả với âm tiết Việt (`the`, `man`, `ai`) bị bỏ sót:
metric là cận dưới.

Lưu ý khi đọc:

- **Quy ước quy lỗi chèn ảnh hưởng ~10 điểm CS-WER** (audit 2.2). Trên C1: Gipformer `cs_wer` 70.25 / `cs_wer_sd` 56.37 /
  `I_en_right_only` 328; VieGem 82.39 / 71.56 / 266. Luôn báo `cs_wer` kèm `cs_wer_sd`. So với số của bên ngoài
  (định nghĩa khác) là không hợp lệ.
- **Số chính để báo: `cs_wer` theo `text_tagged`** (nhãn chính của benchmark). Đổi `--label-field` (vd `text_tagged_orig`
  của C0) làm CS-WER đổi ~6 điểm theo hai chiều ngược nhau cho hai model. Final mode chỉ nhận nhãn chính hoặc trường trong
  `alt_label_fields` của benchmark, và `summary.json` ghi `label_field` + `overrides`.
- `vi_only_wer` / `en_hallucination_per100` chỉ có nghĩa khi bench có câu thuần Việt: G0 (VIVOS, CV, G-Group không có từ
  Anh), không có ở C1 (mọi câu đều có span Anh).

### 2c. Provenance trong `summary.json` (scorer 1.1.0, audit R07 / R13 / IP-03)

| Trường | Nội dung |
| --- | --- |
| `scorer_code_sha256` | sha256 mã nguồn chấm điểm (`core/{textnorm,normprofiles,align,scoring,evalstats}.py` + `bench/evaluate.py`, đọc bytes, `\r\n` → `\n`). Code chấm đổi thì hash đổi |
| `profile.code_sha256` | sha256 `normprofiles.py` + `textnorm.py`. `profile.sha256` vẫn chỉ băm spec (không đổi, để bản khoá còn hiệu lực) |
| `label_field`, `overrides` | trường nhãn đã chấm; mọi giá trị manifest / catalog / hồ sơ / nhãn khác mặc định của benchmark |
| `baseline_hyps_sha256`, `baseline_hyp_check` | baseline của so cặp được kiểm như hyps chính |
| `metric_definitions` | định nghĩa các metric ở mục 2b |

<!-- share:omit -->
## 3. Vì sao CS-WER strict = lenient trên v0-draft (đã điều tra)

Catalog `data/bench_v0_draft/catalog_v0_draft.tsv` có 1 933 thuật ngữ, **0** dòng có cột `phonetic`, **0** dòng có cột
`variants`. Bảng lenient vì vậy trùng bảng strict, nên hai số bằng nhau **do dữ liệu**, không phải lỗi scorer.
score.py có ghi cờ `lenient_available: false` cho trường hợp này, nhưng `summary.json` của các lần chạy v0-draft được tạo
trước khi có cờ đó. Muốn có số lenient thì phải soạn cột phonetic **từ nguồn không phải test** (vd từ điển phiên âm).
Không được lấy từ lỗi của model trên test.

<!-- /share:omit -->

## 4. Kiểm hyps

`check_hyps`: id thiếu, id thừa, id trùng (trong hyps và trong manifest), số hyp rỗng. Final mode: lệch bất kỳ → thoát mã 2.
Diagnostic: hyp thiếu tính là rỗng và ghi cảnh báo (giống score.py). Từ scorer 1.1.0, `--baseline-hyps` được kiểm y hệt
(trước đó baseline thiếu câu vẫn chạy: cắt baseline C1 còn 630 / 1 260 câu làm Δ CS-WER đổi từ +12.13 thành −2.57 mà
vẫn thoát mã 0).

## 5. Khoảng tin cậy

- `cluster_bootstrap`: lấy mẫu lại đơn vị (`boot_unit` của benchmark: speaker / session / source / utt /
  boot_unit_legacy = speaker_cluster nếu có), có hoàn lại, 1 000 lần, seed cố định. Tỉ lệ tính lại từ tổng mỗi mẫu. CI =
  phân vị 2.5 / 97.5.
- `paired_bootstrap`: Δ = metric(hệ) − metric(baseline) trên các id chung, cùng mẫu đơn vị cho hai hệ.
- Giới hạn: G-Group không có id người nói, nên CI ở đó là theo câu (hẹp hơn thực tế nếu một người nói nhiều câu).
  `speaker_cluster` (ước lượng bằng embedding) chỉ dùng cho phân tích độ nhạy CI, không chứng minh tách người nói.

## 6. Test

`tests/core/test_scoring_v1.py` (30 ca: 1 từ Anh → 2 token, chèn ở ranh giới, API / KPI / A.P.I, vitamin, Zalo,
pull request, code / cốt strict vs lenient, "cốt lõi" không bị đổi thành "code", số, dấu câu, hoà / hòa, span giữ nguyên,
CER, kiểm hyps, bootstrap, final mode; thêm 1.1.0: phân rã S / D / I span Anh, ranh giới, câu thuần Việt, từ Anh chèn,
bất biến `err_en`, kiểm baseline, `label_field`, `scorer_code_sha256`), `tests/core/test_normalization.py` (15 ví dụ + bất biến),
`tests/core/test_leakage.py`, `tests/bench/test_registry.py`, `tests/infer/test_adapters.py`.
