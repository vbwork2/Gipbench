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

## 3. Vì sao CS-WER strict = lenient trên v0-draft (đã điều tra)

Catalog `data/bench_v0_draft/catalog_v0_draft.tsv` có 1 933 thuật ngữ, **0** dòng có cột `phonetic`, **0** dòng có cột
`variants`. Bảng lenient vì vậy trùng bảng strict, nên hai số bằng nhau **do dữ liệu**, không phải lỗi scorer.
score.py có ghi cờ `lenient_available: false` cho trường hợp này, nhưng `summary.json` của các lần chạy v0-draft được tạo
trước khi có cờ đó. Muốn có số lenient thì phải soạn cột phonetic **từ nguồn không phải test** (vd từ điển phiên âm).
Không được lấy từ lỗi của model trên test.

## 4. Kiểm hyps

`check_hyps`: id thiếu, id thừa, id trùng (trong hyps và trong manifest), số hyp rỗng. Final mode: lệch bất kỳ → thoát mã 2.
Diagnostic: hyp thiếu tính là rỗng và ghi cảnh báo (giống score.py).

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
CER, kiểm hyps, bootstrap, final mode), `tests/core/test_normalization.py` (15 ví dụ + bất biến),
`tests/core/test_leakage.py`, `tests/bench/test_registry.py`, `tests/infer/test_adapters.py`.
