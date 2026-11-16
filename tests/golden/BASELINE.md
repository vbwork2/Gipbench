# Kết quả test trước refactor (commit 1570a42, 16/11/2026, WSL venv ~/.venvs/gipbench, Python 3.11.18, CPU)

| Test | Kết quả | Thời gian |
| --- | --- | --- |
| `bash tools/selftest.sh` | ĐẠT (rc 0) | 5 s |
| `python tools/test_converters.py` | ĐẠT: "TẤT CẢ KIỂM THỬ CONVERTER ĐẠT" | 23 s |
| `python tools/test_tts.py` | ĐẠT: "toàn bộ kiểm tra chuỗi TTS" | 10 s |
| `python tools/test_span_metrics.py` | ĐẠT: "Tất cả đạt" | 2 s |
| `python tools/test_dev_tools.py` | ĐẠT: 11/11 | < 1 s |
| `python tools/test_hard_mining.py` | ĐẠT: "Tất cả ca đạt" | 2 s |

Golden (`tests/golden/capture.py --layout old`): 90 file, 1 phút 37 giây; chạy lần 2 giống hệt lần 1.

Hành vi sẵn có được giữ nguyên trong mốc (không phải lỗi do refactor):
- `validate.py` trên bench nháp + train / dev của E3 v1b thoát mã 1 với 3 734 LỖI, phần lớn là "RÒ RỈ: thuật ngữ
  reserved_b3 có trong train" (ViMedCSS train chứa thuật ngữ B3). `ft/run_e3_v1b.sh` chạy validate với `|| true`.
