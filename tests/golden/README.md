# Golden test (refactor 3 pipeline)

Kiểm rằng refactor không đổi hành vi: chạy cùng các bước trên dữ liệu thật đang có (chỉ đọc `data/`, `results/`), so output
từng byte với mốc chụp trước refactor.

```bash
# WSL, venv CPU, tại gốc repo
source ~/.venvs/gipbench/bin/activate && export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=
python tests/golden/capture.py --layout new --out /tmp/golden_new        # ~2 phút CPU
python tests/golden/compare.py <thư mục mốc> /tmp/golden_new             # thoát mã 1 nếu khác
```

**Mốc** (`--layout old`) phải chụp từ code trước refactor (commit `1570a42`), ghi vào **cùng đường dẫn output** với lần chụp
mới (vài file chứa đường dẫn audio tương đối theo thư mục output), rồi mới đổi tên:

```bash
cp tests/golden/capture.py /tmp/capture.py && git checkout -q 1570a42
python /tmp/capture.py --repo . --layout old --out $SP/golden_run && mv $SP/golden_run $SP/golden_old
git checkout -q refactor/pipelines
python tests/golden/capture.py --layout new --out $SP/golden_run && mv $SP/golden_run $SP/golden_new
```
Mốc chụp ngày 16/11/2026 đã chạy 2 lần cho ra 90/90 file giống nhau (không có trường thay đổi theo lần chạy).

Các bước và lệnh old / new: bảng `SCRIPTS` trong `capture.py`. Kết quả test hiện có trước refactor: `BASELINE.md`.
