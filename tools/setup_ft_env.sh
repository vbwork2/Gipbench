#!/usr/bin/env bash
# A1 (decisions §12.7): venv riêng cho fine-tune: torch 2.15.0+cu132, k2 (wheel khớp torch/CUDA), lhotse, requirements
# của icefall (bỏ style check / cantonese; onnx* tuỳ chọn), requirements.txt của gipbench. icefall: git clone --depth 1
# https://github.com/k2-fsa/icefall ~/icefall (commit đã thử: f8c8578c, 12/11/2026). Không có torchaudio 2.15.0: không cần.
# Kiểm tra CUDA bằng lệnh nhỏ: python -c "import torch; print(torch.cuda.is_available())"; python -m k2.version
set -e
export PATH=$HOME/.local/bin:$PATH
V=$HOME/.venvs/gipbench-ft
echo "== $(date '+%F %T') tạo venv $V"
[ -x $V/bin/python ] || uv venv -q -p 3.11 $V
. $V/bin/activate
echo "== torch 2.15.0 cu132 (cache uv)"
uv pip install -q "torch==2.15.0" --index-url https://download.pytorch.org/whl/cu132
uv pip install -q "torchaudio==2.15.0" --index-url https://download.pytorch.org/whl/cu132 || echo "LƯU Ý: không cài được torchaudio 2.15.0 (icefall zipformer không bắt buộc)"
echo "== k2"
uv pip install -q "k2==1.24.4.dev20261112+cuda13.2.torch2.15.0" -f https://k2-fsa.github.io/k2/cuda.html
echo "== lhotse + scipy (Resample của lhotse khi không có torchaudio) + requirements icefall (bỏ style check, cantonese)"
grep -v "^#\|black\|isort\|flake8\|pycantonese\|pypinyin\|onnx\|^$" $HOME/icefall/requirements.txt > /tmp/icefall_req.txt
uv pip install -q lhotse scipy "pypinyin==0.50.0" -r /tmp/icefall_req.txt
uv pip install -q "onnx>=1.15.0" "onnxruntime>=1.16.3" || echo "LƯU Ý: onnx / onnxruntime lỗi"
uv pip install -q onnxoptimizer onnxsim onnxconverter_common || echo "LƯU Ý: onnxoptimizer / onnxsim / onnxconverter_common lỗi (chỉ cần khi export tối ưu)"
echo "== requirements gipbench"
uv pip install -q -r /mnt/d/code/ZAIC/gipbench/requirements.txt || echo "LƯU Ý: một phần requirements.txt lỗi"
echo "== phiên bản"
python -c "import torch, k2, lhotse; print('torch', torch.__version__, 'cuda', torch.version.cuda); print("k2", k2.__dev_version__); print('lhotse', lhotse.__version__)"
uv pip list 2>/dev/null | grep -i "^torch\|^k2\|^lhotse\|^kaldi\|^sentencepiece\|^onnx\|^sherpa" || true
echo "== $(date '+%F %T') XONG"
