#!/usr/bin/env bash
# Sao lưu dữ liệu KHÔNG có trong git và khó tạo lại (decisions §12.6), KHÔNG nén audio:
#   data/raw/**/*.jsonl, data/raw/**/*_review.csv, data/raw/**/leak_block.txt, data/raw/**/speaker_clusters.tsv,
#   configs/, results/asr_hyps/ (cache hyps ASR, nhiều GPU-giờ)
# vào <dest>/<ngày>.tar.gz (đã có thì thêm giờ phút: <ngày>_HHMM.tar.gz).
#
#   bash tools/backup.sh                       # dest mặc định: D:\gipbench_backup (/mnt/d/... trong WSL, /d/... Git Bash)
#   bash tools/backup.sh --dest DIR --root DIR  # thử / kiểm thử
#
# File đang được ghi (vd hyps của job ASR nền) được chép ở trạng thái lúc đọc; tar không dừng vì file đổi trong lúc đọc.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    *) echo "tham số lạ: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$DEST" ]; then
  if [ -d /mnt/d ]; then DEST=/mnt/d/gipbench_backup; elif [ -d /d ]; then DEST=/d/gipbench_backup; else
    echo "không tìm thấy ổ D:, truyền --dest" >&2; exit 2; fi
fi
mkdir -p "$DEST"
STAMP="$(date +%F)"
OUT="$DEST/$STAMP.tar.gz"
[ -e "$OUT" ] && OUT="$DEST/${STAMP}_$(date +%H%M).tar.gz"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
cd "$ROOT"
{
  [ -d data/raw ] && find data/raw -type f \( -name "*.jsonl" -o -name "*_review.csv" -o -name "leak_block.txt" \
      -o -name "speaker_clusters.tsv" \) -not -path "*/audio/*"
  [ -d configs ] && find configs -type f
  [ -d results/asr_hyps ] && find results/asr_hyps -type f
} | sort > "$LIST"
N="$(wc -l < "$LIST")"
if [ "$N" = 0 ]; then echo "không có file nào để sao lưu trong $ROOT" >&2; exit 1; fi
tar --warning=no-file-changed -czf "$OUT" -T "$LIST" || [ $? = 1 ]
SIZE="$(du -h "$OUT" | cut -f1)"
RAW="$(tr '\n' '\0' < "$LIST" | du -ch --files0-from=- 2>/dev/null | tail -1 | cut -f1)"
echo "Đã sao lưu $N file ($RAW trước nén) -> $OUT ($SIZE)"
