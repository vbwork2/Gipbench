"""Đọc manifest JSONL của bench (1 utterance / dòng). Đặc tả: xem SCHEMA.md."""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REQUIRED = ["id", "split", "audio", "duration", "text_tagged", "speaker", "source"]
META = ["region", "gender", "channel", "domain"]
BENCH_SPLITS = ["B1", "B2", "B3", "B4", "B5"]
CS_SPLITS = ["B1", "B2", "B3", "B4"]
TARGET_HOURS = {"B1": 3.0, "B2": 2.0, "B3": 1.5, "B4": 2.5, "B5": 2.0}


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    base = Path(path).resolve().parent
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: JSON lỗi: {e}")
            r["_line"] = ln
            if "audio" in r:
                p = Path(r["audio"])
                r["_audio_path"] = str(p if p.is_absolute() else base / p)
            rows.append(r)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")


def span_meta(r: Dict, i: int) -> Dict:
    m = r.get("en_meta") or []
    return m[i] if i < len(m) and isinstance(m[i], dict) else {}


def filter_splits(rows: List[Dict], splits: Optional[List[str]]) -> List[Dict]:
    if not splits:
        return rows
    s = set(splits)
    return [r for r in rows if r.get("split") in s]
