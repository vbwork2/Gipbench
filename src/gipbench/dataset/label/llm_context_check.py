#!/usr/bin/env python3
"""LLM đọc một lượt phiếu nghe (listen.csv) và đánh dấu từ nghi sai theo ngữ cảnh câu, để người nghe lại kỹ chỗ đó.

Nó bắt được loại lỗi mà so khớp ASR (cột `hint`) bỏ sót: cả nhãn lẫn ASR cùng nghe nhầm một từ thành từ khác đọc gần giống,
nhưng từ đó không hợp nghĩa với các từ trước / sau. Ba loại:
  ctx    từ (tiếng Việt) không hợp ngữ cảnh / ngữ pháp với từ xung quanh, có thể là nghe nhầm;
  en     từ trong <en> không hợp ngữ cảnh, có thể là một từ Anh khác (hoặc vốn là tiếng Việt);
  untag  từ / cụm trông như tiếng Anh hoặc phiên âm tiếng Anh nhưng không gắn <en>.
LLM chỉ ĐÁNH DẤU để nghe lại, không sửa nhãn. Quyết định vẫn là của người nghe.

Kết quả ghi vào cột `llm_suspects` của phiếu (JSON: [{"i": vị trí từ, "w": từ, "kind", "maybe", "why"}]). review_bench.html
gạch chân sóng các từ đó, liệt kê lý do, và phím F nhảy tới câu kế tiếp có từ bị nghi. Các cột khác giữ nguyên.

Gọi OpenAI Chat Completions (key ở biến môi trường OPENAI_API_KEY; OPENAI_BASE_URL nếu dùng endpoint khác). Câu đã hỏi được
lưu ở <dir>/llm_check.jsonl (khoá: id + sha nhãn + model + phiên bản prompt), nên chạy lại chỉ hỏi câu mới / câu đổi nhãn.

  $env:OPENAI_API_KEY = "..."       (bash: export OPENAI_API_KEY=...)
  python -m gipbench.dataset.label.llm_context_check --dir data/listen/pending_all --limit 5 --dry-run   # xem prompt
  python -m gipbench.dataset.label.llm_context_check --dir data/listen/pending_all --limit 20             # thử 20 câu
  python -m gipbench.dataset.label.llm_context_check --dir data/listen/pending_all                        # cả phiếu
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

PROMPT_VERSION = "ctx-v2"
KINDS = ("ctx", "en", "untag")
MAX_SUSPECTS = 6
COL = "llm_suspects"
CACHE = "llm_check.jsonl"

SYSTEM = f"""Bạn kiểm tra nhãn phiên âm (transcript) của audio tiếng Việt nói tự nhiên (podcast, phỏng vấn), có lúc chen \
từ tiếng Anh (code-switch). Nhãn do máy sinh và người sẽ nghe lại. Việc của bạn: chỉ ra các từ CÓ VẺ bị nghe nhầm, dựa vào \
ngữ cảnh của câu, để người nghe lại đúng chỗ đó. Bạn KHÔNG nghe được audio.

Câu được đưa dưới dạng danh sách từ đánh số "i:từ"; từ tiếng Anh đã gắn tag hiện là "i:[EN]từ". Có thể kèm bản của các ASR \
khác (cũng có thể sai) để tham khảo.

Chỉ có một câu hỏi: từ này có thể đã bị nghe nhầm thành một từ KHÁC đọc gần giống không? Đánh dấu một từ khi:
- "ctx": từ tiếng Việt không hợp nghĩa với các từ ngay trước và sau, và có một từ khác đọc gần giống (sai dấu, sai phụ \
âm đầu / vần) hợp nghĩa hơn hẳn. "maybe" phải là từ đọc gần giống đó;
- "en": từ tiếng Anh (đã gắn [EN]) không hợp ngữ cảnh câu, nhiều khả năng là một từ Anh khác đọc gần giống, hoặc vốn là \
tiếng Việt;
- "untag": từ / cụm trông như tiếng Anh hoặc phiên âm từ tiếng Anh (vd "mít tinh", "sô", "ca mê ra" đọc theo kiểu Anh) \
mà chưa gắn [EN]; hoặc tên riêng / thương hiệu tiếng Anh viết như tiếng Việt.

KHÔNG đánh dấu (người nói thật sự nói như vậy, nhãn phải ghi đúng như thế):
- từ đệm và tiếng đệm của văn nói (à, ừ, ờ, thì, mà, là, á, nha, nè, hả, ạ, đấy, đó...);
- lặp từ / lặp cụm, nói vấp, sửa lời giữa chừng (vd "một cái connection một cái sự kết nối", "chia sẻ chia sẻ", \
"có đang có", "nhiều hơn cái"), từ thừa hoặc câu thiếu từ: đó là cách nói, không phải lỗi nghe;
- câu bị cắt ở đầu / cuối, văn nói lủng củng nhưng vẫn hiểu được, chọn từ chưa hay nhưng vẫn hợp nghĩa;
- biến thể văn nói, chính tả và phương ngữ: ông ý / ông ấy, dĩa / đĩa, kỉ / kỷ, ba / bố, tụi / bọn, ngàn / nghìn;
- số đọc thành chữ, kể cả đọc từng chữ số (vd "hai không hai hai" = 2022), tên riêng tiếng Việt viết thường.
Chỉ đánh dấu khi khá chắc; phần lớn câu sẽ không có gì, khi đó trả danh sách rỗng. Tối đa {MAX_SUSPECTS} từ mỗi câu. \
Một lỗi trải nhiều từ thì chỉ đánh dấu từ đầu tiên.

Trả về đúng một JSON object:
{{"suspects": [{{"i": <số thứ tự từ>, "w": "<từ tại vị trí i, chép đúng>", "kind": "ctx|en|untag", \
"maybe": "<từ / cụm có lẽ đúng, hoặc rỗng>", "why": "<lý do ngắn, tiếng Việt, tối đa 15 từ>"}}]}}"""

EN_RE = re.compile(r"<en(?:\s[^>]*)?>([\s\S]*?)</en>")


def tokens(text_tagged: str) -> List[Dict]:
    """Như tokenizeTagged trong review_bench.html: [{w, en}], cùng thứ tự và chỉ số với từ hiện trên trang."""
    out, last, t = [], 0, text_tagged or ""
    for m in EN_RE.finditer(t):
        out += [{"w": w, "en": False} for w in t[last:m.start()].split()]
        out += [{"w": w, "en": True} for w in m.group(1).split()]
        last = m.end()
    out += [{"w": w, "en": False} for w in t[last:].split()]
    return out


def asr_alts(hint: str) -> Dict[str, str]:
    """Bản ASR khác trong cột hint ("tên (lệch n): ... || ..."), bỏ dấu ⟦ ⟧ và ⟦∅⟧."""
    alts = {}
    for part in (hint or "").split("||"):
        m = re.match(r"\s*([\w.\-]+) \(lệch \d+\):\s*(.*)$", part.strip())
        if m:
            txt = re.sub(r"⟦∅⟧", " ", m.group(2))
            alts[m.group(1)] = " ".join(re.sub(r"[⟦⟧]", "", txt).split())
    return alts


def user_prompt(row: Dict) -> str:
    ts = tokens(row.get("text_tagged", ""))
    lines = ["Câu:", " ".join(f"{k}:{'[EN]' if t['en'] else ''}{t['w']}" for k, t in enumerate(ts))]
    alts = asr_alts(row.get("hint", ""))
    if alts:
        lines += ["", "ASR khác (tham khảo, có thể sai):"] + [f"- {k}: {v}" for k, v in alts.items()]
    return "\n".join(lines)


def text_sha(row: Dict) -> str:
    return hashlib.sha256((row.get("text_tagged") or "").encode("utf-8")).hexdigest()[:16]


def clean_suspects(raw, ts: List[Dict]) -> List[Dict]:
    """Kiểm trả lời của LLM: vị trí phải trỏ đúng từ (lệch thì lấy chỗ có từ đó gần nhất), loại hợp lệ, không trùng."""
    items = raw.get("suspects") if isinstance(raw, dict) else None
    out, seen = [], set()
    for s in items if isinstance(items, list) else []:
        if not isinstance(s, dict):
            continue
        w = str(s.get("w") or "").strip()
        try:
            i = int(s.get("i"))
        except (TypeError, ValueError):
            i = -1
        if not (0 <= i < len(ts) and ts[i]["w"] == w):
            hits = [k for k, t in enumerate(ts) if t["w"] == w]
            if not hits:
                continue
            i = min(hits, key=lambda k: abs(k - i))
        kind = str(s.get("kind") or "").strip()
        if kind not in KINDS:
            kind = "en" if ts[i]["en"] else "ctx"
        maybe = str(s.get("maybe") or "").strip()[:60]
        # "nghe nhầm" phải có từ khác để so: ctx không có đề xuất (thường là chê thừa / thiếu từ) hoặc đề xuất trùng chính
        # từ đó thì bỏ
        if i in seen or maybe.lower() == ts[i]["w"].lower() or (kind == "ctx" and not maybe):
            continue
        seen.add(i)
        out.append({"i": i, "w": ts[i]["w"], "kind": kind, "maybe": maybe,
                    "why": str(s.get("why") or "").strip()[:160]})
    return sorted(out, key=lambda s: s["i"])[:MAX_SUSPECTS]


def openai_chat(model: str, system: str, user: str, *, api_key: str, base_url: str,
                reasoning_effort: Optional[str] = None, timeout: float = 120, retries: int = 5) -> Dict:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            content = resp["choices"][0]["message"]["content"] or "{}"
            return {"json": json.loads(content), "usage": resp.get("usage", {})}
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            raise RuntimeError(f"OpenAI lỗi: {e}") from None
    raise RuntimeError("OpenAI: hết lượt thử")


def read_sheet(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def write_sheet(path: Path, header: List[str], rows: List[Dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in header})
    os.replace(tmp, path)


def load_cache(path: Path) -> Dict[str, Dict]:
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                out[x["key"]] = x
    return out


def decided_ids(paths: List[str]) -> set:
    out = set()
    for p in paths or []:
        _, rows = read_sheet(Path(p))
        out |= {r["id"] for r in rows if (r.get("decision") or "").strip().lower() in ("ok", "fix", "drop")}
    return out


def load_dotenv(path: Path) -> List[str]:
    """Nạp biến OPENAI_* từ file .env (KEY=VALUE, bỏ dòng # và ngoặc kép) nếu môi trường chưa có. Trả tên biến đã nạp."""
    got = []
    if not path.is_file():
        return got
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith("OPENAI_") and v and not os.environ.get(k):
            os.environ[k] = v
            got.append(k)
    return got


def run(args, chat: Optional[Callable[[str, str], Dict]] = None) -> int:
    d = Path(args.dir)
    sheet = d / args.sheet
    header, rows = read_sheet(sheet)
    if "id" not in header or "text_tagged" not in header:
        raise SystemExit(f"{sheet}: thiếu cột id / text_tagged")
    skip = decided_ids(args.skip_decided)
    todo = [r for r in rows if r["id"] not in skip and (r.get("decision") or "").strip().lower() not in ("ok", "fix", "drop")]
    if args.limit:
        todo = todo[:args.limit]
    key = lambda r: f"{r['id']}|{text_sha(r)}|{args.model}|{PROMPT_VERSION}"  # noqa: E731
    cache_path = d / CACHE
    cache = load_cache(cache_path)
    ask = [r for r in todo if key(r) not in cache]
    print(f"{len(rows)} câu trong phiếu, kiểm {len(todo)} (bỏ {len(rows) - len(todo)} đã quyết định / ngoài --limit), "
          f"cần hỏi LLM {len(ask)}, có sẵn trong cache {len(todo) - len(ask)}; model {args.model}")
    if args.dry_run:
        for r in ask[:3]:
            print(f"\n--- {r['id']} ---\n{user_prompt(r)}")
        print(f"\n--- system ({len(SYSTEM)} ký tự) ---\n{SYSTEM}")
        return 0
    if ask and chat is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Chưa có OPENAI_API_KEY (biến môi trường hoặc .env ở gốc repo)")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        chat = lambda s, u: openai_chat(args.model, s, u, api_key=api_key, base_url=base,  # noqa: E731
                                        reasoning_effort=args.reasoning_effort)
    errors, tok_in, tok_out = [], 0, 0
    if ask:
        with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(chat, SYSTEM, user_prompt(r)): r for r in ask}
            for n, fu in enumerate(as_completed(futs), 1):
                r = futs[fu]
                try:
                    res = fu.result()
                except Exception as e:  # một câu lỗi không làm hỏng cả lượt; chạy lại sẽ hỏi lại câu đó
                    errors.append((r["id"], str(e)))
                    continue
                sus = clean_suspects(res.get("json"), tokens(r.get("text_tagged", "")))
                u = res.get("usage") or {}
                tok_in += u.get("prompt_tokens", 0) or 0
                tok_out += u.get("completion_tokens", 0) or 0
                rec = {"key": key(r), "id": r["id"], "model": args.model, "prompt_version": PROMPT_VERSION,
                       "suspects": sus, "raw": res.get("json"), "usage": u}
                cache[rec["key"]] = rec
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cf.flush()
                if n % 25 == 0 or n == len(ask):
                    print(f"  {n}/{len(ask)} câu, token vào {tok_in} / ra {tok_out}, lỗi {len(errors)}", flush=True)
    if COL not in header:
        header.append(COL)
    by_id = {r["id"]: key(r) for r in todo}
    n_flag, n_word, kinds = 0, 0, {k: 0 for k in KINDS}
    for r in rows:
        rec = cache.get(by_id.get(r["id"], ""))
        if rec is None:
            continue  # câu không kiểm lượt này: giữ giá trị cũ của cột
        # lọc lại từ trả lời gốc, để đổi luật lọc không phải hỏi lại API
        sus = clean_suspects(rec["raw"], tokens(r.get("text_tagged", ""))) if "raw" in rec else rec["suspects"]
        r[COL] = json.dumps(sus, ensure_ascii=False, separators=(",", ":")) if sus else ""
        n_flag += bool(sus)
        n_word += len(sus)
        for s in sus:
            kinds[s["kind"]] += 1
    write_sheet(sheet, header, rows)
    print(f"LLM nghi {n_word} từ ở {n_flag} / {len(todo)} câu ({', '.join(f'{k} {v}' for k, v in kinds.items())}) "
          f"-> {sheet} (cột {COL})")
    for i, e in errors[:5]:
        print(f"  LỖI {i}: {e[:300]}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} câu lỗi, chạy lại lệnh để hỏi lại", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[4] / ".env")  # gốc repo; trước argparse để OPENAI_MODEL trong .env có hiệu lực
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="thư mục phiếu, vd data/listen/pending_all")
    ap.add_argument("--sheet", default="listen.csv")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    ap.add_argument("--reasoning-effort", default=None, help="vd low / medium (chỉ cho model suy luận)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="chỉ kiểm N câu đầu (để thử)")
    ap.add_argument("--skip-decided", nargs="*", default=[], help="phiếu đã nghe (vd listen_reviewed.csv): bỏ câu đã quyết định")
    ap.add_argument("--dry-run", action="store_true", help="in prompt, không gọi API")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
