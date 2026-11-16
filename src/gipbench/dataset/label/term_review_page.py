#!/usr/bin/env python3
"""Trang duyệt NHANH theo thuật ngữ (thay cho duyệt từng câu): mỗi thuật ngữ còn cờ một thẻ, kèm vài câu ví dụ có audio,
bấm en / vi / bỏ tag. Kết quả xuất ra TSV đúng định dạng configs/dataset/labeling/term_decisions.tsv.

  # 1. sinh trang (ghi cạnh manifest để đường dẫn audio tương đối chạy được khi mở bằng file://)
  python -m gipbench.dataset.label.term_review_page --manifest data/raw/ggroup/b1_candidates.jsonl
  # 2. mở data/raw/ggroup/term_review.html bằng Chrome / Edge, duyệt, bấm "Xuất TSV" (tải về term_decisions_review.tsv)
  # 3. gộp vào file quyết định (giữ dòng cũ, thêm / cập nhật dòng mới), rồi chạy lại converter + build_bench
  python -m gipbench.dataset.label.term_review_page --merge ~/Downloads/term_decisions_review.tsv

Thuật ngữ cần duyệt = needed_terms() của core.term_decisions (cờ abbr_reading / vi_word / suspect / maybe_name / name /
ambiguous_joined), giống luật `build_bench --accept trusted-text`. Đề xuất ban đầu lấy từ reports/history/
term_decisions_ask.csv (cột decision, do Claude điền); quyết định đã có trong term_decisions.tsv được hiện là "đã chốt".
Trang tự chứa (không mạng, không server); tiến độ lưu tạm trong localStorage của trình duyệt.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.manifest import read_jsonl
from gipbench.core.term_decisions import (DECISIONS, DEFAULT_TERM_DECISIONS, load_term_decisions, needed_terms,
                                          norm_term, read_term_rows, write_term_rows)

DEFAULT_ASK = P.REPO / "reports" / "history" / "term_decisions_ask.csv"


def load_ask(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {norm_term(r["term"]): {"decision": (r.get("decision") or "").strip(), "note": r.get("note") or ""}
                for r in csv.DictReader(f) if r.get("term")}


def collect(manifests, out_dir: Path, n_examples: int):
    terms = defaultdict(lambda: {"rows": [], "flags": set()})
    for m in manifests:
        for r in read_jsonl(m):
            need, _ = needed_terms(r)
            for t, kinds in need.items():
                terms[t]["rows"].append(r)
                terms[t]["flags"] |= kinds
    out = []
    for t, v in terms.items():
        rows = sorted(v["rows"], key=lambda r: (abs((r.get("duration") or 5) - 4), r["id"]))[:n_examples]
        ex = [{"id": r["id"], "text": r.get("text_tagged", ""), "dur": round(r.get("duration") or 0, 1),
               "audio": os.path.relpath(r["_audio_path"], out_dir).replace(os.sep, "/") if r.get("_audio_path") else ""}
              for r in rows]
        out.append({"term": t, "n": len(v["rows"]), "flags": sorted(v["flags"]), "examples": ex})
    return sorted(out, key=lambda x: (-x["n"], x["term"]))


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duyệt thuật ngữ</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#dcdcd6;--en:#0b57d0;--enbg:#e8f0fe;--hl:#fff1b8;
 --ok:#1e7d32;--vi:#8a5a00;--drop:#b3261e;--sel:#0b57d0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ececea;--mut:#a3a39d;--card:#212120;
 --line:#3a3a37;--en:#8ab4f8;--enbg:#1f2a3d;--hl:#4a3f10;--ok:#6dd58c;--vi:#ffd27a;--drop:#f2b8b5;--sel:#8ab4f8}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ececea;--mut:#a3a39d;--card:#212120;--line:#3a3a37;--en:#8ab4f8;--enbg:#1f2a3d;
 --hl:#4a3f10;--ok:#6dd58c;--vi:#ffd27a;--drop:#f2b8b5;--sel:#8ab4f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;
 padding:10px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0}button,select{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;
 padding:5px 10px;font:inherit;cursor:pointer}.bar{flex:1;min-width:120px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}.mut{color:var(--mut);font-size:13px}
main{max-width:980px;margin:0 auto;padding:12px 16px 80px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.card.cur{outline:2px solid var(--sel)}.top{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.term{font-size:20px;font-weight:600}.tag{font-size:12px;border:1px solid var(--line);border-radius:10px;padding:0 7px;color:var(--mut)}
.dec{display:flex;gap:6px;margin-left:auto}.dec button.on.en{background:var(--en);color:var(--bg)}
.dec button.on.vi{background:var(--vi);color:var(--bg)}.dec button.on.drop_tag{background:var(--drop);color:var(--bg)}
.ex{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}.ex .t{flex:1;min-width:220px}
en{color:var(--en);background:var(--enbg);border-radius:3px;padding:0 2px}mark{background:var(--hl);color:inherit;border-radius:3px}
.sug{font-size:13px;color:var(--mut)}.sug b{color:var(--fg)}.help{font-size:13px;color:var(--mut);margin:8px 0}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
@media (max-width:600px){.dec{margin-left:0;width:100%}.dec button{flex:1}}
</style></head><body>
<header><h1>Duyệt thuật ngữ</h1><span id="cnt" class="mut"></span><div class="bar"><i id="prog"></i></div>
<select id="flt"><option value="todo">Chưa quyết</option><option value="all">Tất cả</option><option value="done">Đã quyết</option></select>
<button id="acc">Nhận mọi đề xuất còn lại</button><button id="exp">Xuất TSV</button></header>
<main><p class="help">Mỗi thẻ là một thuật ngữ; quyết định áp cho MỌI câu chứa nó. <b>en</b> = gắn &lt;en&gt; (đọc / là từ tiếng Anh),
<b>vi</b> = không gắn (đọc kiểu Việt / từ Việt), <b>bỏ tag</b> = không phải từ Anh, cũng không phải từ Việt (lỗi gõ, mã...).
Phím: <kbd>1</kbd> en · <kbd>2</kbd> vi · <kbd>3</kbd> bỏ tag · <kbd>A</kbd> nhận đề xuất · <kbd>U</kbd> bỏ quyết định ·
<kbd>Space</kbd> nghe câu đầu · <kbd>J</kbd>/<kbd>K</kbd> thẻ sau/trước. Tiến độ tự lưu trong trình duyệt; xong thì bấm
<b>Xuất TSV</b>. Chữ xanh = đang được gắn &lt;en&gt;; nền vàng = thuật ngữ đang duyệt.</p><div id="list"></div></main>
<script>
const DATA = __DATA__;
const KEY = "term_review:" + DATA.name;
const LABEL = {en: "en", vi: "vi", drop_tag: "bỏ tag"};
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) {} };
let cur = 0, shown = [];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]);
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function renderText(t, term) {
  const h = esc(t).replace(/&lt;en(?:\s[^&]*)?&gt;(.*?)&lt;\/en&gt;/g, "<en>$1</en>");
  const re = new RegExp("(^|[^\\p{L}\\p{N}])(" + term.split(" ").map(reEsc).join("[\\s\\-]+") + ")(?=$|[^\\p{L}\\p{N}])", "giu");
  return h.replace(re, "$1<mark>$2</mark>");
}
const decided = t => dec[t.term] || t.locked || "";
function draw() {
  const f = document.getElementById("flt").value;
  shown = DATA.terms.filter(t => f === "all" || (f === "todo" ? !decided(t) : !!decided(t)));
  if (cur >= shown.length) cur = Math.max(0, shown.length - 1);
  document.getElementById("list").innerHTML = shown.map((t, i) => {
    const d = decided(t);
    const btn = k => `<button class="${k} ${d === k ? "on" : ""}" data-i="${i}" data-k="${k}">${LABEL[k]}</button>`;
    const sug = t.locked ? `đã chốt trong term_decisions.tsv: <b>${LABEL[t.locked]}</b>` :
      (t.suggest ? `đề xuất: <b>${LABEL[t.suggest] || esc(t.suggest)}</b>${t.note ? " · " + esc(t.note) : ""}` : "chưa có đề xuất");
    return `<div class="card ${i === cur ? "cur" : ""}" id="c${i}"><div class="top"><span class="term">${esc(t.term)}</span>
      <span class="tag">${t.n} câu</span>${t.flags.map(x => `<span class="tag">${esc(x)}</span>`).join("")}
      <div class="dec">${btn("en")}${btn("vi")}${btn("drop_tag")}</div></div><div class="sug">${sug}</div>
      ${t.examples.map((e, j) => `<div class="ex"><audio id="a${i}_${j}" controls preload="none" src="${esc(e.audio)}"></audio>
      <span class="t">${renderText(e.text, t.term)} <span class="mut">(${e.dur} s)</span></span></div>`).join("")}</div>`;
  }).join("") || "<p>Không còn thuật ngữ nào trong bộ lọc này.</p>";
  const done = DATA.terms.filter(decided).length;
  document.getElementById("cnt").textContent = `${done} / ${DATA.terms.length} đã quyết`;
  document.getElementById("prog").style.width = (100 * done / DATA.terms.length) + "%";
}
function focusCur() {
  document.querySelectorAll(".card").forEach((c, i) => c.classList.toggle("cur", i === cur));
  const el = document.getElementById("c" + cur); if (el) el.scrollIntoView({block: "nearest"});
}
function setDec(i, k) {
  const t = shown[i]; if (!t) return;
  if (k) dec[t.term] = k; else delete dec[t.term];
  save();
  const f = document.getElementById("flt").value;
  draw();
  if (f !== "todo" && k) cur = Math.min(i + 1, shown.length - 1);
  focusCur();
}
function notice(msg) { const p = document.createElement("p"); p.className = "card"; p.textContent = msg;
  document.querySelector("main").prepend(p); setTimeout(() => p.remove(), 6000); }
document.getElementById("list").onclick = e => {
  const b = e.target.closest("button[data-k]"); if (!b) return;
  cur = +b.dataset.i; const t = shown[cur];
  setDec(cur, (dec[t.term] === b.dataset.k) ? "" : b.dataset.k);
};
document.getElementById("flt").onchange = () => { cur = 0; draw(); };
document.getElementById("acc").onclick = () => {
  let n = 0; for (const t of DATA.terms) if (!decided(t) && t.suggest in LABEL) { dec[t.term] = t.suggest; n++; }
  save(); draw(); notice(`Đã nhận ${n} đề xuất. Nên lướt lại ở bộ lọc "Đã quyết".`);
};
document.getElementById("exp").onclick = () => {
  const day = new Date().toISOString().slice(0, 10);
  const rows = DATA.terms.filter(t => dec[t.term]).map(t => `${t.term}\t${dec[t.term]}\tngười dùng duyệt ${day} (term_review)`);
  const blob = new Blob(["term\tdecision\tnote\n" + rows.join("\n") + "\n"], {type: "text/tab-separated-values"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "term_decisions_review.tsv"; a.click();
  notice(`Đã xuất ${rows.length} quyết định.`);
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "AUDIO") return;
  const k = e.key.toLowerCase(), t = shown[cur];
  if (k === "1") setDec(cur, "en"); else if (k === "2") setDec(cur, "vi"); else if (k === "3") setDec(cur, "drop_tag");
  else if (k === "u") setDec(cur, ""); else if (k === "a" && t && t.suggest in LABEL) setDec(cur, t.suggest);
  else if (k === "j" || e.key === "ArrowDown") { cur = Math.min(cur + 1, shown.length - 1); focusCur(); }
  else if (k === "k" || e.key === "ArrowUp") { cur = Math.max(cur - 1, 0); focusCur(); }
  else if (e.key === " ") { const a = document.getElementById(`a${cur}_0`); if (a) { a.paused ? a.play() : a.pause(); } }
  else return;
  e.preventDefault();
});
draw();
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", help="manifest ứng viên (vd data/raw/ggroup/b1_candidates.jsonl)")
    ap.add_argument("--out", help="file HTML (mặc định: term_review.html cạnh manifest đầu tiên)")
    ap.add_argument("--ask", default=str(DEFAULT_ASK), help="phiếu đề xuất (cột term, decision, note)")
    ap.add_argument("--decisions", default=str(DEFAULT_TERM_DECISIONS))
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--merge", help="TSV xuất từ trang: gộp vào --decisions (thêm / cập nhật theo term)")
    args = ap.parse_args(argv)
    if args.merge:
        old = read_term_rows(args.decisions)
        new = {norm_term(r["term"]): r for r in read_term_rows(args.merge) if (r.get("decision") or "").strip()}
        bad = [t for t, r in new.items() if r["decision"].strip() not in DECISIONS]
        if bad:
            raise SystemExit(f"decision lạ cho: {bad[:5]}")
        seen, rows, changed = set(), [], 0
        for r in old:
            t = norm_term(r["term"])
            seen.add(t)
            if t in new and (r.get("decision") or "").strip() != new[t]["decision"].strip():
                changed += 1
                rows.append(new[t])
            else:
                rows.append(r)
        added = [r for t, r in new.items() if t not in seen]
        write_term_rows(args.decisions, rows + added)
        print(f"{args.decisions}: +{len(added)} thuật ngữ mới, {changed} thuật ngữ đổi quyết định")
        return
    if not args.manifest:
        ap.error("cần --manifest (sinh trang) hoặc --merge")
    out = Path(args.out or Path(args.manifest[0]).parent / "term_review.html")
    terms = collect(args.manifest, out.parent, args.examples)
    ask, locked = load_ask(args.ask), load_term_decisions(args.decisions)
    for t in terms:
        a = ask.get(t["term"], {})
        t.update(suggest=a.get("decision", ""), note=a.get("note", ""), locked=locked.get(t["term"], ""))
    data = {"name": out.parent.name + "/" + out.name, "created": time.strftime("%Y-%m-%d %H:%M"), "terms": terms}
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    todo = sum(1 for t in terms if not t["locked"])
    print(f"{out}: {len(terms)} thuật ngữ ({todo} chưa chốt, {sum(1 for t in terms if t['suggest'])} có đề xuất), "
          f"{sum(t['n'] for t in terms)} lượt câu")


if __name__ == "__main__":
    main()
