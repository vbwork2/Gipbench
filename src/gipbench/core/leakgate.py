"""Cổng rò rỉ train -> bench khoá, ĐÓNG KHI LỖI (audit 17/11/2026, R02 / R03; IP-02). Dùng trong model.run stage 0.

  run_gate(...)  kiểm MỌI bench khoá (flags.locked trong configs/dataset/releases.yaml) hợp với `leak_guard` của experiment,
                 so với mọi pool + dev + file text (TTS / LM) của experiment, bằng core.decontam.audit.

Khác `core.leak.LeakGuard` (lọc câu lúc dựng cut, chỉ so với một bench, so chuỗi chính xác): đây là cổng dừng cả experiment.
  * bench khoá không resolve được (chưa register / sha256 sai) -> critical "bench khoá chưa được kiểm" (không bỏ qua im lặng);
  * finding critical chưa được miễn trừ -> blocking khi mode = fail (model.run thoát mã 3 TRƯỚC khi curate / train);
  * mode = report vẫn ghi báo cáo, không dừng (chỉ dùng cho chẩn đoán có chủ ý).
Báo cáo ghi leak_report.json + leak_report.md; stage 6 chép vào artifact và ghi sha256 vào model_card.json.
Kiểm audio: audio_sha = hash (đọc file khi thiếu audio_sha256) | metadata (chỉ dùng audio_sha256 có sẵn, không đọc file) |
skip. Hash audio KHÔNG bắt được audio đã resample / cắt khác; trùng nguồn và clip gốc (source_key / clip_key) mới là kiểm
chính, và báo cáo luôn ghi rõ chế độ audio đã dùng.
"""
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from gipbench.core import release as R
from gipbench.core.decontam import audit, load_exemptions, overlap_counts, read_lines, sha256_file
from gipbench.core.manifest import read_jsonl

AUDIO_SHA_MODES = ("hash", "metadata", "skip")
DEFAULTS = {"mode": "fail", "exemptions": None, "reserved_terms": None, "text_files": [], "audio_sha": "metadata",
            "require_audio_sha": False, "near": False}


def gate_config(exp: Dict) -> Dict:
    """`leak_gate` của experiment YAML + mặc định; kiểm giá trị."""
    g = dict(DEFAULTS, **(exp.get("leak_gate") or {}))
    if g["mode"] not in ("fail", "report"):
        raise ValueError(f"leak_gate.mode phải là fail | report, không phải {g['mode']!r}")
    if g["audio_sha"] not in AUDIO_SHA_MODES:
        raise ValueError(f"leak_gate.audio_sha phải thuộc {AUDIO_SHA_MODES}, không phải {g['audio_sha']!r}")
    if g["require_audio_sha"] and g["audio_sha"] == "skip":
        raise ValueError("leak_gate.require_audio_sha mâu thuẫn với audio_sha: skip")
    return g


def guard_refs(exp: Dict) -> List[str]:
    """`leak_guard` có thể là một chuỗi hoặc danh sách."""
    g = exp.get("leak_guard")
    return [g] if isinstance(g, str) else list(g or [])


def plan_refs(guard: Sequence[str], locked: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(mọi bench sẽ kiểm theo thứ tự: leak_guard trước rồi bench khoá còn lại, bench khoá KHÔNG có trong leak_guard)."""
    refs = list(dict.fromkeys([*guard, *locked]))
    return refs, [r for r in locked if r not in guard]


def _compact(rep: Dict) -> Dict:
    return {"critical": rep["critical"], "review": rep["review"], "exempted": rep.get("exempted", 0),
            "stats": rep["stats"], "findings": rep["findings"]}


def run_gate(name: str, guard: Sequence[str], inputs: Sequence[Tuple[str, str]], *, locked: Optional[Sequence[str]] = None,
             mode: str = "fail", exemptions: Optional[str] = None, reserved_terms: Optional[str] = None,
             text_files: Sequence[str] = (), audio_sha: str = "metadata", require_audio_sha: bool = False,
             near: bool = False, out_dir: Optional[Path] = None) -> Dict:
    """inputs: [(nhãn, đường dẫn manifest)] của pool + dev (audio tuyệt đối). Trả về dict kết quả; `blocking` = nên dừng."""
    t0 = time.time()
    locked = R.locked_bench_refs() if locked is None else list(locked)
    refs, extra = plan_refs(guard, locked)
    ex = load_exemptions(exemptions)
    res: Dict = {"experiment": name, "mode": mode, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "git_commit": R.git_commit(), "audio_sha": audio_sha, "require_audio_sha": require_audio_sha, "near": near,
                 "guard_refs": list(guard), "locked_refs": locked, "extra_locked": extra,
                 "exemptions": exemptions, "exemptions_sha256": R.sha256(exemptions) if exemptions else None,
                 "reserved_terms": reserved_terms, "reserved_terms_sha256": R.sha256(reserved_terms) if reserved_terms else None,
                 "inputs": [], "text_files": list(text_files), "benches": {}, "unresolved": {}, "reserved": None}
    train: List[Dict] = []
    per_input: List[Tuple[str, List[Dict]]] = []
    for label, path in inputs:
        rows = read_jsonl(str(path))
        per_input.append((label, rows))
        train += rows
        res["inputs"].append({"label": label, "rows": len(rows)})
    for f in text_files:
        rows = [{"id": f"{Path(f).name}:{i}", "text": t} for i, t in enumerate(read_lines(f))]
        per_input.append((f"text:{Path(f).name}", rows))
        train += rows
        res["inputs"].append({"label": f"text:{Path(f).name}", "rows": len(rows)})
    if audio_sha == "hash":
        # băm audio train MỘT lần (audit chạy riêng từng bench, nếu không sẽ đọc lại mọi file train cho mỗi bench)
        for r in train:
            p = r.get("_audio_path")
            if not r.get("audio_sha256") and p and Path(p).exists():
                r["audio_sha256"] = sha256_file(p)
    for ref in refs:
        try:
            rel = R.resolve(ref)
            if rel.manifest is None:
                raise R.ReleaseError(f"{ref} không có manifest")
        except R.ReleaseError as e:
            res["unresolved"][ref] = str(e)
            continue
        test = read_jsonl(str(rel.manifest))
        rep = audit(test, train, check_audio_sha=audio_sha != "skip", hash_files=audio_sha == "hash",
                    require_audio_sha=require_audio_sha and audio_sha != "skip", near=near, exemptions=ex)
        b = _compact(rep)
        b["n_test"], b["n_train"] = len(test), len(train)
        b["by_input"] = [dict(input=label, **overlap_counts(test, rows)) for label, rows in per_input]
        res["benches"][ref] = b
    if reserved_terms:
        terms = read_lines(reserved_terms)
        if not terms:
            res["unresolved"]["reserved_terms"] = f"{reserved_terms} rỗng: danh sách thuật ngữ giữ riêng phải có ít nhất 1 mục"
        else:
            rep = audit([], train, check_audio_sha=False, near=False, reserved_terms=terms, exemptions=ex)
            res["reserved"] = dict(_compact(rep), n_terms=len(terms))
    crit = (sum(b["critical"] for b in res["benches"].values()) + len(res["unresolved"])
            + (res["reserved"]["critical"] if res["reserved"] else 0))
    res.update(critical=crit, review=sum(b["review"] for b in res["benches"].values()),
               exempted=sum(b["exempted"] for b in res["benches"].values()) + (res["reserved"]["exempted"] if res["reserved"] else 0),
               passed=crit == 0, blocking=mode == "fail" and crit > 0, sec=round(time.time() - t0, 1))
    if out_dir is not None:
        write_report(res, out_dir)
    return res


def _row(f: Dict) -> str:
    ex = json.dumps(f["examples"][:1], ensure_ascii=False)[:140]
    return f"| {f['kind']} | {f['severity']} | {f['count']} | {f.get('test_rows', '–')} | `{ex}` |"


def render_md(res: Dict) -> str:
    L = [f"# Cổng rò rỉ: {res['experiment']}", "",
         f"- Chế độ: **{res['mode']}**" + (" (CHỈ BÁO CÁO, không dừng)" if res["mode"] == "report" else " (critical chưa miễn trừ -> thoát mã 3)"),
         f"- Kết quả: **{'ĐẠT' if res['passed'] else 'KHÔNG ĐẠT'}**: {res['critical']} critical, {res['review']} cần xem, "
         f"{res['exempted']} được miễn trừ" + (" · **DỪNG** trước khi curate / train" if res["blocking"] else ""),
         f"- Bench kiểm: {len(res['benches'])} / {len(res['guard_refs']) + len(res['extra_locked'])} "
         f"(leak_guard: {', '.join(res['guard_refs']) or '–'}; bench khoá thêm: {', '.join(res['extra_locked']) or '–'})",
         f"- Audio: `{res['audio_sha']}`" + {"hash": " (đọc file khi thiếu audio_sha256)",
                                             "metadata": " (chỉ dùng audio_sha256 có sẵn trong manifest, KHÔNG đọc file: trùng audio chỉ kiểm được cho dòng có hash)",
                                             "skip": " (KHÔNG kiểm trùng audio)"}[res["audio_sha"]]
         + (", thiếu hash = critical" if res["require_audio_sha"] else "")
         + ". Hash không bắt được audio đã resample / cắt khác: trùng nguồn (video / người nói) và mã clip gốc là kiểm chính.",
         f"- Text gần trùng: {'có' if res['near'] else 'KHÔNG kiểm (leak_gate.near: false)'}",
         f"- Miễn trừ: {res['exemptions'] or '–'} · reserved_terms: {res['reserved_terms'] or '**chưa khai báo**'}", "",
         "Đầu vào so với bench: " + ", ".join(f"{i['label']} ({i['rows']})" for i in res["inputs"]), ""]
    if res["unresolved"]:
        L += ["## Bench khoá CHƯA được kiểm (critical)", "", *[f"- `{k}`: {v}" for k, v in res["unresolved"].items()], ""]
    for ref, b in res["benches"].items():
        L += [f"## {ref}: {b['critical']} critical, {b['review']} cần xem, {b['exempted']} miễn trừ "
              f"({b['n_test']} câu test × {b['n_train']} dòng train / dev / text)", "",
              "| Loại | Mức | Số giá trị | Câu test bị ảnh hưởng | Ví dụ |", "| --- | --- | --- | --- | --- |"]
        L += [_row(f) for f in b["findings"]] or ["| (không phát hiện) | | | | |"]
        hot = [x for x in b["by_input"] if any(x[k] for k in ("id", "source_norm", "clip", "text_exact"))]
        if hot:
            L += ["", "Câu test trùng, theo đầu vào:", "", "| Đầu vào | id | nguồn chuẩn hoá | clip gốc | text |", "| --- | --- | --- | --- | --- |"]
            L += [f"| {x['input']} | {x['id']} | {x['source_norm']} | {x['clip']} | {x['text_exact']} |" for x in hot]
        L.append("")
    if res["reserved"]:
        r = res["reserved"]
        L += [f"## reserved_terms ({r['n_terms']} thuật ngữ) trong train / dev / text: {r['critical']} critical", "",
              "| Loại | Mức | Số giá trị | Câu | Ví dụ |", "| --- | --- | --- | --- | --- |",
              *([_row(f) for f in r["findings"]] or ["| (không phát hiện) | | | | |"]), ""]
    return "\n".join(L) + "\n"


def write_report(res: Dict, out_dir) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pj, pm = out / "leak_report.json", out / "leak_report.md"
    pj.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pm.write_text(render_md(res), encoding="utf-8")
    return {"json": pj, "md": pm}


def card_summary(res: Dict, report_json: Path) -> Dict:
    """Phần ghi vào model_card.json: báo cáo cổng đã dùng + sha256 của nó."""
    return {"mode": res["mode"], "passed": res["passed"], "critical": res["critical"], "review": res["review"],
            "exempted": res["exempted"], "benches_checked": sorted(res["benches"]), "unresolved": sorted(res["unresolved"]),
            "audio_sha": res["audio_sha"], "reserved_terms_sha256": res["reserved_terms_sha256"],
            "exemptions_sha256": res["exemptions_sha256"], "report_sha256": hashlib.sha256(Path(report_json).read_bytes()).hexdigest()}
