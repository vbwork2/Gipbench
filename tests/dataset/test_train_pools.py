#!/usr/bin/env python3
"""Dọn dữ liệu train (TODO 2c, audit IP-06): clean_pool (lọc pool theo bench khoá), curate dedupe audio, mix_cuts
--max-views, ingest commonvoice_pool. Dữ liệu GIẢ trong thư mục tạm, CPU, không cần lhotse.

  python tests/dataset/test_train_pools.py
"""
import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from gipbench.core.decontam import audit
from gipbench.core.paths import REPO as G

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-400:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def row(i, text, **kw):
    return dict({"id": i, "dataset": "x", "domain": "medical", "speaker": f"s_{i}", "source": f"src_{i}",
                 "duration": 2.0, "text_tagged": text}, **kw)


def test_clean_pool():
    from gipbench.dataset.release.clean_pool import clean
    bench = [row("b1", "uống <en>vitamin</en> mỗi ngày", speaker="https://www.youtube.com/watch?v=AAAAAAAAAAA",
                 source="https://www.youtube.com/watch?v=AAAAAAAAAAA"),
             row("b2", "đọc một câu", dataset="vivos", domain="read", speaker="vivos_VIVOSDEV01", source="vivos:VIVOSDEV01"),
             row("b3", "câu riêng của test", dataset="cv17_vi", domain="general", speaker="cv_aaaaaaaaaaaaaaaa")]
    pool = [row("p1", "câu khác hẳn", speaker="yt_AAAAAAAAAAA", source="yt_AAAAAAAAAAA"),        # cùng video (định dạng khác)
            row("p2", "câu khác nữa", speaker="yt_AAAAAAAAAAA", source="yt_AAAAAAAAAAA:seg2"),   # cùng video
            row("p3", "đọc một câu", dataset="vivos", domain="read", speaker="vivos_VIVOSSPK01"),  # trùng chữ, cùng bộ đọc
            row("p4", "câu riêng của test", dataset="commonvoice", domain="read", speaker="cv_bbbbbbbbbbbbbbbb"),  # trùng text
            row("p5", "câu sạch", speaker="yt_BBBBBBBBBBB", source="yt_BBBBBBBBBBB"),
            row("p6", "câu sạch thứ hai", speaker="yt_BBBBBBBBBBB", source="yt_BBBBBBBBBBB")]
    keep, drop, st = clean(pool, [("bench/x", bench)])
    ids = {r["id"] for r in keep}
    check("clean_pool: bỏ mọi câu cùng video bench dù khác định dạng id (yt_ vs URL)", not {"p1", "p2"} & ids, ids)
    check("clean_pool: trùng chữ trong cùng bộ đọc (read) được giữ, trùng text khác bộ bị bỏ",
          "p3" in ids and "p4" not in ids, ids)
    check("clean_pool: câu sạch giữ nguyên", {"p5", "p6"} <= ids, ids)
    rep = audit(bench, keep, check_audio_sha=False, near=False)
    check("clean_pool: pool sau lọc qua decontam.audit 0 critical", rep["critical"] == 0,
          [f["kind"] for f in rep["findings"] if f["severity"] == "critical"])
    check("clean_pool: báo cáo đếm theo khoá", st["rows_by_key"].get("source_norm") == 2 and st["rows_by_key"].get("text") == 1, st)
    # trùng text không lan ra cả nguồn: người nói p4 còn câu khác thì câu đó vẫn được giữ
    keep2, _, _ = clean(pool + [row("p7", "câu không trùng", dataset="commonvoice", domain="read", speaker="cv_bbbbbbbbbbbbbbbb")],
                        [("bench/x", bench)])
    check("clean_pool: trùng chỉ do text không bỏ cả người nói", "p7" in {r["id"] for r in keep2})


def test_clean_pool_audio_paths(d: Path):
    """Lỗi 17/11: clean_pool chép nguyên `audio` tương đối của manifest nguồn (thư mục release, sâu 4 cấp) sang file ra ở
    độ sâu khác (data/pools/<x>_v2, 3 cấp) -> lệch một cấp, cả 3 pool v2 trỏ vào file không tồn tại."""
    from gipbench.core.manifest import read_jsonl
    from gipbench.dataset.release.clean_pool import write_pool
    wav = d / "dl" / "vivos" / "a.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF")
    src = d / "releases" / "pool" / "x@v1" / "manifest.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(row("p1", "câu", audio="../../../dl/vivos/a.wav")) + "\n", encoding="utf-8")
    rows = read_jsonl(str(src))
    check("clean_pool: manifest nguồn đọc đúng file", Path(rows[0]["_audio_path"]).exists(), rows[0].get("_audio_path"))
    out = d / "pools" / "x_v2" / "manifest.jsonl"
    write_pool(out, rows, rows)
    for f in (out, Path(str(out) + ".dropped.jsonl")):
        back = read_jsonl(str(f))
        check(f"clean_pool: {f.name} ghi audio tương đối theo thư mục file ra, đọc lại đúng file",
              Path(back[0]["_audio_path"]).resolve() == wav.resolve(), (back[0]["audio"], back[0].get("_audio_path")))


def test_curate_dedupe(d: Path):
    a = d / "a.jsonl"
    b = d / "b.jsonl"
    rows_a = [row("v1", "một hai ba", dataset="vivos:train", domain="read", audio=str(d / "w" / "x.wav")),
              row("v2", "bốn năm", dataset="vivos:train", domain="read", audio=str(d / "w" / "y.wav"))]
    rows_b = [row("c1", "một hai ba", dataset="confusable:vivos:train", domain="confusable",
                  audio=str(d / "w" / ".." / "w" / "x.wav")),                                   # cùng file, đường dẫn khác
              row("c2", "sáu bảy", dataset="confusable:vivos:train", domain="confusable", audio=str(d / "w" / "z.wav"),
                  audio_sha256="f" * 64),
              row("c3", "sáu bảy", dataset="other", domain="other", audio=str(d / "w2" / "q.wav"), audio_sha256="f" * 64)]
    for p, rs in ((a, rows_a), (b, rows_b)):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    out = d / "cur.jsonl"
    env = dict(os.environ, PYTHONPATH=str(G / "src"))
    common = ["--max-per-term", "0", "--max-speaker-share", "0", "--max-domain-share", "0", "--max-synthetic-share", "0",
              "--block-text"]
    # thứ tự --manifest đặt confusable trước, --dedupe-priority vẫn ưu tiên vivos
    r = subprocess.run([sys.executable, "-m", "gipbench.model.recipe.curate", "--manifest", str(b), str(a), "--out", str(out),
                        "--dedupe-priority", "vivos", *common], capture_output=True, text=True, env=env, cwd=G)
    got = [json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    check("curate: 2 câu cùng audio (đường dẫn khác dạng) -> giữ 1, theo --dedupe-priority", r.returncode == 0
          and "v1" in got and "c1" not in got, (got, r.stderr[-400:]))
    check("curate: trùng audio_sha256 -> giữ câu đứng trước theo thứ tự manifest", "c2" in got and "c3" not in got, got)
    rep = Path(str(out) + ".report.md").read_text(encoding="utf-8") if out.exists() else ""
    check("curate: báo cáo ghi cặp dataset giữ <- bỏ", "vivos <- confusable: 1 câu" in rep, rep[-600:])
    r2 = subprocess.run([sys.executable, "-m", "gipbench.model.recipe.curate", "--manifest", str(a), str(b), "--out", str(d / "c2.jsonl"),
                         "--no-dedupe-audio", *common], capture_output=True, text=True, env=env, cwd=G)
    n = len((d / "c2.jsonl").read_text(encoding="utf-8").splitlines()) if (d / "c2.jsonl").exists() else -1
    check("curate: --no-dedupe-audio giữ cả 5 câu", r2.returncode == 0 and n == 5, (n, r2.stderr[-300:]))


def test_mix_cap():
    from gipbench.model.features.mix_cuts import cap_weights, source_id
    check("mix_cuts: source_id bỏ hậu tố speed perturb", source_id("abc_sp0.9") == "abc" and source_id("abc_sp1.1") == "abc"
          and source_id("abc") == "abc")
    # nhóm nhỏ: 1 h cut (3 bản speed perturb / audio), trọng số 0.35; 10 h × 6 epoch -> 0.35*60/1*3 = 63 lượt / audio gốc
    g = [{"name": "cs_real", "w": 0.35, "h": 1.0, "k": 3}, {"name": "replay", "w": 0.65, "h": 100.0, "k": 3}]
    share, eh, warn = cap_weights(g, 10, 6, 8)
    views = share["cs_real"] * 10 * 6 / 1.0 * 3
    check("mix_cuts --max-views 8: nhóm nhỏ bị hạ đúng tới trần, phần dư sang nhóm lớn",
          abs(views - 8) < 1e-6 and abs(sum(share.values()) - 1) < 1e-9 and share["replay"] > 0.65 and not warn, (share, views))
    share0, _, _ = cap_weights(g, 10, 6, 0)
    check("mix_cuts: max_views = 0 chỉ chuẩn hoá trọng số", abs(share0["cs_real"] - 0.35) < 1e-9, share0)
    # mọi nhóm đều nhỏ -> epoch ảo ngắn lại, có cảnh báo
    g2 = [{"name": "a", "w": 0.5, "h": 1.0, "k": 1}, {"name": "b", "w": 0.5, "h": 1.0, "k": 1}]
    share2, eh2, warn2 = cap_weights(g2, 10, 6, 8)
    check("mix_cuts: mọi nhóm chạm trần -> epoch ngắn lại + cảnh báo", bool(warn2) and eh2 < 10
          and all(s * eh2 * 6 / 1.0 <= 8 + 1e-6 for s in share2.values()), (share2, eh2, warn2))
    # nhóm nhận phần dư cũng vượt trần ở vòng sau -> cũng bị hạ
    g3 = [{"name": "a", "w": 0.5, "h": 1.0, "k": 1}, {"name": "b", "w": 0.3, "h": 2.0, "k": 1}, {"name": "c", "w": 0.2, "h": 1000, "k": 1}]
    share3, _, _ = cap_weights(g3, 10, 6, 8)
    check("mix_cuts: trần áp lặp tới khi ổn định", all(share3[x["name"]] * 60 / x["h"] * x["k"] <= 8 + 1e-6 for x in g3)
          and abs(sum(share3.values()) - 1) < 1e-9, share3)


def test_cv_pool(d: Path):
    from gipbench.core.textnorm import Catalog
    from gipbench.dataset.ingest import local as L
    from gipbench.dataset.ingest.common import Writer
    root = d / "cv"
    (root / "clips").mkdir(parents=True)
    hdr = "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
    line = lambda c, p, s, u=0, dn=0: f"{c}\t{p}\tx\t{s}\t\t{u}\t{dn}\t\t\t\t\tvi\t\n"  # noqa: E731
    (root / "test.tsv").write_text(hdr + line("T" * 40, "t1.mp3", "câu test"), encoding="utf-8")
    (root / "dev.tsv").write_text(hdr + line("D" * 40, "d1.mp3", "câu dev"), encoding="utf-8")
    (root / "validated.tsv").write_text(hdr + line("T" * 40, "t1.mp3", "câu test") + "".join(
        line("A" * 40, f"a{i}.mp3", f"câu số {i}", 2, 0) for i in range(5)), encoding="utf-8")
    (root / "other.tsv").write_text(hdr + line("D" * 40, "d2.mp3", "người dev") + line("B" * 40, "b1.mp3", "câu bê", 0, 0)
                                    + line("C" * 40, "c1.mp3", "câu xê", 0, 2), encoding="utf-8")
    (root / "clip_durations.tsv").write_text("clip\tduration[ms]\n" + "".join(
        f"{p}\t3000\n" for p in ["t1.mp3", "d1.mp3", "d2.mp3", "b1.mp3", "c1.mp3"] + [f"a{i}.mp3" for i in range(5)]), encoding="utf-8")
    w, stats = Writer(str(d / "cvout"), "commonvoice_pool"), {}
    L.commonvoice_pool(Namespace(root=str(root), cv_parts=["validated", "other"], cv_min_up=0, cv_max_down=0,
                                 cv_max_per_speaker=3, seed=0), w, Catalog.load(None), stats)
    rows = w.rows.get("pool_cv", [])
    spk = {r["speaker"] for r in rows}
    check("commonvoice_pool: loại người nói test / dev ở mọi phần", not any(s.startswith(("cv_TTTT", "cv_DDDD")) for s in spk), spk)
    check("commonvoice_pool: trần câu / người nói và luật phiếu (down_votes > 0 bị bỏ)",
          sum(r["speaker"] == "cv_" + "A" * 16 for r in rows) == 3 and "cv_" + "B" * 16 in spk and "cv_" + "C" * 16 not in spk,
          [(r["speaker"], r["id"]) for r in rows])
    check("commonvoice_pool: ghi label_source + upstream_id + duration từ clip_durations",
          bool(rows) and all(r["label_source"].startswith("human_prompt:cv27_") and r["upstream_id"] and r["duration"] == 3.0
                             for r in rows), rows[:1])


def main():
    d = Path(tempfile.mkdtemp())
    test_clean_pool()
    test_clean_pool_audio_paths(d)
    test_curate_dedupe(d)
    test_mix_cap()
    test_cv_pool(d)
    print(f"\n{'HỎNG ' + str(len(FAIL)) if FAIL else 'ĐẠT tất cả'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
