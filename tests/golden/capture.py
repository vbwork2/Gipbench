#!/usr/bin/env python3
"""Golden test cho refactor 3 pipeline: chạy cùng một bộ bước trên dữ liệu THẬT đang có (chỉ đọc), ghi output ra --out.

  python tests/golden/capture.py --layout old --out /tmp/golden_old     # code trước refactor (commit 1570a42)
  python tests/golden/capture.py --layout new --out /tmp/golden_new     # code sau refactor
  python tests/golden/compare.py /tmp/golden_old /tmp/golden_new

Chạy trong WSL, venv CPU (~/.venvs/gipbench), tại gốc repo. Không ghi gì vào data/, results/. Bước nào thiếu input thì bỏ
qua và ghi vào <out>/SKIPPED.txt. Mỗi bước có lệnh "old" (script cũ) và "new" (module sau refactor), nhận CÙNG input.
Các heredoc của ft/run_e3_v1b.sh được chép nguyên logic ở dưới (old_*), chỉ đổi đường dẫn ghi ra.
Layout "old" chỉ chạy được khi checkout commit trước refactor (git worktree), xem tests/golden/README.md.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

G = Path(__file__).resolve().parents[2]  # đổi bằng --repo (chụp mốc từ checkout cũ)
BENCH = "data/bench_v0_draft/bench_v0_draft.jsonl"
CATALOG = "data/bench_v0_draft/catalog_v0_draft.tsv"
E3 = "data/e3_v1"
TRAIN_IN = [f"{E3}/train_vimedcss_abs.jsonl", f"{E3}/train_vss_cs.jsonl", f"{E3}/train_vivos.jsonl",
            "data/raw/cv/train_cv.jsonl", f"{E3}/distill.jsonl", f"{E3}/train_confusable.jsonl"]
DEV = [f"{E3}/dev_cs.jsonl", f"{E3}/dev_vi.jsonl"]
ITERS = "200"

# bước -> (lệnh old, lệnh new)
SCRIPTS = {
    "score": ("score.py", "-m gipbench.bench.score"),
    "leaderboard": ("leaderboard.py", "-m gipbench.bench.leaderboard"),
    "validate": ("validate.py", "-m gipbench.dataset.release.validate"),
    "curate": ("ft/curate_train.py", "-m gipbench.model.recipe.curate"),
    "mix_plan": ("tools/mix_plan.py", "-m gipbench.model.recipe.mix_plan"),
    "draft_tables": ("tools/draft_tables.py", "-m gipbench.bench.report.draft_tables"),
    "select_ckpt": ("ft/select_ckpt.py", "-m gipbench.model.select.select_ckpt"),
    "eda": ("eda.py", "-m gipbench.bench.report.eda"),
    "prep_e3_v1": ("ft/prep_e3_v1.py", "-m gipbench.dataset.release.prep_e3_v1"),
    "make_dev_split": ("tools/make_dev_split.py", "-m gipbench.dataset.release.make_dev_split"),
    "split_groups": (None, "-m gipbench.model.recipe.split_groups"),
    "build_dev_cs": (None, "-m gipbench.dataset.release.build_dev_cs"),
}
MIX_OLD = "configs/train_mix.tsv"
EXP_NEW = "configs/model/experiments/e3_v1b.yaml"


def py(layout, step):
    s = SCRIPTS[step][0 if layout == "old" else 1]
    return [sys.executable] + s.split()


def run(argv, log):
    t = time.time()
    p = subprocess.run([str(a) for a in argv], cwd=G, capture_output=True, text=True, encoding="utf-8")
    log.write_text(f"rc={p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}", encoding="utf-8")
    print(f"  rc={p.returncode} {time.time() - t:.1f}s  {log.name}")
    return p.returncode


def exists(*paths):
    return all((G / p).exists() for p in paths)


# ---- logic heredoc cũ của ft/run_e3_v1b.sh (chép nguyên, chỉ đổi đường dẫn đọc / ghi)
def old_stage0_dev_cs(out):
    rows = [json.loads(l) for l in open(G / "data/raw/vimedcss_full/dev_vimedcss.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    base = "/mnt/d/code/ZAIC/gipbench/data/raw/vimedcss_full/"
    o = [dict(r, audio=base + r["audio"]) for r in rows[:600]]
    o += [json.loads(l) for l in open(G / "data/e3_v1/dev_cs_vss.jsonl", encoding="utf-8")]
    o = [r for r in o if float(r["duration"]) >= 0.5]
    with open(out / "dev_cs.jsonl", "w", encoding="utf-8") as f:
        for r in o:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("dev_cs:", len(o), "câu", round(sum(float(r["duration"]) for r in o) / 3600, 2), "h")
    with open(out / "train_vimedcss_abs.jsonl", "w", encoding="utf-8") as f:
        for l in open(G / "data/raw/vimedcss_full/train_vimedcss.jsonl", encoding="utf-8"):
            r = json.loads(l)
            f.write(json.dumps(dict(r, audio=base + r["audio"]), ensure_ascii=False) + "\n")


def old_stage2_groups(curated, out):
    groups = {}
    for l in open(G / MIX_OLD, encoding="utf-8"):
        if l.startswith("#") or l.startswith("group\t") or not l.strip():
            continue
        g, w, _, ds, _ = l.rstrip("\n").split("\t")
        for d in ds.split(","):
            groups[d.strip().lower()] = g
    o = {}
    for l in open(curated, encoding="utf-8"):
        r = json.loads(l)
        d = r["dataset"].split(":")[0].lower()
        g = groups.get(d) or ("distill" if r.get("distill") else None)
        if g is None:
            raise SystemExit(f"không biết nhóm của dataset {r['dataset']}")
        o.setdefault(g, []).append(l)
    for g, ls in o.items():
        open(out / f"group_{g}.jsonl", "w", encoding="utf-8").writelines(ls)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=["old", "new"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=[], help="chỉ chạy các bước này")
    ap.add_argument("--repo", help="gốc repo cần chạy (mặc định: repo chứa file này)")
    args = ap.parse_args()
    global G
    if args.repo:
        G = Path(args.repo).resolve()
    L, O = args.layout, Path(args.out).resolve()
    if O.exists():
        shutil.rmtree(O)
    O.mkdir(parents=True)
    (O / "ROOT.txt").write_text(str(O), encoding="utf-8")  # compare.py chuẩn hoá đường dẫn này
    skipped = []

    def want(step):
        return not args.only or step in args.only

    # 1. score lại hyps đã có của mọi system trong results/v0_draft (copy hyps + decode_summary sang O)
    systems = sorted(p.parent.name for p in (G / "results/v0_draft").glob("*/hyps.jsonl"))
    if want("score") and exists(BENCH, CATALOG) and systems:
        for s in systems:
            d = O / "score" / s
            d.mkdir(parents=True)
            shutil.copy(G / "results/v0_draft" / s / "hyps.jsonl", d / "hyps.jsonl")
            if (G / "results/v0_draft" / s / "decode_summary.json").exists():
                shutil.copy(G / "results/v0_draft" / s / "decode_summary.json", d / "decode_summary.json")
            print(f"score {s}")
            run(py(L, "score") + ["--bench", BENCH, "--catalog", CATALOG, "--hyps", d / "hyps.jsonl", "--iters", ITERS],
                O / f"score_{s}.log")
    else:
        skipped.append("score")
    if want("leaderboard") and (O / "score").exists():
        print("leaderboard")
        run(py(L, "leaderboard") + ["--results", O / "score", "--baseline", "e0_baseline", "--out", O / "leaderboard",
                                    "--iters", ITERS], O / "leaderboard.log")
    if want("draft_tables") and (O / "score").exists():
        print("draft_tables")
        sy = [O / "score" / s for s in systems]
        run(py(L, "draft_tables") + ["--bench", BENCH, "--catalog", CATALOG, "--systems", *sy, "--baseline", "e0_baseline",
                                     "--classify", "e0_baseline", "--by-domain", "--title", "golden",
                                     "--out", O / "draft_tables.md"], O / "draft_tables.log")
    # 2. EDA bench nháp
    if want("eda") and exists(BENCH, CATALOG):
        print("eda")
        run(py(L, "eda") + ["--bench", BENCH, "--catalog", CATALOG, "--out", O / "eda"], O / "eda.log")
    # 3. validate như stage 0 của run_e3_v1b.sh
    if want("validate") and exists(BENCH, CATALOG, *TRAIN_IN, *DEV):
        print("validate")
        run(py(L, "validate") + ["--bench", BENCH, "--catalog", CATALOG, "--train", *TRAIN_IN, "--dev", *DEV],
            O / "validate.log")
    else:
        skipped.append("validate")
    # 4. curate như stage 1 của run_e3_v1b.sh, rồi tách nhóm (stage 2) và mix_plan
    cur = O / "curate" / "train_curated.jsonl"
    if want("curate") and exists(CATALOG, *TRAIN_IN):
        print("curate")
        cur.parent.mkdir(parents=True)
        run(py(L, "curate") + ["--manifest", *TRAIN_IN, "--catalog", CATALOG, "--out", cur, "--holdout-domain", "finance",
                               "--max-domain-share", "0.50", "--cap-exempt", "other"], O / "curate.log")
    else:
        skipped.append("curate")
    if want("split_groups") and cur.exists():
        print("split_groups")
        gd = O / "groups"
        gd.mkdir()
        if L == "old":
            old_stage2_groups(cur, gd)
        else:
            run(py(L, "split_groups") + ["--experiment", EXP_NEW, "--manifest", cur, "--out-dir", gd],
                O / "split_groups.log")
    if want("mix_plan") and cur.exists():
        print("mix_plan")
        mix = ["--mix", MIX_OLD] if L == "old" else ["--experiment", EXP_NEW]
        run(py(L, "mix_plan") + mix + ["--manifest", cur, "--domain-cap", "0.5", "--fit-cap", "--out", O / "mix_plan.md",
                                       "--out-json", O / "mix_plan.json"], O / "mix_plan.log")
    # 5. chọn checkpoint trên dev (results/e3_v1_dev)
    rd = G / "results/e3_v1_dev"
    cands = sorted(p for p in rd.glob("ck*") if p.is_dir()) if rd.exists() else []
    if want("select_ckpt") and cands and exists(CATALOG, *DEV):
        print("select_ckpt")
        run(py(L, "select_ckpt") + ["--catalog", CATALOG, "--dev-cs", DEV[0], "--dev-vi", DEV[1], "--base", rd / "base",
                                    "--cand", *cands, "--out", O / "select_ckpt"], O / "select_ckpt.log")
        wise = sorted(p for p in rd.glob("wise_*") if p.is_dir())
        if wise:
            run(py(L, "select_ckpt") + ["--catalog", CATALOG, "--dev-cs", DEV[0], "--dev-vi", DEV[1], "--base",
                                        rd / "base", "--cand", *wise, "--out", O / "select_wise"], O / "select_wise.log")
    else:
        skipped.append("select_ckpt")
    # 6. chuẩn bị dữ liệu E3 v1 (chỉ lấy mẫu theo seed, không đụng audio) + heredoc stage 0
    pre = ["data/raw/vss_full/vss_cs_clean_dom.jsonl", "data/raw/vivos/train_vivos.jsonl",
           "data/raw/confusable/train_confusable.jsonl", "data/raw/vss_full/train_vss_dom.jsonl",
           "data/raw/vss_full/vss_cs_candidates.jsonl"]
    if want("prep_e3_v1") and exists(*pre):
        print("prep_e3_v1")
        run(py(L, "prep_e3_v1") + ["--out", O / "prep_e3_v1", "--distill-hours", "10"], O / "prep_e3_v1.log")
    else:
        skipped.append("prep_e3_v1")
    s0 = ["data/raw/vimedcss_full/dev_vimedcss.jsonl", "data/e3_v1/dev_cs_vss.jsonl",
          "data/raw/vimedcss_full/train_vimedcss.jsonl"]
    if want("build_dev_cs") and exists(*s0):
        print("build_dev_cs")
        d = O / "stage0"
        d.mkdir()
        if L == "old":
            old_stage0_dev_cs(d)
        else:
            run(py(L, "build_dev_cs") + ["--vimedcss-dev", s0[0], "--vss-dev", s0[1], "--vimedcss-train", s0[2],
                                         "--audio-base", "/mnt/d/code/ZAIC/gipbench/data/raw/vimedcss_full/",
                                         "--out-dir", d], O / "build_dev_cs.log")
    else:
        skipped.append("build_dev_cs")
    # 7. tách dev bench theo cụm người nói (như docstring tools/make_dev_split.py)
    md = ["data/raw/vimedcss_full/dev_vimedcss.jsonl", "data/e3_v1/dev_cs_vss.jsonl", "data/e3_v1/train_curated.jsonl"]
    if want("make_dev_split") and exists(BENCH, CATALOG, *md):
        print("make_dev_split")
        (O / "dev_split").mkdir()
        run(py(L, "make_dev_split") + ["--bench", BENCH, "--splits", "B1", "--dev-frac", "0.3", "--seed", "0",
                                       "--extra-dev", md[0], md[1], "--train", md[2], "--catalog", CATALOG,
                                       "--out-dev", O / "dev_split/dev_bench.jsonl",
                                       "--out-test", O / "dev_split/bench_test.jsonl"], O / "make_dev_split.log")
    else:
        skipped.append("make_dev_split")
    (O / "SKIPPED.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    print("xong:", O)


if __name__ == "__main__":
    main()
