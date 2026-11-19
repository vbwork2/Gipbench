#!/usr/bin/env python3
"""Trộn nhiều CutSet theo trọng số thành MỘT manifest dài gồm K "epoch ảo" (cho finetune_cs.py --num-epochs 1).

  python -m gipbench.model.features.mix_cuts --group cs_real=data/lhotse/cs_real_cuts_raw.jsonl.gz:0.35 \
      --group replay=data/lhotse/replay_cuts_raw.jsonl.gz:0.25 ... \
      --epoch-hours 160 --epochs 6 --out data/lhotse/train_mix_cuts.jsonl.gz

Vì sao không dùng CutSet.mux(stop_early=True): mux dừng khi nhóm nhỏ nhất hết (vd confusable 0.9 h), nên mỗi epoch chỉ
dùng lại đúng phần đầu của các nhóm lớn. Ở đây mỗi epoch ảo lấy w_i * --epoch-hours giờ của nhóm i (trọng số được
chuẩn hoá lại theo các nhóm có mặt), đi vòng qua một thứ tự xáo trộn riêng của nhóm: nhóm lớn cho dữ liệu mới ở mỗi
epoch, nhóm nhỏ lặp lại (báo số lần lặp). Trong mỗi epoch ảo các cut được xáo trộn; các epoch ghi nối tiếp nhau, nên
checkpoint mỗi --save-every-n (= số step của một epoch ảo) tương ứng một epoch. id cut được đổi thành <id>__e<k>_<n>
để không trùng khi lặp. In ra bảng giờ / số lần lặp theo nhóm và ghi <out>.json.

Độ phơi (audit R15): `views_per_source_audio` = số lần mỗi audio GỐC được dùng (gộp mọi bản speed perturb: cut id
`<id>_sp0.9` / `_sp1.1` tính về `<id>`). --max-views N (0 = tắt): nhóm nào dự kiến vượt N lượt / audio gốc thì bị hạ trọng
số đúng tới trần, phần trọng số dư chia lại cho các nhóm còn lại theo tỉ lệ cũ (lặp tới khi ổn định: nhóm nhận thêm có
thể vượt trần ở vòng sau). Nếu MỌI nhóm chạm trần thì epoch ảo ngắn lại (cảnh báo). Mọi điều chỉnh ghi vào
`<out>.json` (`weight_requested`, `weight`, `capped`, `warnings`).
"""
import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

_SP = re.compile(r"_sp\d+(?:\.\d+)?$")


def source_id(cut_id: str) -> str:
    """id audio gốc của một cut (bỏ hậu tố speed perturb của lhotse)."""
    return _SP.sub("", cut_id)


def cap_weights(groups: List[Dict], epoch_hours: float, epochs: int, max_views: float) -> Tuple[Dict[str, float], float, List[str]]:
    """groups: [{name, w, h (giờ cut trong pool, đã gồm speed perturb), k (số cut / audio gốc)}].
    Lượt / audio gốc dự kiến của nhóm i = share_i * epoch_hours * epochs / h_i * k_i. Trả về (share đã chuẩn hoá theo nhóm,
    số giờ một epoch ảo sau điều chỉnh, cảnh báo). max_views <= 0: chỉ chuẩn hoá."""
    wsum = sum(g["w"] for g in groups)
    share = {g["name"]: g["w"] / wsum for g in groups}
    if max_views <= 0:
        return share, epoch_hours, []
    cap = {g["name"]: max_views * g["h"] / (epoch_hours * epochs * g["k"]) for g in groups}
    fixed: Dict[str, float] = {}
    for _ in range(len(groups) + 1):
        free = [g for g in groups if g["name"] not in fixed]
        rest = 1 - sum(fixed.values())
        fw = sum(g["w"] for g in free)
        new = {g["name"]: cap[g["name"]] for g in free if fw and rest * g["w"] / fw > cap[g["name"]] + 1e-12}
        if not new:
            return dict(fixed, **{g["name"]: (rest * g["w"] / fw if fw else 0.0) for g in free}), epoch_hours, []
        fixed.update(new)
        if len(fixed) == len(groups) or sum(fixed.values()) >= 1:
            break
    tot = sum(fixed.values())
    if len(fixed) < len(groups):  # các nhóm trần đã chiếm hết epoch: nhóm còn lại = 0 (không xảy ra khi tot < 1)
        fixed.update({g["name"]: 0.0 for g in groups if g["name"] not in fixed})
    if tot >= 1:
        return {n: v / tot for n, v in fixed.items()}, epoch_hours, []
    warn = [f"mọi nhóm chạm trần {max_views:g} lượt / audio gốc: epoch ảo giảm {epoch_hours:g} -> {epoch_hours * tot:.2f} h"]
    return {n: v / tot for n, v in fixed.items()}, epoch_hours * tot, warn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="TÊN=ĐƯỜNG_DẪN_CUTS:TRỌNG_SỐ")
    ap.add_argument("--epoch-hours", type=float, required=True, help="số giờ cut (đã tính speed perturb) mỗi epoch ảo")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-views", type=float, default=0, help="trần lượt dùng mỗi audio gốc (0 = tắt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from lhotse import CutSet, load_manifest_lazy

    rnd = random.Random(args.seed)
    groups = []
    for g in args.group:
        name, rest = g.split("=", 1)
        path, _, w = rest.rpartition(":")
        cuts = list(load_manifest_lazy(path))
        if not cuts:
            print(f"BỎ nhóm {name}: không có cut")
            continue
        rnd.shuffle(cuts)
        n_src = len({source_id(c.id) for c in cuts})
        groups.append({"name": name, "cuts": cuts, "w": float(w), "h": sum(c.duration for c in cuts) / 3600, "pos": 0,
                       "taken": 0.0, "k": len(cuts) / n_src, "n_src": n_src, "uses": {}})
    wsum = sum(g["w"] for g in groups)
    share, epoch_hours, warns = cap_weights(groups, args.epoch_hours, args.epochs, args.max_views)
    for wmsg in warns:
        print("CẢNH BÁO", wmsg)
    stats = {"epoch_hours": round(epoch_hours, 4), "epoch_hours_requested": args.epoch_hours, "epochs": args.epochs,
             "max_views": args.max_views, "warnings": warns, "groups": [], "seed": args.seed}
    view_limit = math.floor(args.max_views) if args.max_views else None
    if view_limit is not None and view_limit < 1:
        ap.error("--max-views must be zero or at least one")
    out_cuts = []
    for e in range(args.epochs):
        block = []
        for g in groups:
            need = share[g["name"]] * epoch_hours * 3600
            got = 0.0
            skipped = 0
            while got < need and skipped < len(g["cuts"]):
                c = g["cuts"][g["pos"] % len(g["cuts"])]
                k = g["pos"] // len(g["cuts"])
                g["pos"] += 1
                s = source_id(c.id)
                uses = g["uses"].get(s, 0)
                if view_limit is not None and uses >= view_limit:
                    skipped += 1
                    continue
                skipped = 0
                block.append(c.with_id(f"{c.id}__e{e}_{k}"))
                g["uses"][s] = uses + 1
                got += c.duration
            g["taken"] += got / 3600
        rnd.shuffle(block)
        out_cuts += block
    for g in groups:
        u = list(g["uses"].values()) or [0]
        vps = sum(u) / g["n_src"]
        req = g["w"] / wsum
        st = {"group": g["name"], "weight_requested": round(req, 4), "weight": round(share[g["name"]], 4),
              "capped": bool(args.max_views) and share[g["name"]] < req - 1e-9, "pool_h": round(g["h"], 2),
              "per_epoch_h": round(g["taken"] / args.epochs, 2), "views_total": round(g["taken"] / g["h"], 2),
              "source_audios": g["n_src"], "cuts_per_source": round(g["k"], 3),
              "views_per_source_audio": round(vps, 2), "views_per_source_audio_max": max(u)}
        stats["groups"].append(st)
        print(f"  {st['group']:12s} w={st['weight']:.3f}{' (trần, yêu cầu %.3f)' % req if st['capped'] else ''}"
              f"  pool {st['pool_h']:7.2f} h  mỗi epoch {st['per_epoch_h']:7.2f} h"
              f"  lượt / cut ~{st['views_total']:.1f}  lượt / audio gốc ~{vps:.1f} (max {max(u)})")
        expected = share[g["name"]] * epoch_hours * args.epochs
        if g["taken"] + 1e-6 < expected:
            msg = f"{g['name']}: actual {g['taken']:.3f} h / planned {expected:.3f} h at the per-recording view cap"
            stats["warnings"].append(msg)
            print("CẢNH BÁO", msg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(out_cuts).to_file(args.out)
    stats["n_cuts"] = len(out_cuts)
    stats["hours"] = round(sum(c.duration for c in out_cuts) / 3600, 2)
    Path(args.out + ".json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out_cuts)} cut, {stats['hours']} h ({args.epochs} epoch ảo) -> {args.out}")


if __name__ == "__main__":
    main()
