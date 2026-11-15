#!/usr/bin/env python3
"""Trộn nhiều CutSet theo trọng số thành MỘT manifest dài gồm K "epoch ảo" (cho finetune_cs.py --num-epochs 1).

  python ft/mix_cuts.py --group cs_real=data/lhotse/cs_real_cuts_raw.jsonl.gz:0.35 \
      --group replay=data/lhotse/replay_cuts_raw.jsonl.gz:0.25 ... \
      --epoch-hours 160 --epochs 6 --out data/lhotse/train_mix_cuts.jsonl.gz

Vì sao không dùng CutSet.mux(stop_early=True): mux dừng khi nhóm nhỏ nhất hết (vd confusable 0.9 h), nên mỗi epoch chỉ
dùng lại đúng phần đầu của các nhóm lớn. Ở đây mỗi epoch ảo lấy w_i * --epoch-hours giờ của nhóm i (trọng số được
chuẩn hoá lại theo các nhóm có mặt), đi vòng qua một thứ tự xáo trộn riêng của nhóm: nhóm lớn cho dữ liệu mới ở mỗi
epoch, nhóm nhỏ lặp lại (báo số lần lặp). Trong mỗi epoch ảo các cut được xáo trộn; các epoch ghi nối tiếp nhau, nên
checkpoint mỗi --save-every-n (= số step của một epoch ảo) tương ứng một epoch. id cut được đổi thành <id>__e<k>_<n>
để không trùng khi lặp. In ra bảng giờ / số lần lặp theo nhóm và ghi <out>.json.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="TÊN=ĐƯỜNG_DẪN_CUTS:TRỌNG_SỐ")
    ap.add_argument("--epoch-hours", type=float, required=True, help="số giờ cut (đã tính speed perturb) mỗi epoch ảo")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
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
        groups.append({"name": name, "cuts": cuts, "w": float(w), "h": sum(c.duration for c in cuts) / 3600, "pos": 0,
                       "taken": 0.0})
    wsum = sum(g["w"] for g in groups)
    stats = {"epoch_hours": args.epoch_hours, "epochs": args.epochs, "groups": []}
    out_cuts = []
    for e in range(args.epochs):
        block = []
        for g in groups:
            need = g["w"] / wsum * args.epoch_hours * 3600
            got = 0.0
            while got < need:
                c = g["cuts"][g["pos"] % len(g["cuts"])]
                k = g["pos"] // len(g["cuts"])
                g["pos"] += 1
                block.append(c.with_id(f"{c.id}__e{e}_{k}"))
                got += c.duration
            g["taken"] += got / 3600
        rnd.shuffle(block)
        out_cuts += block
    for g in groups:
        st = {"group": g["name"], "weight": round(g["w"] / wsum, 4), "pool_h": round(g["h"], 2),
              "per_epoch_h": round(g["taken"] / args.epochs, 2), "views_total": round(g["taken"] / g["h"], 2)}
        stats["groups"].append(st)
        print(f"  {st['group']:12s} w={st['weight']:.3f}  pool {st['pool_h']:7.2f} h  mỗi epoch {st['per_epoch_h']:7.2f} h"
              f"  tổng số lần dùng mỗi cut ~{st['views_total']:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(out_cuts).to_file(args.out)
    stats["n_cuts"] = len(out_cuts)
    stats["hours"] = round(sum(c.duration for c in out_cuts) / 3600, 2)
    Path(args.out + ".json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out_cuts)} cut, {stats['hours']} h ({args.epochs} epoch ảo) -> {args.out}")


if __name__ == "__main__":
    main()
