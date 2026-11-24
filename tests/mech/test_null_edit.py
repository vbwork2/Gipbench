#!/usr/bin/env python3
"""Phép sửa hàng output Joiner trong không gian con năng lượng thấp (scripts/10_joiner_null_edit.py). CPU, số giả.

  python tests/mech/test_null_edit.py      # thoát mã 1 nếu có ca hỏng
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("null_edit", ROOT / "scripts/10_joiner_null_edit.py")
ne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne)

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


rng = np.random.default_rng(0)
dim = 8
# Frame thường chỉ dùng 4 chiều đầu, nên không gian con năng lượng thấp là 4 chiều cuối.
ordinary = np.zeros((500, dim))
ordinary[:, :4] = rng.standard_normal((500, 4))
basis = ne.low_energy_basis(ordinary, 0.5)
check("basis có 4 vector", basis.shape == (dim, 4))
check("basis trực giao với dữ liệu thường", np.abs(ordinary @ basis).max() < 1e-9)

events = [{"target": 3, "h": rng.standard_normal(dim), "deficit": d} for d in (2.0, 4.0)]
events.append({"target": 5, "h": rng.standard_normal(dim), "deficit": 1.0})
deltas = ne.fit_edit(events, basis, 1.0)
check("sửa đúng các piece đích", sorted(deltas) == [3, 5])
m3 = np.mean([e["h"] for e in events if e["target"] == 3], axis=0)
check("trung bình tăng logit đích = alpha * deficit trung bình", abs(deltas[3] @ m3 - 3.0) < 1e-9, deltas[3] @ m3)
check("alpha 2 nhân đôi", abs(ne.fit_edit(events, basis, 2.0)[3] @ m3 - 6.0) < 1e-9)
check("frame thường không bị ảnh hưởng", np.abs(ordinary @ deltas[3]).max() < 1e-9)
free = ne.fit_edit(events, None, 1.0)
check("bản không ràng buộc cũng đạt alpha * deficit", abs(free[3] @ m3 - 3.0) < 1e-9)

rand = ne.random_like(deltas, basis, 1)
check("hướng ngẫu nhiên cùng chuẩn", all(abs(np.linalg.norm(rand[t]) - np.linalg.norm(deltas[t])) < 1e-9 for t in deltas))
check("hướng ngẫu nhiên nằm trong không gian con", np.abs(ordinary @ rand[3]).max() < 1e-9)
check("seed tái lập", np.allclose(rand[3], ne.random_like(deltas, basis, 1)[3]))

weight = rng.standard_normal((6, dim)).astype(np.float32)
bias = np.zeros(6, dtype=np.float32)
edited = ne.apply(weight, deltas)
check("apply chỉ đổi hàng đã sửa", np.array_equal(edited[[0, 1, 2, 4]], weight[[0, 1, 2, 4]]))
s = ne.static(edited, weight, bias, [], ordinary.astype(np.float32))
check("static: không đổi argmax ở frame thường", s["change"] == 0.0, s)

sys.exit(1 if FAIL else 0)
