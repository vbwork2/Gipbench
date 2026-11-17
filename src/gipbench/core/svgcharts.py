"""Biểu đồ SVG tự viết cho report HTML (không phụ thuộc thư viện ngoài).

Màu theo vai trò qua CSS variables (định nghĩa trong report.css của eda.py):
  --s1..--s4 : categorical theo thứ tự cố định; --q1..--q4 : ramp thứ bậc (1 hue, nhạt -> đậm)
Mỗi mark có <title> để hover hiện giá trị; mỗi biểu đồ đi kèm bảng số liệu.
"""
from html import escape
from typing import List, Optional, Sequence, Tuple

FONT = 'font-family="inherit"'


def _bar_path(x, y, w, h, r=4):
    """Thanh ngang: gốc vuông ở baseline, đầu dữ liệu bo tròn."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


def _vbar_path(x, y0, w, h, r=3):
    if h <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    y = y0 - h
    return (f"M{x:.1f},{y0:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} "
            f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y0:.1f} Z")


def _fmt(v, unit=""):
    if isinstance(v, float):
        s = f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"
    else:
        s = f"{v:,}"
    return s + unit


def hbar(rows: Sequence[Tuple[str, float]], unit: str = "", targets: Optional[Sequence[Optional[float]]] = None,
         width: int = 640, label_w: int = 150, row_h: int = 26, color: str = "--s1") -> str:
    """Thanh ngang 1 series; `targets` vẽ vạch mục tiêu."""
    vmax = max([v for _, v in rows] + [t for t in (targets or []) if t] + [1e-9])
    plot_w = width - label_w - 120
    h = row_h * len(rows) + 8
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" {FONT}>']
    for i, (lab, v) in enumerate(rows):
        y = 4 + i * row_h
        bw = plot_w * v / vmax
        out.append(f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" class="lab">{escape(str(lab))}</text>')
        tip = f"{lab}: {_fmt(v, unit)}"
        t = targets[i] if targets else None
        if t:
            tip += f" (mục tiêu {_fmt(t, unit)})"
        out.append(f'<path d="{_bar_path(label_w, y + 5, bw, row_h - 10)}" style="fill:var({color})"><title>{escape(tip)}</title></path>')
        if t:
            tx = label_w + plot_w * t / vmax
            out.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{y + 2}" y2="{y + row_h - 2}" class="target"><title>mục tiêu {_fmt(t, unit)}</title></line>')
        out.append(f'<text x="{label_w + bw + 6:.1f}" y="{y + row_h / 2 + 4}" class="val">{_fmt(v, unit)}</text>')
    out.append(f'<line x1="{label_w}" x2="{label_w}" y1="2" y2="{h - 2}" class="axis"/></svg>')
    return "".join(out)


LIGHT_FILLS = {"--s3", "--s4", "--q1", "--q2"}  # nền sáng -> chữ tối


def stacked100(rows: Sequence[Tuple[str, Sequence[float]]], cats: Sequence[str], colors: Sequence[str],
               width: int = 640, label_w: int = 110, row_h: int = 28, unit: str = "",
               labels: Optional[dict] = None) -> str:
    """Thanh ngang 100% chồng (thành phần của tổng); legend ở trên, nhãn % trong đoạn đủ rộng."""
    plot_w = width - label_w - 10
    leg_h = 26
    h = leg_h + row_h * len(rows) + 6
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" {FONT}>']
    lx = label_w
    labels = labels or {}
    for c, col in zip(cats, colors):
        name = labels.get(c, c)
        out.append(f'<rect x="{lx}" y="6" width="12" height="12" rx="3" style="fill:var({col})"/>'
                   f'<text x="{lx + 17}" y="16" class="lab">{escape(name)}</text>')
        lx += 30 + 7.5 * len(name)
    for i, (lab, vals) in enumerate(rows):
        y = leg_h + i * row_h
        tot = sum(vals) or 1
        out.append(f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" class="lab">{escape(lab)}</text>')
        x = label_w
        nz = [k for k, v in enumerate(vals) if v > 0]
        for k, (v, c, col) in enumerate(zip(vals, cats, colors)):
            if v <= 0:
                continue
            w = plot_w * v / tot
            gap = 2 if k != nz[-1] else 0
            tip = f"{lab} · {labels.get(c, c)}: {_fmt(v, unit)} ({100 * v / tot:.0f}%)"
            if k == nz[-1]:
                d = _bar_path(x, y + 4, max(0.0, w), row_h - 8)
                out.append(f'<path d="{d}" style="fill:var({col})"><title>{escape(tip)}</title></path>')
            else:
                out.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{max(0.0, w - gap):.1f}" height="{row_h - 8}" style="fill:var({col})"><title>{escape(tip)}</title></rect>')
            ib_cls = "inbar dark" if col in LIGHT_FILLS else "inbar"
            if w > 38:
                out.append(f'<text x="{x + w / 2:.1f}" y="{y + row_h / 2 + 4}" text-anchor="middle" class="{ib_cls}">{100 * v / tot:.0f}%</text>')
            x += w
    out.append("</svg>")
    return "".join(out)


def hist(counts: Sequence[int], edges: Sequence[float], title: str = "", width: int = 300, height: int = 140,
         color: str = "--s1", xunit: str = "s") -> str:
    """Histogram cột dọc nhỏ (dùng làm small multiple)."""
    pad_l, pad_b, pad_t = 30, 20, 18
    pw, ph = width - pad_l - 22, height - pad_b - pad_t
    cmax = max(list(counts) + [1])
    n = len(counts)
    bw = pw / n
    y0 = pad_t + ph
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" {FONT}>',
           f'<text x="{pad_l}" y="12" class="lab strong">{escape(title)}</text>',
           f'<line x1="{pad_l}" x2="{pad_l + pw}" y1="{y0}" y2="{y0}" class="axis"/>',
           f'<text x="{pad_l - 4}" y="{pad_t + 8}" text-anchor="end" class="tick">{cmax}</text>']
    for i, c in enumerate(counts):
        bh = ph * c / cmax
        d = _vbar_path(pad_l + i * bw + 1, y0, max(1.0, bw - 2), bh)
        if d:
            out.append(f'<path d="{d}" style="fill:var({color})"><title>{edges[i]:g}–{edges[i + 1]:g}{xunit}: {c}</title></path>')
    for k in (0, n // 2, n):
        out.append(f'<text x="{pad_l + k * bw:.1f}" y="{height - 5}" text-anchor="middle" class="tick">{edges[k]:g}{xunit}</text>')
    out.append("</svg>")
    return "".join(out)


def table(header: Sequence[str], rows: List[Sequence], num_from: int = 1, cls: str = "") -> str:
    h = "".join(f"<th>{escape(str(x))}</th>" for x in header)
    body = []
    for r in rows:
        cells = []
        for i, x in enumerate(r):
            txt = _fmt(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)
            cells.append(f'<td class="{"num" if i >= num_from else ""}">{txt if str(txt).startswith("<") else escape(txt)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'
