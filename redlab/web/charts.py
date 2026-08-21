"""Minimal hand-rolled SVG chart helpers.

No charting library, no CDN: the page must render standalone for a judge
running the repo offline. These functions return small strings of SVG markup
consumed directly by the templates.
"""

from typing import List, Optional, Sequence, Tuple

W, H = 640, 220
PAD_L, PAD_R, PAD_T, PAD_B = 46, 20, 16, 30


def _scale(values: Sequence[float], lo: Optional[float] = None,
          hi: Optional[float] = None) -> Tuple[float, float]:
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def line_chart(series: List[Tuple[str, Sequence[float], str]],
               x_labels: Sequence[str], y_fmt=lambda v: f"{v:.0f}",
               y_domain: Optional[Tuple[float, float]] = None) -> str:
    """series: list of (name, values, color). All series share x positions."""
    n = len(x_labels)
    all_vals = [v for _, vals, _ in series for v in vals]
    lo, hi = y_domain or _scale(all_vals, lo=0)
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def xy(i: int, v: float) -> Tuple[float, float]:
        x = PAD_L + (i / max(n - 1, 1)) * plot_w
        y = PAD_T + (1 - (v - lo) / (hi - lo)) * plot_h
        return x, y

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="SF Mono, Menlo, monospace" font-size="10">']

    # gridlines + y labels
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + (1 - frac) * plot_h
        val = lo + frac * (hi - lo)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                    f'stroke="#232c3a" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+3:.1f}" text-anchor="end" '
                    f'fill="#8a97a8">{y_fmt(val)}</text>')

    for i, lbl in enumerate(x_labels):
        x, _ = xy(i, lo)
        parts.append(f'<text x="{x:.1f}" y="{H-8}" text-anchor="middle" '
                    f'fill="#8a97a8">{lbl}</text>')

    for name, values, color in series:
        pts = [xy(i, v) for i, v in enumerate(values)]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>')

    parts.append("</svg>")
    return "".join(parts)


def bar_chart(labels: Sequence[str], values: Sequence[float], color: str = "#4f9dff",
             y_fmt=lambda v: f"{v:.2f}", y_domain: Optional[Tuple[float, float]] = None,
             highlight_negative: bool = False, neg_color: str = "#ff6b6b") -> str:
    n = len(labels)
    lo, hi = y_domain or _scale(list(values) + [0])
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    bw = plot_w / n * 0.6
    zero_y = PAD_T + (1 - (0 - lo) / (hi - lo)) * plot_h

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="SF Mono, Menlo, monospace" font-size="9.5">']
    for frac in (0, 0.5, 1.0):
        y = PAD_T + (1 - frac) * plot_h
        val = lo + frac * (hi - lo)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                    f'stroke="#232c3a" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+3:.1f}" text-anchor="end" '
                    f'fill="#8a97a8">{y_fmt(val)}</text>')

    for i, (lbl, v) in enumerate(zip(labels, values)):
        cx = PAD_L + (i + 0.5) / n * plot_w
        y = PAD_T + (1 - (v - lo) / (hi - lo)) * plot_h
        top, bot = (y, zero_y) if v >= 0 else (zero_y, y)
        c = neg_color if (highlight_negative and v < 0) else color
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                    f'height="{max(bot-top,1):.1f}" fill="{c}" rx="2"/>')
        parts.append(f'<text x="{cx:.1f}" y="{H-8}" text-anchor="middle" '
                    f'fill="#8a97a8" transform="rotate(0)">{lbl}</text>')
    parts.append("</svg>")
    return "".join(parts)
