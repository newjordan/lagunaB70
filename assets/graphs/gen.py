#!/usr/bin/env python3
"""Generate the Laguna B70 README figure (light + dark SVG).

Source: one matched ladder run, results/{base,tip}/ladder.md.
Regenerate:  python3 assets/graphs/gen.py assets/graphs
"""

import sys

W, H = 880, 480
PAD_L, PAD_R = 32, 848
LABEL_X = 150          # right edge of the row-label column
BAR_X = 166            # left edge of every bar
P1_MAX_W = 440         # px for the largest decode bar
P2_MAX_W = 360         # px for the largest prefill bar (different scale: shorter)
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

# results/base/ladder.md vs results/tip/ladder.md
DECODE = [("stock", 108.29), ("lagunaB70", 136.27)]
DELTA = "+26% vs stock"
PREFILL = [("512", 1142.45, 1173.85), ("2048", 1954.15, 2028.85),
           ("4096", 1932.54, 2002.31), ("8192", 1879.59, 1950.12)]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e",
                  rule="#e1e0d9", series="#2a78d6", stock="#898781",
                  delta="#006300"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7",
                 rule="#2c2c2a", series="#3987e5", stock="#6b6a65",
                 delta="#0ca30c"),
}


def bar(x, y, w, h, fill, r=4):
    """Rect with only the data end (right) rounded, flat on the zero baseline."""
    w = max(w, r + 0.5)
    return (f'<path d="M{x} {y} H{x + w - r:.1f} A{r} {r} 0 0 1 {x + w:.1f} {y + r} '
            f'V{y + h - r} A{r} {r} 0 0 1 {x + w - r:.1f} {y + h} H{x} Z" fill="{fill}"/>')


def text(x, y, s, size, fill, weight=400, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{extra}>{s}</text>')


def build(mode):
    c = THEMES[mode]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d" '
         f"font-family='{FONT}'>",
         '<title id="t">Laguna B70 speed vs the stock build</title>',
         '<desc id="d">Writing an answer runs at 136 tokens per second versus 108 '
         'for the stock build, 26 percent faster. Reading a prompt is about 3 '
         'percent faster at 512, 2048, 4096 and 8192 tokens.</desc>',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>']

    # ---- header -----------------------------------------------------------
    o.append(text(PAD_L, 44, "Laguna B70 vs the stock build", 23, c["ink"], 600))
    o.append(text(PAD_L, 68, "Laguna-XS-2.1 Q4_K_M · Intel Arc Pro B70 · one request "
                  "at a time · tok/s, higher is better", 13.5, c["secondary"]))

    # legend — identity never rests on color alone
    lx = PAD_R
    for name, col in (("lagunaB70", c["series"]), ("stock", c["stock"])):
        o.append(text(lx, 44, name, 13, c["secondary"], 500, anchor="end"))
        wpx = 7.1 * len(name)
        o.append(f'<rect x="{lx - wpx - 18:.1f}" y="35" width="10" height="10" '
                 f'rx="2" fill="{col}"/>')
        lx -= wpx + 40

    # ---- panel 1: writing (the headline) -----------------------------------
    o.append(text(PAD_L, 112, "WRITING AN ANSWER — 128 tokens", 11.5, c["secondary"],
                  600, extra=' letter-spacing="0.09em"'))
    p1max = max(v for _, v in DECODE)
    y = 126
    for name, val in DECODE:
        win = name != "stock"
        w = val / p1max * P1_MAX_W
        o.append(bar(BAR_X, y, w, 34, c["series"] if win else c["stock"]))
        o.append(text(LABEL_X, y + 22, name, 14, c["secondary"],
                      600 if win else 400, anchor="end"))
        o.append(text(BAR_X + w + 14, y + 24, f"{val:.0f}", 21, c["ink"], 600))
        o.append(text(BAR_X + w + 14 + 13 * len(f"{val:.0f}"), y + 24, "tok/s", 13,
                      c["secondary"]))
        if win:
            o.append(text(PAD_R, y + 25, DELTA, 17, c["delta"], 700, anchor="end"))
        y += 46

    # ---- panel 2: reading (subordinate, its own scale) ---------------------
    o.append(f'<line x1="{PAD_L}" y1="242" x2="{PAD_R}" y2="242" '
             f'stroke="{c["rule"]}" stroke-width="1"/>')
    o.append(text(PAD_L, 272, "READING A PROMPT — by prompt length · separate scale",
                  11.5, c["secondary"], 600, extra=' letter-spacing="0.09em"'))
    p2max = max(max(a, b) for _, a, b in PREFILL)
    y = 292
    for name, stock, tip in PREFILL:
        o.append(text(LABEL_X, y + 18, f"{name} tok", 13, c["secondary"], 400,
                      anchor="end"))
        for val, col in ((stock, c["stock"]), (tip, c["series"])):
            w = val / p2max * P2_MAX_W
            o.append(bar(BAR_X, y, w, 11, col))
            o.append(text(BAR_X + w + 10, y + 9.5, f"{val:,.0f}", 12, c["secondary"]))
            y += 14
        y += 12

    o.append(text(PAD_L, 462, "Stock is the same binary and the same model file with "
                  "these kernels off. Each panel has its own scale; every bar starts "
                  "at zero.", 11.5, c["secondary"]))
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for mode in THEMES:
        with open(f"{out}/benchmark-{mode}.svg", "w") as f:
            f.write(build(mode) + "\n")
        print(f"wrote {out}/benchmark-{mode}.svg")
