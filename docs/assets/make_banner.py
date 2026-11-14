"""csi5341-vla-jepa banner: one pick-and-place arm, rendered at falling bit depth.

Run: uv run --with pillow --with numpy python make_banner.py
The scene is a 2-link arm placing its gripper on a block. It is quantised with
ordered (Bayer) dithering to 16, 8, 4, 3 and 2 bits (the bit sweep of the study) from left to
right: the silhouette, the behaviour, survives while the bits go. Numbers in the
caption are from report/data/numbers.json (B16 stock vs C2: peak 5.61 -> 1.51 GB,
median latency 79 -> 32 ms, LIBERO-Spatial success 95.0 -> 98.0 %).
"""

import itertools
import math
import os

import numpy as np
from PIL import Image, ImageDraw

W, H = 1600, 480
CELL = 4  # one dither cell = 4 px
INK, PAPER, MUTE, LINE, ACC = (
    (14, 19, 26),
    (236, 231, 218),
    (107, 118, 134),
    (38, 48, 60),
    (63, 208, 192),
)
RAMP = [(0.0, INK), (0.35, (20, 52, 60)), (0.72, ACC), (1.0, PAPER)]  # luminance -> colour

FONT = """
c 2 .#### #.... #.... #.... .####
s 2 .#### #.... .###. ....# ####.
i 0 ..#.. ..... .##.. ..#.. ..#.. ..#.. .###.
j 0 ...#. ..... ..##. ...#. ...#. ...#. ...#. #..#. .##..
v 2 #...# #...# #...# .#.#. ..#..
l 0 .##.. ..#.. ..#.. ..#.. ..#.. ..#.. .###.
a 2 .###. ....# .#### #...# .####
e 2 .###. #...# ##### #.... .####
p 2 ####. #...# #...# ####. #.... #.... #....
g 2 .#### #...# #...# .#### ....# #...# .###.
b 0 #.... #.... ####. #...# #...# #...# ####.
t 1 .#... ####. .#... .#... .#... ..###
o 2 .###. #...# #...# #...# .###.
m 2 ##.#. #.#.# #.#.# #.#.# #.#.#
- 4 .###.
. 6 ..#..
0 0 .###. #...# #..## #.#.# ##..# #...# .###.
1 0 ..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.
2 0 .###. #...# ....# ...#. ..#.. .#... #####
3 0 .###. #...# ....# ..##. ....# #...# .###.
4 0 ...#. ..##. .#.#. #..#. ##### ...#. ...#.
5 0 ##### #.... ####. ....# ....# #...# .###.
6 0 .###. #.... #.... ####. #...# #...# .###.
7 0 ##### ....# ...#. ..#.. .#... .#... .#...
8 0 .###. #...# #...# .###. #...# #...# .###.
9 0 .###. #...# #...# .#### ....# ...#. .##..
"""
GLYPHS = {}
for ln in FONT.strip().splitlines():
    c, r0, *rows = ln.split()
    GLYPHS[c] = (int(r0), rows)


def adv(c, s):
    return (3 if c == "." else 6) * s


def text(d, x, y, s, scale, fill):
    """Pixel lettering on a 5-wide grid; y is the top of the ascender row."""
    for ch in s:
        r0, rows = GLYPHS.get(ch, (0, []))
        for j, row in enumerate(rows):
            for i, c in enumerate(row):
                if c == "#":
                    px, py = x + i * scale, y + (r0 + j) * scale
                    d.rectangle([px, py, px + scale - 1, py + scale - 1], fill=fill)
        x += adv(ch, scale)


def text_w(s, scale):
    return sum(adv(c, scale) for c in s) - scale


# --- scene (luminance field on the dither grid) ---------------------------------
AX0, AX1 = 600, 1560  # art extent in px
GW, GH = (AX1 - AX0) // CELL, H // CELL  # 240 x 120 cells
SSF = 3  # scene supersampling per cell


def grid(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx + 0.5) / (SSF * 1.0), (yy + 0.5) / (SSF * 1.0)  # in cell units


def seg(x, y, a, b, r, shade):
    """Rounded link from a to b (cell units): returns coverage and cylindrical shading."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0, 1)
    px, py = ax + t * dx, ay + t * dy
    dist = np.hypot(x - px, y - py)
    side = ((x - ax) * dy - (y - ay) * dx) / math.sqrt(L2)  # signed offset across the link
    cov = np.clip((r - dist) * SSF + 0.5, 0, 1)
    cyl = np.sqrt(np.clip(1 - (np.clip(side / r, -1, 1)) ** 2, 0, 1))
    lit = 0.5 + 0.5 * np.clip(-side / r, -1, 1) * 0.6 + 0.0  # light from upper left
    return cov, shade * (0.45 + 0.55 * cyl) * (0.75 + 0.25 * lit)


def scene():
    x, y = grid(GW * SSF, GH * SSF)
    v = np.zeros_like(x)
    base = GH * 0.84  # table line
    sh = (GW * 0.095, base - 8)  # shoulder
    l1, l2 = 78.0, 80.0
    bowl = (GW * 0.60, base - 4.5)  # object to pick
    plate = (GW * 0.895, base - 1.5)  # place target
    v += (y < base) * 0.34 * np.exp(
        -(((x - bowl[0]) / 95) ** 2 + ((y - bowl[1]) / 50) ** 2)
    ) + 0.20 * np.clip(y / base, 0, 1) ** 1.5 * (y < base)
    v = np.where(np.abs(y - base) < 0.5, 0.55, v)
    v = np.where((y > base) & (y < base + 14), 0.06, v)

    def target(s):
        """End-effector path: home -> above the bowl (s=1) -> carry to the plate (s=2)."""
        if s <= 1:
            k = s
            return (
                sh[0] + 62 + (bowl[0] - sh[0] - 62) * k,
                (sh[1] - 60) + (bowl[1] - 9 - (sh[1] - 60)) * k**1.6,
            )
        k = s - 1
        return (bowl[0] + (plate[0] - bowl[0]) * k, bowl[1] - 9 - 38 * math.sin(math.pi * k))

    def pose(s):
        tx, ty = target(s)
        dx, dy = tx - sh[0], ty - sh[1]
        d = min(math.hypot(dx, dy), l1 + l2 - 0.01)
        a = math.atan2(dy, dx) - math.acos((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))  # elbow up
        return (sh[0] + l1 * math.cos(a), sh[1] + l1 * math.sin(a)), (tx, ty)

    def blob(c, r, val):
        cv = np.clip((r - np.hypot(x - c[0], y - c[1])) * SSF + 0.5, 0, 1)
        return cv, val

    # plate: flat ellipse; bowl: half-round vessel
    pe = ((x - plate[0]) / 21) ** 2 + ((y - plate[1]) / 2.6) ** 2 < 1
    v = np.where(pe, 0.78, v)
    bw = (((x - bowl[0]) / 8) ** 2 + ((y - bowl[1] + 2) / 6.5) ** 2 < 1) & (y > bowl[1] - 2.5)
    v = np.where(bw, 0.9 - 0.35 * np.clip((x - bowl[0]) / 8, 0, 1), v)
    for s_, op in ((0.0, 0.14), (0.35, 0.2), (0.7, 0.3), (1.0, 1.0), (1.45, 0.2), (1.8, 0.14)):
        e, g = pose(s_)
        for a_, b_, r, sh_v in ((sh, e, 4.6, 0.9), (e, g, 3.6, 0.95)):
            c, col = seg(x, y, a_, b_, r, sh_v)
            v = v * (1 - c * op) + c * op * col
        for p, r in ((e, 5.2), (g, 3.0)):
            c, _ = blob(p, r, 1)
            v = v * (1 - c * op) + c * op * 0.98
        if s_ == 1.0:  # gripper fingers around the bowl
            for off in (-9, 9):
                c, _ = seg(x, y, (g[0] + off, g[1] - 1), (g[0] + off * 0.9, g[1] + 7), 1.6, 0.9)
                v = v * (1 - c) + c * 0.92
    ped = (np.abs(x - sh[0]) < 8) & (y > sh[1] - 3) & (y < base)
    v = np.where(ped, 0.5 + 0.3 * np.clip((x - sh[0] + 8) / 16, 0, 1), v)
    for k in np.linspace(0, 2, 61):  # dotted end-effector path
        _, g = pose(k)
        v = np.maximum(v, np.clip((1.0 - np.hypot(x - g[0], y - g[1])) * SSF + 0.5, 0, 1) * 0.85)
    v = v.reshape(GH, SSF, GW, SSF).mean(axis=(1, 3))
    return v * np.clip(np.arange(GW) / 16, 0, 1)[None, :] ** 1.5  # fade into the page on the left


BAYER = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ]
    )
    / 64.0
)


def ramp(v):
    out = np.zeros(v.shape + (3,))
    for (a, ca), (b, cb) in itertools.pairwise(RAMP):
        m = (v >= a) & (v <= b)
        t = ((v - a) / (b - a))[..., None]
        out = np.where(m[..., None], np.array(ca) * (1 - t) + np.array(cb) * t, out)
    return out


BITS = [16, 8, 4, 3, 2]  # the bit-depth sweep of the study


def main():
    v = scene()
    bd = np.tile(BAYER, (GH // 8 + 1, GW // 8 + 1))[:GH, :GW]
    q = np.zeros_like(v)
    band = GW // len(BITS)
    for i, b in enumerate(BITS):
        lv = 2**b - 1
        sl = slice(i * band, (i + 1) * band)
        q[:, sl] = np.floor(v[:, sl] * lv + bd[:, sl]) / lv
    q = np.clip(q, 0, 1)
    art = np.kron(ramp(q), np.ones((CELL, CELL, 1))).astype(np.uint8)
    img = Image.new("RGB", (W, H), INK)
    img.paste(Image.fromarray(art), (AX0, 0))
    d = ImageDraw.Draw(img)
    # bit-depth labels under the table
    for i, b in enumerate(BITS):
        x0 = AX0 + i * band * CELL
        if i:
            d.line([(x0, 28), (x0, 452)], fill=INK, width=2)
        lab = f"{b} bit"
        text(d, x0 + band * CELL // 2 - text_w(lab, 3) // 2, 438, lab, 3, (150, 160, 175))
    # title + caption
    sc = 10
    for ln, y0 in (("csi5341-", 118), ("vla-jepa", 118 + 9 * sc + 14)):
        text(d, 72 + sc // 2, y0 + sc // 2, ln, sc, (6, 9, 13))
        text(d, 72, y0, ln, sc, PAPER)
    d.rectangle(
        [72, 118 + 2 * (9 * sc + 14) - 4, 72 + text_w("vla-jepa", sc), 118 + 2 * (9 * sc + 14) + 6],
        fill=ACC,
    )
    text(d, 72, 392, "5.61 gb to 1.51 gb", 3, MUTE)
    text(d, 72, 420, "79 ms to 32 ms", 3, MUTE)
    img = img.convert("RGBA")
    mask = Image.new("L", (W * 2, H * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W * 2 - 1, H * 2 - 1], 32, fill=255)
    img.putalpha(mask.resize((W, H), Image.LANCZOS))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, W - 1, H - 1], 16, outline=LINE + (255,), width=1)
    img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.png"), optimize=True)


if __name__ == "__main__":
    main()
