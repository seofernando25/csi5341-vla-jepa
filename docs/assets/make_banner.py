"""Generate the abstract mathematical & robotics banner for the csi5341-vla-jepa README.

Run: uv run --no-sync python docs/assets/make_banner.py
"""

from __future__ import annotations

import math
import os

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw
from scipy.special import ndtri

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H = 1600, 440
DPI = 200
INK_HEX = "#0b0f17"
PAPER_HEX = "#f3efe6"
TEAL_HEX = "#4cc3b8"  # Vision encoder
INDIGO_HEX = "#6384f2"  # Language decoder
AMBER_HEX = "#f2a93b"  # Flow-matching action head
ROSE_HEX = "#eb5e83"  # Divergence / threshold accent
MUTE_HEX = "#8ea3c2"
BORDER = (36, 46, 66)


def _rounded_card(im: Image.Image, radius: int = 18) -> Image.Image:
    im = im.convert("RGBA")
    w, h = im.size
    mask = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], radius * 2, fill=255)
    im.putalpha(mask.resize((w, h), Image.LANCZOS))
    ImageDraw.Draw(im).rounded_rectangle(
        [0, 0, w - 1, h - 1], radius, outline=BORDER + (255,), width=2
    )
    return im


def _ik_2link(
    base: tuple[float, float],
    target: tuple[float, float],
    l1: float,
    l2: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    bx, by = base
    tx, ty = target
    dx, dy = tx - bx, ty - by
    d = min(math.hypot(dx, dy), l1 + l2 - 0.02)
    cos_a = max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)))
    a = math.atan2(dy, dx) + math.acos(cos_a)
    ex = bx + l1 * math.cos(a)
    ey = by + l1 * math.sin(a)
    wx = bx + d * math.cos(math.atan2(dy, dx))
    wy = by + d * math.sin(math.atan2(dy, dx))
    return (ex, ey), (wx, wy)


def main() -> None:
    plt.rcParams["mathtext.fontset"] = "stix"

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI, facecolor=INK_HEX)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=INK_HEX)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    # 1. Subtle tri-modal radial glow (Vision Teal -> Language Indigo -> Action Amber)
    gx, gy = np.meshgrid(np.linspace(0, 16, 640), np.linspace(0, 4.4, 176))
    g_vis = np.exp(-(((gx - 2.4) / 3.3) ** 2 + ((gy - 2.5) / 1.7) ** 2))
    g_lang = np.exp(-(((gx - 8.0) / 4.2) ** 2 + ((gy - 2.2) / 1.8) ** 2))
    g_act = np.exp(-(((gx - 13.6) / 3.2) ** 2 + ((gy - 2.0) / 1.7) ** 2))

    bg = np.zeros((*gx.shape, 4), dtype=float)
    bg[..., 0] = (11 + 8 * g_vis + 12 * g_lang + 25 * g_act) / 255.0
    bg[..., 1] = (15 + 23 * g_vis + 16 * g_lang + 16 * g_act) / 255.0
    bg[..., 2] = (23 + 22 * g_vis + 34 * g_lang + 8 * g_act) / 255.0
    bg[..., 3] = 1.0
    ax.imshow(bg, extent=[0, 16, 0, 4.4], origin="lower", aspect="auto", zorder=0)

    # 2. Orthogonal phase-space grid & diagonal transport axes
    for x in np.arange(1.0, 16.0, 1.0):
        ax.axvline(x, color=MUTE_HEX, alpha=0.05, lw=0.7, zorder=1)
    for y in np.arange(0.55, 4.4, 0.55):
        ax.axhline(y, color=MUTE_HEX, alpha=0.05, lw=0.7, zorder=1)
    ax.plot([0.3, 15.7], [0.55, 3.85], color=MUTE_HEX, ls=(0, (6, 6)), lw=0.7, alpha=0.08, zorder=1)
    ax.plot([0.8, 15.2], [3.85, 0.55], color=MUTE_HEX, ls=(0, (4, 8)), lw=0.7, alpha=0.07, zorder=1)

    # 3. Delicate flow-matching ODE velocity field (small hairline vectors)
    qx, qy = np.meshgrid(np.linspace(0.65, 15.35, 25), np.linspace(0.50, 3.90, 8))
    target_y = np.where(qy > 2.2, 2.82 + 0.32 * np.sin(0.45 * qx), 1.48 - 0.28 * np.cos(0.42 * qx))
    u = 1.0 + 0.16 * np.cos(0.4 * qx)
    v = 0.52 * (target_y - qy) + 0.14 * np.sin(0.65 * qx + 0.7 * qy)
    norm = np.hypot(u, v)
    ax.quiver(
        qx,
        qy,
        u / norm,
        v / norm,
        color=MUTE_HEX,
        alpha=0.065,
        scale=52,
        width=0.0010,
        headwidth=3.2,
        headlength=4.0,
        zorder=1,
    )

    # 4. Flow-matching probability transport trajectories (noise prior -> action chunks)
    ts = np.linspace(0.0, 1.0, 320)
    xs_flow = -0.2 + 16.4 * ts
    n_streams = 26
    for idx in range(n_streams):
        y0 = 0.25 + 3.90 * (idx / (n_streams - 1))
        mode = 1 if idx % 2 == 0 else -1
        y1 = 2.20 + mode * (0.48 + 0.42 * np.sin(idx * 0.7))
        blend = 3 * ts**2 - 2 * ts**3
        ripple = (
            (0.26 + 0.11 * np.sin(idx)) * np.sin(np.pi * ts) * np.sin(2.4 * np.pi * ts + 0.35 * idx)
        )
        ys_flow = (1.0 - blend) * y0 + blend * y1 + ripple

        pts = np.column_stack([xs_flow, ys_flow]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = []
        alpha_line = 0.16 if idx % 3 else 0.23
        for s in ts[:-1]:
            if s < 0.5:
                k = s * 2.0
                c = (
                    (76 * (1 - k) + 99 * k) / 255.0,
                    (195 * (1 - k) + 132 * k) / 255.0,
                    (184 * (1 - k) + 242 * k) / 255.0,
                    alpha_line,
                )
            else:
                k = (s - 0.5) * 2.0
                c = (
                    (99 * (1 - k) + 242 * k) / 255.0,
                    (132 * (1 - k) + 169 * k) / 255.0,
                    (242 * (1 - k) + 59 * k) / 255.0,
                    alpha_line,
                )
            cols.append(c)
        lw = 1.25 if idx % 5 == 0 else 0.78
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=2))

        # Highlight 4-step Euler flow waypoints on select outer trajectories
        if idx in (8, 21):
            for st in (0.11, 0.32, 0.68, 0.93):
                k_idx = int(st * (len(ts) - 1))
                ax.scatter(
                    [xs_flow[k_idx]],
                    [ys_flow[k_idx]],
                    s=20,
                    facecolors=INK_HEX,
                    edgecolors=AMBER_HEX if st > 0.5 else TEAL_HEX,
                    linewidths=1.05,
                    alpha=0.52,
                    zorder=3,
                )

    # 5. Left: Vision Patch Grid projecting into V-JEPA 2 Latent World-Model Rings
    patch_x0, patch_y0 = 0.48, 2.58
    ps = 0.155
    for r in range(5):
        for c in range(5):
            active = (r + c) % 3 == 0 or (r == 2 and c in (1, 2, 3))
            rect = Rectangle(
                (patch_x0 + c * (ps + 0.03), patch_y0 + r * (ps + 0.03)),
                ps,
                ps,
                facecolor=TEAL_HEX if active else MUTE_HEX,
                edgecolor=TEAL_HEX if active else MUTE_HEX,
                linewidth=0.6,
                alpha=0.15 if active else 0.06,
                zorder=2,
            )
            ax.add_patch(rect)

    jepa_c = (2.78, 2.38)
    # Faint projection rays from patch grid corners into latent space
    for py_end in (patch_y0 + 0.08, patch_y0 + 4 * (ps + 0.03) + 0.08):
        ax.plot(
            [patch_x0 + 5 * (ps + 0.03) - 0.02, jepa_c[0] - 0.52],
            [py_end, jepa_c[1]],
            color=TEAL_HEX,
            ls=(0, (3, 4)),
            lw=0.75,
            alpha=0.18,
            zorder=2,
        )

    for rad, col, al, ls in (
        (0.70, TEAL_HEX, 0.24, "-"),
        (0.48, INDIGO_HEX, 0.27, "--"),
        (0.26, PAPER_HEX, 0.20, ":"),
    ):
        ax.add_patch(
            Circle(jepa_c, rad, fill=False, edgecolor=col, lw=1.0, ls=ls, alpha=al, zorder=3)
        )
    ax.add_patch(
        FancyArrowPatch(
            (jepa_c[0] - 0.34, jepa_c[1] - 0.18),
            (jepa_c[0] + 0.40, jepa_c[1] + 0.25),
            arrowstyle="-|>",
            mutation_scale=9.5,
            color=TEAL_HEX,
            lw=1.1,
            alpha=0.45,
            zorder=3,
        )
    )
    ax.scatter(
        [jepa_c[0] - 0.34, jepa_c[0] + 0.40],
        [jepa_c[1] - 0.18, jepa_c[1] + 0.25],
        s=18,
        c=[INDIGO_HEX, AMBER_HEX],
        alpha=0.60,
        zorder=4,
    )

    # 6. Right: Geometric Robotic Arm Kinematics & 7-Step Action Chunk Arc
    base_pt = (12.20, 0.62)
    l1, l2 = 1.62, 1.42
    s_arc = np.linspace(0.0, 1.0, 80)
    ee_xs = 12.50 + 2.55 * s_arc
    ee_ys = 1.24 + 1.50 * np.sin(np.pi * s_arc) + 0.24 * s_arc
    ax.plot(ee_xs, ee_ys, color=AMBER_HEX, ls=(0, (3, 3)), lw=1.15, alpha=0.36, zorder=3)

    # Minimalist geometric pick/place target ellipses
    ax.add_patch(
        Ellipse(
            (12.50, 1.18), 0.34, 0.12, fill=False, edgecolor=TEAL_HEX, lw=0.95, alpha=0.32, zorder=3
        )
    )
    ax.add_patch(
        Ellipse(
            (15.05, 1.44),
            0.44,
            0.14,
            fill=False,
            edgecolor=AMBER_HEX,
            lw=0.95,
            alpha=0.36,
            zorder=3,
        )
    )

    # 7-action chunk waypoints along the trajectory
    for ci, cs in enumerate(np.linspace(0.06, 0.96, 7)):
        cx_i = 12.50 + 2.55 * cs
        cy_i = 1.24 + 1.50 * np.sin(np.pi * cs) + 0.24 * cs
        ax.scatter(
            [cx_i],
            [cy_i],
            s=26 if ci in (0, 3, 6) else 14,
            c=AMBER_HEX if ci >= 3 else TEAL_HEX,
            alpha=0.50,
            linewidths=0,
            zorder=4,
        )

    # Ghosted kinematic poses along the arc
    for cs, op in ((0.18, 0.13), (0.52, 0.23), (0.86, 0.36)):
        tx = 12.50 + 2.55 * cs
        ty = 1.24 + 1.50 * np.sin(np.pi * cs) + 0.24 * cs
        elbow, wrist = _ik_2link(base_pt, (tx, ty), l1, l2)
        ax.plot(
            [base_pt[0], elbow[0], wrist[0]],
            [base_pt[1], elbow[1], wrist[1]],
            color=PAPER_HEX,
            lw=1.55,
            alpha=op,
            zorder=3,
        )
        ax.scatter(
            [base_pt[0], elbow[0], wrist[0]],
            [base_pt[1], elbow[1], wrist[1]],
            s=22,
            facecolors=INK_HEX,
            edgecolors=AMBER_HEX,
            lw=1.05,
            alpha=min(0.68, op * 1.8),
            zorder=4,
        )
        ang = math.atan2(wrist[1] - elbow[1], wrist[0] - elbow[0])
        nx, ny = -math.sin(ang), math.cos(ang)
        fx, fy = math.cos(ang), math.sin(ang)
        for sign in (-1, 1):
            p1 = (wrist[0] + sign * 0.10 * nx, wrist[1] + sign * 0.10 * ny)
            p2 = (p1[0] + 0.16 * fx, p1[1] + 0.16 * fy)
            ax.plot(
                [wrist[0], p1[0], p2[0]],
                [wrist[1], p1[1], p2[1]],
                color=AMBER_HEX,
                lw=0.95,
                alpha=op * 1.35,
                zorder=3,
            )

    ax.add_patch(
        Arc(
            base_pt,
            0.80,
            0.80,
            angle=0,
            theta1=18,
            theta2=76,
            color=AMBER_HEX,
            lw=0.9,
            alpha=0.30,
            zorder=3,
        )
    )
    ax.plot(
        [base_pt[0] - 0.30, base_pt[0] + 0.30],
        [base_pt[1], base_pt[1]],
        color=MUTE_HEX,
        lw=1.4,
        alpha=0.35,
        zorder=3,
    )

    # 7. Bottom-left: Bit-Width Compression Ladder (left) & NF4 Quantile Bell Curve (right of ladder)
    for k, (bw, col) in enumerate(
        [
            (1.12, MUTE_HEX),
            (0.84, INDIGO_HEX),
            (0.46, TEAL_HEX),
            (0.35, AMBER_HEX),
            (0.24, ROSE_HEX),
        ]
    ):
        yb = 0.36 + k * 0.115
        ax.plot([0.42, 0.42 + bw], [yb, yb], color=col, lw=1.8, alpha=0.36, zorder=3)
        ax.scatter(
            [0.42, 0.42 + bw], [yb, yb], s=8, color=PAPER_HEX, alpha=0.36, linewidths=0, zorder=3
        )

    wx = np.linspace(-2.35, 2.35, 200)
    wy = np.exp(-0.5 * wx**2)
    gx_center = 3.05
    gx_plot = gx_center + wx * 0.50
    gy_base = 0.34
    ax.plot(gx_plot, gy_base + 0.72 * wy, color=INDIGO_HEX, lw=1.15, alpha=0.34, zorder=3)
    ax.fill_between(gx_plot, gy_base, gy_base + 0.72 * wy, color=INDIGO_HEX, alpha=0.06, zorder=2)
    for qi, qv in enumerate(ndtri((np.arange(16) + 0.5) / 16.0)):
        qx_pos = gx_center + qv * 0.50
        qh = 0.72 * math.exp(-0.5 * qv**2)
        ax.plot(
            [qx_pos, qx_pos],
            [gy_base, gy_base + qh],
            color=TEAL_HEX if qi in (7, 8) else INDIGO_HEX,
            lw=0.85,
            alpha=0.34,
            zorder=3,
        )

    # 8. Top-right: CUDA Graph Kernel Fusion DAG motif
    dag_nodes = [
        (13.65, 3.92),
        (14.05, 4.10),
        (14.05, 3.74),
        (14.52, 4.04),
        (14.52, 3.80),
        (15.05, 3.92),
        (15.48, 3.92),
    ]
    for u_i, v_i in ((0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (3, 5), (4, 5), (5, 6)):
        ax.plot(
            [dag_nodes[u_i][0], dag_nodes[v_i][0]],
            [dag_nodes[u_i][1], dag_nodes[v_i][1]],
            color=INDIGO_HEX if v_i < 5 else AMBER_HEX,
            lw=0.95,
            alpha=0.32,
            zorder=3,
        )
    for ni, (nx_p, ny_p) in enumerate(dag_nodes):
        ax.scatter(
            [nx_p],
            [ny_p],
            s=26 if ni in (0, 5, 6) else 16,
            c=AMBER_HEX if ni >= 5 else TEAL_HEX,
            alpha=0.42,
            linewidths=0,
            zorder=4,
        )

    # 9. Mathematical background equations (cleanly spaced in quiet zones)
    equations = [
        (
            2.45,
            3.92,
            r"$\frac{\mathrm{d}a_\tau}{\mathrm{d}\tau} = v_\theta(a_\tau, \tau \mid z_t, \ell)$",
            13.0,
            0.34,
            PAPER_HEX,
            -1.0,
        ),
        (
            6.05,
            3.88,
            r"$a_{\tau+\Delta\tau} = a_\tau + \Delta\tau\, v_\theta(a_\tau, \tau)$",
            12.5,
            0.31,
            TEAL_HEX,
            1.0,
        ),
        (
            9.75,
            3.90,
            r"$\mathcal{L}_{\mathrm{JEPA}} = \|f_\psi(z_t, a_t) - \mathrm{sg}(\bar{z}_{t+\Delta})\|_2^2$",
            12.5,
            0.32,
            INDIGO_HEX,
            -1.2,
        ),
        (
            12.20,
            3.46,
            r"$\pi_\theta(o_t, \ell) \mapsto \mathbf{a}_{t:t+7} \in \mathbb{R}^{7 \times 7}$",
            12.0,
            0.31,
            AMBER_HEX,
            -1.5,
        ),
        (
            4.55,
            3.06,
            r"$\mathcal{Q}_{\mathrm{NF4}}: \; q_i = \Phi^{-1}\!\left(\frac{i + 1/2}{16}\right)\sigma_B$",
            12.0,
            0.30,
            PAPER_HEX,
            1.5,
        ),
        (10.50, 3.02, r"$u_t(a \mid a_1) = a_1 - a_0$", 12.0, 0.29, TEAL_HEX, -1.5),
        (14.65, 3.12, r"$T(\theta) \in \mathrm{SE}(3)$", 12.5, 0.31, PAPER_HEX, 1.0),
        (
            1.18,
            1.66,
            r"$\hat{W} = s_2 \cdot \mathrm{round}(s_1^{-1} W_{\mathrm{BF16}})$",
            11.8,
            0.30,
            INDIGO_HEX,
            -1.5,
        ),
        (
            5.20,
            1.24,
            r"$\mathrm{CKA}(K, L) = \frac{\|L^\top K\|_F^2}{\|K^\top K\|_F \|L^\top L\|_F}$",
            12.0,
            0.30,
            MUTE_HEX,
            1.2,
        ),
        (
            6.25,
            0.50,
            r"$\Delta a = \frac{1}{N}\sum_{i=1}^{N} \|\pi_q(o_i) - \pi_{\mathrm{BF16}}(o_i)\|_2^2$",
            12.0,
            0.31,
            ROSE_HEX,
            -0.8,
        ),
        (
            10.05,
            0.52,
            r"$M(b) = \sum_{m \in \{V,L,A\}} \frac{b_m}{8} P_m$",
            12.5,
            0.31,
            PAPER_HEX,
            1.0,
        ),
        (14.55, 0.86, r"$\theta_1, \theta_2 \mapsto p_{\mathrm{ee}}$", 12.0, 0.29, AMBER_HEX, -1.5),
    ]
    for ex, ey, tex, fs, al, col, rot in equations:
        ax.text(
            ex,
            ey,
            tex,
            fontsize=fs,
            color=col,
            alpha=al,
            ha="center",
            va="center",
            rotation=rot,
            zorder=4,
        )

    # 10. Soft central backdrop so 'csi5341-vla-jepa' pops cleanly
    cx, cy = np.meshgrid(np.linspace(0, 16, 320), np.linspace(0, 4.4, 88))
    halo = np.exp(-(((cx - 8.0) / 3.85) ** 2 + ((cy - 2.2) / 0.85) ** 2))
    veil = np.zeros((*cx.shape, 4), dtype=float)
    veil[..., 0] = 11 / 255.0
    veil[..., 1] = 15 / 255.0
    veil[..., 2] = 23 / 255.0
    veil[..., 3] = 0.86 * halo
    ax.imshow(veil, extent=[0, 16, 0, 4.4], origin="lower", aspect="auto", zorder=5)

    # 11. Title text: csi5341-vla-jepa
    ax.text(
        8.04,
        2.26,
        "csi5341-vla-jepa",
        fontsize=46,
        fontfamily="Noto Sans Mono",
        fontweight="bold",
        color="#05080d",
        ha="center",
        va="center",
        alpha=0.92,
        zorder=6,
    )
    ax.text(
        8.0,
        2.30,
        "csi5341-vla-jepa",
        fontsize=46,
        fontfamily="Noto Sans Mono",
        fontweight="bold",
        color=PAPER_HEX,
        ha="center",
        va="center",
        zorder=7,
    )

    # Smooth continuous tri-color gradient bar (Vision Teal -> Language Indigo -> Action Amber)
    t = np.linspace(0.0, 1.0, 512)[None, :]
    grad = np.zeros((2, 512, 4), dtype=float)
    t1 = np.clip(t * 2.0, 0.0, 1.0)
    t2 = np.clip((t - 0.5) * 2.0, 0.0, 1.0)
    r_c = np.where(t < 0.5, 76 * (1 - t1) + 99 * t1, 99 * (1 - t2) + 242 * t2)
    g_c = np.where(t < 0.5, 195 * (1 - t1) + 132 * t1, 132 * (1 - t2) + 169 * t2)
    b_c = np.where(t < 0.5, 184 * (1 - t1) + 242 * t1, 242 * (1 - t2) + 59 * t2)
    grad[..., 0] = r_c / 255.0
    grad[..., 1] = g_c / 255.0
    grad[..., 2] = b_c / 255.0
    grad[..., 3] = 1.0
    ax.imshow(grad, extent=[4.75, 11.25, 1.62, 1.69], origin="lower", aspect="auto", zorder=7)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(out_dir, "_banner_raw.png")
    fig.savefig(raw_path, dpi=DPI, facecolor=INK_HEX)
    plt.close(fig)

    im = Image.open(raw_path).resize((W, H), Image.LANCZOS)
    im = _rounded_card(im, radius=18)
    im.save(os.path.join(out_dir, "banner.png"), optimize=True)
    os.remove(raw_path)


if __name__ == "__main__":
    main()
