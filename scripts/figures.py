"""Figures of the paper and Supplementary Movie 1.

    python scripts/figures.py               # all figures
    python scripts/figures.py s06 s07a      # selected ones
    python scripts/figures.py movie         # needs ffmpeg
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse, Patch, Polygon
from scipy.integrate import solve_ivp

from bladder import P, U1, max_real_eig, rhs, rL_on_B0
from bladder.model import saddle, theta_star
from common import DATA, FIGURES, RA, RLB, U0

DEG = 180 / np.pi
TW = 16 / 2.54
INK, INK2 = "#111111", "#555555"
C1, C2 = "#2a78d6", "#eb6834"
GOOD, BAD = "#0072B2", "#E69F00"
SB_COLOR, TR_COLOR = "#123f7a", "#0b5a3b"
MAP_COLORS = ["#f2f1ec", "#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7", "#8a8983"]
MAP_LABELS = ["равновесие", "симметричный цикл", "несимметричный цикл", "цикл кратного периода",
              "не установилось (у TR)", "хаотический аттрактор", "нет аттрактора"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": "#e4e4e4", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42,
    })


def comma(ax):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: ("%g" % v).replace(".", ",").replace("-", "−"))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(name)


def solve(x0, T, U, rL, **kw):
    return solve_ivp(rhs, (0, T), x0, args=(U, RA, rL, P), method="DOP853",
                     rtol=kw.pop("rtol", 1e-9), atol=kw.pop("atol", 1e-11), **kw)


def fish_outline(L, H):
    u = np.linspace(-1, 1, 240)
    half = 0.5 * H * (1 - u**2) ** 0.45 * np.where(u < 0, 1 - 0.55 * u**2, 1.0)
    return 0.5 * L * u, half


def belly(x, L, H):
    s = 2 * x / L
    return -0.5 * H * (1 - s**2) ** 0.45 * (1 - 0.55 * s**2)


def s00():
    th, ga = np.radians(17), np.radians(7)
    fig, ax = plt.subplots(figsize=(0.8 * TW, 0.4 * TW))
    ax.set_axis_off()
    ax.set_aspect("equal")
    L, H = 10.0, 2.3
    c, s = np.cos(th), np.sin(th)
    R = lambda px, py: (px * c - py * s, px * s + py * c)

    xs, half = fish_outline(L, H)
    body = [R(x, y) for x, y in zip(xs, half)] + [R(x, -y) for x, y in zip(xs[::-1], half[::-1])]
    ax.add_patch(Polygon(body, closed=True, fc="#eef1f4", ec="#5b6570", lw=1.2, zorder=2))
    tail = [(-0.5 * L + 0.2, 0), (-0.5 * L - 1.1, 0.9), (-0.5 * L - 0.7, 0), (-0.5 * L - 1.1, -0.9)]
    ax.add_patch(Polygon([R(*q) for q in tail], closed=True, fc="#dfe4ea", ec="#5b6570", lw=1.1, zorder=1))
    xa, xl = -3.2, -1.6
    ax.add_patch(Ellipse(R(xa, 0.1), 1.5, 0.75, angle=np.degrees(th), fc="#cfe0f6", ec=C1, lw=1.3, zorder=3))
    yb = belly(xl, L, H)
    fin = [(xl + 0.5, yb + 0.1), (xl - 0.3, yb + 0.1), (xl - 0.8, yb - 0.75), (xl - 0.1, yb - 0.7)]
    ax.add_patch(Polygon([R(*q) for q in fin], closed=True, fc="#f8d9c8", ec=C2, lw=1.2, zorder=3))

    ax.plot([-7.2, 9.2], [0, 0], color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=4)
    ax.plot(*zip(R(-6.6, 0), R(8.6, 0)), color=INK2, lw=0.7, ls=(0, (6, 2, 1, 2)), zorder=4)
    ax.plot([0, 9.0 * np.cos(ga)], [0, 9.0 * np.sin(ga)], color=INK, lw=0.8, ls=(0, (5, 2)), zorder=6)
    arrow = dict(arrowstyle="-|>", mutation_scale=12, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(9.6, -1.0), xytext=(7.0, -1.0), zorder=6, arrowprops=dict(arrow, color=INK, lw=1.4))
    ax.text(9.75, -1.0, "$U$", fontsize=11, va="center")
    for r, t1, t2, lab, col in ((6.4, 0, ga, r"$\gamma$", INK), (7.6, ga, th, r"$\alpha$", C2),
                                (3.2, 0, th, r"$\theta$", INK)):
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=np.degrees(t1), theta2=np.degrees(t2),
                         color=col, lw=1.0, zorder=6))
        m = (t1 + t2) / 2
        ax.text((r + 0.35) * np.cos(m), (r + 0.35) * np.sin(m), lab, color=col, fontsize=10.5,
                va="center", ha="left", zorder=7)
    ax.plot(0, 0, "o", color=INK, ms=4.5, zorder=8)

    pa = R(xa, 0.1)
    ax.annotate("", xy=(pa[0], pa[1] + 2.0), xytext=pa, zorder=7, arrowprops=dict(arrow, color=C1, lw=1.6))
    ax.text(pa[0] - 0.2, pa[1] + 2.0, "$F_A$", color=C1, fontsize=11, ha="right", va="top")
    pf = R(xl - 0.25, yb - 0.35)
    tip = (pf[0] - 2.6 * np.sin(ga), pf[1] + 2.6 * np.cos(ga))
    ax.annotate("", xy=tip, xytext=pf, zorder=7, arrowprops=dict(arrow, color=C2, lw=1.6))
    ax.text(tip[0] + 0.2, tip[1], "$F_L$", color=C2, fontsize=11, va="top")

    for xe, off, lab, col in ((xl, -2.3, r"$r_L$", C2), (xa, -3.1, r"$r_A$", C1)):
        ax.annotate("", xy=R(xe, off), xytext=R(0, off),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.0, shrinkA=0, shrinkB=0))
        for xx, top in ((0, 0), (xe, yb - 0.2 if xe == xl else -0.4)):
            q0, q1 = R(xx, top), R(xx, off - 0.15)
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=col, lw=0.6, ls=":")
        m = R(xe / 2, off - 0.45)
        ax.text(m[0], m[1], lab, color=col, fontsize=11, ha="center", va="center")

    ax.annotate("", xy=(-8.3, 1.9), xytext=(-8.3, -1.4),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=10))
    ax.text(-8.05, 1.75, "$z$", fontsize=11)
    ax.set_xlim(-8.8, 10.2)
    ax.set_ylim(-4.5, 4.3)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "s00_scheme")


# Fish drawn along a computed trajectory (U = 2, d = -0.15, z0 = -0.3).
# Pitch and bladder volume are exaggerated so that the phases are visible.
PITCH_GAIN = 2.2
KX, KZ = 0.95, 3.4


def draw_fish(ax, x, y, th, vol, L=1.6):
    H = 0.30 * L
    c, s = np.cos(th), np.sin(th)
    R = lambda px, py: (x + px * c - py * s, y + px * s + py * c)
    xs, half = fish_outline(L, H)
    body = [R(a, b) for a, b in zip(xs, half)] + [R(a, -b) for a, b in zip(xs[::-1], half[::-1])]
    ax.add_patch(Polygon(body, closed=True, fc="#eef1f4", ec="#5b6570", lw=1.0, zorder=3))
    e = -0.5 * L
    tail = [(e + 0.03 * L, 0), (e - 0.17 * L, 0.15 * L), (e - 0.10 * L, 0), (e - 0.17 * L, -0.15 * L)]
    ax.add_patch(Polygon([R(*q) for q in tail], closed=True, fc="#dfe4ea", ec="#5b6570", lw=0.9, zorder=2))
    ax.add_patch(Ellipse(R(-0.27 * L, 0.0), 0.22 * L * vol, 0.12 * L * vol, angle=np.degrees(th),
                         fc="#cfe0f6", ec=C1, lw=1.0, zorder=4))
    xf = -0.12 * L
    yb = belly(xf, L, H)
    fin = [(xf + 0.06 * L, yb + 0.01 * L), (xf - 0.04 * L, yb + 0.01 * L),
           (xf - 0.10 * L, yb - 0.09 * L), (xf - 0.01 * L, yb - 0.085 * L)]
    ax.add_patch(Polygon([R(*q) for q in fin], closed=True, fc="#f8d9c8", ec=C2, lw=0.9, zorder=4))
    ax.plot(*R(0.33 * L, 0.04 * L), "o", ms=2.0, color="#5b6570", zorder=5)


def bladder_volume(z, th):
    return float(np.clip(1 + 0.6 * (z + RA * np.sin(th)), 0.4, 1.45))


def mechanism_solution(T):
    return solve([-0.3, 0, 0, 0], T, 2.0, RLB - 0.15, dense_output=True).sol


def s01():
    T = 16.4
    sol = mechanism_solution(T)
    t = np.linspace(0, T, 800)
    z = sol(t)[0]
    x0, x1, y0, y1 = -1.45, KX * T + 1.0, -1.8, 1.2
    fig, ax = plt.subplots(figsize=(TW, TW * (y1 - y0) / (x1 - x0)))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.plot([x0, x1], [0, 0], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.plot(KX * t, KZ * z, color="#9aa3ad", lw=1.0, zorder=1)
    for tk in (0.0, 2.6, 5.2, 8.4, 11.8, 15.4):
        zk, _, thk, _ = sol(tk)
        draw_fish(ax, KX * tk, KZ * zk, PITCH_GAIN * thk, bladder_volume(zk, thk))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "s01_mechanism")


def movie(path=None, fps=30, T=26.0, play=9.0, dpi=200):
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    sol = mechanism_solution(T)
    kx = 0.7
    t = np.linspace(0, T, 1600)
    X, Z = kx * t, KZ * sol(t)[0]
    W, y0, y1 = 8.0, -1.75, 1.15
    fig, ax = plt.subplots(figsize=(8.0, 8.0 * (y1 - y0) / W), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    n = int(fps * play)

    def frame(i):
        ax.clear()
        ax.set_axis_off()
        ax.set_aspect("equal")
        tk = min(T, T * i / (n - 1))
        left = kx * tk - 0.38 * W
        ax.set_xlim(left, left + W)
        ax.set_ylim(y0, y1)
        ax.plot([left - 1, left + W + 1], [0, 0], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.plot(X, Z, color="#e3e6ea", lw=1.2, zorder=0.5)
        ax.plot(X[t <= tk], Z[t <= tk], color="#8a939d", lw=1.6, zorder=1)
        zk, _, thk, _ = sol(tk)
        draw_fish(ax, kx * tk, KZ * zk, PITCH_GAIN * thk, bladder_volume(zk, thk))

    frames = list(range(n)) + [n - 1] * int(fps * 1.5)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2500,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    FuncAnimation(fig, frame, frames=frames).save(path or FIGURES / "supplementary_movie_1.mp4", writer=writer)
    plt.close(fig)


def s03():
    U, n = 2.0, 320
    rAs = np.linspace(-9, -0.3, n)
    rLs = np.linspace(-9, -0.05, n)
    stable = np.array([[max_real_eig(U, a, b) < 0 for a in rAs] for b in rLs], float)
    fig, ax = plt.subplots(figsize=(0.8 * TW, 0.66 * TW))
    ax.contourf(rAs, rLs, stable, levels=[-0.5, 0.5, 1.5], colors=["#f6e3cc", "#cde4f5"])
    ra = np.linspace(-9, -0.3, 500)
    ax.plot(ra, [rL_on_B0(r) for r in ra], color=C2, lw=1.8)
    rl4 = ra + P.kth / (P.K * U**2)
    ok = rl4 < -0.05
    ax.plot(ra[ok], rl4[ok], color=C1, lw=1.8)
    ax.set_xlim(-9, -0.3)
    ax.set_ylim(-9, -0.05)
    ax.set_xlabel("плечо пузыря $r_A$, м")
    ax.set_ylabel("плечо плавников $r_L$, м")
    ax.grid(False)
    fig.legend(handles=[Patch(fc="#cde4f5", label="равновесие устойчиво"),
                        Patch(fc="#f6e3cc", label="равновесие неустойчиво"),
                        Line2D([], [], color=C1, lw=1.8, label="$a_4=0$: свободный член характеристического многочлена"),
                        Line2D([], [], color=C2, lw=1.8, label=r"$B=0$: старший минор Гурвица $\Delta_3$")],
               loc="lower center", ncol=1, handlelength=1.6)
    comma(ax)
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    save(fig, "s03_wedge")


def panel_label(ax, text, x=-0.06, y=1.05):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, weight="bold")


def s04a():
    rL = RLB - 0.3
    u1 = U1(RA, rL)
    Us = np.linspace(1.3, 2.4, 600)
    th = np.array([theta_star(u, RA, rL) or np.nan for u in Us]) * DEG
    fig, ax = plt.subplots(figsize=(0.49 * TW, 0.6 * TW))
    ax.plot([u1, 2.4], [0, 0], color="k", lw=2.0, label=r"$\theta=0$ устойчиво")
    ax.plot([1.3, u1], [0, 0], color="k", lw=1.4, ls="--", label=r"$\theta=0$ неустойчиво")
    ax.plot(Us, th, color=C1, lw=1.5, ls="--", label=r"сёдла $\pm\theta^*$")
    ax.plot(Us, -th, color=C1, lw=1.5, ls="--")
    ax.set_xlim(1.3, 2.4)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("скорость $U$, м с$^{-1}$")
    ax.set_ylabel("равновесный тангаж, град")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1, handlelength=1.8)
    panel_label(ax, "a")
    comma(ax)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    save(fig, "s04a_pitchfork")


def s04b():
    rL, U = -2.75, 1.70
    xs = saddle(U, RA, rL, theta_star(U, RA, rL))

    def far(t, x, *args):
        return 30.0 - abs(x[0])
    far.terminal = True

    fig, ax = plt.subplots(figsize=(0.49 * TW, 0.6 * TW))
    for s, col in [(0.6, GOOD), (0.9, GOOD), (0.99, GOOD), (1.01, BAD), (1.1, BAD), (1.4, BAD)]:
        r = solve(s * xs, 200, U, rL, t_eval=np.linspace(0, 200, 4000), events=far)
        ax.plot(r.t, r.y[0], color=col, lw=1.3)
    ax.axhline(xs[0], color=INK2, ls=":", lw=1.0)
    ax.set_xlim(0, 200)
    ax.set_ylim(-3, 0.6)
    ax.set_xlabel("время $t$, с")
    ax.set_ylabel("глубина $z$, м")
    fig.legend(handles=[Line2D([], [], color=GOOD, label="$s<1$: возврат"),
                        Line2D([], [], color=BAD, label="$s>1$: выход из окрестности"),
                        Line2D([], [], color=INK2, ls=":", label="глубина седла $z^*$")],
               loc="lower center")
    panel_label(ax, "b")
    comma(ax)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    save(fig, "s04b_basin")


def s05():
    br = np.load(DATA / "branch_U2.npy")
    d, A, h = br[:, 0], br[:, 2] * DEG, br[:, 5]
    k = int(np.argmax(h >= 1)) if (h >= 1).any() else len(d)
    fig, ax = plt.subplots(figsize=(0.8 * TW, 0.52 * TW))
    ax.plot([-0.15, 0], [0, 0], color="k", lw=2.0, label="равновесие устойчиво")
    ax.plot([0, 1.2], [0, 0], color="k", lw=1.4, ls="--", label="равновесие неустойчиво")
    ax.plot(np.r_[0, d[:k]], np.r_[0, A[:k]], color=C1, lw=2.0, label="предельный цикл устойчив")
    ax.plot(d[k - 1:], A[k - 1:], color=C1, lw=1.4, ls="--", label="предельный цикл неустойчив")
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-2, 65)
    ax.set_xlabel("сдвиг плавников $d$, м")
    ax.set_ylabel(r"амплитуда $\theta$, град")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, columnspacing=1.0,
               handlelength=1.6)
    comma(ax)
    fig.tight_layout(rect=(0.01, 0.2, 0.99, 0.99))
    save(fig, "s05_hopf")


WINDOW = (1.5375, 1.7225, -0.065, 0.245)


def map_grid():
    R = np.load(DATA / "regime_map.npy")[:, :3]
    Us, Ds = np.unique(R[:, 0]), np.unique(R[:, 1])
    M = np.full((len(Ds), len(Us)), np.nan)
    M[np.searchsorted(Ds, R[:, 1]), np.searchsorted(Us, R[:, 0])] = R[:, 2]
    return Us, Ds, M


def numerical_lines():
    near = np.load(DATA / "sb_corner.npy")[:4]
    sb = np.load(DATA / "sb_line.npy")
    sb = sb[np.isfinite(sb[:, 1])]
    U = np.r_[U0, near[:, 0], sb[:, 0]]
    d = np.r_[0.0, near[:, 2], sb[:, 1]]
    o = np.argsort(U)
    tr = np.load(DATA / "tr_line.npy")
    tr = tr[np.isfinite(tr[:, 1])]
    return (U[o], d[o]), (np.r_[U0, tr[:, 1]], np.r_[0.0, tr[:, 0]])


def s06():
    Us, Ds, M = map_grid()
    fig = plt.figure(figsize=(TW, 0.56 * TW))
    ax = fig.add_axes([0.13, 0.13, 0.49, 0.83])
    ax.pcolormesh(Us, Ds, M, cmap=ListedColormap(MAP_COLORS), norm=BoundaryNorm(np.arange(-0.5, 7.5), 7),
                  shading="nearest", rasterized=True)
    dd = np.linspace(WINDOW[2], WINDOW[3], 300)
    ax.plot([U1(RA, RLB + d) for d in dd], dd, color=INK, lw=1.8, zorder=4)
    ax.axhline(0, color=INK, lw=1.0, ls=(0, (1, 1.2)), zorder=4)

    for (u, d), col in zip(numerical_lines(), (SB_COLOR, TR_COLOR)):
        ax.plot(u, d, color="white", lw=3.4, zorder=5)
        ax.plot(u, d, color=col, lw=1.6, zorder=6)

    nf = np.load(DATA / "nf_lines.npy")
    nf = nf[np.all(np.isfinite(nf), axis=1) & (nf[:, 0] <= 0.06)]
    for j, col in ((2, SB_COLOR), (3, TR_COLOR)):
        ax.plot(np.r_[U0, nf[:, j]], np.r_[0.0, nf[:, 0]], color=col, lw=0.9, ls=(0, (2.5, 1.5)), zorder=7)

    ax.plot(U0, 0, "*", ms=11, color="#eda100", mec=INK, mew=0.8, zorder=8)
    ax.set_xlim(WINDOW[0], WINDOW[1])
    ax.set_ylim(WINDOW[2], WINDOW[3])
    ax.set_xlabel("скорость $U$, м с$^{-1}$")
    ax.set_ylabel("сдвиг плавников $d$, м")
    ax.grid(False)
    comma(ax)

    dashed = (0, (2.5, 1.5))
    handles = [Patch(fc=c, label=l) for c, l in zip(MAP_COLORS, MAP_LABELS)] + [
        Line2D([], [], color=INK, lw=1.8, label="$a_4=0$"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 1.2)), label="$B=0$"),
        Line2D([], [], color=SB_COLOR, lw=1.6, label="SB (численно)"),
        Line2D([], [], color=TR_COLOR, lw=1.6, label="TR (численно)"),
        Line2D([], [], color=SB_COLOR, lw=0.9, ls=dashed, label="SB (нормальная форма)"),
        Line2D([], [], color=TR_COLOR, lw=0.9, ls=dashed, label="TR (нормальная форма)"),
        Line2D([], [], ls="none", marker="*", ms=9, color="#eda100", mec=INK, mew=0.7,
               label="точка Хопф-вилки")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.645, 0.55), fontsize=9.5,
               handlelength=1.6, labelspacing=0.45)
    save(fig, "s06_diagram")


def s07():
    U, cases = 1.6, (0.03, 0.10, 0.150, 0.157, 0.20)
    fig, axs = plt.subplots(1, len(cases), figsize=(TW, 0.27 * TW), sharey=True)
    x = np.array([0.0, 0.0, 0.05, 0.0])
    for i, (ax, d) in enumerate(zip(axs, cases)):
        for dd in np.linspace(0.005, d, max(2, int(d / 0.005))):
            x = solve(x, 300, U, RLB + dd).y[:, -1]
        s = solve(x, 900, U, RLB + d, rtol=1e-10, atol=1e-12, t_eval=np.linspace(300, 900, 24000))
        ax.plot(s.y[2] * DEG, s.y[3] * DEG, color=C1, lw=0.35)
        if i > 0:
            ax.plot(-s.y[2] * DEG, -s.y[3] * DEG, color=C2, lw=0.35)
        ax.set_xlabel(r"$\theta$, град")
        ax.text(0.03, 0.97, "abcde"[i], transform=ax.transAxes, va="top", fontsize=10.5, weight="bold")
        comma(ax)
        x = s.y[:, -1]
    axs[0].set_ylabel(r"$\dot\theta$, град/с")
    fig.tight_layout(w_pad=0.6)
    save(fig, "s07_attractors")


def sweep_figure(name, job, xlim, xlabel, labels, mark_escape=False):
    z = np.load(DATA / f"sweep_{job}.npz")
    rows, pmax, pmin = z["rows"], z["pmax"], z["pmin"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(0.49 * TW, 0.52 * TW), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    if mark_escape:
        for u in rows[rows[:, 1] == 1, 0]:
            a1.axvspan(u - 0.001, u + 0.001, color="#bdbdbd", lw=0, zorder=0.5)
    a1.scatter(pmax[:, 0], pmax[:, 1] * DEG, s=0.5, c=C1, alpha=0.8, lw=0, rasterized=True)
    a1.scatter(pmin[:, 0], -pmin[:, 1] * DEG, s=0.5, c=C1, alpha=0.8, lw=0, rasterized=True)
    a1.set_ylabel(r"$\theta_{\max}$, град")
    ok = np.isfinite(rows[:, 7])
    a2.plot(rows[ok, 0], rows[ok, 7], color="#2b2b29", lw=0.8)
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_ylabel(r"$\lambda_1$, с$^{-1}$")
    a2.set_xlabel(xlabel)
    a2.set_xlim(*xlim)
    for ax, lab in zip((a1, a2), labels):
        comma(ax)
        ax.text(-0.3, 1.0, lab, transform=ax.transAxes, fontsize=11, weight="bold", va="bottom")
    fig.align_ylabels((a1, a2))
    fig.tight_layout(h_pad=0.4)
    save(fig, name)


def s07a():
    sweep_figure("s07a_sweep_d", "d_up", (0, 0.4), "сдвиг плавников $d$, м", ("a", "c"))


def s07b():
    sweep_figure("s07b_sweep_U", "U_down", (1.5, 2.6), "скорость $U$, м с$^{-1}$", ("b", "d"),
                 mark_escape=True)


ALL = ["s00", "s01", "s03", "s04a", "s04b", "s05", "s06", "s07", "s07a", "s07b"]

if __name__ == "__main__":
    style()
    for name in sys.argv[1:] or ALL:
        globals()[name]()
