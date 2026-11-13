"""Figures of the Nature version (main text and Extended Data), English and Russian.

Copy of scripts/figures.py of the repository swim-bladder-bifurcations (07 Код для GitHub), changed so that
every panel is drawn into axes passed by the caller and each multi-panel figure is assembled into ONE PDF
with a common style: DejaVu Sans 7-8 pt, lowercase bold 9 pt panel letters in the top-left corner outside
the axes. The repository itself is not changed; its package `bladder` and its data are used read-only.

    python figures_src/figures.py                 # all figures, English -> figures/
    python figures_src/figures.py ru              # all figures, Russian -> ../nature_ru_fish/figures/*_ru.pdf
    python figures_src/figures.py en fig3 edfig2  # selected ones

The repository is found at ../../../../07 Код для GitHub/swim-bladder-bifurcations relative to nature_fish/
or at $BLADDER_REPO. Fig. 5 (figures/s08_compliance.pdf) is made by "11 Два пузыря/fig5_nature_fish.py" (since
10.11.2026; the earlier fig5_nature.py is kept) and is not redrawn here. Since 13.11.2026 Fig. 1a is a micro-CT
zebrafish (data/fig1_ct, made by scripts/fish_render/ct_zebrafish.py from Hur et al. 2017, CC0) and Figs. 1b,c
use the vector zebrafish of zebrafish_draw.py; the functions s00 and s01 (BarramundiFish renders) are kept but
no longer used. The same file is kept in the repository as scripts/nature_figures/figures.py; there the output
goes to figures/nature/.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Patch, Wedge
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent            # nature_fish/figures_src
NATURE = HERE.parent                              # nature_fish
IN_REPO = (HERE.parents[1] / "bladder").is_dir()     # copy in scripts/nature_figures of the repository
REPO = Path(os.environ.get("BLADDER_REPO", HERE.parents[1] if IN_REPO else
                           NATURE.parents[3] / "07 Код для GitHub" / "swim-bladder-bifurcations"))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "fish_render"), str(HERE)]

from bladder import P, U1, max_real_eig, rhs, rL_on_B0  # noqa: E402
from bladder.model import saddle, theta_star  # noqa: E402

DATA = REPO / "data"
RA = -4.6
RLB = rL_on_B0(RA, P)
U0 = U1(RA, RLB, P)

DEG = 180 / np.pi
CM = 1 / 2.54
W2, W1 = 18.0, 8.9                                # Nature double and single column, cm
INK, INK2 = "#111111", "#555555"
C1, C2 = "#2a78d6", "#eb6834"
GOOD, BAD = C1, C2                                # one palette in all figures (10.11-b)
SB_COLOR, TR_COLOR = "#123f7a", "#0b5a3b"
MAP_COLORS = ["#f2f1ec", "#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7", "#8a8983"]

LANG = "en"
TXT = {
    "speed": ("Speed $U$ (m s$^{-1}$)", "скорость $U$, м с$^{-1}$"),
    "shift": ("Fin shift $d$ (m)", "сдвиг плавников $d$, м"),
    "eq_pitch": ("Equilibrium pitch (deg)", "равновесный тангаж, град"),
    "th0_st": (r"$\theta=0$ stable", r"$\theta=0$ устойчиво"),
    "th0_un": (r"$\theta=0$ unstable", r"$\theta=0$ неустойчиво"),
    "saddles": (r"saddles $\pm\theta^*$", r"сёдла $\pm\theta^*$"),
    "time": ("Time $t$ (s)", "время $t$, с"),
    "depth": ("Vertical position $z$ (m)", "положение по вертикали $z$, м"),
    "ret": ("$s<1$: return", "$s<1$: возврат"),
    "esc": ("$s>1$: escape", "$s>1$: выход из окрестности"),
    "zsad": ("saddle position $z^*$", "положение седла $z^*$"),
    "amp": (r"$\theta$ amplitude (deg)", r"амплитуда $\theta$, град"),
    "eq_st": ("stable equilibrium", "равновесие устойчиво"),
    "eq_un": ("unstable equilibrium", "равновесие неустойчиво"),
    "lc_st": ("stable limit cycle", "предельный цикл устойчив"),
    "lc_un": ("unstable limit cycle", "предельный цикл неустойчив"),
    "wx": ("Bladder offset $r_A/\\ell$", "плечо пузыря $r_A/\\ell$"),
    "wy": ("Lift-centre offset $r_L/\\ell$", "плечо подъёмной силы $r_L/\\ell$"),
    "a4": ("$a_4=0$, shaded speed", "$a_4=0$ при скорости заливки"),
    "a4f": ("$a_4=0$, higher speeds", "$a_4=0$ при больших скоростях"),
    "th_deg": (r"$\theta$ (deg)", r"$\theta$, град"),
    "om_deg": (r"$\dot\theta$ (deg s$^{-1}$)", r"$\dot\theta$, град с$^{-1}$"),
    "thmax": (r"$\theta_{\max}$ (deg)", r"$\theta_{\max}$, град"),
    "lyap": (r"$\lambda_1$ (s$^{-1}$)", r"$\lambda_1$, с$^{-1}$"),
    "sbn": ("SB (numerical)", "SB (численно)"),
    "trn": ("TR (numerical)", "TR (численно)"),
    "sbf": ("SB (normal form)", "SB (нормальная форма)"),
    "trf": ("TR (normal form)", "TR (нормальная форма)"),
    "hp": ("Hopf–pitchfork point", "точка Хопфа-вилки"),
    "veh": ("vehicle-like", "аппарат"),
    "fish": ("fish-like", "рыбоподобный набор"),
}
MAP_LABELS = {
    "en": ["equilibrium", "symmetric cycle", "asymmetric cycle", "higher-period cycle", "not settled",
           "chaotic attractor", "no attractor"],
    "ru": ["равновесие", "симметричный цикл", "несимметричный цикл", "цикл кратного периода", "не установилось",
           "хаотический аттрактор", "нет аттрактора"],
}


def t(key):
    return TXT[key][0 if LANG == "en" else 1]


def sans_font():
    """Arial if installed, otherwise a metrically compatible Liberation Sans or Nimbus Sans (10.11-b)."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans", "Nimbus Sans", "Nimbus Sans L", "Helvetica"):
        if name in names:
            return name
    return "DejaVu Sans"


def style():
    font = sans_font()
    plt.rcParams.update({
        "font.family": font, "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": f"{font}:italic", "mathtext.bf": f"{font}:bold", "mathtext.sf": font,
        "font.size": 6.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "axes.grid": True,
        "grid.color": "#e4e4e4", "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.1, "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35,
        "hatch.linewidth": 0.4, "pdf.fonttype": 42,
    })


def fmt(v, _=None):
    s = "%g" % v
    if LANG == "ru":
        s = s.replace(".", ",")
    return s.replace("-", "−")


def plain_ticks(ax):
    f = matplotlib.ticker.FuncFormatter(fmt)
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


# ---------- layout helpers (all in cm, origin at the top-left corner of the figure) ----------
def new_fig(w, h):
    fig = plt.figure(figsize=(w * CM, h * CM))
    fig._cm = (w, h)
    return fig


def add_ax(fig, x, y, w, h, **kw):
    W, H = fig._cm
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def letter(fig, x, y, s):
    """Panel letter with its top-left corner at (x, y) cm from the top-left corner of the figure."""
    W, H = fig._cm
    fig.text(x / W, 1 - y / H, s, fontsize=8, weight="bold", ha="left", va="top")


def save(fig, name, dpi=None):
    if IN_REPO:                                    # run from the repository: figures/nature/
        out = REPO / "figures" / "nature" / (f"{name}.pdf" if LANG == "en" else f"{name}_ru.pdf")
    elif LANG == "en":
        out = NATURE / "figures" / f"{name}.pdf"
    else:
        out = NATURE.parent / "nature_ru_fish" / "figures" / f"{name}_ru.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, **({"dpi": dpi} if dpi else {}))   # dpi: resolution of the embedded fish renders
    plt.close(fig)
    print(out)


def solve(x0, T, U, rL, **kw):
    return solve_ivp(rhs, (0, T), x0, args=(U, RA, rL, P), method="DOP853",
                     rtol=kw.pop("rtol", 1e-9), atol=kw.pop("atol", 1e-11), **kw)


# ---------- drawings of the fish ----------
# Since 10.11.2026 the fish is a side-view render of the BarramundiFish model (CC0, Khronos glTF Sample Assets)
# made by figures_src/render_fish_side.py (Blender) and post-processed by figures_src/fishart.py: the swim
# bladder is shown in a cut-away window, the pelvic fin is tinted orange. Geometry in body lengths L from the
# centre of mass: bladder at r_A = -0.125 L (fish-like set, rho_A = -0.5, l = 0.25 L), lift centre at
# r_L = -0.053 L (goldfish of Table 1, as in Fig. 5). The renders are in figures_src/fish_renders/.
from fishart import RENDERS, Render  # noqa: E402

RA_FISH, RL_FISH = -0.125, -0.053
BL_Z = 0.06                                       # bladder raised towards the spine (10.11-b)


def halo(lw=1.8):
    return [pe.withStroke(linewidth=lw, foreground="white", alpha=0.85)]


def s00(ax, fs=7.0):
    """Model on the fish (panel a of Fig. 1)."""
    r = Render(RENDERS / "fig1a", "a")
    th, ga = r.pitch, np.radians(7)
    Ld = 10.0
    ax.set_axis_off()
    ax.set_aspect("equal")
    r.draw(ax, 0, 0, Ld, zorder=2, stiff=())
    B = lambda px, pz: r.body(0, 0, Ld, px / Ld, pz / Ld)  # noqa: E731  (body frame in drawing units)
    xa, xl = RA_FISH * Ld, RL_FISH * Ld
    yb = r.belly(RL_FISH) * Ld                    # ventral contour at the lift centre (body frame)
    r.line(ax, (-6.6, 0), (9.2, 0), 0, 0, Ld, INK2, 0.6, (0, (4, 3)), 4)                  # horizontal
    r.line(ax, B(-6.3, 0), B(8.6, 0), 0, 0, Ld, INK2, 0.6, (0, (6, 2, 1, 2)), 4)          # body axis
    r.line(ax, (0, 0), (9.0 * np.cos(ga), 9.0 * np.sin(ga)), 0, 0, Ld, INK, 0.7, (0, (5, 2)), 6)  # velocity
    arrow = dict(arrowstyle="-|>", mutation_scale=9, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(11.2, -1.1), xytext=(8.8, -1.1), zorder=6, arrowprops=dict(arrow, color=INK, lw=1.1))
    ax.text(11.35, -1.1, "$U$", fontsize=fs, va="center")
    for rr, t1, t2, lab, col in ((6.4, 0, ga, r"$\gamma$", INK), (7.6, ga, th, r"$\alpha$", C2),
                                 (5.1, 0, th, r"$\theta$", INK)):
        ax.add_patch(Arc((0, 0), 2 * rr, 2 * rr, theta1=np.degrees(t1), theta2=np.degrees(t2),
                         color=col, lw=0.8, zorder=6))
        m = (t1 + t2) / 2
        ax.text((rr + 0.3) * np.cos(m), (rr + 0.3) * np.sin(m), lab, color=col, fontsize=fs,
                va="center", ha="left", zorder=7)
    ax.plot(0, 0, "o", color=INK, mec="white", mew=0.7, ms=4.2, zorder=8)

    pa = B(xa, BL_Z * Ld)
    ax.plot(*pa, "o", color=C1, mec="white", mew=0.5, ms=2.6, zorder=8)
    ax.annotate("", xy=(pa[0], pa[1] + 2.1), xytext=pa, zorder=7,
                arrowprops=dict(arrow, color=C1, lw=1.2, path_effects=halo(2.6)))
    ax.text(pa[0] - 0.2, pa[1] + 2.1, "$F_A$", color=C1, fontsize=fs, ha="right", va="top")
    pf = B(xl, yb)                                # centre of lift on the ventral contour; F_L pushes it up
    tail = (pf[0] + 1.4 * np.sin(ga), pf[1] - 1.4 * np.cos(ga))
    ax.annotate("", xy=pf, xytext=tail, zorder=7,
                arrowprops=dict(arrow, color=C2, lw=1.2, shrinkB=2.2, path_effects=halo(2.6)))
    ax.plot(*pf, "o", mfc="white", mec=C2, mew=1.0, ms=3.6, zorder=9)               # centre of lift
    ax.text(tail[0] - 0.15, tail[1] + 0.35, "$F_L$", color=C2, fontsize=fs, ha="right", va="center")

    # offsets: two separate levels below the body; extension lines from the CM and the points of action
    for xe, off, lab, col, top in ((xl, -3.0, r"$r_L$", C2, yb), (xa, -3.75, r"$r_A$", C1, BL_Z * Ld)):
        ax.annotate("", xy=B(xe, off), xytext=B(0, off),
                    arrowprops=dict(arrowstyle="<|-|>", color=col, lw=0.8 if xe == xa else 0.6, shrinkA=0,
                                    shrinkB=0, mutation_scale=6 if xe == xa else 4))
        for xx, t0 in ((0, 0), (xe, top)):
            q0, q1 = B(xx, t0), B(xx, off - 0.2)
            r.line(ax, q0, q1, 0, 0, Ld, col, 0.6, (0, (1.5, 1.5)), 5, inner_alpha=0.9)
        m = B(0.3, off)
        ax.text(m[0], m[1], lab, color=col, fontsize=fs, ha="left", va="center")

    ax.annotate("", xy=(-6.6, -2.4), xytext=(-6.6, -3.9),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7))
    ax.text(-6.4, -2.55, "$z$", fontsize=fs)
    ax.set_xlim(*S00_LIMITS[:2])
    ax.set_ylim(*S00_LIMITS[2:])


S00_LIMITS = (-7.0, 11.9, -4.35, 3.05)


# Fish drawn along a computed trajectory (U = 2, d = -0.15, z0 = -0.3).
# Pitch and bladder volume are exaggerated so that the phases are visible.
PITCH_GAIN = 2.2
KX, KZ = 0.95, 3.4
S01_LIMITS = (-1.45, KX * 16.4 + 1.0, -1.8, 1.2)
S01_TIMES = (0.0, 2.6, 5.2, 8.4, 11.8, 15.4)
S01_L = 1.6


def bladder_volume(z, th):
    return float(np.clip(1 + 0.6 * (z + RA * np.sin(th)), 0.4, 1.45))


def mechanism_solution(T):
    return solve([-0.3, 0, 0, 0], T, 2.0, RLB - 0.15, dense_output=True).sol


def s01_states():
    """Pitch (deg, exaggerated) and bladder scale of the drawn positions; written for make_renders.py."""
    sol = mechanism_solution(16.4)
    out = []
    for tk in S01_TIMES:
        zk, _, thk, _ = sol(tk)
        out.append([round(float(np.degrees(PITCH_GAIN * thk)), 3), round(bladder_volume(zk, thk), 4)])
    return out


def s01(ax):
    """Successive positions of the body (formerly s01_mechanism.pdf)."""
    T = 16.4
    sol = mechanism_solution(T)
    tt = np.linspace(0, S01_TIMES[-1], 800)        # path ends at the centre of mass of the last fish
    z = sol(tt)[0]
    x0, x1, y0, y1 = S01_LIMITS
    states = json.loads((RENDERS / "fig1b_states.json").read_text())
    assert np.allclose(states, s01_states(), atol=1e-3), "fish renders are out of date: run make_renders.py fig1b"
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.plot([x0, x1], [0, 0], color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.plot(KX * tt, KZ * z, color="#9aa3ad", lw=0.8, zorder=1)
    for i, tk in enumerate(S01_TIMES):
        zk = sol(tk)[0]
        r = Render(RENDERS / "fig1b", f"b{i}")
        r.draw(ax, KX * tk, KZ * zk, S01_L, lw=0.35, zorder=3 + i)
        ax.plot(KX * tk, KZ * zk, "o", color=INK, mec="white", mew=0.4, ms=2.0, zorder=10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


# ---------- stability region ----------
# (kappa, beta, shading speed, faster speeds, r_A range, r_L range); speed in units of sqrt(k_theta/m)
WEDGE_PANELS = (
    (0.1, 1.0, 2 / np.sqrt(5), (1.5, 3.0), (-9.0, -0.3), (-9.0, -0.05)),
    (0.002, 1.0, 2.0, (4.0, 10.0), (-1.1, 0.02), (-1.1, 0.05)),
)


def wedge_panel(ax, kappa, beta, U, faster, xr, yr, n=300):
    from bladder import Params
    p = Params(K=beta, kb=kappa, kth=1.0)
    rA, rL = np.linspace(*xr, n), np.linspace(*yr, n)
    stable = np.array([[max_real_eig(U, a, b, p) < 0 for a in rA] for b in rL], float)
    ax.contourf(rA, rL, stable, levels=[-0.5, 0.5, 1.5], colors=["#f6e3cc", "#cde4f5"])
    ra = np.linspace(*xr, 600)
    ax.plot(ra, [rL_on_B0(r, p) for r in ra], color=C2, lw=1.3)
    for u, ls, lw in [(U, "-", 1.3)] + [(v, (0, (4, 2)), 0.8) for v in faster]:
        line = ra + p.kth / (p.K * u**2)
        ok = line < yr[1]
        ax.plot(ra[ok], line[ok], color=C1, lw=lw, ls=ls)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.grid(False)
    plain_ticks(ax)


def wedge_handles():
    return [Patch(fc="#cde4f5", label=t("eq_st")), Patch(fc="#f6e3cc", label=t("eq_un")),
            Line2D([], [], color=C1, lw=1.3, label=t("a4")),
            Line2D([], [], color=C1, lw=0.8, ls=(0, (4, 2)), label=t("a4f")),
            Line2D([], [], color=C2, lw=1.3, label="$B=0$")]


# ---------- loss of stability ----------
def s04a(ax):
    rL = RLB - 0.3
    u1 = U1(RA, rL)
    Us = np.linspace(1.3, 2.4, 600)
    th = np.array([theta_star(u, RA, rL) or np.nan for u in Us]) * DEG
    ax.plot([u1, 2.4], [0, 0], color="k", lw=1.5, label=t("th0_st"))
    ax.plot([1.3, u1], [0, 0], color="k", lw=1.0, ls="--", label=t("th0_un"))
    ax.plot(Us, th, color=C1, lw=1.1, ls="--", label=t("saddles"))
    ax.plot(Us, -th, color=C1, lw=1.1, ls="--")
    ax.set_xlim(1.3, 2.4)
    ax.set_ylim(-60, 60)
    ax.set_xlabel(t("speed"))
    ax.set_ylabel(t("eq_pitch"))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03, -0.2))
    plain_ticks(ax)


def s04b(ax):
    rL, U = -2.75, 1.70
    xs = saddle(U, RA, rL, theta_star(U, RA, rL))

    def far(tt, x, *args):
        return 30.0 - abs(x[0])
    far.terminal = True

    for s, col in [(0.6, GOOD), (0.9, GOOD), (0.99, GOOD), (1.01, BAD), (1.1, BAD), (1.4, BAD)]:
        r = solve(s * xs, 200, U, rL, t_eval=np.linspace(0, 200, 4000), events=far)
        ax.plot(r.t, r.y[0], color=col, lw=1.0)
    ax.axhline(xs[0], color=INK2, ls=":", lw=0.8)
    ax.set_xlim(0, 200)
    ax.set_ylim(-3, 0.6)
    ax.set_xlabel(t("time"))
    ax.set_ylabel(t("depth"))
    ax.legend(handles=[Line2D([], [], color=GOOD, label=t("ret")),
                       Line2D([], [], color=BAD, label=t("esc")),
                       Line2D([], [], color=INK2, ls=":", label=t("zsad"))],
              loc="upper left", bbox_to_anchor=(-0.03, -0.2))
    plain_ticks(ax)


def s05(ax):
    br = np.load(DATA / "branch_U2.npy")
    d, A, h = br[:, 0], br[:, 2] * DEG, br[:, 5]
    k = int(np.argmax(h >= 1)) if (h >= 1).any() else len(d)
    ax.plot([-0.15, 0], [0, 0], color="k", lw=1.5, label=t("eq_st"))
    ax.plot([0, 1.2], [0, 0], color="k", lw=1.0, ls="--", label=t("eq_un"))
    ax.plot(np.r_[0, d[:k]], np.r_[0, A[:k]], color=C1, lw=1.5, label=t("lc_st"))
    ax.plot(d[k - 1:], A[k - 1:], color=C1, lw=1.0, ls="--", label=t("lc_un"))
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-2, 65)
    ax.set_xlabel(t("shift"))
    ax.set_ylabel(t("amp"))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03, -0.2), handlelength=1.8)
    plain_ticks(ax)


# ---------- bifurcation diagram ----------
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


def s06(ax):
    Us, Ds, M = map_grid()
    ax.pcolormesh(Us, Ds, M, cmap=ListedColormap(MAP_COLORS), norm=BoundaryNorm(np.arange(-0.5, 7.5), 7),
                  shading="nearest", rasterized=True)
    dd = np.linspace(WINDOW[2], WINDOW[3], 300)
    ax.plot([U1(RA, RLB + d) for d in dd], dd, color=INK, lw=1.3, zorder=4)
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (1, 1.2)), zorder=4)
    for (u, d), col in zip(numerical_lines(), (SB_COLOR, TR_COLOR)):
        ax.plot(u, d, color="white", lw=2.6, zorder=5)
        ax.plot(u, d, color=col, lw=1.2, zorder=6)
    nf = np.load(DATA / "nf_lines.npy")
    nf = nf[np.all(np.isfinite(nf), axis=1) & (nf[:, 0] <= 0.06)]
    for j, col in ((2, SB_COLOR), (3, TR_COLOR)):
        ax.plot(np.r_[U0, nf[:, j]], np.r_[0.0, nf[:, 0]], color=col, lw=0.7, ls=(0, (2.5, 1.5)), zorder=7)
    ax.plot(U0, 0, "*", ms=8, color="#eda100", mec=INK, mew=0.6, zorder=8)
    ax.set_xlim(WINDOW[0], WINDOW[1])
    ax.set_ylim(WINDOW[2], WINDOW[3])
    ax.set_xlabel(t("speed"))
    ax.set_ylabel(t("shift"))
    ax.grid(False)
    plain_ticks(ax)
    dashed = (0, (2.5, 1.5))
    return [Patch(fc=c, label=lab) for c, lab in zip(MAP_COLORS, MAP_LABELS[LANG])] + [
        Line2D([], [], color=INK, lw=1.3, label="$a_4=0$"),
        Line2D([], [], color=INK, lw=0.8, ls=(0, (1, 1.2)), label="$B=0$"),
        Line2D([], [], color=SB_COLOR, lw=1.2, label=t("sbn")),
        Line2D([], [], color=TR_COLOR, lw=1.2, label=t("trn")),
        Line2D([], [], color=SB_COLOR, lw=0.7, ls=dashed, label=t("sbf")),
        Line2D([], [], color=TR_COLOR, lw=0.7, ls=dashed, label=t("trf")),
        Line2D([], [], ls="none", marker="*", ms=7, color="#eda100", mec=INK, mew=0.6, label=t("hp"))]


# ---------- route to chaos ----------
S07_CASES = (0.03, 0.10, 0.150, 0.157, 0.20)
_S07 = {}


def s07_data():
    """Attractors at U = 1.6 for the shifts S07_CASES, reached by continuation in d (as in s07)."""
    if not _S07:
        U = 1.6
        x = np.array([0.0, 0.0, 0.05, 0.0])
        for d in S07_CASES:
            for dd in np.linspace(0.005, d, max(2, int(d / 0.005))):
                x = solve(x, 300, U, RLB + dd).y[:, -1]
            s = solve(x, 900, U, RLB + d, rtol=1e-10, atol=1e-12, t_eval=np.linspace(300, 900, 24000))
            _S07[d] = (s.y[2] * DEG, s.y[3] * DEG)
            x = s.y[:, -1]
    return _S07


def s07(axs):
    data = s07_data()
    for i, (ax, d) in enumerate(zip(axs, S07_CASES)):
        th, om = data[d]
        ax.plot(th, om, color=C1, lw=0.3)
        if i > 0:
            ax.plot(-th, -om, color=C2, lw=0.3)
        ax.set_xlabel(t("th_deg"))
        plain_ticks(ax)
    axs[0].set_ylabel(t("om_deg"))


def sweep_panels(a1, a2, job, xlim, xlabel, mark_escape=False, xlabel_top=None):
    z = np.load(DATA / f"sweep_{job}.npz")
    rows, pmax, pmin = z["rows"], z["pmax"], z["pmin"]
    if mark_escape:
        for u in rows[rows[:, 1] == 1, 0]:
            a1.axvspan(u - 0.001, u + 0.001, color="#bdbdbd", lw=0, zorder=0.5)
    a1.scatter(pmax[:, 0], pmax[:, 1] * DEG, s=0.3, c=C1, alpha=0.8, lw=0, rasterized=True)
    a1.scatter(pmin[:, 0], -pmin[:, 1] * DEG, s=0.3, c=C1, alpha=0.8, lw=0, rasterized=True)
    a1.set_ylabel(t("thmax"))
    ok = np.isfinite(rows[:, 7])
    a2.plot(rows[ok, 0], rows[ok, 7], color="#2b2b29", lw=0.6)
    a2.axhline(0, color=INK2, lw=0.5)
    a2.set_ylabel(t("lyap"))
    a2.set_xlabel(xlabel)
    if xlabel_top:
        a1.set_xlabel(xlabel_top)
    for ax in (a1, a2):
        ax.set_xlim(*xlim)
        plain_ticks(ax)


# ---------- assembled figures ----------
# ---------- Fig. 1 (since 13.11.2026): a, micro-CT zebrafish; b, c, drawn zebrafish ----------
CT_DATA = DATA / "fig1_ct" / "zebrafish_nr16.npz"     # made by scripts/fish_render/ct_zebrafish.py
import zebrafish_draw as zd  # noqa: E402

CT_LAB = {"dors": ("dorsal fin", "спинной\nплавник"), "anal": ("anal fin", "анальный\nплавник"),
       "caud": ("caudal fin", "хвостовой\nплавник"), "pelv": ("pelvic fin", "брюшной\nплавник"),
       "pect": ("pectoral fin", "грудной\nплавник"), "ant": ("anterior chamber", "передняя\nкамера"),
       "post": ("posterior chamber", "задняя\nкамера"), "spine": ("vertebral column", "позвоночник"),
       "cm": ("CM", "ЦМ"), "cb": ("CB", "ЦП"), "mm": ("5 mm", "5 мм")}


# ---------------- a: micro-CT zebrafish ----------------
def ct_panel(ax, lang):
    T = lambda k: CT_LAB[k][0 if lang == "en" else 1]  # noqa: E731
    d = np.load(CT_DATA); vx = float(d["vx"])
    P = d["proj"].T                              # rows: dorsoventral (down +), cols: snout -> tail
    sb = d["sb"].T.astype(float)
    pel = d["pelvic"].T.astype(float)
    ny, nx = P.shape
    # coordinates in mm: x from the snout to the right is the tail; y up (flip rows)
    ext = (0, nx * vx, -ny * vx, 0)
    img = 1 - np.clip(P / np.percentile(P[P > 0], 99.6), 0, 1) ** 0.8
    alpha = np.clip(P / np.percentile(P[P > 0], 5), 0, 1)
    ax.imshow(np.dstack([img, img, img, alpha]), extent=ext, interpolation="lanczos", zorder=1)
    X = (np.arange(nx) + 0.5) * vx; Y = -(np.arange(ny) + 0.5) * vx
    ax.contourf(X, Y, sb, [0.5, 1.5], colors=[C1], alpha=0.30, zorder=2)
    ax.contour(X, Y, sb, [0.5], colors=[C1], linewidths=0.7, zorder=2.1)
    ax.contourf(X, Y, pel, [0.5, 1.5], colors=[C2], alpha=0.55, zorder=2.2)
    ax.contour(X, Y, pel, [0.5], colors=[C2], linewidths=1.0, zorder=2.3)
    cm = d["cm"]; cb = d["cb"]
    xc, yc = float(cm[0]), -float(cm[1])
    r = 0.24
    ax.add_patch(Circle((xc, yc), r, fc="white", ec=INK, lw=0.6, zorder=6))
    for a0 in (0, 180):
        ax.add_patch(Wedge((xc, yc), r, a0, a0 + 90, fc=INK, ec="none", zorder=6.1))
    ycb = yc + 0.62                              # drawn slightly apart for visibility
    ax.add_patch(Circle((float(cb[0]), ycb), 0.19, fc="white", ec=C1, lw=0.8, zorder=6.2))
    ax.plot(float(cb[0]), ycb, "o", color=C1, ms=1.2, zorder=6.3)
    lead = dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=1.5, shrinkB=0.5)

    def lab(k, xy, xyt, ha="center", va="bottom", color=INK):
        ax.annotate(T(k), xy=xy, xytext=xyt, fontsize=6.5, ha=ha, va=va, color=color, arrowprops=lead, zorder=7,
                    linespacing=1.0)
    top, bot = 1.6, -9.2
    lab("ant", (8.6, -3.9), (4.6, top), color=C1)
    lab("cb", (float(cb[0]) + 0.1, ycb + 0.18), (9.4, top - 1.15), color=C1)
    lab("post", (14.0, -4.4), (13.4, top), color=C1)
    lab("dors", (17.6, -0.7), (19.6, top))
    lab("spine", (22.0, -2.95), (25.4, top))
    lab("caud", (28.0, -4.6), (27.4, bot), va="top")
    lab("pect", (7.0, -5.6), (4.2, bot), va="top")
    lab("cm", (xc, yc - r), (10.2, bot), va="top")
    lab("pelv", (13.6, -6.9), (14.4, bot), va="top", color=C2)
    lab("anal", (18.5, -6.3), (20.0, bot), va="top")
    ys = -12.3
    ax.plot([1.0, 6.0], [ys, ys], color=INK2, lw=0.8, solid_capstyle="butt")
    ax.text(3.5, ys - 0.3, T("mm"), fontsize=6.5, ha="center", va="top", color=INK2)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_xlim(0, 29.6); ax.set_ylim(-13.6, 3.6)


# ---------------- b: model on the drawn zebrafish ----------------
S00Z = (-9.8, 11.9, -4.35, 3.05)
def line(ax, p, q, col, lw, ls, z):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw, ls=ls, zorder=z)


def s00z(ax, fs=7.0):
    th = np.radians(17.0); ga = np.radians(7)
    Ld = 13.0
    ax.set_axis_off(); ax.set_aspect("equal")
    body, belly = zd.draw_fish_at(ax, 0, 0, Ld, np.degrees(th), stiff="both")
    B = lambda px, pz: body(px / Ld, pz / Ld)  # noqa: E731
    xa, xl = RA_FISH * Ld, RL_FISH * Ld
    ycm = float(zd.spine(zd.X_CM)) - 0.045
    yc = float(zd.spine(0.30)) - 0.040 - 0.006
    za = (yc - ycm) * Ld                          # bladder level relative to the CM
    yb = belly(RL_FISH) * Ld
    line(ax, (-9.4, 0), (9.2, 0), INK2, 0.6, (0, (4, 3)), 4)
    line(ax, B(-9.0, 0), B(8.6, 0), INK2, 0.6, (0, (6, 2, 1, 2)), 4.5)
    line(ax, (0, 0), (9.0 * np.cos(ga), 9.0 * np.sin(ga)), INK, 0.7, (0, (5, 2)), 6)
    arrow = dict(arrowstyle="-|>", mutation_scale=9, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(11.2, -1.1), xytext=(8.8, -1.1), zorder=6, arrowprops=dict(arrow, color=INK, lw=1.1))
    ax.text(11.35, -1.1, "$U$", fontsize=fs, va="center")
    for rr, t1, t2, lab, col in ((6.4, 0, ga, r"$\gamma$", INK), (7.6, ga, th, r"$\alpha$", C2),
                                 (5.1, 0, th, r"$\theta$", INK)):
        ax.add_patch(Arc((0, 0), 2 * rr, 2 * rr, theta1=np.degrees(t1), theta2=np.degrees(t2), color=col, lw=0.8,
                         zorder=6))
        m = (t1 + t2) / 2
        ax.text((rr + 0.3) * np.cos(m), (rr + 0.3) * np.sin(m), lab, color=col, fontsize=fs, va="center",
                ha="left", zorder=7)
    pa = B(xa, za)
    ax.plot(*pa, "o", color=C1, mec="white", mew=0.5, ms=2.6, zorder=8)
    ax.annotate("", xy=(pa[0], pa[1] + 2.1), xytext=pa, zorder=7,
                arrowprops=dict(arrow, color=C1, lw=1.2, path_effects=halo(2.6)))
    ax.text(pa[0] - 0.2, pa[1] + 2.1, "$F_A$", color=C1, fontsize=fs, ha="right", va="top")
    pf = B(xl, yb)
    tail = (pf[0] + 1.4 * np.sin(ga), pf[1] - 1.4 * np.cos(ga))
    ax.annotate("", xy=pf, xytext=tail, zorder=7,
                arrowprops=dict(arrow, color=C2, lw=1.2, shrinkB=2.2, path_effects=halo(2.6)))
    ax.plot(*pf, "o", mfc="white", mec=C2, mew=1.0, ms=3.6, zorder=9)
    ax.text(tail[0] - 0.15, tail[1] + 0.35, "$F_L$", color=C2, fontsize=fs, ha="right", va="center")
    for xe, off, lab, col, top in ((xl, -3.0, r"$r_L$", C2, yb), (xa, -3.75, r"$r_A$", C1, za)):
        ax.annotate("", xy=B(xe, off), xytext=B(0, off),
                    arrowprops=dict(arrowstyle="<|-|>", color=col, lw=0.8 if xe == xa else 0.6, shrinkA=0,
                                    shrinkB=0, mutation_scale=6 if xe == xa else 4))
        for xx, t0 in ((0, 0), (xe, top)):
            line(ax, B(xx, t0), B(xx, off - 0.2), col, 0.6, (0, (1.5, 1.5)), 7)
        m = B(0.3, off)
        ax.text(m[0], m[1], lab, color=col, fontsize=fs, ha="left", va="center")
    ax.plot(0, 0, "o", color=INK, mec="white", mew=0.7, ms=4.2, zorder=8)
    ax.annotate("", xy=(-9.4, -2.4), xytext=(-9.4, -3.9),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7))
    ax.text(-9.2, -2.55, "$z$", fontsize=fs)
    ax.set_xlim(*S00Z[:2]); ax.set_ylim(*S00Z[2:])


# ---------------- c: positions along the path ----------------
def s01z(ax):
    sol = mechanism_solution(16.4)
    tt = np.linspace(0, S01_TIMES[-1], 800)
    z = sol(tt)[0]
    x0, x1, y0, y1 = S01_LIMITS
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.plot([x0, x1], [0, 0], color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.plot(KX * tt, KZ * z, color="#9aa3ad", lw=0.8, zorder=1)
    for tk, (pitch, vol) in zip(S01_TIMES, s01_states()):
        zk = sol(tk)[0]
        k = float(np.sqrt(vol))
        n0 = len(ax.patches) + len(ax.lines)
        zd.draw_fish_at(ax, KX * tk, KZ * zk, S01_L * 1.15, pitch, stiff="both", scale=(k, k))
        ax.plot(KX * tk, KZ * zk, "o", color=INK, mec="white", mew=0.4, ms=2.0, zorder=10)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def fig1():
    """a, x-ray side view of a zebrafish from micro-CT (Hur et al. 2017, CC0); b, model on the drawn zebrafish;
    c, mechanism (positions along the computed path)."""
    lang = LANG
    W = W2
    wa_ = 7.9; ha_ = wa_ * (3.6 + 13.6) / 29.6
    x0, x1, y0, y1 = S00Z
    wb_ = 9.6; hb_ = wb_ * (y1 - y0) / (x1 - x0)
    X0, X1, Y0, Y1 = S01_LIMITS
    wc = 17.6; hc = wc * (Y1 - Y0) / (X1 - X0)
    top_c = 0.15 + max(ha_, hb_) + 0.25
    H = top_c + hc + 0.05
    fig = new_fig(W, H)
    ct_panel(add_ax(fig, 0.2, 0.15, wa_, ha_), lang)
    s00z(add_ax(fig, W - wb_ - 0.05, 0.15 + (ha_ - hb_) / 2, wb_, hb_))
    s01z(add_ax(fig, 0.4, top_c, wc, hc))
    letter(fig, 0.0, 0.05, "a"); letter(fig, W - wb_ - 0.35, 0.05, "b"); letter(fig, 0.0, top_c - 0.1, "c")
    save(fig, "fig1", dpi=600)


def fig2():
    W, H = W2, 6.6
    top, h, w = 0.5, 5.0, 4.9
    xa, xb = 1.25, 7.45
    fig = new_fig(W, H)
    axs = [add_ax(fig, xa, top, w, h), add_ax(fig, xb, top, w, h)]
    for ax, args in zip(axs, WEDGE_PANELS):
        wedge_panel(ax, *args)
        ax.set_xlabel(t("wx"))
    axs[0].set_ylabel(t("wy"))
    fig.legend(handles=wedge_handles(), loc="upper left", bbox_to_anchor=((xb + w + 0.45) / W, 1 - top / H),
               handlelength=2.0)
    letter(fig, 0.0, 0.0, "a")
    letter(fig, xb - 1.05, 0.0, "b")
    save(fig, "fig2")


def fig3():
    W, H = W2, 6.85
    top, h, w = 0.45, 4.1, 4.45
    xs = (1.15, 7.05, 12.95)
    fig = new_fig(W, H)
    axs = [add_ax(fig, x, top, w, h) for x in xs]
    s04a(axs[0])
    s04b(axs[1])
    s05(axs[2])
    for x, s in zip(xs, "abc"):
        letter(fig, x - 1.15, 0.0, s)
    save(fig, "fig3")


def fig4():
    W = W2
    top1, h1, w1, gap = 0.45, 2.75, 2.95, 0.38
    x1 = 1.25
    top2 = top1 + h1 + 0.85 + 0.45
    h2, w2 = 3.0, 7.0
    xf, xg = 1.25, 10.25
    H = top2 + h2 + 0.85
    fig = new_fig(W, H)
    row = []
    for i in range(5):
        ax = add_ax(fig, x1 + i * (w1 + gap), top1, w1, h1, sharey=row[0] if row else None)
        if i:
            plt.setp(ax.get_yticklabels(), visible=False)
        row.append(ax)
    s07(row)
    for i, s in enumerate("abcde"):
        letter(fig, (x1 - 1.25) if i == 0 else (x1 + i * (w1 + gap) - 0.3), 0.0, s)
    af = add_ax(fig, xf, top2, w2, h2)
    ag = add_ax(fig, xg, top2, w2, h2)
    sweep_panels(af, ag, "d_up", (0, 0.4), t("shift"), xlabel_top=t("shift"))
    letter(fig, 0.0, top2 - 0.45, "f")
    letter(fig, xg - 1.25, top2 - 0.45, "g")
    save(fig, "fig4")


def edfig1():
    W, H = W2, 7.4
    fig = new_fig(W, H)
    ax = add_ax(fig, 1.4, 0.3, 9.6, 6.2)
    handles = s06(ax)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(11.5 / W, 1 - 0.3 / H), handlelength=1.8,
               labelspacing=0.45)
    save(fig, "ed_fig1")


def edfig2():
    W = W1
    top, h1, h2, gap = 0.45, 3.6, 1.8, 0.55
    H = top + h1 + gap + h2 + 0.85
    fig = new_fig(W, H)
    a1 = add_ax(fig, 1.4, top, 7.2, h1)
    a2 = add_ax(fig, 1.4, top + h1 + gap, 7.2, h2, sharex=a1)
    plt.setp(a1.get_xticklabels(), visible=False)
    sweep_panels(a1, a2, "U_down", (1.5, 2.6), t("speed"), mark_escape=True)
    fig.align_ylabels((a1, a2))
    letter(fig, 0.0, 0.0, "a")
    letter(fig, 0.0, top + h1 + gap - 0.4, "b")
    save(fig, "ed_fig2")


ALL = ["fig1", "fig2", "fig3", "fig4", "edfig1", "edfig2"]

if __name__ == "__main__":
    args = sys.argv[1:]
    langs = [a for a in args if a in ("en", "ru")] or ["en"]
    names = [a for a in args if a not in ("en", "ru", "both")] or ALL
    if "both" in args:
        langs = ["en", "ru"]
    style()
    for LANG in langs:
        for name in names:
            globals()[name]()
