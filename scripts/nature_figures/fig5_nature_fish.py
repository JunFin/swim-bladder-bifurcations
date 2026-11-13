"""Fig. 5 of the Nature version, 10.11.2026: where the gas is compressed. Two panels (former a and c of
fig5_nature.py, which is kept unchanged): a, mechanism on a realistic fish (BarramundiFish render, CC0, see
nature_fish/figures_src/README.md); b, depth change after a fin step of -2 deg at U = 2 U0.
Data: fig5.json (fig5_nature_data.py, goldfish of Table 1). Style of nature_fish/figures_src/figures.py.
10.11-b: one palette (blue, orange #eb6834, dark grey), Arial-compatible font, stiff chamber hatched, centre of
lift marked, vertical axis "vertical displacement", fin step and new equilibrium marked on panel b.

    python fig5_nature_fish.py [en|ru|both] [--png DIR]
-> figures/nature/s08_compliance.pdf, figures/nature/s08_compliance_ru.pdf (repository copy of
"11 Два пузыря/fig5_nature_fish.py"; data data/two_chambers/fig5.json, renders of scripts/fish_render)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent                       # scripts/nature_figures of the repository
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "fish_render")]
from fishart import RENDERS, Render  # noqa: E402
from figures import style  # noqa: E402  (common style of the figures: Arial-compatible font, 6.5-7 pt)
from make_renders import FRONT, REAR  # noqa: E402  (chamber geometry of the renders)

CM = 1 / 2.54
INK, INK2 = "#111111", "#555555"
C1, C2 = "#2a78d6", "#eb6834"
CP, CI, CA = C1, "#52514e", C2                               # posterior, incompressible, anterior
RL = -0.053                                                  # lift centre, L from the centre of mass (fig5.json)
d = json.loads((REPO / "data" / "two_chambers" / "fig5.json").read_text())
assert abs(d["rL_L"] - RL) < 1e-3


def make(lang):
    ru = lang == "ru"
    T = (lambda en, r: r if ru else en)  # noqa: E731
    num = FuncFormatter(lambda v, _: (("%g" % v).replace(".", ",") if ru else "%g" % v).replace("-", "−"))
    W, H = 18.0, 6.45
    fig = plt.figure(figsize=(W * CM, H * CM))
    add = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])  # noqa: E731

    # ---------- a: mechanism ----------
    aw, y0, y1 = 7.9, -0.55, 5.75
    ax = add(0.15, 0.1, aw, y1 - y0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(0, aw)
    ax.set_ylim(y0, y1)
    ax.grid(False)
    Ld = 3.9
    arrow = dict(arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0)
    rows = (("post", 4.05, 1, T("Compliant at the rear", "Податливая задняя камера"), CP, T("returns", "возвращается"),
             [FRONT]),
            ("ant", 1.05, -1, T("Compliant in front", "Податливая передняя камера"), CA, T("sinks", "уходит глубже"),
             [REAR]))
    for name, yc, up, title, col, fate, stiff in rows:
        r = Render(RENDERS / "fig5", name)
        xc = 2.45
        yeq = yc + 1.25                                      # equilibrium depth: the body is below it
        ax.plot([0.05, aw - 0.05], [yeq, yeq], color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.text(0.05, yeq + 0.08, title, fontsize=7, color=col, va="bottom", ha="left")
        r.draw(ax, xc, yc, Ld, zorder=3, stiff=[s[:4] for s in stiff])
        ax.plot(xc, yc, "o", color=INK, mec="white", mew=0.5, ms=2.8, zorder=8)
        pf = r.body(xc, yc, Ld, RL, r.belly(RL))           # centre of lift on the ventral contour
        q = (pf[0], pf[1] - 0.65)
        if up > 0:
            ax.annotate("", xy=pf, xytext=q, zorder=7, arrowprops=dict(arrow, color=C2, lw=1.1, shrinkB=2.0),
                        annotation_clip=False)
        else:
            ax.annotate("", xy=q, xytext=pf, zorder=7, arrowprops=dict(arrow, color=C2, lw=1.1, shrinkA=2.0),
                        annotation_clip=False)
        ax.plot(*pf, "o", mfc="white", mec=C2, mew=0.9, ms=3.2, zorder=9)
        ax.text(pf[0] - 0.1, pf[1] - 0.42, "$F_L$", color=C2, fontsize=7, ha="right", va="center")
        # subsequent motion of the centre of mass
        p0 = (xc + 1.95, yc + up * 0.45)
        p1 = (xc + 3.15, yeq - 0.2) if up > 0 else (xc + 3.15, yc - 1.0)
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={-0.25 * up}", arrowstyle="-|>",
                                     mutation_scale=7, color=col, lw=0.9, zorder=6))
        ax.text(p1[0] + 0.1, p1[1], fate, fontsize=6.5, color=col, ha="left", va="center")
    ax.text(aw - 0.05, rows[0][1] + 1.25 + 0.08, T("equilibrium depth", "равновесная глубина"), fontsize=6,
            color=INK2, ha="right", va="bottom")

    # ---------- b: response to a fin step ----------
    bx = add(9.75, 0.3, 7.95, 5.25)
    t = np.array(d["t"])
    for k, col in (("posterior", CP), ("incompressible", CI), ("anterior", CA)):
        bx.plot(t, d["z_" + k], color=col)
    ya = -2.2
    bx.set_ylim(ya, 0.1)
    bx.set_xlim(0, 150)
    bx.set_yticks(np.arange(0, -2.01, -0.5))
    za = np.array(d["z_anterior"])
    i = int(np.argmax(za < ya))                              # the curve leaves the axes: small arrow
    bx.annotate("", xy=(t[i], ya), xytext=(t[i] - 2.0, ya + 0.17), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=0))
    bx.set_xlabel(T("Time $t$ (s)", "время $t$, с"))
    bx.set_ylabel(T(r"Vertical displacement $\Delta z$ (m)", r"смещение по вертикали $\Delta z$, м"))
    bx.xaxis.set_major_formatter(num)
    bx.yaxis.set_major_formatter(num)
    bx.text(148, 0.05, T("fin step −2° at $t=0$", "ступень плавника −2° при $t=0$"), fontsize=6.5,
            color=INK2, ha="right", va="top")
    bx.text(148, -0.29, T("compliant at the rear", "податливая задняя"), color=CP, fontsize=6.5, ha="right",
            va="bottom")
    bx.text(148, -0.31, T("approaches a new equilibrium", "к новому равновесию"), color=CP, fontsize=6.5,
            ha="right", va="top", style="italic")
    bx.text(148, -1.17, T("incompressible", "несжимаемый"), color=CI, fontsize=6.5, ha="right", va="top")
    bx.text(72, -1.75, T("compliant in front\n(zebrafish)", "податливая передняя\n(данио)"), color=CA,
            fontsize=6.5, ha="right", va="center")
    for lab, x, y in (("a", 0.0, 0.0), ("b", 8.55, 0.0)):
        fig.text(x / W, 1 - y / H, lab, fontsize=8, weight="bold", ha="left", va="top")
    out = REPO / "figures" / "nature" / ("s08_compliance_ru.pdf" if ru else "s08_compliance.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)                                # resolution of the embedded fish renders
    if PNG:
        fig.savefig(Path(PNG) / (out.stem + ".png"), dpi=300)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    PNG = args[args.index("--png") + 1] if "--png" in args else None
    langs = ["en", "ru"] if "both" in args else [a for a in args if a in ("en", "ru")] or ["en"]
    style()
    for lg in langs:
        make(lg)
