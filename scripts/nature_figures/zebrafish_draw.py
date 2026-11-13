"""Side view of an adult zebrafish (Danio rerio) with the two-chambered swim bladder, the ducts, the vertebral
column and the fins: panels b and c of Fig. 1 of the Nature version (since 13.11.2026; drawn with draw_fish_at; the pelvic fin is
clearly orange). Own vector drawing in Matplotlib.

Coordinates: x from the snout tip in total lengths TL (0 = snout, 1 = tip of the caudal fin), y up, also in TL.
Positions taken from the literature (and only these are meant as data):
  * anterior chamber 0.17-0.30 TL, posterior chamber 0.30-0.47 TL, connecting duct at the constriction
    between them: our estimate from the schematic of Fig. 1A of Stoyek, Smith & Croll 2011 (J Exp Biol 214:
    2962), digitization accuracy about 0.03 TL;
  * centre of mass at 0.33 TL, at the connecting duct, nearly coincident with the centre of buoyancy
    (Robertson et al. 2008, J Morphol 269: 666; pivot of Stoyek et al. 2011 at 0.33 TL);
  * the pneumatic duct joins only the posterior chamber to the oesophagus, opening in its anteroventral wall
    just below the connecting duct (Stoyek et al. 2011; Dumbarton et al. 2010; Mohajer 2011, carp).
Everything else (body outline, body depth, positions and shapes of the fins, oesophagus and gut) is drawn to
be plausible for an adult zebrafish but is not taken from measurements and carries no numbers on the figure.

    draw_anatomy(ax, lang="en")       # into existing axes, aspect equal, axis off
"""
import numpy as np
from matplotlib.patches import Circle, PathPatch, Polygon, Wedge
from matplotlib.path import Path
from scipy.interpolate import PchipInterpolator

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2 = "#111111", "#555555"
EDGE = "#3c4148"                      # body outline
FIN_FILL, FIN_EDGE = "#e3e6ea", "#6b737c"
LIFT_FILL = "#f6b48f"                 # pelvic fins: the lifting surface of the model (light orange); the pectoral
                                      # fins lie in front of the centre of mass and are not part of the model (grey)
CHAMBER_A, CHAMBER_P = "#bcd6f3", "#e7eef7"

# ---- data from the literature (see the docstring) ----
ANT = (0.17, 0.30)                    # anterior chamber, TL (Stoyek et al. 2011, Fig. 1A, our estimate)
POST = (0.30, 0.47)                   # posterior chamber, TL (same)
X_CM = 0.33                           # centre of mass, TL (Robertson et al. 2008)

LABELS = {
    "ant": ("anterior chamber\n(compliant)", "передняя камера\n(податливая)"),
    "post": ("posterior chamber", "задняя камера"),
    "cduct": ("connecting duct", "соединительный\nпроток"),
    "pduct": ("pneumatic duct", "пневматический\nпроток"),
    "oes": ("oesophagus", "пищевод"),
    "spine": ("vertebral column", "позвоночник"),
    "pect": ("pectoral fin", "грудной\nплавник"),
    "pelv": ("pelvic fin", "брюшной\nплавник"),
    "dors": ("dorsal fin", "спинной\nплавник"),
    "anal": ("anal fin", "анальный\nплавник"),
    "caud": ("caudal fin", "хвостовой\nплавник"),
    "cm": ("CM", "ЦМ"),
    "cb": ("CB", "ЦП"),
}

# body outline up to the caudal peduncle (illustrative)
_DX = [0.0, 0.012, 0.035, 0.08, 0.14, 0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.75, 0.80, 0.815]
_DY = [0.004, 0.020, 0.036, 0.056, 0.072, 0.082, 0.090, 0.094, 0.093, 0.088, 0.076, 0.058, 0.040, 0.031, 0.031]
_VY = [-0.004, -0.018, -0.032, -0.048, -0.060, -0.068, -0.076, -0.082, -0.084, -0.079, -0.066, -0.047, -0.033, -0.027,
       -0.027]
dorsal = PchipInterpolator(_DX, _DY)
ventral = PchipInterpolator(_DX, _VY)
X_PED = 0.815


def body_path():
    x = np.linspace(0, X_PED, 200)
    up = np.column_stack([x, dorsal(x)])
    lo = np.column_stack([x[::-1], ventral(x[::-1])])
    return np.vstack([up, lo])


def spine(x):
    """Vertebral column: a smooth line in the upper half of the body (illustrative)."""
    return 0.035 + (dorsal(x) + ventral(x)) * 0.5 * 0.9 - 0.012 * (x - 0.2)


def chamber(x0, x1, yc, h, taper_front, taper_back, n=160):
    """Closed outline of a bladder chamber between x0 and x1 with half-height h."""
    s = np.linspace(0, np.pi, n)
    x = x0 + (x1 - x0) * (1 - np.cos(s)) / 2
    u = (x - x0) / (x1 - x0)
    prof = np.sin(s) ** 0.75 * (1 - taper_front * (1 - u) ** 3) * (1 - taper_back * u ** 3)
    top = np.column_stack([x, yc + h * prof])
    bot = np.column_stack([x[::-1], (yc - h * prof)[::-1]])
    return np.vstack([top, bot])


def fin(points, fill, edge, ax, z=2, lw=0.6, rays=0, base=None, tipline=None):
    poly = Polygon(points, closed=True, fc=fill, ec=edge, lw=lw, zorder=z, joinstyle="round")
    ax.add_patch(poly)
    if rays and base is not None and tipline is not None:
        base, tipline = np.asarray(base, float), np.asarray(tipline, float)
        for k in np.linspace(0.08, 0.92, rays):
            b = base[0] + k * (base[1] - base[0])
            t = tipline[0] + k * (tipline[1] - tipline[0])
            ax.plot([b[0], t[0]], [b[1], t[1]], color=edge, lw=0.25, alpha=0.6, zorder=z + 0.1)
    return poly


def draw_anatomy(ax, lang="en", fs=6.5, labels=True, scale=(1.0, 1.0), stiff=None, set_limits=True, gradient=True):
    """scale: semi-axis factors of the (anterior, posterior) chamber; stiff: "ant"/"post" chamber drawn hatched
    grey (not compliant), the other one blue; None: anterior blue, posterior hatched (as in Fig. 1a)."""
    T = lambda k: LABELS[k][0 if lang == "en" else 1]  # noqa: E731
    dec = (lambda v: ("%g" % v).replace(".", ",")) if lang == "ru" else (lambda v: "%g" % v)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ---- unpaired fins (behind the body) ----
    yd, yv = float(dorsal(X_PED)), float(ventral(X_PED))
    caud = [(X_PED - 0.01, yd), (0.88, 0.060), (0.955, 0.082), (1.0, 0.078), (0.965, 0.040), (0.925, 0.004),
            (0.965, -0.036), (1.0, -0.074), (0.955, -0.078), (0.88, -0.056), (X_PED - 0.01, yv)]
    fin(caud, FIN_FILL, FIN_EDGE, ax, z=1)
    for (bx, by), (tx, ty) in (((0.83, 0.015), (0.99, 0.072)), ((0.83, 0.0), (0.92, 0.006)),
                               ((0.83, -0.015), (0.99, -0.068)), ((0.83, 0.008), (0.96, 0.05)),
                               ((0.83, -0.008), (0.96, -0.047))):
        ax.plot([bx, tx], [by, ty], color=FIN_EDGE, lw=0.25, alpha=0.6, zorder=1.1)
    d0, d1 = 0.53, 0.63
    dors = [(d0, float(dorsal(d0)) - 0.004), (0.565, 0.148), (0.60, 0.138), (0.645, 0.090),
            (d1, float(dorsal(d1)) - 0.004)]
    fin(dors, FIN_FILL, FIN_EDGE, ax, z=1, rays=6, base=((d0, float(dorsal(d0))), (d1, float(dorsal(d1)))),
        tipline=((0.565, 0.146), (0.64, 0.092)))
    a0, a1 = 0.56, 0.71
    anal = [(a0, float(ventral(a0)) + 0.004), (0.60, -0.140), (0.66, -0.128), (0.725, -0.075),
            (a1, float(ventral(a1)) + 0.004)]
    fin(anal, FIN_FILL, FIN_EDGE, ax, z=1, rays=8, base=((a0, float(ventral(a0))), (a1, float(ventral(a1)))),
        tipline=((0.60, -0.138), (0.72, -0.078)))
    p0, p1 = 0.40, 0.44
    pelv = [(p0, float(ventral(p0)) + 0.004), (0.455, -0.135), (0.475, -0.128), (p1, float(ventral(p1)) + 0.004)]
    fin(pelv, LIFT_FILL, C2, ax, z=1, lw=1.1, rays=4, base=((p0, float(ventral(p0))), (p1, float(ventral(p1)))),
        tipline=((0.452, -0.133), (0.474, -0.127)))

    # ---- body with a soft vertical gradient ----
    bp = body_path()
    xo = np.linspace(0, X_PED, 200)
    ax.plot(np.r_[xo[::-1], xo], np.r_[dorsal(xo[::-1]), ventral(xo)], color=EDGE, lw=0.8, zorder=4,
            solid_joinstyle="round")
    if not gradient:                                 # solid fill (for rotated copies of the drawing)
        ax.add_patch(PathPatch(Path(bp, closed=True), fc="#dde1e6", ec="none", zorder=2))
    grad = np.linspace(0, 1, 256)[:, None]
    cm_ = np.array([[0.80, 0.83, 0.86], [0.965, 0.97, 0.975]])
    img = cm_[0] + (cm_[1] - cm_[0]) * grad[..., None]
    im = ax.imshow(img, extent=(0, X_PED, -0.1, 0.1), origin="upper", zorder=2, interpolation="bicubic",
                   visible=gradient)
    clip = PathPatch(Path(bp, closed=True), transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    # faint horizontal stripes of the zebrafish, clipped to the body
    for yy in (0.03, 0.0, -0.03):
        xs = np.linspace(0.20, X_PED, 120)
        line, = ax.plot(xs, yy * (1 - 0.55 * (xs - 0.2) / 0.62), color="#c9d0d8", lw=1.6, zorder=2.2,
                        solid_capstyle="round")
        line.set_clip_path(clip)
    # head: operculum and eye
    th = np.linspace(-0.95, 0.95, 60)
    ax.plot(0.185 - 0.035 * np.cos(th) ** 2, 0.072 * np.sin(th) - 0.004, color=EDGE, lw=0.5, zorder=4)
    ax.add_patch(Circle((0.062, 0.017), 0.014, fc="#ffffff", ec=EDGE, lw=0.5, zorder=4))
    ax.add_patch(Circle((0.063, 0.017), 0.008, fc="#2b2f33", ec="none", zorder=4.1))
    ax.plot([0.0, 0.022], [0.002, -0.004], color=EDGE, lw=0.5, zorder=4)

    # ---- internal: gut and oesophagus (illustrative), ducts, bladder, spine ----
    gx = np.linspace(0.12, 0.56, 160)                  # oesophagus and gut (schematic)
    gc = -0.024 - 0.034 * np.sin(np.clip((gx - 0.12) / 0.30, 0, 1) * np.pi / 2) + 0.012 * np.clip((gx - 0.46) / 0.10, 0, 1)
    gw = (0.006 + 0.008 * np.sin(np.clip((gx - 0.26) / 0.26, 0, 1) * np.pi) ** 0.7) * np.clip((0.56 - gx) / 0.03, 0.15, 1)
    ax.add_patch(Polygon(np.vstack([np.column_stack([gx, gc + gw]), np.column_stack([gx[::-1], (gc - gw)[::-1]])]),
                         closed=True, fc="#efe6dd", ec="#b9ada1", lw=0.4, zorder=3))
    xs = np.linspace(0.16, 0.80, 200)
    ys = spine(xs)
    ax.plot(xs, ys, color="#6b737c", lw=1.1, zorder=3.2, solid_capstyle="round")
    for xv in np.arange(0.18, 0.80, 0.022):
        yv_ = float(spine(xv))
        ax.plot([xv, xv], [yv_ - 0.006, yv_ + 0.006], color="#6b737c", lw=0.35, zorder=3.25)

    yc = float(spine(0.30)) - 0.040
    # anterior chamber shorter but slightly larger in volume (54.5 % of the bladder, Stoyek et al. 2011):
    # taller and rounder; posterior chamber long and low (proportions only, no numbers on the figure)
    def scaled(c, k):
        cx, cy = c.mean(0)
        return np.column_stack([cx + k * (c[:, 0] - cx), cy + k * (c[:, 1] - cy)])
    a_out = scaled(chamber(ANT[0], ANT[1] - 0.004, yc - 0.004, 0.037, 0.12, 0.20), scale[0])
    p_out = scaled(chamber(POST[0] + 0.004, POST[1], yc - 0.006, 0.025, 0.30, 0.55), scale[1])
    if stiff is None:
        ax.add_patch(Polygon(a_out, closed=True, fc=CHAMBER_A, ec=C1, lw=0.8, zorder=3.4))
        ax.add_patch(Polygon(p_out, closed=True, fc=CHAMBER_P, ec=C1, lw=0.8, zorder=3.4, hatch="//////"))
    else:
        for c, st in ((a_out, stiff == "ant"), (p_out, stiff == "post")):
            if st:
                ax.add_patch(Polygon(c, closed=True, fc="#eceef1", ec="#7b8794", lw=0.8, zorder=3.4, hatch="//////"))
            else:
                ax.add_patch(Polygon(c, closed=True, fc=CHAMBER_A, ec=C1, lw=0.8, zorder=3.4))
    ax.plot([ANT[1] - 0.006, POST[0] + 0.006], [yc, yc], color=C1, lw=1.6, zorder=3.35, solid_capstyle="butt")
    # pneumatic duct: anteroventral wall of the posterior chamber -> oesophagus
    t = np.linspace(0, 1, 40)
    px = POST[0] + 0.012 - 0.035 * t
    py = yc - 0.024 - 0.018 * t + 0.010 * t * (1 - t)
    ax.plot(px, py, color=C1, lw=0.7, zorder=3.45)

    # ---- paired fins over the body ----
    pect = [(0.195, -0.040), (0.210, -0.028), (0.300, -0.062), (0.305, -0.072), (0.285, -0.074), (0.200, -0.050)]
    fin(pect, FIN_FILL, FIN_EDGE, ax, z=5, rays=5, base=((0.198, -0.046), (0.208, -0.030)),
        tipline=((0.29, -0.073), (0.302, -0.064)))

    # ---- centre of mass (quartered disc) and centre of buoyancy ----
    ycm = float(spine(X_CM)) - 0.045
    r = 0.0105
    ax.add_patch(Circle((X_CM, ycm), r, fc="white", ec=INK, lw=0.6, zorder=6))
    for a0_ in (0, 180):
        ax.add_patch(Wedge((X_CM, ycm), r, a0_, a0_ + 90, fc=INK, ec="none", zorder=6.1))
    ycb = ycm + 0.028
    ax.add_patch(Circle((X_CM, ycb), 0.0085, fc="white", ec=C1, lw=0.8, zorder=6))
    ax.plot(X_CM, ycb, "o", color=C1, ms=1.2, zorder=6.1)

    if not labels:
        return ycm
    # ---- labels with leader lines ----
    lead = dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=1.5, shrinkB=0.5)

    def lab(key, xy, xyt, ha="center", va="bottom", color=INK):
        ax.annotate(T(key), xy=xy, xytext=xyt, fontsize=fs, ha=ha, va=va, color=color, arrowprops=lead,
                    zorder=7, linespacing=1.0)

    t2, t1, b1, b2 = 0.248, 0.168, -0.168, -0.232
    lab("ant", (0.19, yc + 0.012), (0.10, t2), color=C1)
    lab("post", (0.41, yc + 0.014), (0.45, t2), color=C1)
    lab("spine", (0.72, float(spine(0.72))), (0.80, t2))
    lab("cduct", (0.300, yc + 0.002), (0.245, t1), color=C1)
    lab("cb", (X_CM + 0.004, ycb + 0.008), (0.40, t1), color=C1)
    lab("dors", (0.60, 0.125), (0.64, t1))
    lab("caud", (0.96, 0.05), (0.95, t1))
    lab("pect", (0.255, -0.067), (0.12, b1), va="top")
    lab("cm", (X_CM, ycm - 0.011), (X_CM, b1), va="top")
    lab("pelv", (0.462, -0.128), (0.50, b1), va="top", color=C2)
    lab("anal", (0.66, -0.125), (0.70, b1), va="top")
    lab("pduct", (POST[0] - 0.02, yc - 0.048), (0.22, b2), va="top", color=C1)

    # ---- scale of fractions of the total length ----
    ysb = -0.305
    ax.plot([0, 1], [ysb, ysb], color=INK2, lw=0.6, zorder=3)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        ax.plot([v, v], [ysb, ysb + 0.012], color=INK2, lw=0.6, zorder=3)
        ax.text(v, ysb - 0.012, dec(v) + (" $L$" if v == 1.0 else ""), fontsize=fs, ha="center", va="top",
                color=INK2)
    if set_limits:
        ax.set_xlim(-0.02, 1.03)
        ax.set_ylim(-0.355, 0.315)
    return ycm


def draw_fish_at(ax, x0, y0, Ld, pitch_deg, **kw):
    """Zebrafish of draw_anatomy (without labels) facing right, centre of mass at (x0, y0), length Ld, pitched
    nose up by pitch_deg. Returns the function mapping body points (x from the centre of mass, forward +, in TL)
    to plot coordinates, and the belly height function."""
    from matplotlib.transforms import Affine2D
    n_p, n_l, n_i = len(ax.patches), len(ax.lines), len(ax.images)
    ycm = draw_anatomy(ax, labels=False, set_limits=False, gradient=False, **kw)
    T = (Affine2D().translate(-X_CM, -ycm).scale(-1, 1).scale(Ld).rotate_deg(pitch_deg).translate(x0, y0)
         + ax.transData)
    for a in list(ax.patches[n_p:]) + list(ax.lines[n_l:]) + list(ax.images[n_i:]):
        a.set_transform(T)
    c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))

    def body(px, pz):
        return x0 + Ld * (px * c - pz * s), y0 + Ld * (px * s + pz * c)

    def belly(px):                       # ventral contour at body abscissa px (forward +), relative to the CM
        return float(ventral(X_CM - px)) - ycm
    return body, belly
