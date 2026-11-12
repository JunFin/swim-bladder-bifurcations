"""Post-processing of the side-view renders of the BarramundiFish model (render_fish_side.py) for the figures:
cut-away of the swim bladder, highlight of the pelvic fin, placement of the image in data coordinates.

All geometry is in body lengths L measured from the centre of mass (head towards +x, up +z), as in
render_fish_side.py; a fish drawn with length Ld at (x0, y0) has the point (x, z) of its body at
x0 + Ld * x, y0 + Ld * z (the pitch is already in the render).
"""
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse
from scipy import ndimage

RENDERS = Path(__file__).resolve().parent / "fish_renders"
CAVITY = np.array(to_rgb("#e9ebee"))          # opened cavity around a stiff (not compliant) chamber
CUT_EDGE = "#4a4f55"                           # thin dark-grey edge of the cut in the body wall
STIFF_EDGE = "#7b8794"
BLADDER_TINT = "#8db7ea"                       # mutes the specular highlight of the rendered bladder


class Render:
    """One fish: the opaque render and its masks, all of the same size."""

    def __init__(self, folder, name):
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        fr = meta["frames"][name]
        self.k, self.cx, self.cy = fr["k_px_per_L"], fr["cx"], fr["cy"]
        self.pitch = np.radians(fr["pitch_deg"])
        rd = lambda s: mpimg.imread(folder / f"{name}_{s}.png") if (folder / f"{name}_{s}.png").exists() else None  # noqa: E731
        self.op = mpimg.imread(folder / f"{name}.png")
        self.bladder_mask = rd("bm")
        self.bladder = rd("bs")
        self.fin_mask = rd("fin")
        self.pect_mask = rd("pect")
        self.xray = rd("xr")
        h, w = self.op.shape[:2]
        self.extent0 = (-self.cx / self.k, (w - self.cx) / self.k, -(h - self.cy) / self.k, self.cy / self.k)
        self.window = None

    def extent(self, x0, y0, Ld):
        a, b, c, d = self.extent0
        return (x0 + Ld * a, x0 + Ld * b, y0 + Ld * c, y0 + Ld * d)

    def compose(self, style="cutaway", fin_color="#eb6834", fin_alpha=0.16, shadow=0.42):
        """RGBA image: textured fish, the bladder in a cut-away window of the body wall, a light tint of the
        pelvic fin, and the near pectoral fin redrawn over the window.  Blue (compliant) chambers come from the
        shaded render; a grey (stiff) chamber is left as an opened light cavity (drawn by draw(stiff=...))."""
        img = (self.xray if style == "xray" else self.op).copy()
        rgb, a = img[..., :3], img[..., 3]
        if self.fin_mask is not None and fin_alpha > 0:
            f = (self.fin_mask[..., 0] * self.fin_mask[..., 3] * a)[..., None]
            rgb[:] = rgb * (1 - fin_alpha * f) + np.array(to_rgb(fin_color)) * fin_alpha * f
        if style == "cutaway" and self.bladder_mask is not None:
            m = self.bladder_mask[..., 3] > 0.5
            win = ndimage.binary_dilation(m, iterations=1) & (a > 0.5)
            self.window = win
            w = ndimage.gaussian_filter(win.astype(float), 0.7)[..., None] * a[..., None]
            fill = np.broadcast_to(CAVITY, rgb.shape).copy()
            if self.bladder is not None:
                b = self.bladder
                # a chamber (connected part of the mask) is blue = compliant if its mean colour is bluish;
                # stiff (grey) chambers are left as the light opened cavity and hatched in draw()
                lab, n = ndimage.label(ndimage.binary_dilation(b[..., 3] > 0.05, iterations=2))
                blue = np.zeros(m.shape, bool)
                for i in range(1, n + 1):
                    part = lab == i
                    if (b[..., 2] - b[..., 0])[part & (b[..., 3] > 0.5)].mean() > 0.08:
                        blue |= part
                ab = np.where(blue[..., None], b[..., 3:4], 0.0)
                col = 0.7 * b[..., :3] + 0.3 * np.array(to_rgb(BLADDER_TINT))
                fill = fill * (1 - ab) + col * ab
            # soft inner shadow along the edge of the cut: the window reads as an opening in the body wall
            dist = ndimage.distance_transform_edt(win)
            r = max(1.5, 0.016 * self.k)
            fill = fill * (1 - shadow * np.exp(-dist / r))[..., None]
            rgb[:] = rgb * (1 - w) + fill * w
        if self.pect_mask is not None:
            pm = (self.pect_mask[..., 3] * (self.pect_mask[..., 0] > 0.5))[..., None]
            rgb[:] = rgb * (1 - pm) + self.op[..., :3] * pm
        img[..., :3] = np.clip(rgb, 0, 1)
        return img

    def draw(self, ax, x0, y0, Ld, style="cutaway", outline=True, c2="#eb6834", lw=0.5, zorder=3,
             stiff=(), **kw):
        """Draw the fish at (x0, y0) with length Ld. stiff: [(x, z, a, c), ...] stiff chambers in body L,
        drawn as hatched light ellipses inside the window."""
        img = self.compose(style, fin_color=c2, **kw)
        ext = self.extent(x0, y0, Ld)
        ax.imshow(img, extent=ext, interpolation="lanczos", zorder=zorder, origin="upper")
        h, w = img.shape[:2]
        X = np.linspace(ext[0], ext[1], w)
        Y = np.linspace(ext[3], ext[2], h)
        for (x, z, ea, ec) in stiff:
            c = self.body(x0, y0, Ld, x, z)
            ax.add_patch(Ellipse(c, 2 * ea * Ld, 2 * ec * Ld, angle=np.degrees(self.pitch), fc="none",
                                 ec=STIFF_EDGE, lw=0.6, hatch="//////", zorder=zorder + 0.15))
        if outline:
            if self.window is not None:
                win = self.window.copy()
                if self.pect_mask is not None:       # the cut edge stops at the pectoral fin
                    win = win | (self.pect_mask[..., 3] > 0.5)
                    edge = ndimage.gaussian_filter(win.astype(float), 1.0)
                    edge = np.where(self.pect_mask[..., 3] > 0.5, np.nan, edge)
                else:
                    edge = ndimage.gaussian_filter(win.astype(float), 1.0)
                ax.contour(X, Y, edge, [0.5], colors=CUT_EDGE, linewidths=lw, zorder=zorder + 0.2)
            if self.fin_mask is not None:
                ax.contour(X, Y, ndimage.gaussian_filter(self.fin_mask[..., 3], 1.0), [0.5], colors=c2,
                           linewidths=0.6 * lw, alpha=0.8, zorder=zorder + 0.2)
        return ext

    def body(self, x0, y0, Ld, px, pz):
        """Plot coordinates of the body point (px, pz) (body frame, L units from the centre of mass)."""
        c, s = np.cos(self.pitch), np.sin(self.pitch)
        return x0 + Ld * (px * c - pz * s), y0 + Ld * (px * s + pz * c)

    def belly(self, px, x0=0.0, y0=0.0, Ld=1.0):
        """Lowest point of the opaque silhouette at the body-frame abscissa px (body frame z, L units)."""
        a = self.op[..., 3] > 0.5
        c, s = np.cos(self.pitch), np.sin(self.pitch)
        best = None
        for pz in np.linspace(-0.3, 0.1, 400):
            X = px * c - pz * s
            Z = px * s + pz * c
            i, j = int(round(self.cy - Z * self.k)), int(round(self.cx + X * self.k))
            if 0 <= i < a.shape[0] and 0 <= j < a.shape[1] and a[i, j]:
                best = pz
                break
        return best

    def inside(self, X, Y, x0, y0, Ld):
        """True where the plot points (X, Y) fall on the fish silhouette (fish drawn at x0, y0 with length Ld)."""
        a = self.op[..., 3] > 0.5
        j = np.round(self.cx + (np.asarray(X) - x0) / Ld * self.k).astype(int)
        i = np.round(self.cy - (np.asarray(Y) - y0) / Ld * self.k).astype(int)
        ok = (i >= 0) & (i < a.shape[0]) & (j >= 0) & (j < a.shape[1])
        out = np.zeros(np.shape(X), bool)
        out[ok] = a[i[ok], j[ok]]
        return out

    def line(self, ax, p, q, x0, y0, Ld, color, lw, ls, zorder, inner="white", inner_alpha=0.75, n=600):
        """Straight line p-q: dark outside the fish, light where it crosses the body."""
        t = np.linspace(0, 1, n)
        X = p[0] + (q[0] - p[0]) * t
        Y = p[1] + (q[1] - p[1]) * t
        ins = self.inside(X, Y, x0, y0, Ld)
        for mask, col, al in ((~ins, color, 1.0), (ins, inner, inner_alpha)):
            Xm, Ym = np.where(mask, X, np.nan), np.where(mask, Y, np.nan)
            ax.plot(Xm, Ym, color=col, lw=lw, ls=ls, alpha=al, zorder=zorder)
