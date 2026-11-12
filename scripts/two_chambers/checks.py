"""Numbers of the two-chamber section and its Methods that are not produced by fig5_nature_data.py.

1. Static pitch response of a body held at a pivot, dtheta/dz = kb*rbar/kth:
   goldfish of Table 1 with the zebrafish partition (Boyle's law at 1 m, as for Fig. 5) and
   zebrafish of Stoyek et al. 2011 (height h that reproduces the measured 0.009 deg per cm of water).
2. Slow root: formula (8) of the paper against the computed eigenvalue, kb*sigma^2/kth.
3. Depth change: time to reach a depth 1 m greater after a fin step, four bladders, beta = 1 and 0.1,
   U = 1.5, 2, 3 U0, three positions of the lifting surface, fin steps 1, 2, 5 deg;
   constant fin command that holds the new depth (static balance).
Writes data/two_chambers/checks.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parents[2] / "data" / "two_chambers"
OUT.mkdir(parents=True, exist_ok=True)
g, rho = 9.81, 1000.0
P0 = 101325.0                 # atmospheric pressure, Pa
P1 = P0 + rho * g * 1.0       # absolute pressure at 1 m
res = {}

# ---------- 1. static pitch response ----------
# goldfish (Di Santo et al. 2025): L, m, h; bladder 6.5 % of body volume; chambers at +0.10 L and -0.12 L
L, m, h = 0.0588, 0.0031428571, 0.00064
kb_tot = (rho * g) ** 2 * (0.065 * m / rho) / P1
kth = m * g * h
f_z = 0.97
rbar = (f_z * 0.10 - (1 - f_z) * 0.12) * L
res["goldfish_rbar_L"] = rbar / L
res["goldfish_zebra_deg_per_m"] = float(np.degrees(kb_tot * rbar / kth))
# zebrafish (Stoyek et al. 2011): slopes of chamber volume, % of total volume per cm of water (isolated bladders)
s_ant, s_post = 0.07, 0.002
f_zf = s_ant / (s_ant + s_post)
# chambers 17-30 % and 30-47 % TL from the snout (estimated from the schematic, Fig. 1A of Stoyek et al.),
# pivot at 33 % TL (text of Stoyek et al.); offsets of the chamber centroids from the pivot (+ forward)
r_ant, r_post = 0.33 - 0.235, 0.33 - 0.385
TL, phi = 0.028, 0.051          # mid-range TL; bladder 5.1 % of body volume (Robertson et al. 2008)
rb_zf = (f_zf * r_ant + (1 - f_zf) * r_post) * TL
target = np.radians(0.009 * 100)   # 0.009 deg per cm of water = 0.9 deg per m
res["zebrafish_anterior_share"] = f_zf
res["zebrafish_rbar_TL"] = rb_zf / TL
for lab, P in (("P0", P0), ("1m", P1)):
    # dtheta/dz = kb*rbar/(m g h) with kb = (rho g)^2 phi m / (rho P): h does not depend on m
    c = phi * rho * g * rb_zf / P
    res[f"zebrafish_h_mm_for_0.9deg_per_m_{lab}"] = 1e3 * c / target
    res[f"zebrafish_deg_per_m_h0.6_0.8mm_{lab}"] = [float(np.degrees(c / hh)) for hh in (0.6e-3, 0.8e-3)]
res["boyle_slope_pct_per_cm_P0"] = 100 * rho * g * 0.01 / P0

# ---------- 2. slow root ----------
ell = 0.25 * L
J = m * ell ** 2
rA = 0.1403 * L               # measured centroid offset (behind the centre of mass)


def B(rA_, rL, kb, K):
    return (rL - rA_) * (J * kb + kb * m * rL * rA_) + kth * rL * m


def setup(beta):
    K = beta * m / ell
    rLB = brentq(lambda r: B(-rA, r, kb_tot, K), -rA + 1e-12, -1e-12, xtol=1e-15)
    U0 = np.sqrt(kth / (K * (rLB + rA)))
    return K, rLB, U0


def rhs(t, x, ch, delta, K, U, rL):
    z, vz, th, om = x
    gam = np.arctan2(vz, U)
    al = th - gam + delta
    FA = sum(k * (z + r * np.sin(th)) for k, r in ch)
    MA = sum(r * np.cos(th) * k * (z + r * np.sin(th)) for k, r in ch)
    FL = K * U ** 2 * al
    return [vz, (FA + FL * np.cos(gam)) / m, om, (MA + rL * FL * np.cos(al) - kth * th) / J]


def AB(ch, K, U, rL):
    f = lambda x, d: np.array(rhs(0, x, ch, d, K, U, rL))
    e = 1e-8
    A = np.column_stack([(f(np.eye(4)[i] * e, 0) - f(-np.eye(4)[i] * e, 0)) / (2 * e) for i in range(4)])
    b = (f(np.zeros(4), e) - f(np.zeros(4), -e)) / (2 * e)
    return A, b


def cham(f):
    return [(kb_tot * f, 0.10 * L), (kb_tot * (1 - f), -0.12 * L)]


K, rLB, U0 = setup(1.0)
U2 = 2 * U0
rL_fig = 0.5 * ((-rA + kth / (K * U2 ** 2)) + rLB)   # middle of the stable interval at 2 U0 (Fig. 5)
res["U0_Lps"] = U0 / L
res["rL_fig5_L"] = rL_fig / L
err = []
for Uf in (1.5, 2.0, 3.0):
    U = Uf * U0
    for f in np.linspace(0, 1, 201):
        A, _ = AB(cham(f), K, U, rL_fig)
        w = np.linalg.eigvals(A)
        real = [z.real for z in w if abs(z.imag) < 1e-9]
        lam = min(real, key=abs)
        rb = f * 0.10 * L - (1 - f) * 0.12 * L
        lam_f = kb_tot / (K * U) * (1 - K * U ** 2 * (rL_fig - rb) / kth)
        if abs(lam) > 1e-4:
            err.append(abs(lam_f / lam - 1))
res["lambda_formula_max_rel_err"] = max(err)
res["kb_sigma2_over_kth_equal"] = kb_tot * 0.25 * (0.22 * L) ** 2 / kth

# ---------- 3. depth change ----------
configs = {"rear": [(kb_tot, -rA)], "front": [(kb_tot, +rA)], "incompressible": [(0.0, 0.0)],
           "zebrafish": cham(f_z)}


def t1m(ch, delta, K, U, rL, Tmax=6000.0):
    def hit(t, x, *a):
        return x[0] + 1.0
    hit.terminal = True
    s = solve_ivp(rhs, (0, Tmax), [0, 0, 0, 0], args=(ch, delta, K, U, rL), events=hit, max_step=0.2,
                  rtol=1e-8, atol=1e-11)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else float("inf")


def hold(ch, K, U, rL):
    A, b = AB(ch, K, U, rL)
    M = np.column_stack([A[:, 1:], b])
    return float(np.degrees(np.linalg.lstsq(M, A[:, 0], rcond=None)[0][-1]))   # z = -1 m


rows = []
for beta in (1.0, 0.1):
    K, rLB, U0b = setup(beta)
    for Uf in (1.5, 2.0, 3.0):
        U = Uf * U0b
        lo = -rA + kth / (K * U ** 2)
        for pos in (0.25, 0.5, 0.75):
            rL = lo + pos * (rLB - lo)
            for dd in (-1.0, -2.0, -5.0):
                t = {k: t1m(ch, np.radians(dd), K, U, rL) for k, ch in configs.items()}
                hd = {k: hold(ch, K, U, rL) for k, ch in configs.items()} if dd == -1.0 else None
                rows.append(dict(beta=beta, U_over_U0=Uf, pos=pos, step_deg=dd, t1m=t, hold_deg=hd))
res["depth_change"] = rows
first = [min(r["t1m"], key=r["t1m"].get) for r in rows]
res["fastest_counts"] = {k: first.count(k) for k in configs}
res["zebrafish_before_rear_and_incompressible"] = sum(
    r["t1m"]["zebrafish"] < min(r["t1m"]["rear"], r["t1m"]["incompressible"]) for r in rows)
res["front_before_rear_and_incompressible"] = sum(
    r["t1m"]["front"] < min(r["t1m"]["rear"], r["t1m"]["incompressible"]) for r in rows)
res["n_cases"] = len(rows)
json.dump(res, open(OUT / "checks.json", "w"), indent=1)
for k, v in res.items():
    if k != "depth_change":
        print(k, v)
hr = [r["hold_deg"] for r in rows if r["hold_deg"]]
print("hold command (deg), |zebrafish| > |rear| in", sum(abs(x["zebrafish"]) > abs(x["rear"]) for x in hr),
      "of", len(hr), "; > |incompressible| in", sum(abs(x["zebrafish"]) > abs(x["incompressible"]) for x in hr))
print("examples:", hr[:2])
