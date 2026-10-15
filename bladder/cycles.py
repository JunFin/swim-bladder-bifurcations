import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

from .model import P, U1, flow, rhs, saddle, theta_star
from .normal_form import hopf_seed, linear_rates

# fsolve warns on every failed seed; failures are handled by the residual check.
warnings.filterwarnings("ignore", "The iteration is not making good progress")


def _monodromy(x0, T, U, rA, rL, p, h=1e-7):
    D = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        D[:, j] = (flow(x0 + e, T, U, rA, rL, p) - flow(x0 - e, T, U, rA, rL, p)) / (2 * h)
    return D


def symmetric_cycle(U, rA, rL, seed, p=P):
    """Cycle with x(t + T/2) = -x(t), found by shooting over half a period.

    seed = [z, vz, theta, T/2] at omega = 0. The multipliers `hmult` belong to
    the half-period map h(x) = -phi_{T/2}(x); a multiplier leaving through -1
    breaks the symmetry.
    """
    def residual(v):
        x0 = np.array([v[0], v[1], v[2], 0.0])
        return flow(x0, v[3], U, rA, rL, p) + x0

    v = fsolve(residual, seed, xtol=1e-12)
    x0 = np.array([v[0], v[1], v[2], 0.0])
    half = float(v[3])
    if half <= 0.1 or np.linalg.norm(x0) < 1e-8 or np.linalg.norm(residual(v)) > 1e-8:
        return None
    Dh = _monodromy(x0, half, U, rA, rL, p)
    s = solve_ivp(rhs, (0, 2 * half), x0, args=(U, rA, rL, p), rtol=1e-11, atol=1e-13,
                  method="DOP853", t_eval=np.linspace(0, 2 * half, 800))
    hmult = np.linalg.eigvals(-Dh)
    return dict(x0=x0, T=2 * half, v=v, hmult=np.array(sorted(hmult, key=abs)[::-1]),
                amp_z=0.5 * float(np.ptp(s.y[0])), amp_th=0.5 * float(np.ptp(s.y[2])))


def cycle_info(U, rA, rL, cyc, p=P):
    s = solve_ivp(rhs, (0, cyc["T"]), cyc["x0"], args=(U, rA, rL, p), rtol=1e-10, atol=1e-12,
                  method="DOP853", t_eval=np.linspace(0, cyc["T"], 1200))
    alpha = s.y[2] - np.arctan2(s.y[1], U)
    hm = cyc["hmult"]
    nontrivial = np.delete(hm, int(np.argmin(np.abs(hm - 1.0))))
    return dict(T=cyc["T"], amp_th=cyc["amp_th"], amp_z=cyc["amp_z"],
                alpha_max=float(np.max(np.abs(alpha))), h=nontrivial,
                hmax=float(np.max(np.abs(nontrivial))))


class SymmetricBranch:
    """Continuation of the symmetric cycle in d = rL - rL(B=0) at fixed U."""

    def __init__(self, U, rA, rLB, p=P):
        self.U, self.rA, self.rLB, self.p = U, rA, rLB, p

    def solve(self, d, seed=None, dprev=None):
        tries = []
        if seed is not None:
            tries.append(seed)
            if dprev and d < 0.1:
                f = np.sqrt(d / dprev)
                tries.append([seed[0] * f, seed[1] * f, seed[2] * f, seed[3]])
        if d < 0.2:
            tries.append(hopf_seed(self.U, self.rA, d, self.p))
        for sd in tries:
            c = symmetric_cycle(self.U, self.rA, self.rLB + d, sd, self.p)
            if c is not None:
                return c
        return None

    def info(self, d, cyc):
        out = cycle_info(self.U, self.rA, self.rLB + d, cyc, self.p)
        out["d"] = d
        return out

    def trace(self, ds):
        out, seed, dprev = [], None, None
        for d in ds:
            c = self.solve(d, seed, dprev)
            if c is None:
                break
            out.append((self.info(d, c), list(c["v"])))
            seed, dprev = list(c["v"]), d
        return out

    def bisect(self, lo, hi, seed_lo, key, target, steps=14):
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            c = self.solve(mid, seed_lo, lo)
            if c is None:
                break
            if key(self.info(mid, c)) < target:
                lo, seed_lo = mid, list(c["v"])
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def crossing(self, branch, key, target):
        for a, b in zip(branch[:-1], branch[1:]):
            if key(a[0]) < target <= key(b[0]):
                return self.bisect(a[0]["d"], b[0]["d"], a[1], key, target)
        return np.nan


def symmetry_breaking_d(U, rA, rLB, dmax, p=P):
    """d at which the symmetric cycle loses stability (h-multiplier -1) at fixed U."""
    br = SymmetricBranch(U, rA, rLB, p)
    ds = np.round(np.linspace(dmax / 40, dmax, 40), 6)
    return br.crossing(br.trace(ds), lambda i: i["hmax"], 1.0)


def full_cycle(U, rA, rL, seed, p=P):
    def residual(v):
        x0 = np.array([v[0], v[1], v[2], 0.0])
        return flow(x0, v[3], U, rA, rL, p) - x0

    v = fsolve(residual, seed, xtol=1e-12)
    x0 = np.array([v[0], v[1], v[2], 0.0])
    T = float(v[3])
    if T <= 0.1 or np.linalg.norm(residual(v)) > 1e-8:
        return None
    mu = np.linalg.eigvals(_monodromy(x0, T, U, rA, rL, p))
    nontrivial = np.delete(mu, int(np.argmin(np.abs(mu - 1))))
    s = solve_ivp(rhs, (0, T), x0, args=(U, rA, rL, p), rtol=1e-11, atol=1e-13,
                  method="DOP853", t_eval=np.linspace(0, T, 600))
    return dict(v=v, T=T, mult=nontrivial, thmean=float(s.y[2].mean()))


def asymmetric_seed(U, rA, rL, p=P):
    th = theta_star(U, rA, rL, p) or 0.05
    x0 = 0.5 * saddle(U, rA, rL, th, p) + np.array([0, 0, 0.01, 0])
    s = solve_ivp(rhs, (0, 2000), x0, args=(U, rA, rL, p), rtol=1e-10, atol=1e-13,
                  method="DOP853", t_eval=np.linspace(1800, 2000, 40000))
    sg = np.sign(s.y[3])
    cr = np.where((sg[:-1] > 0) & (sg[1:] <= 0))[0]
    i = cr[-1]
    return [s.y[0, i], s.y[1, i], s.y[2, i], float(np.mean(np.diff(s.t[cr])))]


def torus_speed(d, rA, rLB, U_start, dU=5e-4, p=P):
    """U on the TR line at fixed d: follow the asymmetric cycle down in U until
    a complex pair of multipliers leaves the unit circle."""
    rL = rLB + d
    seed = asymmetric_seed(U_start, rA, rL, p)
    prev = None
    U = U_start
    while U > U1(rA, rL, p):
        cyc = full_cycle(U, rA, rL, seed, p)
        if cyc is None or abs(cyc["thmean"]) < 1e-3:
            return np.nan
        mu = cyc["mult"][int(np.argmax(np.abs(cyc["mult"])))]
        if abs(mu) > 1:
            if prev is None:
                return np.nan
            U_prev, m_prev = prev
            return U_prev + (U - U_prev) * (1 - m_prev) / (abs(mu) - m_prev)
        prev = (U, abs(mu))
        seed = list(cyc["v"])
        U -= dU
    return np.nan


def root_scan(g, lo, hi, n=400):
    xs = np.linspace(lo, hi, n)
    prev = None
    for x in xs:
        v = g(x)
        if not np.isfinite(v):
            return np.nan
        if prev is not None and np.sign(v) != np.sign(prev[1]):
            return brentq(g, prev[0], x, xtol=1e-10)
        prev = (x, v)
    return np.nan


def normal_form_line(slope, rA, rL, U_lo, U_hi, p=P, n=800):
    """U where mu2 = slope * mu1 at fixed rL (a line of the normal form mapped to (U, d))."""
    g = lambda U: linear_rates(U, rA, rL, p)[1] - slope * linear_rates(U, rA, rL, p)[0]
    return root_scan(g, U_lo, U_hi, n)
