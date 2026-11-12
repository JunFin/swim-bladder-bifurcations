"""Pitch damping by the fin: alpha = theta - arctan((zdot + eta*rL*thetadot*cos(theta))/U).

Linear check: the critical eigenvalues on a4 = 0 and B = 0 do not depend on eta.
First Lyapunov coefficient on B = 0 for eta = 0 and 1.
Attractor sweeps in d (reference set) for given eta and U:
    python scripts/pitch_damping.py linear
    python scripts/pitch_damping.py sweep ETA U        # d = 0.01..0.40, step 0.01
Results used in the paper: data/pitch_damping.json (linear, l1) and data/pitch_damping_sweeps.json.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

from bladder import P, U1
from bladder.model import Params, birth_frequency, rL_on_B0
from bladder.normal_form import first_lyapunov, numerical_jacobian
from common import DATA

RA = -4.6


def rhs(t, x, U, rA, rL, p, eta):
    z, vz, th, om = x
    gamma = np.arctan2(vz, U)
    alpha = th - np.arctan2(vz + eta * rL * om * np.cos(th), U)
    FA = p.kb * (z + rA * np.sin(th))
    FL = p.K * U**2 * alpha
    return np.array([vz, (FA + FL * np.cos(gamma)) / p.m, om,
                     (rA * np.cos(th) * FA + rL * FL * np.cos(alpha) - p.kth * th) / p.J])


def field(U, rA, rL, p, eta):
    return lambda x: rhs(0.0, x, U, rA, rL, p, eta)


def linear():
    rLB = rL_on_B0(RA, P)
    out = dict(omega0=birth_frequency(RA, rLB, P), linear=[], l1_ref=[], l1_fish={})
    for U in (1.6, 2.0, 3.0, 5.0):
        for eta in (0.0, 0.5, 1.0):
            ev = np.linalg.eigvals(numerical_jacobian(field(U, RA, rLB, P, eta), np.zeros(4)))
            ev4 = np.linalg.eigvals(numerical_jacobian(field(U1(RA, -1.0, P), RA, -1.0, P, eta), np.zeros(4)))
            c = ev[np.argmin(np.abs(ev.real))]
            out["linear"].append(dict(U=U, eta=eta, crit_re=float(c.real), crit_im=float(abs(c.imag)),
                                      min_abs_on_a4=float(np.min(np.abs(ev4)))))
    for U in (1.6, 1.8, 2.0, 3.0, 5.0):
        for eta in (0.0, 1.0):
            out["l1_ref"].append(dict(U=U, eta=eta, l1=first_lyapunov(field(U, RA, rLB, P, eta))[0]))
    for name, kap, bet, ra in (("kappa0.002_beta1_rho-0.5", 0.002, 1.0, -0.5),
                               ("kappa0.002_beta0.3_rho-0.7", 0.002, 0.3, -0.7)):
        pf = Params(m=1, J=1, K=bet, kb=kap, kth=1)
        rlb = rL_on_B0(ra, pf)
        U0 = U1(ra, rlb, pf)
        out["l1_fish"][name] = [dict(U_over_U0=k, eta=eta, l1=first_lyapunov(field(k * U0, ra, rlb, pf, eta))[0])
                                for k in (1.05, 1.2, 1.5, 2.0) for eta in (0.0, 1.0)]
    (DATA / "pitch_damping.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def esc(t, x, *a):
    return 1.0 - max(abs(x[0]) / 50.0, abs(x[2]) / (np.pi / 2))


esc.terminal = True


def run(x0, T, args, dense=False):
    return solve_ivp(rhs, (0, T), x0, args=args, method="DOP853", rtol=1e-9, atol=1e-11,
                     events=esc, dense_output=dense)


def lyap(x0, args, T=600.0, dt=2.0, eps=1e-8):
    x = np.array(x0, float)
    y = x + eps * np.array([0, 0, 1, 0.0])
    s = 0.0
    for _ in range(int(T / dt)):
        a, b = run(x, dt, args), run(y, dt, args)
        if a.status == 1 or b.status == 1:
            return None
        x, y = a.y[:, -1], b.y[:, -1]
        d = np.linalg.norm(y - x)
        s += np.log(d / eps)
        y = x + (y - x) * eps / d
    return s / T


def sweep(eta, U):
    rLB = rL_on_B0(RA, P)
    x = np.array([0.0, 0.0, 0.05, 0.0])
    rows = []
    for d in np.linspace(0.01, 0.40, 40):
        args = (U, RA, rLB + d, P, eta)
        s = run(x, 1200.0, args)
        if s.status == 1:
            rows.append(dict(d=round(d, 4), cls="escape"))
            x = np.array([0.0, 0.0, 0.05, 0.0])
            continue
        x = s.y[:, -1]
        r = run(x, 600.0, args, dense=True)
        th = r.sol(np.linspace(0, 600, 60001))[2]
        i = np.where((th[1:-1] > th[:-2]) & (th[1:-1] > th[2:]))[0] + 1
        nmax = len(np.unique(np.round(th[i][-40:], 3)))
        lam = lyap(r.y[:, -1], args) if nmax > 8 else None
        rows.append(dict(d=round(d, 4), cls="chaos" if (lam or 0) > 0.01 else f"cycle({nmax})", lam=lam,
                         mean_theta_deg=float(np.degrees(th.mean()))))
        x = r.y[:, -1]
        print(rows[-1], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "linear":
        linear()
    else:
        sweep(float(sys.argv[2]), float(sys.argv[3]))
