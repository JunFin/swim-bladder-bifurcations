"""Limit of bounded motion beyond the curve B = 0 for the fish-like set, with long integration.

For kappa << 1 the depth mode is nearly neutral, and trajectories leave the cycle only after
1e4-1e5 s, so short runs count them as bounded. The shift d is given as a fraction of the width of the
stable interval at the same speed, W = r_L^B - r_L(a4 = 0).

Usage: python fish_escape.py <U/U0> <T> <d/W in %, comma-separated>; prints one JSON line per shift.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

from bladder import Params, U1, rhs, rL_on_B0
from bladder.model import escaped

KAPPA, BETA, RHO_A, KTH = 0.002, 1.0, -0.5, 0.4
P = Params(K=BETA, kb=KAPPA * KTH, kth=KTH)
THETA0 = (0.1, 0.5)


def run(f, T, frac):
    rLB = rL_on_B0(RHO_A, P)
    U = f * U1(RHO_A, rLB, P)
    W = rLB - (RHO_A + P.kth / (P.K * U**2))
    d = frac / 100 * W
    out = dict(U_over_U0=f, T=T, d_over_W_pct=frac, W=W, runs=[])
    for th0 in THETA0:
        s = solve_ivp(rhs, (0, T), [0, 0, th0, 0], args=(U, RHO_A, rLB + d, P), rtol=1e-9, atol=1e-11,
                      method="DOP853", events=escaped)
        if s.status == 1:
            out["runs"].append(dict(theta0=th0, escaped=True, t=float(s.t[-1])))
        else:
            late = s.t > 0.9 * T
            out["runs"].append(dict(theta0=th0, escaped=False,
                                    theta_max_deg=float(np.degrees(np.abs(s.y[2][late]).max())),
                                    z_max=float(np.abs(s.y[0][late]).max())))
    return out


if __name__ == "__main__":
    f, T = float(sys.argv[1]), float(sys.argv[2])
    for frac in map(float, sys.argv[3].split(",")):
        print(json.dumps(run(f, T, frac)), flush=True)
