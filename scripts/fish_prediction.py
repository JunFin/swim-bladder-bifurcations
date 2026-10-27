"""Checks behind the prediction for fish: U0 is the lowest speed at which any position of the lifting
surface gives a stable equilibrium, it hardly depends on depth, and the order of species by U0 does not
depend on the assumed bladder volume, radius of gyration and lift parameter.

Writes data/fish_prediction.json.
"""
import json
from itertools import product

import numpy as np

from bladder import Params, U1, rL_on_B0
from bladder.model import max_real_eig
from common import DATA
from fish_parameters import BETA, G, L_FRAC, PHI, RHO, species

DEPTHS = (0.0, 1.0, 10.0, 100.0)


def params(s, phi, l_frac, beta, depth):
    l, m = l_frac * s["TL"], s["mass"]
    p_abs = 101325.0 + RHO * G * depth
    kb = (RHO * G) ** 2 * phi * (m / RHO) / p_abs
    kth = m * G * s["dz"] * s["depth"]
    return Params(m=m, J=m * l**2, K=beta * m / l, kb=kb, kth=kth), s["dx"] * s["TL"], l


def threshold(p, rA):
    return U1(rA, rL_on_B0(rA, p), p)


def stable_somewhere(U, rA, p, n=4000):
    rL = np.linspace(rA, 0.0, n + 2)[1:-1]
    return any(max_real_eig(U, rA, r, p) < 0 for r in rL)


def main():
    used = [s for s in species() if s["dx"] < 0 and s["dz"] > 0]
    below_ok = above_ok = 0
    depth_shift, approx_err = 0.0, 0.0
    orders = set()
    for phi, lf, beta in product(PHI, L_FRAC, BETA):
        u = {}
        for s in used:
            p, rA, l = params(s, phi, lf, beta, 1.0)
            U0 = threshold(p, rA)
            u[s["name"]] = U0 / s["TL"]
            below_ok += not stable_somewhere(0.98 * U0, rA, p)
            above_ok += stable_somewhere(1.02 * U0, rA, p)
            kappa = p.kb * l**2 / p.kth
            h = p.kth / (p.m * G)
            approx = np.sqrt(G * h * l * (1 + kappa) / (beta * abs(rA)))
            approx_err = max(approx_err, abs(approx / U0 - 1))
            for d in DEPTHS:
                pd, _, _ = params(s, phi, lf, beta, d)
                depth_shift = max(depth_shift, abs(threshold(pd, rA) / U0 - 1))
        orders.add(tuple(sorted(u, key=u.get)))
    n = len(PHI) * len(L_FRAC) * len(BETA) * len(used)
    out = dict(n_cases=n, below_never_stable=below_ok, above_stable=above_ok,
               depth_shift_max=depth_shift, depths=DEPTHS, approx_err_max=approx_err,
               order=list(orders)[0] if len(orders) == 1 else None, n_orders=len(orders))
    print(out)
    (DATA / "fish_prediction.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
