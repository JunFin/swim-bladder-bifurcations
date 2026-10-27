"""Hopf-pitchfork point for fish-like geometry (kappa = 0.002, beta = 1, rho_A = -0.5) versus the
vehicle-like reference set of Table 1.

The time unit is chosen so that the period of the emerging cycle is close to that of the reference set.
Writes data/fish_corner.json.
"""
import json

import numpy as np

from bladder import P as P_REF, Params, U1, rL_on_B0
from bladder.attractors import RESTART, classify
from bladder.cycles import symmetry_breaking_d
from bladder.model import birth_frequency
from bladder.normal_form import corner_coefficients, hopf_amplitude_coefficients
from common import DATA, RA as RA_REF

KAPPA, BETA, RHO_A, KTH = 0.002, 1.0, -0.5, 0.4
P_FISH = Params(K=BETA, kb=KAPPA * KTH, kth=KTH)
SPEEDS = (1.05, 1.3, 2.0)
SHIFTS = (0.0002, 0.0004, 0.001, 0.002, 0.004, 0.01, 0.03, 0.1)


def invariant(c):
    return c["p12"] * c["p21"] / (c["p11"] * c["p22"])


def geometry(rA, p):
    rLB = rL_on_B0(rA, p)
    U0 = U1(rA, rLB, p)
    c = corner_coefficients(rA, p)
    width = rLB - rA
    out = dict(rLB=rLB, U0=U0, T0=2 * np.pi / birth_frequency(rA, rLB, p), X=invariant(c),
               coefficients={k: c[k] for k in ("p11", "p22", "p12", "p21")}, width=width, speeds={})
    for f in SPEEDS:
        k_th = hopf_amplitude_coefficients(f * U0, rA, p)[0]
        out["speeds"][str(f)] = dict(c_theta=k_th, d10_over_width=np.radians(10) ** 2 / k_th / width)
    return out


def escape_scan(rA, p, U0):
    res = {}
    for f in SPEEDS:
        U, rLB = f * U0, rL_on_B0(rA, p)
        d_sb = symmetry_breaking_d(U, rA, rLB, 0.01, p)
        x, classes = RESTART, []
        for d in SHIFTS:
            cls, x, _ = classify(U, rA, rLB + d, x, d, p)
            classes.append(int(cls))
        res[str(f)] = dict(d_sb=d_sb, shifts=list(SHIFTS), classes=classes)
        print(f"U = {f} U0: SB at d = {d_sb:.5f}, classes {classes}", flush=True)
    return res


def main():
    fish = geometry(RHO_A, P_FISH)
    ref = geometry(RA_REF, P_REF)
    print(f"X: fish {fish['X']:.3f}, reference {ref['X']:.3f}")
    fish["scan"] = escape_scan(RHO_A, P_FISH, fish["U0"])
    (DATA / "fish_corner.json").write_text(json.dumps(dict(fish=fish, reference=ref), indent=1))


if __name__ == "__main__":
    main()
