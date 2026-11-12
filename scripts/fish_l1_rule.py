"""Rule for the sign of l1 on B = 0 over the parameter range of fish: l1 > 0 when |rho_A| > beta (U/U0)^2.

m = J = 1 (l = 1), kb = 1, kth = 1/kappa, K = beta (as in fish_l1.py).
1. Check of the rule on a grid inside the ranges of kappa and rho_A of the six species
   (Extended Data Table 2), beta = 0.07 ... 1, U/U0 = 1, 1.5, 2, 3.
2. Boundary |rho_A| at which l1 changes sign (bisection), in units of beta (U/U0)^2, for kappa up to 0.05.
Writes data/fish_l1_rule.json.
"""
import json

import numpy as np

from bladder import Params, U1, rL_on_B0
from bladder.model import vector_field
from bladder.normal_form import first_lyapunov
from common import DATA

SPECIES = {  # kappa range, rho_A range (Extended Data Table 2)
    "Atherina presbyter": ((0.012, 0.044), (-0.11, -0.077)),
    "Gasterosteus aculeatus": ((0.00065, 0.0023), (-1.05, -0.7)),
    "Carassius auratus": ((0.00095, 0.0034), (-0.7, -0.47)),
    "Dermogenys pusilla": ((0.00097, 0.0035), (-0.41, -0.27)),
    "Poecilia latipinna": ((0.00019, 0.00069), (-0.75, -0.5)),
    "Trigonostigma heteromorpha": ((0.00028, 0.00099), (-0.01, -0.0067)),
}
BETA = (0.07, 0.1, 0.17, 0.2, 0.3, 0.5, 1.0)
SPEED = (1.0, 1.5, 2.0, 3.0)


def l1(beta, rho, kappa, s):
    p = Params(K=beta, kb=1.0, kth=1.0 / kappa)
    rL = rL_on_B0(rho, p)
    U0 = U1(rho, rL, p)
    r = first_lyapunov(vector_field(max(s, 1.0005) * U0, rho, rL, p))
    return np.nan if r is None else r[0]


def boundary(beta, s, kappa):
    t = beta * s * s
    lo, hi = 0.3 * t, min(3 * t, 3.0)
    if not (l1(beta, -lo, kappa, s) < 0 < l1(beta, -hi, kappa, s)):
        return None
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if l1(beta, -mid, kappa, s) > 0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi) / t)


def main():
    total, wrong = 0, []
    for name, (K, R) in SPECIES.items():
        for kappa in np.geomspace(*K, 6):
            for rho in np.linspace(*R, 7):
                for beta in BETA:
                    for s in SPEED:
                        v = l1(beta, rho, kappa, s)
                        total += 1
                        if (v > 0) != (abs(rho) > beta * s * s):
                            wrong.append((name, kappa, rho, beta, s, v))
    print(f"rule checked at {total} points, disagreements: {len(wrong)}")
    bnd = {}
    for kappa in (0.0002, 0.003, 0.01, 0.03, 0.044, 0.05):
        for beta in (0.1, 0.2, 0.3, 1.0):
            for s in (1.0, 2.0):
                bnd[f"kappa={kappa} beta={beta} U/U0={s}"] = boundary(beta, s, kappa)
    for k, v in bnd.items():
        print(k, "boundary / (beta (U/U0)^2) =", "outside |rho_A| <= 3" if v is None else round(v, 3))
    json.dump({"points": total, "disagreements": wrong, "boundary_ratio": bnd},
              open(DATA / "fish_l1_rule.json", "w"), indent=1)


if __name__ == "__main__":
    main()
