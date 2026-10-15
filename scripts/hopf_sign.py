"""Sign of the first Lyapunov value on B = 0.

1. Grid r_A in [-12, -2] m, U up to 8 m/s for the parameters of Table 1.
2. Map over kappa = kb J / (m kth) and rho_A = r_A / l for beta = K l / m in (0.3, 1, 3),
   at the Hopf-pitchfork point (U = U0) and at U = 2 U0.

Writes data/hopf_sign.json and data/l1_map.npz.
"""
import json

import numpy as np

from bladder import P, Params, U1, a4, rL_on_B0
from bladder.model import vector_field
from bladder.normal_form import first_lyapunov
from common import DATA

KAPPAS = np.geomspace(0.01, 3.0, 28)
RHO_A = -np.linspace(1.0, 12.0, 28)
BETAS = (0.3, 1.0, 3.0)


def l1_value(U, rA, rL, p):
    r = first_lyapunov(vector_field(U, rA, rL, p))
    return np.nan if r is None else r[0]


def table_grid():
    values = []
    for rA in np.linspace(-2.0, -12.0, 30):
        rL = rL_on_B0(rA)
        U_low = np.sqrt(-P.kb * rA / P.K)
        for U in np.linspace(1.02 * U_low, 8.0, 40):
            if a4(U, rA, rL) > 0:
                values.append(l1_value(U, rA, rL, P))
    v = np.array(values)
    v = v[np.isfinite(v)]
    return dict(points=len(v), positive=int((v > 0).sum()), max_l1=float(v.max()))


def similar_params(kappa, beta, kb=1.0):
    # m = J = 1, so l = 1 and r_A = rho_A
    return Params(K=beta, kb=kb, kth=kb / kappa)


def kappa_map(beta):
    corner = np.full((len(KAPPAS), len(RHO_A)), np.nan)
    double = corner.copy()
    for i, kappa in enumerate(KAPPAS):
        p = similar_params(kappa, beta)
        for j, rA in enumerate(RHO_A):
            rL = rL_on_B0(rA, p)
            U0 = U1(rA, rL, p)
            corner[i, j] = l1_value(1.0005 * U0, rA, rL, p)
            double[i, j] = l1_value(2 * U0, rA, rL, p)
    return corner, double


def summary(M):
    pos = M > 0
    out = dict(points=int(np.isfinite(M).sum()), positive=int(pos.sum()))
    if pos.any():
        i, j = np.where(pos)
        out.update(kappa_max=float(KAPPAS[i].max()), rho_max=float(-RHO_A[j].min()))
    return out


def main():
    res = dict(table=table_grid(), map={})
    print("Table 1:", res["table"])
    maps = {}
    for beta in BETAS:
        corner, double = kappa_map(beta)
        maps[f"beta{beta}"] = np.stack([corner, double])
        res["map"][str(beta)] = dict(corner=summary(corner), double=summary(double))
        print(f"beta = {beta}:", res["map"][str(beta)])
    np.savez(DATA / "l1_map.npz", kappa=KAPPAS, rho_A=RHO_A, **maps)
    (DATA / "hopf_sign.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
