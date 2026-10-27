"""Sign of l1 on B = 0 over the parameter range of fish (rho_A = -0.01 ... -1.05).

m = J = 1 (l = 1), kb = 1, kth = 1/kappa, K = beta. Points: the Hopf-pitchfork point and U = 2 U0.
Writes data/fish_l1.npy with columns beta, rho_A, kappa, l1 at U0, l1 at 2 U0.
"""
import numpy as np

from bladder import Params, U1, rL_on_B0
from bladder.model import vector_field
from bladder.normal_form import first_lyapunov
from common import DATA

KAPPA = np.geomspace(0.0003, 1.0, 22)
BETA = (0.1, 0.2, 0.3, 0.5, 1.0, 2.6)
RHO_A = (-0.01, -0.08, -0.3, -0.5, -0.7, -1.05)


def l1(U, rA, rL, p):
    r = first_lyapunov(vector_field(U, rA, rL, p))
    return np.nan if r is None else r[0]


def main():
    rows = []
    for beta in BETA:
        for rho in RHO_A:
            for kappa in KAPPA:
                p = Params(K=beta, kb=1.0, kth=1.0 / kappa)
                rL = rL_on_B0(rho, p)
                U0 = U1(rho, rL, p)
                rows.append((beta, rho, kappa, l1(1.0005 * U0, rho, rL, p), l1(2 * U0, rho, rL, p)))
    A = np.array(rows)
    np.save(DATA / "fish_l1.npy", A)
    for beta in BETA:
        s = A[A[:, 0] == beta]
        print(f"beta = {beta}: l1 > 0 at U0 in {int((s[:, 3] > 0).sum())} of {len(s)} points, "
              f"at 2U0 in {int((s[:, 4] > 0).sum())}")


if __name__ == "__main__":
    main()
