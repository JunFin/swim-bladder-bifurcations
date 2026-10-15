"""Between a4 = 0 and the TR line no bounded attractor is found: trajectories from random
initial states escape. Checked just below the numerical TR line at several d."""
import itertools

import numpy as np
from scipy.integrate import solve_ivp

from bladder import P, U1, rhs
from bladder.model import escaped
from common import DATA, RA, RLB

T = 3000.0


def escape_times(U, d, n=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x0 = rng.uniform(-1, 1, 4) * [0.5, 0.2, 0.2, 0.2]
        s = solve_ivp(rhs, (0, T), x0, args=(U, RA, RLB + d, P), method="DOP853",
                      rtol=1e-9, atol=1e-11, events=escaped)
        out.append(s.t[-1] if s.status == 1 else np.inf)
    return np.array(out)


def main():
    tr = np.load(DATA / "tr_line.npy")
    for (d, U_tr), frac in itertools.product(tr[[1, 4, 9, 13]], (0.05, 0.5)):
        U = U_tr - frac * (U_tr - U1(RA, RLB + d))
        t = escape_times(U, d)
        print(f"d = {d:.2f}, U = {U:.5f} (TR at {U_tr:.5f}): escaped {np.isfinite(t).sum()}/{len(t)}, "
              f"latest after {t[np.isfinite(t)].max():.0f} s")


if __name__ == "__main__":
    main()
