"""One-parameter sweeps by continuation (Fig. 7).

    python scripts/sweeps.py d_up | d_down | d_zoom | U_down | U_up

Each point starts from the final state of the previous one. Writes data/sweep_<name>.npz:
rows = (value, escaped, theta amplitude, z amplitude, <z>, <theta>, max |alpha|, lambda_1),
pmax / pmin = (value, theta) at pitch maxima / minima.
"""
import sys

import numpy as np

from bladder.attractors import attractor, largest_lyapunov
from common import DATA, RA, RLB

X0 = np.array([0.0, 0.0, 0.05, 0.0])


def sweep(name, values, fixed, var, x0, with_lyapunov):
    rows, pmax, pmin = [], [], []
    x = np.array(x0, float)
    for v in values:
        U, d = (fixed, v) if var == "d" else (v, fixed)
        a = attractor(U, RA, RLB + d, x)
        if a is None:
            rows.append((v, 1) + (np.nan,) * 6)
            x = X0.copy()
            continue
        x = a["x_end"]
        lam = largest_lyapunov(U, RA, RLB + d, x) if with_lyapunov else np.nan
        rows.append((v, 0, a["th_amp"], a["z_amp"], a["zmean"], a["thmean"], a["alpha_max"], lam))
        pmax += [(v, t) for t in a["th_max"]]
        pmin += [(v, t) for t in a["th_min"]]
    np.savez(DATA / f"sweep_{name}.npz", rows=np.array(rows), pmax=np.array(pmax),
             pmin=np.array(pmin), var=var, fixed=fixed)


def settled(U, d, T=3000.0):
    return attractor(U, RA, RLB + d, X0, T_transient=T)


def main(job):
    if job == "d_up":
        sweep(job, np.round(np.arange(0.0005, 0.4001, 0.0005), 4), 1.6, "d", X0, True)
    elif job == "d_down":
        x = settled(1.6, 0.40)["x_end"]
        sweep(job, np.round(np.arange(0.40, 0.0004, -0.0005), 4), 1.6, "d", x, False)
    elif job == "d_zoom":
        x = X0.copy()
        for d in np.arange(0.01, 0.13, 0.01):
            x = attractor(1.6, RA, RLB + d, x)["x_end"]
        sweep(job, np.round(np.arange(0.1300, 0.20001, 0.0001), 4), 1.6, "d", x, True)
    elif job == "U_down":
        x = settled(2.6, 0.20)["x_end"]
        sweep(job, np.round(np.arange(2.60, 1.4999, -0.002), 3), 0.20, "U", x, True)
    elif job == "U_up":
        a = settled(1.58, 0.20)
        sweep(job, np.round(np.arange(1.50, 2.6001, 0.002), 3), 0.20, "U",
              a["x_end"] if a else X0, False)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
