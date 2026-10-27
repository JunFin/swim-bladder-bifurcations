"""Where the equilibrium can be stable: scan of the bladder and lift arms r_A, r_L in [-3, 3] l.

Writes data/stability_quadrants.json: number of stable grid points with r_A >= 0 or r_L >= 0.
"""
import json

import numpy as np

from bladder import P, Params, max_real_eig
from common import DATA

ARMS = np.linspace(-3, 3, 241)
SETS = {"reference": P, "fish": Params(K=1.0, kb=0.0008, kth=0.4)}


def main():
    out = {}
    for name, p in SETS.items():
        for U in (0.5, 1.0, 2.0, 5.0, 10.0):
            stable = [(a, b) for a in ARMS for b in ARMS if max_real_eig(U, a, b, p) < 0]
            forward = sum(1 for a, b in stable if a >= 0 or b >= 0)
            out[f"{name} U={U}"] = dict(stable=len(stable), with_forward_arm=forward)
            print(name, U, len(stable), forward)
    (DATA / "stability_quadrants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
