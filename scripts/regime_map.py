"""Regime map in the (U, d) plane (Fig. 6).

Along each column U the shift d is increased and the final state of one cell starts the next.
Writes data/regime_map.npy with columns U, d, class, pitch amplitude.
"""
from multiprocessing import Pool

import numpy as np

from bladder.attractors import RESTART, classify
from common import DATA, RA, RLB

SPEEDS = np.round(np.arange(1.54, 1.72001, 0.0025), 5)
SHIFTS = np.round(np.arange(-0.06, 0.24001, 0.005), 5)


def column(U):
    x, rows = RESTART, []
    for d in SHIFTS:
        cls, x, amp = classify(U, RA, RLB + d, x, d)
        rows.append((U, d, cls, amp))
    print(f"U = {U:.4f}: " + "".join(str(int(r[2])) for r in rows), flush=True)
    return rows


def main():
    with Pool() as pool:
        cols = pool.map(column, SPEEDS)
    np.save(DATA / "regime_map.npy", np.concatenate([np.array(c) for c in cols]))


if __name__ == "__main__":
    main()
