"""SB and TR lines of the full system in the window of Fig. 6.

SB: at fixed U, the d where the symmetric cycle loses stability.
TR: at fixed d, the U where the asymmetric cycle loses stability through a complex pair.
Writes data/sb_line.npy (U, d) and data/tr_line.npy (d, U).
"""
from multiprocessing import Pool

import numpy as np

from bladder import U1
from bladder.cycles import symmetry_breaking_d, torus_speed
from common import DATA, RA, RLB

SPEEDS = np.round(np.arange(1.580, 1.7201, 0.005), 4)
SHIFTS = np.round(np.arange(0.01, 0.1401, 0.01), 3)

# The asymmetric band is narrow near the corner, so the descent starts closer to it.
U_START = {0.01: 1.5805, 0.02: 1.582, 0.03: 1.583, 0.04: 1.584, 0.05: 1.585, 0.14: 1.580}


def sb(U):
    return U, symmetry_breaking_d(U, RA, RLB, 0.26)


def tr(d):
    U_start = U_START.get(float(d), U1(RA, RLB + d) + 0.045)
    return d, torus_speed(d, RA, RLB, U_start)


def main():
    with Pool() as pool:
        np.save(DATA / "sb_line.npy", np.array(pool.map(sb, SPEEDS)))
        np.save(DATA / "tr_line.npy", np.array(pool.map(tr, SHIFTS)))


if __name__ == "__main__":
    main()
