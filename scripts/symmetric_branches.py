"""Symmetric cycle born at B = 0: amplitude near onset and thresholds along the branch.

Writes data/symmetric_branches.json and data/branch_U2.npy (Fig. 5).
"""
import json

import numpy as np

from bladder.cycles import SymmetricBranch, symmetric_cycle
from bladder.normal_form import hopf_amplitude_coefficients, hopf_seed
from common import DATA, RA, RLB

SPEEDS = (1.6, 1.7, 2.0)
D_MAX = -RLB - 0.02
DS = np.round(np.r_[[0.002, 0.005, 0.01, 0.02, 0.035], np.arange(0.05, D_MAX, 0.025)], 4)


def amplitude_check(U=2.0, d=1e-3):
    k_th, k_z = hopf_amplitude_coefficients(U, RA)
    cyc = symmetric_cycle(U, RA, RLB + d, hopf_seed(U, RA, d))
    return dict(U=U, d=d, kappa_th_nf=k_th, kappa_th=cyc["amp_th"]**2 / d,
                kappa_z_nf=k_z, kappa_z=cyc["amp_z"]**2 / d)


def thresholds(U):
    br = SymmetricBranch(U, RA, RLB)
    branch = br.trace(DS)
    out = dict(U=U,
               d_alpha15=br.crossing(branch, lambda i: i["alpha_max"], np.radians(15)),
               d_sb=br.crossing(branch, lambda i: i["hmax"], 1.0))
    return out, branch


def main():
    res = dict(amplitude=amplitude_check(), thresholds=[])
    for U in SPEEDS:
        row, branch = thresholds(U)
        res["thresholds"].append(row)
        print(f"U = {U}: alpha = 15 deg at d = {row['d_alpha15']:.4f}, SB at d = {row['d_sb']:.4f}")
        if U == 2.0:
            np.save(DATA / "branch_U2.npy",
                    np.array([(i["d"], i["T"], i["amp_th"], i["amp_z"], i["alpha_max"], i["hmax"])
                              for i, _ in branch]))
    (DATA / "symmetric_branches.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
