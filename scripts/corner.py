"""Hopf-pitchfork point: normal-form coefficients, SB near the corner, normal-form lines in (U, d).

Writes data/corner.json, data/sb_corner.npy (U, d_nf, d_numerical), data/nf_lines.npy
(d, U on a4 = 0, U on SB, TR, SH).
"""
import json

import numpy as np

from bladder import U1
from bladder.cycles import normal_form_line, root_scan, symmetry_breaking_d
from bladder.normal_form import corner_coefficients, line_slopes, linear_rates
from common import DATA, RA


def sb_near_corner(c, slopes):
    rows = []
    for eps in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04):
        U = c["U0"] + eps
        mu = lambda d: linear_rates(U, RA, c["rL0"] + d)
        d_nf = root_scan(lambda d: mu(d)[1] - slopes["SB"] * mu(d)[0], 1e-7, 1.0)
        d_num = symmetry_breaking_d(U, RA, c["rL0"], 3 * d_nf if np.isfinite(d_nf) else 0.2)
        rows.append((U, d_nf, d_num))
        print(f"U - U0 = {eps:.4f}: d_SB = {d_nf:.6f} (normal form), {d_num:.6f} (shooting)")
    return np.array(rows)


def lines_in_Ud(c, slopes):
    rows = []
    for d in np.linspace(0.002, 0.2, 60):
        rL = c["rL0"] + d
        Ua4 = U1(RA, rL)
        rows.append([d, Ua4] + [normal_form_line(slopes[k], RA, rL, Ua4 + 1e-7, Ua4 + 0.4)
                                for k in ("SB", "TR", "SH")])
    return np.array(rows)


def main():
    c = corner_coefficients(RA)
    slopes = line_slopes(c)
    X = c["p12"] * c["p21"] / (c["p11"] * c["p22"])
    print(", ".join(f"{k} = {c[k]:+.4f}" for k in ("p11", "p22", "p12", "p21")), f"X = {X:.3f}")
    print(", ".join(f"{k}: mu2 = {v:+.4f} mu1" for k, v in slopes.items()))

    opt = corner_coefficients(-np.sqrt(11.0))
    (DATA / "corner.json").write_text(json.dumps(dict(
        nominal=dict(c, X=X, slopes=slopes),
        optimum=dict(opt, X=opt["p12"] * opt["p21"] / (opt["p11"] * opt["p22"]))), indent=2))
    np.save(DATA / "sb_corner.npy", sb_near_corner(c, slopes))
    np.save(DATA / "nf_lines.npy", lines_in_Ud(c, slopes))


if __name__ == "__main__":
    main()
