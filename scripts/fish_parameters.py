"""Model parameters estimated for fish from the morphology data of Di Santo et al. 2025
(PNAS 122, e2420015122; Dryad doi:10.5061/dryad.9cnp5hqw7, copied to data/disanto2025).

The data give, for 13 species, the longitudinal offset of the centre of buoyancy from the centre of mass
(fraction of total length, negative: centre of mass in front) and the vertical offset (fraction of body
depth, positive: centre of mass below). The model needs the bladder behind the centre of mass and positive
pitch stiffness, so only species with both offsets of the right sign are used. Assumptions:
the bladder arm is approximated by the longitudinal offset of the centre of buoyancy; bladder volume
5-8 % of body volume; pitch radius of gyration 0.2-0.3 TL; lift parameter beta = 0.1 (pelvic fins
of trout) and 1 (reference value); depth 1 m.

Writes data/fish_parameters.json.
"""
import json
from itertools import product

import numpy as np

from bladder import Params, U1, rL_on_B0
from bladder.model import birth_frequency
from common import DATA

RHO, G, PRESSURE = 1000.0, 9.81, 101325.0 + 1000.0 * 9.81
PHI = (0.05, 0.08)
L_FRAC = (0.2, 0.3)
BETA = (0.1, 1.0)


def species():
    lines = (DATA / "disanto2025" / "hovering_morphology.txt").read_text().strip().splitlines()[1:]
    for line in lines:
        c = line.split("\t")
        yield dict(name=c[0].strip(), dx=float(c[1]), dz=float(c[2]), TL=float(c[5]) / 100,
                   depth=float(c[7]) / 100, mass=float(c[9]) / 1000)


def estimate(s, phi, l_frac, beta):
    l, m = l_frac * s["TL"], s["mass"]
    kb = (RHO * G) ** 2 * phi * (m / RHO) / PRESSURE
    kth = m * G * s["dz"] * s["depth"]
    rA = s["dx"] * s["TL"]
    p = Params(m=m, J=m * l**2, K=beta * m / l, kb=kb, kth=kth)
    rLB = rL_on_B0(rA, p)
    return dict(kappa=kb * l**2 / kth, rho_A=rA / l, U0_TL=U1(rA, rLB, p) / s["TL"],
                T0=2 * np.pi / birth_frequency(rA, rLB, p))


def span(rows, key, sel=lambda r: True):
    v = [r[key] for r in rows if sel(r)]
    return [min(v), max(v)]


def main():
    all_species = list(species())
    used = [s for s in all_species if s["dx"] < 0 and s["dz"] > 0]
    out = dict(n_species=len(all_species), n_behind=sum(s["dx"] < 0 for s in all_species),
               n_stable=sum(s["dz"] > 0 for s in all_species), species={})
    for s in used:
        rows = [dict(beta=b, **estimate(s, ph, lf, b)) for ph, lf, b in product(PHI, L_FRAC, BETA)]
        out["species"][s["name"]] = dict(
            h_mm=1e3 * s["dz"] * s["depth"], kappa=span(rows, "kappa"), rho_A=span(rows, "rho_A"),
            U0_TL=span(rows, "U0_TL"), U0_TL_beta1=span(rows, "U0_TL", lambda r: r["beta"] == 1.0),
            U0_TL_beta01=span(rows, "U0_TL", lambda r: r["beta"] == 0.1),
            T0=span(rows, "T0"))
        print(s["name"], {k: np.round(v, 3).tolist() for k, v in out["species"][s["name"]].items()})
    (DATA / "fish_parameters.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
