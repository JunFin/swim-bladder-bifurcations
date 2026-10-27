"""LaTeX macros for every number quoted in the paper: paper/numbers.tex.

Cheap quantities are recomputed; the rest are read from data/ written by
symmetric_branches.py, corner.py, hopf_sign.py and sweeps.py.
"""
import json

import numpy as np

from bladder import P, U1
from bladder.model import birth_frequency
from common import DATA, RA, RLB, ROOT

SPEED_NAMES = {1.6: "Usixteen", 1.7: "Useventeen", 2.0: "Utwenty"}


def fmt(v, digits):
    s = f"{v:.{digits}g}"
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}\\times10^{{{int(e)}}}"
    return s


def collect():
    n = {}
    for k in ("m", "J", "K", "kb", "kth"):
        n[k] = (getattr(P, k), 2)
    n["rA"] = (RA, 2)
    n["rLBzero"] = (RLB, 4)
    n["Ucorner"] = (U1(RA, RLB), 4)
    n["TBirth"] = (2 * np.pi / birth_frequency(RA, RLB), 4)
    n["rAopt"] = (-np.sqrt(11.0), 4)
    n["rLopt"] = (-1.0, 2)
    n["Umin"] = (U1(-np.sqrt(11.0), -1.0), 4)

    corner = json.loads((DATA / "corner.json").read_text())["nominal"]
    for key, name in (("p11", "pOneOne"), ("p22", "pTwoTwo"), ("p12", "pOneTwo"), ("p21", "pTwoOne")):
        n[name] = (corner[key], 4)

    sb = json.loads((DATA / "symmetric_branches.json").read_text())
    n["cthNFUtwo"] = (sb["amplitude"]["kappa_th_nf"], 4)
    n["cthNumUtwo"] = (sb["amplitude"]["kappa_th"], 4)
    for row in sb["thresholds"]:
        name = SPEED_NAMES[row["U"]]
        n["dSB" + name] = (row["d_sb"], 3)
        n["dAlpha" + name] = (row["d_alpha15"], 3)

    branch = np.load(DATA / "branch_U2.npy")
    d_sb = next(r["d_sb"] for r in sb["thresholds"] if r["U"] == 2.0)
    n["alphaSBUtwenty"] = (np.degrees(np.interp(d_sb, branch[:, 0], branch[:, 4])), 2)

    l1 = json.loads((DATA / "hopf_sign.json").read_text())["map"]
    n["kappaCornerBetaLow"] = (l1["0.3"]["corner"]["kappa_max"], 2)
    n["kappaDoubleBetaLow"] = (l1["0.3"]["double"]["kappa_max"], 2)
    n["kappaCornerBetaOne"] = (l1["1.0"]["corner"]["kappa_max"], 2)

    fish = json.loads((DATA / "fish_parameters.json").read_text())
    n["fishNspecies"], n["fishNbehind"], n["fishNstable"] = (fish["n_species"], 2), (fish["n_behind"], 2), (fish["n_stable"], 2)
    sp = fish["species"]
    n["fishNused"] = (len(sp), 1)
    n["fishNunstable"] = (fish["n_species"] - fish["n_stable"], 1)
    passive = {k: v for k, v in sp.items() if v["rho_A"][1] < -0.05}
    lo = lambda key, d=passive: min(v[key][0] for v in d.values())
    hi = lambda key, d=passive: max(v[key][1] for v in d.values())
    n["fishKappaMin"], n["fishKappaMax"] = (lo("kappa", sp), 1), (hi("kappa", sp), 2)
    n["fishRhoMin"], n["fishRhoMax"] = (lo("rho_A"), 3), (hi("rho_A"), 1)
    n["fishUMinOne"], n["fishUMaxOne"] = (lo("U0_TL_beta1"), 2), (hi("U0_TL_beta1"), 2)
    n["fishUMin"], n["fishUMax"] = (lo("U0_TL"), 1), (hi("U0_TL"), 2)
    n["fishUMinTenth"], n["fishUMaxTenth"] = (lo("U0_TL_beta01"), 2), (hi("U0_TL_beta01"), 2)
    n["fishTMin"], n["fishTMax"] = (lo("T0"), 1), (hi("T0"), 1)
    slow = {k: v for k, v in sp.items() if k not in passive}
    n["fishURasboraMin"] = (lo("U0_TL_beta1", slow), 2)
    n["fishRhoRasbora"] = (hi("rho_A", slow), 1)

    fp = json.loads((DATA / "fish_prediction.json").read_text())
    assert fp["below_never_stable"] == fp["above_stable"] == fp["n_cases"] and fp["n_orders"] == 1
    n["fishDepthShiftPct"] = (100 * fp["depth_shift_max"], 1)
    n["fishDepthMax"] = (max(fp["depths"]), 3)
    n["fishApproxErr"] = (fp["approx_err_max"], 1)
    n["fishNcases"] = (fp["n_cases"], 2)
    n["fishPressureRatio"] = ((101325.0 + 1000.0 * 9.81 * max(fp["depths"])) / 101325.0, 2)
    n["fishNexcluded"] = (fish["n_species"] - len(sp), 1)
    fc = json.loads((DATA / "fish_corner.json").read_text())
    n["Xfish"], n["Xref"] = (fc["fish"]["X"], 2), (fc["reference"]["X"], 2)
    ratio = [fc["fish"]["speeds"][f]["c_theta"] / fc["reference"]["speeds"][f]["c_theta"] for f in fc["fish"]["speeds"]]
    n["cthRatioOrder"] = (int(np.floor(np.log10(min(ratio)))), 1)
    runs = [json.loads(s) for s in (DATA / "fish_escape.jsonl").read_text().splitlines() if s.strip()]
    escaped = lambda r: any(x["escaped"] for x in r["runs"])
    speeds = {r["U_over_U0"] for r in runs}
    n["fishEscBounded"] = (min(max(r["d_over_W_pct"] for r in runs if r["U_over_U0"] == s and not escaped(r))
                               for s in speeds), 1)
    n["fishEscOut"] = (max(min(r["d_over_W_pct"] for r in runs if r["U_over_U0"] == s and escaped(r))
                           for s in speeds), 1)
    times = [x["t"] for r in runs for x in r["runs"] if x["escaped"]]
    n["fishEscTmin"], n["fishEscTmax"] = (min(times), 1), (max(times), 1)
    n["fishEscT"] = (runs[0]["T"], 1)

    n["Ufixcascade"] = (float(np.load(DATA / "sweep_d_up.npz")["fixed"]), 2)
    n["dfixU"] = (float(np.load(DATA / "sweep_U_down.npz")["fixed"]), 2)
    return n


def fish_table():
    """Rows of Table 2 in the order of the predicted threshold speed: species, bladder offset, h, kappa,
    rho_A, U0/TL at beta = 1 and over beta = 0.1-2.6."""
    sp = json.loads((DATA / "fish_parameters.json").read_text())["species"]
    dx = {}
    for line in (DATA / "disanto2025" / "hovering_morphology.txt").read_text().strip().splitlines()[1:]:
        c = line.split("\t")
        dx[c[0].strip()] = -100 * float(c[1])
    r = lambda lo, hi, d: f"{fmt(lo, d)}--{fmt(hi, d)}"
    rho = lambda v: fmt(v, 3 if abs(v) >= 1 else 2)
    rows = []
    for name, v in sorted(sp.items(), key=lambda kv: kv[1]["U0_TL_beta1"][0]):
        rows.append(f"\\emph{{{name}}} & {fmt(dx[name], 2)} & {v['h_mm']:.2f} & {r(*v['kappa'], 2)} & "
                    f"${rho(v['rho_A'][0])}$ to ${rho(v['rho_A'][1])}$ & "
                    f"{r(*v['U0_TL_beta1'], 2)} & {r(*v['U0_TL_beta01'], 2)} \\\\")
    head = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "Species & $\\Delta x/L$ (\\%) & $h$ (mm) & $\\kappa$ & $\\rho_A$ & "
            "$U_0/L$ ($\\beta=1$, s$^{-1}$) & $U_0/L$ ($\\beta=0.1$, s$^{-1}$) \\\\\n"
            "\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    (ROOT / "paper" / "fish_table.tex").write_text(fish_table())
    lines = [f"\\newcommand{{\\numgr{k}}}{{{fmt(v, d)}}}" for k, (v, d) in sorted(collect().items())]
    out = ROOT / "paper" / "numbers.tex"
    out.write_text("% Generated by scripts/paper_numbers.py\n" + "\n".join(lines) + "\n")
    print(f"{len(lines)} macros -> {out}")


if __name__ == "__main__":
    main()
