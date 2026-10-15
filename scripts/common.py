from pathlib import Path

from bladder import P, U1, rL_on_B0

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

RA = -4.6
RLB = rL_on_B0(RA, P)
U0 = U1(RA, RLB, P)
