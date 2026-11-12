"""Data of Fig. 5 (two-chamber bladder, goldfish of Table 1): lambda(rbar) at U = 1.5, 2, 3 U0 and the
response to a fin step of -2 deg for posterior compliance, an incompressible bladder and the zebrafish
partition. Writes data/two_chambers/fig5.json."""
import numpy as np, json
from pathlib import Path
from scipy.integrate import solve_ivp
exec(open(Path(__file__).resolve().with_name("dive.py"), encoding="utf-8").read().split("configs = {")[0])
def jacU(cham, UU):
    def f(x):
        z, vz, th, om = x; gam = np.arctan2(vz, UU); al = th - gam
        FA = sum(kb*(z + r*np.sin(th)) for kb, r in cham); MA = sum(r*np.cos(th)*kb*(z + r*np.sin(th)) for kb, r in cham)
        FL = K*UU**2*al
        return np.array([vz, (FA + FL*np.cos(gam))/m, om, (MA + rL*FL*np.cos(al) - kth*th)/J])
    e = 1e-7; return np.column_stack([(f(np.eye(4)[i]*e) - f(-np.eye(4)[i]*e))/(2*e) for i in range(4)])
fs = np.linspace(0, 1, 201); rb = (0.10*fs - 0.12*(1-fs))   # in L
out = {"rb": rb.tolist(), "U0_Lps": U0/L, "rL_L": rL/L}
for Uf in (1.5, 2.0, 3.0):
    lam = [max(np.linalg.eigvals(jacU([(kb_tot*f, 0.10*L), (kb_tot*(1-f), -0.12*L)], Uf*U0)).real) for f in fs]
    out[f"lam_{Uf}"] = [float(v) for v in lam]
cases = {"posterior": [(kb_tot*0.0, 0.10*L), (kb_tot, -0.12*L)], "incompressible": [(0.0, 0.0)],
         "anterior": [(kb_tot*0.97, 0.10*L), (kb_tot*0.03, -0.12*L)]}
t = np.linspace(0, 150, 601); out["t"] = t.tolist()
for k, ch in cases.items():
    s = solve_ivp(rhs, (0, 150), [0,0,0,0], args=(ch, np.radians(-2.0)), max_step=0.1, rtol=1e-8, atol=1e-11, dense_output=True)
    out["z_"+k] = (s.sol(t)[0]).tolist(); out["th_"+k] = np.degrees(s.sol(t)[2]).tolist()
json.dump(out, open(OUT / "fig5.json", "w"))
print("U0", U0/L, "L/s; U", U/L, "rL", rL/L)
for Uf in (1.5,2.0,3.0):
    l=np.array(out[f"lam_{Uf}"]); i=np.where(np.diff(np.sign(l)))[0]
    print(Uf, "lam(f=0)=%.4f lam(zebra)=%.4f"%(l[0], l[int(0.97*200)]), "zero at rb=", [round(rb[j],4) for j in i])
for k in cases: z=np.array(out["z_"+k]); print(k, "z(60)=%.2f z(150)=%.2f"%(z[240], z[-1]), "th(60)=%.2f"%out["th_"+k][240])
