"""Сдвоенный пузырь на оси. Камеры (kb_i, r_i). Проверки:
A) линейно: две камеры = одна камера в центре податливости rbar = Σkb_i r_i / kb
   с жёсткостью по тангажу kθ_eff = kθ − kb·σ², σ² = Σkb_i (r_i − rbar)² / kb;
B) карта (f, U): max Re λ; C) погружение и подъём на 1 м; D) компромисс."""
import json, numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
exec(open(Path(__file__).resolve().with_name("dive.py"), encoding="utf-8").read().split("configs = {")[0])
r1, r2 = 0.10*L, -0.12*L          # передняя и задняя камеры (данио: ЦМ у перетяжки)
def cham(f): return [(kb_tot*f, r1), (kb_tot*(1-f), r2)]
def jac_c(ch, UU=None, kth_=None):
    UU = U if UU is None else UU; kt = kth if kth_ is None else kth_
    def f_(x):
        z, vz, th, om = x; gam = np.arctan2(vz, UU); al = th - gam
        FA = sum(kb*(z + r*np.sin(th)) for kb, r in ch); MA = sum(r*np.cos(th)*kb*(z + r*np.sin(th)) for kb, r in ch)
        FL = K*UU**2*al
        return np.array([vz, (FA + FL*np.cos(gam))/m, om, (MA + rL*FL*np.cos(al) - kt*th)/J])
    e = 1e-8; return np.column_stack([(f_(np.eye(4)[i]*e) - f_(-np.eye(4)[i]*e))/(2*e) for i in range(4)])
out = {}
# A
errs = []
for f in (0.0, 0.3, 0.55, 0.8, 1.0):
    ch = cham(f); kb = sum(k for k, _ in ch); rbar = sum(k*r for k, r in ch)/kb
    s2 = sum(k*(r - rbar)**2 for k, r in ch)/kb
    e2 = np.sort_complex(np.linalg.eigvals(jac_c(ch)))
    e1 = np.sort_complex(np.linalg.eigvals(jac_c([(kb, rbar)], kth_=kth - kb*s2)))
    errs.append(float(np.max(np.abs(e2 - e1))))
print("A: max |Δλ| =", max(errs), " kb·σ²/kθ (f=0.5) =", kb_tot*0.25*(r1-r2)**2/kth)
out["A_max_err"] = max(errs)
fstar = -r2/(r1 - r2); print("f* (центр податливости над ЦМ) =", round(fstar, 4))
out["fstar"] = fstar
# B: карта
fs = np.linspace(0, 1, 101); Us = np.linspace(0.5, 6, 111)*L
M = np.array([[max(np.linalg.eigvals(jac_c(cham(f), UU)).real) for f in fs] for UU in Us])
np.savez(OUT / "map.npz", fs=fs, Us=Us/L, M=M)
# скорость потери устойчивости при U = 3.6 L/s
lam = [max(np.linalg.eigvals(jac_c(cham(f))).real) for f in fs]
out["lam_U"] = dict(f=fs.tolist(), lam=[float(v) for v in lam])
# C/D: погружение и подъём, delta = ∓2°
def t1m(ch, delta, sign):
    def hit(t, x, *a): return sign*x[0] - 1.0
    hit.terminal = True
    s = solve_ivp(rhs, (0, 5000), [0,0,0,0], args=(ch, delta), events=hit, max_step=0.1, rtol=1e-8, atol=1e-11)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else float("inf")
fd = np.linspace(0, 1, 21)
out["dive"] = [t1m(cham(f), np.radians(-2), -1) for f in fd]
out["climb"] = [t1m(cham(f), np.radians(+2), +1) for f in fd]
out["fd"] = fd.tolist()
for f, a, b in zip(fd, out["dive"], out["climb"]): print(f"f={f:.2f} вниз {a:7.1f} с, вверх {b:7.1f} с, λ={max(np.linalg.eigvals(jac_c(cham(f))).real):+.4f}")
json.dump(out, open(OUT / "two.json", "w"), indent=1)
