"""Пузырь сзади / спереди / несжимаемый у золотой рыбки (Di Santo 2025: L=5.88 см, m=3.14 г, h=0.64 мм, |dx|=0.14 L).
Модель статьи (без демпфирования), F_A = sum kb_i (z + r_i sin th). Манёвр: шаг угла плавника delta (погружение)."""
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / "two_chambers"
OUT.mkdir(parents=True, exist_ok=True)
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
g, rho, P = 9.81, 1000.0, 101325 + 1000*9.81
L, m, h = 0.0588, 0.0031428571, 0.00064
ell = 0.25*L; J = m*ell**2; kth = m*g*h
phi = 0.065; Vb = phi*m/rho
kb_tot = (rho*g)**2*Vb/P
K = 1.0*m/ell                     # beta = 1
rA = 0.1403*L
def B(rA_, rL, kb): return (rL-rA_)*(J*kb+kb*m*rL*rA_)+kth*rL*m
rLB = brentq(lambda r: B(-rA, r, kb_tot), -rA+1e-9, -1e-9, xtol=1e-14)
U0 = np.sqrt(kth/(K*(rLB+rA)))
print(f"kb={kb_tot:.3e} N/m  U0={U0:.4f} m/s = {U0/L:.2f} L/s  rLB={rLB/L:.4f} L")
U = 2.0*U0
U1 = lambda rL: np.sqrt(kth/(K*(rL+rA)))
rL_lo = -rA + kth/(K*U**2)        # a4=0 at this U
rL = 0.5*(rL_lo + rLB)            # middle of the stable interval for the rear bladder
print(f"U={U:.4f} m/s={U/L:.2f} L/s; rL={rL/L:.4f} L (interval {rL_lo/L:.4f}..{rLB/L:.4f})")
def rhs(t, x, cham, delta):
    z, vz, th, om = x
    gam = np.arctan2(vz, U); al = th - gam + delta
    FA = sum(kb*(z + r*np.sin(th)) for kb, r in cham)
    MA = sum(r*np.cos(th)*kb*(z + r*np.sin(th)) for kb, r in cham)
    FL = K*U**2*al
    return [vz, (FA + FL*np.cos(gam))/m, om, (MA + rL*FL*np.cos(al) - kth*th)/J]
def jac(cham):
    f = lambda x: np.array(rhs(0, x, cham, 0.0)); e = 1e-7
    return np.column_stack([(f(np.eye(4)[i]*e) - f(-np.eye(4)[i]*e))/(2*e) for i in range(4)])
configs = {
  "сзади": [(kb_tot, -rA)],
  "спереди": [(kb_tot, +rA)],
  "несжимаемый": [(0.0, 0.0)],
  "две камеры (данио: перед. податлива)": [(kb_tot*0.97, +0.10*L), (kb_tot*0.03, -0.12*L)],
}
out = {"U": U, "U_Lps": U/L, "rL_L": rL/L, "kb": kb_tot, "cases": {}}
delta = np.radians(-2.0)
for name, cham in configs.items():
    ev = np.linalg.eigvals(jac(cham)); lam = float(max(ev.real))
    def hit(t, x, *a): return x[0] + 1.0       # 1 m deeper
    hit.terminal = True
    s = solve_ivp(rhs, (0, 3000), [0, 0, 0, 0], args=(cham, delta), events=hit, max_step=0.05, rtol=1e-8, atol=1e-11, dense_output=True)
    t1 = float(s.t_events[0][0]) if len(s.t_events[0]) else None
    zs = {T: float(s.sol(T)[0]) if T <= s.t[-1] else None for T in (10, 30, 60)}
    out["cases"][name] = dict(max_re_eig=lam, t_1m=t1, z_at=zs, z_end=float(s.y[0, -1]), t_end=float(s.t[-1]))
    print(f"{name:40s} max Re λ={lam:+.4f} 1/s  t(1 м)={t1}  z(10,30,60 с)={[None if v is None else round(v,3) for v in zs.values()]}")
json.dump(out, open(OUT / "dive.json", "w"), ensure_ascii=False, indent=1)
