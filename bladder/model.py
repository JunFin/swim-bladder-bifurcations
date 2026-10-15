from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class Params:
    m: float = 1.0
    J: float = 1.0
    K: float = 1.0
    kb: float = 0.5
    kth: float = 5.0


P = Params()


def rhs(t, x, U, rA, rL, p=P):
    z, vz, th, om = x
    gamma = np.arctan2(vz, U)
    alpha = th - gamma
    FA = p.kb * (z + rA * np.sin(th))
    FL = p.K * U**2 * alpha
    return [vz,
            (FA + FL * np.cos(gamma)) / p.m,
            om,
            (rA * np.cos(th) * FA + rL * FL * np.cos(alpha) - p.kth * th) / p.J]


def vector_field(U, rA, rL, p=P):
    return lambda x: np.asarray(rhs(0.0, x, U, rA, rL, p), float)


def flow(x0, T, U, rA, rL, p=P, rtol=1e-11, atol=1e-13):
    sol = solve_ivp(rhs, (0, T), np.asarray(x0, float), args=(U, rA, rL, p),
                    rtol=rtol, atol=atol, method="DOP853")
    return sol.y[:, -1]


def escaped(t, x, *args):
    # |z| > 50 m or |theta| > pi/2
    return 50.0 - max(abs(x[0]), 50.0 * abs(x[2]) / (np.pi / 2))


escaped.terminal = True


def jacobian_at_rest(U, rA, rL, p=P):
    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [p.kb / p.m, -p.K * U / p.m, (p.kb * rA + p.K * U**2) / p.m, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [rA * p.kb / p.J, -rL * p.K * U / p.J, (-p.kth + rA**2 * p.kb + rL * p.K * U**2) / p.J, 0.0],
    ])


def max_real_eig(U, rA, rL, p=P):
    return float(np.max(np.linalg.eigvals(jacobian_at_rest(U, rA, rL, p)).real))


def B(rA, rL, p=P):
    return (rL - rA) * (p.J * p.kb + p.kb * p.m * rL * rA) + p.kth * rL * p.m


def a4(U, rA, rL, p=P):
    return p.kb * (p.K * U**2 * (rL - rA) - p.kth) / (p.J * p.m)


def U1(rA, rL, p=P):
    return float(np.sqrt(p.kth / (p.K * (rL - rA))))


def rL_on_B0(rA, p=P):
    try:
        return brentq(lambda r: B(rA, r, p), rA + 1e-12, -1e-12, xtol=1e-15, rtol=8.9e-16)
    except ValueError:
        return np.nan


def birth_frequency(rA, rL, p=P):
    return float(np.sqrt((p.kth + p.kb * rA * (rL - rA)) / p.J))


def theta_star(U, rA, rL, p=P):
    S = p.K * U**2 * (rL - rA)
    c = p.kth / S if S > 0 else np.inf
    return float(np.arccos(c)) if 0 < c < 1 else None


def saddle(U, rA, rL, th, p=P):
    z = -rA * np.sin(th) - (p.K * U**2 / p.kb) * th
    return np.array([z, 0.0, th, 0.0])
