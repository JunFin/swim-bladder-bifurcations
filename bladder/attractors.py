import numpy as np
from scipy.integrate import solve_ivp

from .model import P, escaped, flow, rhs


def settle(U, rA, rL, x0, T_transient, T_record, p=P, n=30000, direction=0):
    """Integrate past the transient and record the attractor; None if the trajectory escapes.

    `crossings` are the states on the section omega = 0 (direction=-1 keeps only pitch maxima).
    """
    def section(t, x, *args):
        return x[3]
    section.direction = direction

    args = (U, rA, rL, p)
    s = solve_ivp(rhs, (0, T_transient), x0, args=args, rtol=1e-9, atol=1e-11,
                  method="DOP853", events=escaped)
    if s.status == 1:
        return None
    s2 = solve_ivp(rhs, (0, T_record), s.y[:, -1], args=args, rtol=1e-10, atol=1e-12,
                   method="DOP853", events=[escaped, section], dense_output=True)
    if s2.status == 1:
        return None
    Y = s2.sol(np.linspace(0, T_record, n))
    return dict(sol=s2, Y=Y, x_end=s2.y[:, -1], crossings=s2.y_events[1])


def attractor(U, rA, rL, x0, T_transient=800.0, T_record=600.0, p=P):
    r = settle(U, rA, rL, x0, T_transient, T_record, p)
    if r is None:
        return None
    ev, Y = r["crossings"], r["Y"]
    args = (U, rA, rL, p)
    acc = np.array([rhs(0, e, *args)[3] for e in ev]) if len(ev) else np.zeros(0)
    th = ev[:, 2] if len(ev) else np.zeros(0)
    alpha = Y[2] - np.arctan2(Y[1], U)
    return dict(x_end=r["x_end"], th_max=th[acc < 0], th_min=th[acc > 0],
                zmean=float(np.mean(Y[0])), thmean=float(np.mean(Y[2])),
                alpha_max=float(np.max(np.abs(alpha))),
                th_amp=0.5 * float(np.ptp(Y[2])), z_amp=0.5 * float(np.ptp(Y[0])))


def largest_lyapunov(U, rA, rL, x0, T=600.0, dt=2.0, eps=1e-8, p=P):
    """Benettin's two-trajectory estimate with renormalisation every dt."""
    x = np.asarray(x0, float)
    y = x + eps * np.array([1.0, 0, 0, 0])
    acc = 0.0
    n = int(T / dt)
    for _ in range(n):
        x = flow(x, dt, U, rA, rL, p, rtol=1e-10, atol=1e-12)
        y = flow(y, dt, U, rA, rL, p, rtol=1e-10, atol=1e-12)
        if abs(x[0]) > 50:
            return np.nan
        dv = y - x
        nrm = np.linalg.norm(dv)
        acc += np.log(nrm / eps)
        y = x + dv * (eps / nrm)
    return acc / (n * dt)


REST, SYMMETRIC, ASYMMETRIC, MULTI_PERIOD, UNSETTLED, CHAOS, ESCAPE = range(7)
RESTART = np.array([0.0, 0.0, 2e-3, 0.0])


def _recheck(U, rA, rL, cls, T_transient=6000.0, p=P):
    r = settle(U, rA, rL, [0, 0, 0.05, 0], T_transient, 1500.0, p, n=20000, direction=-1)
    if r is None:
        return ESCAPE
    pts = r["crossings"][:, 2][-40:]
    amp = 0.5 * np.ptp(r["Y"][2])
    if np.ptp(pts) < 1e-6:
        return ASYMMETRIC if abs(np.mean(r["Y"][2])) > 0.05 * amp + 1e-3 else SYMMETRIC
    return cls


def classify(U, rA, rL, x, d, p=P):
    """Regime at (U, rL) starting from x. Returns (class, final state, amplitude)."""
    if np.linalg.norm(x) < 1e-4:
        x = RESTART
    r = settle(U, rA, rL, x, 1200.0, 500.0, p, n=10000, direction=-1)
    if r is None:
        return ESCAPE, RESTART, np.nan
    Y, xe, ev = r["Y"], r["x_end"], r["crossings"]
    amp = 0.5 * float(np.ptp(Y[2]))
    if abs(d) < 1e-9 or (amp < 2e-3 and np.linalg.norm(Y[:, -1]) < 5e-3):
        return REST, xe, amp
    if ev.ndim < 2 or len(ev) < 3:
        return (REST if amp < 1e-2 else MULTI_PERIOD), xe, amp
    tail = ev[len(ev) // 3:, 2]
    n = len(np.unique(np.round(tail, 3)))
    if n == 1:
        return (ASYMMETRIC if abs(np.mean(Y[2])) > 0.01 * amp + 1e-4 else SYMMETRIC), xe, amp
    if n <= 8:
        cls = MULTI_PERIOD
    else:
        cls = CHAOS if largest_lyapunov(U, rA, rL, xe, T=800.0, p=p) > 0.01 else UNSETTLED
    if cls in (MULTI_PERIOD, UNSETTLED):
        cls = _recheck(U, rA, rL, cls, p=p)
    return cls, xe, amp
