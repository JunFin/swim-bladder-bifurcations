import numpy as np

from .model import P, U1, jacobian_at_rest, rL_on_B0, vector_field


def numerical_jacobian(f, x0, h=1e-6):
    x0 = np.asarray(x0, float)
    E = np.eye(len(x0))
    return np.column_stack([(f(x0 + h * e) - f(x0 - h * e)) / (2 * h) for e in E])


def _tensors(f, x0, h):
    n = len(x0)
    E = np.eye(n)
    Bt = np.zeros((n, n, n))
    Ct = np.zeros((n, n, n, n))
    signs = (1, -1)
    for j in range(n):
        for k in range(n):
            acc = np.zeros(n)
            for s1 in signs:
                for s2 in signs:
                    acc += s1 * s2 * f(x0 + h * (s1 * E[j] + s2 * E[k]))
            Bt[:, j, k] = acc / (4 * h * h)
            for l in range(n):
                acc = np.zeros(n)
                for s1 in signs:
                    for s2 in signs:
                        for s3 in signs:
                            acc += s1 * s2 * s3 * f(x0 + h * (s1 * E[j] + s2 * E[k] + s3 * E[l]))
                Ct[:, j, k, l] = acc / (8 * h**3)
    return Bt, Ct


def derivative_tensors(f, x0, h=2e-2):
    # Richardson extrapolation of central differences
    x0 = np.asarray(x0, float)
    B1, C1 = _tensors(f, x0, h)
    B2, C2 = _tensors(f, x0, h / 2)
    return (4 * B2 - B1) / 3, (4 * C2 - C1) / 3


def hopf_vectors(A, tol=1e-7):
    w, V = np.linalg.eig(A)
    idx = [i for i in range(len(w)) if abs(w[i].real) < tol and w[i].imag > tol]
    if len(idx) != 1:
        return None
    i = idx[0]
    omega = float(w[i].imag)
    q = V[:, i] / np.linalg.norm(V[:, i])
    wl, Vl = np.linalg.eig(A.T)
    p = Vl[:, int(np.argmin(np.abs(wl + 1j * omega)))]
    p = p / np.conj(np.vdot(p, q))
    others = [w[k] for k in range(len(w)) if k != i and abs(w[k] - np.conj(w[i])) > tol]
    return omega, q, p, others


def first_lyapunov(f, x0=None, h=2e-2):
    x0 = np.zeros(4) if x0 is None else np.asarray(x0, float)
    A = numerical_jacobian(f, x0)
    hv = hopf_vectors(A)
    if hv is None:
        return None
    omega, q, p, others = hv
    if not all(np.real(o) < -1e-9 for o in others):
        return None
    Bt, Ct = derivative_tensors(f, x0, h)
    bil = lambda u, v: np.einsum("ijk,j,k->i", Bt, u, v)
    tri = lambda u, v, w: np.einsum("ijkl,j,k,l->i", Ct, u, v, w)
    qb = np.conj(q)
    n = len(x0)
    t1 = np.vdot(p, tri(q, q, qb))
    t2 = np.vdot(p, bil(q, np.linalg.solve(A.astype(complex), bil(q, qb))))
    t3 = np.vdot(p, bil(qb, np.linalg.solve(2j * omega * np.eye(n) - A, bil(q, q))))
    return float(np.real(t1 - 2 * t2 + t3) / (2 * omega)), omega


def l1(U, rA, rL, p=P):
    r = first_lyapunov(vector_field(U, rA, rL, p))
    return None if r is None else r[0]


def _oscillatory_rate(U, rA, rL, p=P):
    w = np.linalg.eigvals(jacobian_at_rest(U, rA, rL, p))
    return max(z.real for z in w if abs(z.imag) > 1e-9)


def _hopf_data(U, rA, p=P):
    rL = rL_on_B0(rA, p)
    f = vector_field(U, rA, rL, p)
    omega, q, pv, _ = hopf_vectors(numerical_jacobian(f, np.zeros(4)))
    _, Ct = derivative_tensors(f, np.zeros(4))
    c1 = np.vdot(pv, np.einsum("ijkl,j,k,l->i", Ct, q, q, np.conj(q))) / 2.0
    h = 1e-5
    dmu = (_oscillatory_rate(U, rA, rL + h, p) - _oscillatory_rate(U, rA, rL - h, p)) / (2 * h)
    return omega, q, c1, dmu


def hopf_amplitude_coefficients(U, rA, p=P):
    """A_theta^2 / d and A_z^2 / d for the cycle born at B = 0 (d -> 0)."""
    omega, q, c1, dmu = _hopf_data(U, rA, p)
    r2 = dmu / (-c1.real)
    return 4 * abs(q[2])**2 * r2, 4 * abs(q[0])**2 * r2


def hopf_seed(U, rA, d, p=P):
    omega, q, c1, dmu = _hopf_data(U, rA, p)
    r = np.sqrt(max(dmu * d / (-c1.real), 1e-12))
    x0 = 2 * r * np.real(q * np.exp(1j * (np.pi / 2 - np.angle(q[3]))))
    return [x0[0], x0[1], x0[2], np.pi / omega]


def corner(rA, p=P):
    rL = rL_on_B0(rA, p)
    return U1(rA, rL, p), rL


def linear_rates(U, rA, rL, p=P):
    """(mu1, mu2): real part of the oscillatory pair and the slow real eigenvalue."""
    w = np.linalg.eigvals(jacobian_at_rest(U, rA, rL, p))
    real = sorted([z.real for z in w if abs(z.imag) < 1e-9], key=abs)
    cplx = [z for z in w if z.imag > 1e-9]
    if not real or not cplx:
        return np.nan, np.nan
    return max(c.real for c in cplx), real[0]


def corner_coefficients(rA, p=P, h=2e-2):
    """Cubic normal form at the Hopf-pitchfork point:
    y' = mu2 y + p22 y^3 + p21 y r^2,  r' = r (mu1 + p11 r^2 + p12 y^2)."""
    U0, rL0 = corner(rA, p)
    f = vector_field(U0, rA, rL0, p)
    A = numerical_jacobian(f, np.zeros(4))
    w, V = np.linalg.eig(A)
    wl, Vl = np.linalg.eig(A.T)
    q0 = np.real(V[:, int(np.argmin(np.abs(w)))])
    q0 /= np.linalg.norm(q0)
    p0 = np.real(Vl[:, int(np.argmin(np.abs(wl)))])
    p0 /= p0 @ q0
    i1 = int(np.argmax(w.imag))
    omega = float(w[i1].imag)
    q1 = V[:, i1] / np.linalg.norm(V[:, i1])
    p1 = Vl[:, int(np.argmin(np.abs(wl + 1j * omega)))]
    p1 = p1 / np.conj(np.vdot(p1, q1))
    _, Ct = derivative_tensors(f, np.zeros(4), h)
    C = lambda a, b, c: np.einsum("ijkl,j,k,l->i", Ct, a, b, c)
    q1b = np.conj(q1)
    return dict(U0=U0, rL0=rL0, omega=omega,
                p22=float(np.real(p0 @ C(q0, q0, q0))) / 6,
                p21=float(np.real(p0 @ C(q0, q1, q1b))),
                p11=float((np.vdot(p1, C(q1, q1, q1b)) / 2).real),
                p12=float((np.vdot(p1, C(q0, q0, q1)) / 2).real))


def line_slopes(c):
    """mu2/mu1 along SB, TR and SH."""
    p11, p12, p21, p22 = c["p11"], c["p12"], c["p21"], c["p22"]
    return dict(SB=p21 / p11,
                TR=p22 * (p21 - p11) / (p11 * (p22 - p12)),
                SH=p22 / p12)
