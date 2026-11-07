"""Modal decomposition (poles / residues) of a homogeneous FDN.

H(z) = d + sum_i rho_i / (1 - lambda_i z^-1)                       (paper eq. 5)

Poles: with A = U Gamma and z = gamma * w, det(diag(z^m) - U Gamma) = det(diag(w^m) - U) * prod(gamma^m),
so the poles are gamma times the eigenvalues of the *lossless* state-transition matrix (orthogonal ->
perfectly conditioned dense eigenproblem of size M = sum(m)). Each pole is then polished with a Newton
step on det P(z) and the residue is taken from the N x N null vectors of P(lambda) = diag(lambda^m) - A:
    rho = (c^T v)(u^H b) / (u^H diag(m lambda^m) v)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import eigvals

from .fdn import FDN


def state_matrix(delays, U) -> np.ndarray:
    """M x M transition matrix of the delay-line state (oldest sample of line i is its output)."""
    m = np.asarray(delays, dtype=np.int64)
    M = int(m.sum())
    start = np.concatenate([[0], np.cumsum(m)[:-1]])
    out_idx = start + m - 1          # s_i[m_i-1]
    T = np.zeros((M, M))
    for i, (s, mi) in enumerate(zip(start, m)):
        T[s, out_idx] = U[i]         # s_i[0] <- sum_j U_ij * out_j
        r = np.arange(s + 1, s + mi)
        T[r, r - 1] = 1.0            # shift
    return T


def _P(fdn: FDN, z: np.ndarray):
    N = fdn.N
    P = np.empty((len(z), N, N), dtype=np.complex128)
    P[:] = -fdn.A
    idx = np.arange(N)
    zm = z[:, None] ** fdn.delays[None, :]
    P[:, idx, idx] += zm
    dP = (fdn.delays[None, :] * zm)  # z * d/dz diag(z^m) = diag(m z^m)
    return P, dP


def polish(fdn: FDN, z: np.ndarray, iters: int = 2) -> np.ndarray:
    """Newton on det P(z): z <- z - 1 / tr(P^-1 P'), P' = diag(m z^(m-1))."""
    for _ in range(iters):
        P, dP = _P(fdn, z)
        s = np.linalg.svd(P, compute_uv=False)
        ok = s[:, -1] > 1e-13 * s[:, 0]      # exactly singular -> already a pole (e.g. Householder at z = gamma)
        Pinv_diag = np.diagonal(np.linalg.inv(P[ok]), axis1=1, axis2=2)
        tr = np.sum(Pinv_diag * dP[ok], axis=1) / z[ok]
        z = z.copy()
        z[ok] = z[ok] - 1.0 / tr
    return z


def residues(fdn: FDN, poles: np.ndarray) -> np.ndarray:
    """Residues of (assumed distinct) poles. A k-fold semisimple pole (null space of dimension k, e.g.
    Householder at z = -gamma) is one exponential rho * lambda^n in time; its combined residue is
    c^T V (U^H diag(m lambda^m) V)^-1 U^H b."""
    P, dP = _P(fdn, poles)
    X, s, Yh = np.linalg.svd(P)
    u = X[:, :, -1]                   # left null vector  (u^H P = 0)
    v = Yh[:, -1, :].conj()           # right null vector (P v = 0)
    num = (v @ fdn.c) * (u.conj() @ fdn.b)
    den = np.sum(u.conj() * dP * v, axis=1)
    rho = num / den
    k = np.sum(s < 1e-8 * s[:, :1], axis=1)
    for i in np.nonzero(k > 1)[0]:
        Ui, Vi = X[i, :, -k[i]:], Yh[i, -k[i]:, :].conj().T
        inner = Ui.conj().T @ (dP[i][:, None] * Vi)
        rho[i] = fdn.c @ Vi @ np.linalg.solve(inner, Ui.conj().T @ fdn.b)
    return rho


def _wrapped_phase_sum(delays, U, omega):
    """sum_k arg(eig(diag(e^{-j omega m}) U)) with phases wrapped to [0, 2pi)."""
    S = np.exp(-1j * omega[:, None] * delays[None, :])[:, :, None] * U[None].astype(np.complex128)
    ev = np.linalg.eigvals(S)
    return np.mod(np.angle(ev), 2 * np.pi), ev


def lossless_pole_angles(delays, U, oversample: int = 4, tol: float = 1e-12) -> np.ndarray:
    """Angles in [0, pi] of the poles of the lossless FDN (U orthogonal), by eigenphase counting.

    The eigenphases theta_k(omega) of S(omega) = diag(e^{-j omega m}) U decrease monotonically
    (d theta/d omega = -v^H diag(m) v), so the number of poles in (0, omega] is the integer
        n(omega) = (M omega + sum_k phi_k(omega) - sum_k phi_k(0)) / 2 pi,   phi_k wrapped to [0, 2pi).
    Poles are isolated on a grid, split by bisection where an interval holds more than one, and
    then located by bisection on n(omega).
    """
    m = np.asarray(delays, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    M = m.sum()
    phi0, ev0 = _wrapped_phase_sum(m, U, np.zeros(1))
    sum0 = phi0.sum()
    eps = 1e-9

    def count(omega):
        return np.rint((M * omega + _wrapped_phase_sum(m, U, omega)[0].sum(-1) - sum0) / (2 * np.pi)).astype(np.int64)

    special = []
    if np.min(np.abs(ev0[0] - 1)) < 1e-7:          # pole at omega = 0
        special.append(0.0)
    _, evpi = _wrapped_phase_sum(m, U, np.array([np.pi]))
    if np.min(np.abs(evpi[0] - 1)) < 1e-7:         # pole at omega = pi
        special.append(np.pi)

    lo_end, hi_end = eps, np.pi - eps
    G = int(oversample * M) + 1
    grid = np.linspace(lo_end, hi_end, G)
    n = count(grid)
    a, b, na, nb = grid[:-1], grid[1:], n[:-1], n[1:]
    keep = nb > na
    a, b, na, nb = a[keep], b[keep], na[keep], nb[keep]
    # split until every interval holds exactly one pole
    for _ in range(60):
        multi = (nb - na) > 1
        if not multi.any():
            break
        mid = 0.5 * (a[multi] + b[multi])
        nm = count(mid)
        A1, B1, NA1, NB1 = a[multi], mid, na[multi], nm
        A2, B2, NA2, NB2 = mid, b[multi], nm, nb[multi]
        a = np.concatenate([a[~multi], A1, A2]); b = np.concatenate([b[~multi], B1, B2])
        na = np.concatenate([na[~multi], NA1, NA2]); nb = np.concatenate([nb[~multi], NB1, NB2])
        keep = nb > na
        a, b, na, nb = a[keep], b[keep], na[keep], nb[keep]
    if ((nb - na) != 1).any():
        raise RuntimeError("could not isolate all poles (degenerate pole pairs?)")
    # bisection on the count
    while np.max(b - a) > tol:
        mid = 0.5 * (a + b)
        left = count(mid) > na
        b = np.where(left, mid, b)
        a = np.where(left, a, mid)
    return np.sort(np.concatenate([0.5 * (a + b), special]))


def modal_decomposition(fdn: FDN, positive_only: bool = True, method: str = "count"):
    """Return (poles, residues). positive_only keeps one of each conjugate pair (0 <= angle <= pi).

    method="count": eigenphase counting on the N x N lossless matrix (fast, exact for orthogonal U).
    method="dense": eigenvalues of the M x M lossless state-transition matrix (slow reference).
    """
    if method == "count":
        try:
            ang = lossless_pole_angles(fdn.delays, fdn.U)
        except RuntimeError:
            return modal_decomposition(fdn, positive_only, method="dense")
        if not positive_only:
            ang = np.concatenate([-ang[(ang > 0) & (ang < np.pi)][::-1], ang])
        w = np.exp(1j * ang)
    else:
        w = eigvals(state_matrix(fdn.delays, fdn.U), check_finite=False, overwrite_a=True)
        if positive_only:
            w = w[np.angle(w) >= -1e-12]
        # merge repeated (semisimple) poles into one mode, as the counting method does
        w = w[np.argsort(np.angle(w))]
        w = w[np.concatenate([[True], np.abs(np.diff(w)) > 1e-7])]
    poles = polish(fdn, fdn.gamma * w)
    order = np.argsort(np.angle(poles))
    poles = poles[order]
    return poles, residues(fdn, poles)


def cached_modal(fdn: FDN, cache: Path):
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache)
        if np.array_equal(d["delays"], fdn.delays) and np.allclose(d["U"], fdn.U) \
                and np.allclose(d["b"], fdn.b) and np.allclose(d["c"], fdn.c) and d["gamma"] == fdn.gamma:
            return d["poles"], d["residues"]
    poles, res = modal_decomposition(fdn)
    np.savez(cache, poles=poles, residues=res, delays=fdn.delays, U=fdn.U, b=fdn.b, c=fdn.c, gamma=fdn.gamma)
    return poles, res
