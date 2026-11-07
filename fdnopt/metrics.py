"""Objective coloration / density measures used to compare FDN parameter sets."""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .fdn import FDN


def modal_excitation_stats(residues: np.ndarray) -> dict:
    """Spread of 20log10|rho| (scale-invariant: everything is relative to the mean in dB, as in Fig. 6)."""
    db = 20 * np.log10(np.abs(residues) + 1e-300)
    excited = db > np.median(db) - 100.0   # e.g. Householder's DC mode is exactly orthogonal to b = c = 1
    c = db[excited] - db[excited].mean()
    return {
        "n_modes": int(len(db)),
        "n_unexcited": int((~excited).sum()),
        "std_db": float(c.std()),
        "max_db": float(c.max()),
        "p99_db": float(np.percentile(c, 99)),
        "p95_db": float(np.percentile(c, 95)),
        "p01_db": float(np.percentile(c, 1)),
        "frac_above_6db": float(np.mean(c > 6.0)),
    }


def frequency_grid(K: int) -> np.ndarray:
    """Points on the upper unit half-circle, exactly as the reference get_frequency_samples(K)."""
    return np.exp(1j * np.pi * np.linspace(0, 1, K))


def magnitude_stats(fdn: FDN, K: int = 96000, fmin=50.0, fmax=20000.0) -> dict:
    z = frequency_grid(K)
    H = fdn.transfer(z)
    f = np.linspace(0, fdn.fs / 2, K)
    sel = (f >= fmin) & (f <= fmax)
    db = 20 * np.log10(np.abs(H[sel]) + 1e-12)
    db = db - db.mean()
    return {"mag_std_db": float(db.std()), "mag_p99_db": float(np.percentile(db, 99))}


def ir_from_frequency(fdn: FDN, K: int) -> np.ndarray:
    """IR via irfft of H on K bins (reference training/inference path, time-aliased beyond 2K samples)."""
    return np.fft.irfft(fdn.transfer(frequency_grid(K)))


def sparsity(h: np.ndarray) -> float:
    """Reference temporal loss ||h||_2 / ||h||_1 (lower = denser)."""
    return float(np.linalg.norm(h, 2) / np.linalg.norm(h, 1))


def normalized_echo_density(h: np.ndarray, fs: int, win_ms: float = 20.0, hop: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Abel & Huang normalized echo density profile (1 = Gaussian-noise-like)."""
    L = int(round(win_ms * 1e-3 * fs)) | 1
    w = np.hanning(L)
    w /= w.sum()
    centers = np.arange(L // 2, len(h) - L // 2, hop)
    ned = np.empty(len(centers))
    for k, n in enumerate(centers):
        seg = h[n - L // 2:n + L // 2 + 1]
        sd = np.sqrt(np.sum(w * seg ** 2))
        ned[k] = np.sum(w * (np.abs(seg) > sd)) / erfc(1 / np.sqrt(2))
    return centers / fs, ned


def ned_stats(fdn: FDN, dur_s: float = 1.0) -> dict:
    h = fdn.impulse_response(int(dur_s * fdn.fs))
    t, ned = normalized_echo_density(h, fdn.fs)
    above = np.nonzero(ned >= 0.9)[0]
    return {
        "ned_mean_0_100ms": float(ned[t < 0.1].mean()),
        "ned_mean_100_500ms": float(ned[(t >= 0.1) & (t < 0.5)].mean()),
        "ned_mean_500_1000ms": float(ned[t >= 0.5].mean()),
        "t_ned_0p9_ms": float(t[above[0]] * 1e3) if len(above) else None,
    }


def mixing_stats(fdn: FDN) -> dict:
    """How much the feedback matrix couples the delay lines: 1 = identity (parallel combs), ~1/N = dense."""
    U = fdn.U
    return {"diag_energy": float(np.sum(np.diag(U) ** 2) / fdn.N),
            "diag_energy_dense_ref": 1.0 / fdn.N}


def all_metrics(fdn: FDN, residues: np.ndarray) -> dict:
    out = modal_excitation_stats(residues)
    out.update(mixing_stats(fdn))
    out.update(magnitude_stats(fdn))
    out["sparsity_ir"] = sparsity(ir_from_frequency(fdn, 96000))
    out.update(ned_stats(fdn))
    return out
