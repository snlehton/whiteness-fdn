"""Plain-numpy homogeneous FDN: parameter container, loaders and response rendering.

Conventions follow the DAFx23 paper: H(z) = c^T [D_m(z)^-1 - U Gamma]^-1 b + d with
D_m(z) = diag(z^-m), Gamma = diag(gamma^m), U orthogonal, d = 0.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.linalg import expm, hadamard


def skew_expm(W: np.ndarray) -> np.ndarray:
    """U = expm(triu(W,1) - triu(W,1)^T), the parametrisation used by the reference code."""
    T = np.triu(np.asarray(W, dtype=np.float64), 1)
    return expm(T - T.T)


def gamma_from_t60(t60: float, fs: float) -> float:
    return 10 ** (-3.0 / (fs * t60))


def t60_from_gamma(gamma: float, fs: float) -> float:
    return -3.0 / (fs * np.log10(gamma))


@dataclass
class FDN:
    delays: np.ndarray            # (N,) integer delay lengths in samples
    U: np.ndarray                 # (N,N) orthogonal (lossless) feedback matrix
    b: np.ndarray                 # (N,) input gains
    c: np.ndarray                 # (N,) output gains
    gamma: float = 0.9999         # gain per sample (homogeneous decay)
    fs: int = 48000
    d: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.delays = np.asarray(self.delays, dtype=np.int64).ravel()
        self.U = np.asarray(self.U, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64).ravel()
        self.c = np.asarray(self.c, dtype=np.float64).ravel()

    @property
    def N(self) -> int:
        return len(self.delays)

    @property
    def order(self) -> int:
        return int(self.delays.sum())

    @property
    def A(self) -> np.ndarray:
        return self.U * (self.gamma ** self.delays)[None, :]

    @property
    def t60(self) -> float:
        return t60_from_gamma(self.gamma, self.fs)

    def with_gamma(self, gamma: float) -> "FDN":
        return replace(self, gamma=gamma)

    def with_t60(self, t60: float) -> "FDN":
        return replace(self, gamma=gamma_from_t60(t60, self.fs))

    # ---------------------------------------------------------------- responses
    def transfer(self, z: np.ndarray, per_channel: bool = False) -> np.ndarray:
        """H(z) at arbitrary complex points; per_channel returns (len(z), N) channel terms c_i [..]^-1 b."""
        z = np.asarray(z, dtype=np.complex128).ravel()
        P = np.zeros((len(z), self.N, self.N), dtype=np.complex128)
        P[:] = -self.A
        idx = np.arange(self.N)
        P[:, idx, idx] += z[:, None] ** self.delays[None, :]
        x = np.linalg.solve(P, np.broadcast_to(self.b.astype(np.complex128), (len(z), self.N))[..., None])[..., 0]
        Hc = x * self.c[None, :]
        return Hc if per_channel else Hc.sum(-1) + self.d

    def impulse_response(self, length: int) -> np.ndarray:
        """Exact time-domain IR (delay-line simulation, block-wise over the shortest delay)."""
        N, m = self.N, self.delays
        A, b, c = self.A, self.b, self.c
        h = np.zeros(length)
        h[0] += self.d
        bufs = [np.zeros(int(mi)) for mi in m]
        ptr = np.zeros(N, dtype=np.int64)
        # block-wise: within a block shorter than min(m) the delay outputs do not depend on the block's inputs
        blk = int(m.min())
        n = 0
        while n < length:
            L = min(blk, length - n)
            outs = np.empty((N, L))
            for i in range(N):
                k = (ptr[i] + np.arange(L)) % m[i]
                outs[i] = bufs[i][k]
            x = np.zeros(L)
            if n == 0:
                x[0] = 1.0
            h[n:n + L] += c @ outs
            v = A @ outs + b[:, None] * x[None, :]
            for i in range(N):
                k = (ptr[i] + np.arange(L)) % m[i]
                bufs[i][k] = v[i]
                ptr[i] = (ptr[i] + L) % m[i]
            n += L
        return h

    def to_json(self) -> dict:
        return {
            "delays": self.delays.tolist(),
            "U": self.U.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "gamma": self.gamma,
            "fs": self.fs,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> "FDN":
        return cls(np.array(d["delays"]), np.array(d["U"]), np.array(d["b"]), np.array(d["c"]),
                   d.get("gamma", 0.9999), d.get("fs", 48000), meta=d.get("meta", {}))


# ------------------------------------------------------------------- loaders
def read_args_txt(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if "," not in line:
            continue
        k, v = line.split(",", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def load_reference_mat(mat_path: Path, delays=None, gamma=None, fs=48000) -> FDN:
    """Load a parameters(.init).mat written by the reference solver (A holds the raw weight W)."""
    mat_path = Path(mat_path)
    p = sio.loadmat(mat_path)
    args_path = mat_path.parent / "args.txt"
    args = read_args_txt(args_path) if args_path.exists() else {}
    if delays is None:
        delays = args.get("delays")
        if delays is None:  # newer solver versions store denormalised m
            delays = p["m"].ravel()
    if gamma is None:
        gamma = float(args.get("gain_per_sample", np.ravel(p.get("gain_per_sample", [0.9999]))[0]))
    return FDN(np.rint(np.ravel(delays)).astype(int), skew_expm(p["A"]), p["B"].ravel(), p["C"].ravel(),
               gamma, fs, meta={"source": str(mat_path)})


# ----------------------------------------------------------------- baselines
def householder(N: int) -> np.ndarray:
    return np.eye(N) - 2.0 / N * np.ones((N, N))


def random_orthogonal(N: int, rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((N, N)))
    return Q * np.sign(np.diag(R))[None, :]


def hadamard_matrix(N: int) -> np.ndarray:
    return hadamard(N) / np.sqrt(N)


def baseline(kind: str, delays, gamma=0.9999, fs=48000, seed=0) -> FDN:
    """Reference conditions used in the paper's listening test (unit input/output gains)."""
    N = len(delays)
    if kind == "householder":
        U = householder(N)
    elif kind == "hadamard":
        U = hadamard_matrix(N)
    elif kind == "random":
        U = random_orthogonal(N, np.random.default_rng(seed))
    elif kind == "diagonal":
        U = np.eye(N)
    else:
        raise ValueError(kind)
    return FDN(delays, U, np.ones(N), np.ones(N), gamma, fs, meta={"kind": kind})


def save_json(fdn: FDN, path: Path):
    Path(path).write_text(json.dumps(fdn.to_json()))
