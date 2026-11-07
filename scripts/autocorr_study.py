"""Autocorrelation ("whiteness") as a training objective: what it measures and how its gradients behave.

whiteness(h) = sum_{k != 0} r[k]^2 / r[0]^2, r = autocorrelation of h. By Wiener-Khinchin and Parseval this
equals Var(|H|^2) / mean(|H|^2)^2 over the DFT grid, so it is a spectral-flatness loss. A Gaussian-noise IR
scores ~1, not 0. Parts:

  metrics    whiteness and periodicity (peak normalised autocorrelation) for every variant / configuration,
             full IR and first 300 ms, next to the modal fourth-moment ratio E|rho|^4 / E|rho|^2^2
  limit      lossless (gamma = 1) whiteness over windows of L samples against the modal prediction
             whiteness + 1 -> (2L/3) sum|rho|^4 / (sum|rho|^2)^2
  grad       FDN unrolled in time in torch (block recursion, depth L / min(m)), gradient of whiteness
             w.r.t. a rotation of U for growing L
  sens       relative sensitivity of the IR to a rotation of U, per 100 ms window (finite differences)
  slice      whiteness along a 1-D rotation path for several window lengths
  train      Adam on whiteness of the unrolled IR, alone and with a mixing penalty sum U_ij^4 / N, from the
             reference init; evaluated with the same modal / density metrics as the reproduction
  train_unit the same with b = c = 1 fixed, training only U
  train_density  whiteness + mixing + early density (sparsity of 20 ms windows up to 0.5 s), fixed and trained gains
  train_pairs    the same at density 3 with gains on per-pair unit circles (b = 1 and c on circles, or both)
  train_quad     the same with b on c's circles 90 degrees ahead (b_i c_i = -/+ sin 2 theta per pair)
  train_flat:w[:d]  whiteness + mixing + density d (default 3) + w * transient flatness (first 100 ms held to the
             decay-compensated tail level, sample by sample), free gains
  train_level:w[:band[:d[:cfgs]]]  the same with w * level band instead: every sample of the first 100 ms driven into
             +-band dB (default 10) around the tail level, gaps costing as much as echoes that stick out
  train_aspread:w[:d[:cfgs[:n]]]  the same with the spread over windows of at least n arrivals (default 8)
  train_build:w[:depth[:rise[:aspread[:d[:cfgs]]]]]  arrival spread (default 0.5) + density + w * build-up: the early
             envelope held to a ramp from -depth dB (default 6) to the tail level after rise s (default 0.15)
  train_direct:w[:lo[:hi[:build[:aspread[:d[:cfgs]]]]]]  that (build 30) + w * a ceiling on the direct echoes b_i c_i
             rising from lo dB (shortest line, default 10) to hi dB (longest, default 22) above the tail level
  train_spread:w[:d[:cfgs[:stop[:lead]]]]  the same with w * early spread instead: the effective number of echoes per
             20 ms window up to stop s (default 0.5) pushed towards that of noise (fourth moment, scale-free per window)
  decay      late T60, early decay time and early energy of DiffFDN vs the trained results (T60 2.5 s playback)

    python scripts/autocorr_study.py [metrics limit grad sens slice train train_unit train_density train_pairs train_quad train_flat:w[:d] train_level:w[:band[:d[:cfgs]]] train_spread:w[:d[:cfgs[:stop[:lead]]]] train_aspread:w[:d[:cfgs[:n]]] train_build:w[:depth[:rise[:aspread[:d[:cfgs]]]]] train_direct:w[:lo[:hi[:build[:aspread[:d[:cfgs]]]]]] decay]

Writes runs/autocorr_study.json and docs/figures/autocorr_*.png.
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import expm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import COLORS, INK2, MUTED, style  # noqa: E402
from fdnopt.fdn import FDN, baseline, gamma_from_t60, load_reference_mat  # noqa: E402
from fdnopt.metrics import all_metrics  # noqa: E402
from fdnopt.modal import cached_modal, modal_decomposition  # noqa: E402

CONFIGS = ["N4_set1", "N4_set2", "N6_set1", "N6_set2", "N8_set1", "N8_set2"]
REF = ROOT / "runs" / "reference"
CACHE = ROOT / "runs" / "cache"
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "runs" / "autocorr_study.json"
FS = 48000
# ordinal blue ramp (steps 250 / 450 / 650) for window lengths
RAMP = ["#86b6ef", "#2a78d6", "#104281"]


# ----------------------------------------------------------------- measures
def autocorr(h):
    n = 1 << int(np.ceil(np.log2(2 * len(h))))
    return np.fft.irfft(np.abs(np.fft.rfft(h, n)) ** 2, n)


def whiteness(h):
    r = autocorr(h)
    return float((r[1:] ** 2).sum() / r[0] ** 2)


def acf_peak(h, lag_min, lag_max):
    """Largest |r[k]| / r[0] over lag_min <= k <= lag_max: strength of the most salient periodicity."""
    r = autocorr(h)
    return float(np.abs(r[lag_min:lag_max + 1]).max() / r[0])


def fourth_moment_ratio(poles, res):
    """E|rho|^4 / (E|rho|^2)^2 over all spectral peaks (complex modes count twice): 1 if all modes are equally
    excited, 2 for complex-Gaussian residues. The lossless whiteness slope is (2/3) * this / n_peaks."""
    a2 = np.abs(res) ** 2
    real = (np.abs(np.angle(poles)) < 1e-9) | (np.abs(np.angle(poles)) > np.pi - 1e-9)
    w = np.where(real, 1.0, 2.0)
    n = w.sum()
    return float((w * a2 ** 2).sum() / n / ((w * a2).sum() / n) ** 2), int(n)


def variants(cfg):
    run = REF / cfg
    opt = load_reference_mat(run / "parameters.mat")
    out = {"init": load_reference_mat(run / "parameters_init.mat"), "optimized": opt}
    for kind in ("householder", "random", "diagonal"):
        out[kind] = baseline(kind, opt.delays, opt.gamma)
    return out


def modal_of(name, fdn, cfg):
    key = "-".join(map(str, fdn.delays))
    if name == "diagonal":  # parallel combs: m_i poles per comb, |rho| = |b_i c_i| / (m_i gamma^(m_i - 1))
        ang = np.concatenate([2 * np.pi * np.arange(m // 2 + 1) / m for m in fdn.delays])
        res = np.concatenate([np.full(m // 2 + 1, fdn.b[i] * fdn.c[i] / (m * fdn.gamma ** (m - 1)))
                              for i, m in enumerate(fdn.delays)])
        return np.exp(1j * ang), res
    path = REF / cfg / f"modal_{name}.npz" if name in ("init", "optimized") else CACHE / f"{name}_{key}_{fdn.gamma}.npz"
    return cached_modal(fdn, path)


# ------------------------------------------------------------------- parts
def part_metrics():
    rng = np.random.default_rng(7)
    rows = []
    for cfg in CONFIGS:
        vs = variants(cfg)
        m = vs["optimized"].delays
        L, early = int(2.5 * FS), int(0.3 * FS)
        lo, hi = FS // 1000, 2 * int(m.max())
        for name, fdn in vs.items():
            h = fdn.impulse_response(L)
            poles, res = modal_of(name, fdn, cfg)
            fm, npk = fourth_moment_ratio(poles, res)
            rows.append({"config": cfg, "variant": name, "white_full": whiteness(h), "white_early": whiteness(h[:early]),
                         "acf_peak_full": acf_peak(h, lo, hi), "acf_peak_early": acf_peak(h[:early], lo, hi),
                         "modal_m4": fm, "n_peaks": npk})
        h = np.zeros(L)
        h[m.min():] = rng.standard_normal(L - m.min()) * vs["optimized"].gamma ** np.arange(m.min(), L)
        rows.append({"config": cfg, "variant": "noise", "white_full": whiteness(h), "white_early": whiteness(h[:early]),
                     "acf_peak_full": acf_peak(h, lo, hi), "acf_peak_early": acf_peak(h[:early], lo, hi),
                     "modal_m4": 2.0, "n_peaks": None})
        for r in rows[-6:]:
            print(f"{cfg:8s} {r['variant']:11s} white {r['white_full']:6.3f} early {r['white_early']:6.3f} "
                  f"acf_peak {r['acf_peak_full']:.3f} early {r['acf_peak_early']:.3f} modal_m4 {r['modal_m4']:.2f}")
    return rows


def part_limit(cfg="N8_set1"):
    out = {}
    for name in ("init", "optimized"):
        fdn = variants(cfg)[name]
        fm, npk = fourth_moment_ratio(*modal_of(name, fdn, cfg))
        lossless = fdn.with_gamma(1.0)
        h = lossless.impulse_response(int(40 * FS))
        rows = []
        for sec in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40):
            L = int(sec * FS)
            w = whiteness(h[:L])
            rows.append({"seconds": sec, "white": w, "predicted": 2 * L / 3 * fm / npk - 1})
            print(f"limit {name:9s} L={sec:5.2f}s whiteness {w:10.3f}  modal prediction {rows[-1]['predicted']:10.3f}")
        out[name] = rows
    return out


def unrolled_ir(U, b, c, delays, gamma, L):
    """Differentiable time-domain IR: the FDN as a deep linear network of depth ceil(L / min(m))."""
    m = [int(x) for x in delays]
    N, B = len(m), min(m)
    A = U * torch.tensor([gamma ** mi for mi in m], dtype=U.dtype)[None, :]
    zeros = torch.zeros(B, dtype=U.dtype)
    delta = torch.zeros(B, dtype=U.dtype)
    delta[0] = 1.0

    def read(xs, i, a):  # x_i[a : a + B], zero for negative indices
        if a + B <= 0:
            return zeros
        if a < 0:
            return torch.cat([zeros[:-a], xs[0][i, :B + a]])
        j, off = divmod(a, B)
        return xs[j][i] if off == 0 else torch.cat([xs[j][i, off:], xs[j + 1][i, :off]])

    xs, ys = [], []
    for k in range(-(-L // B)):
        s = torch.stack([read(xs, i, k * B - m[i]) for i in range(N)])
        x = A @ s
        if k == 0:
            x = x + b[:, None] * delta[None, :]
        xs.append(x)
        ys.append(c @ s)
    return torch.cat(ys)[:L]


def whiteness_t(h):
    n = 1 << int(np.ceil(np.log2(2 * len(h))))
    r = torch.fft.irfft(torch.fft.rfft(h, n).abs() ** 2, n)
    return (r[1:] ** 2).sum() / r[0] ** 2


def rotated(U0, T):
    T = torch.triu(T, 1)
    return U0 @ torch.linalg.matrix_exp(T - T.T)


def part_grad(cfg="N8_set1"):
    torch.set_num_threads(4)
    fdn = variants(cfg)["init"]
    U0 = torch.tensor(fdn.U)
    b, c = torch.tensor(fdn.b), torch.tensor(fdn.c)
    # the unrolled network reproduces the plain simulation
    h_ref = fdn.impulse_response(FS // 2)
    h_t = unrolled_ir(U0, b, c, fdn.delays, fdn.gamma, FS // 2).numpy()
    err = float(np.abs(h_t - h_ref).max() / np.abs(h_ref).max())
    print(f"grad: unrolled vs simulation, max rel. error {err:.2e}")
    out = {"check_rel_err": err}
    for gamma in (fdn.gamma, 1.0):
        rows, prev = [], None
        for sec in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10):
            L = int(sec * FS)
            T = torch.zeros(fdn.N, fdn.N, dtype=torch.float64, requires_grad=True)
            t0 = time.perf_counter()
            loss = whiteness_t(unrolled_ir(rotated(U0, T), b, c, fdn.delays, gamma, L))
            loss.backward()
            dt = time.perf_counter() - t0
            g = torch.triu(T.grad, 1).flatten()
            cos = float(g @ prev / (g.norm() * prev.norm())) if prev is not None else None
            prev = g
            rows.append({"seconds": sec, "depth": -(-L // int(fdn.delays.min())), "loss": loss.item(),
                         "grad_norm": g.norm().item(), "rel_grad": g.norm().item() / loss.item(),
                         "cos_prev": cos, "time_s": dt})
            print(f"grad gamma={gamma} L={sec:5.2f}s depth {rows[-1]['depth']:5d} loss {loss.item():9.3f} "
                  f"|g| {g.norm().item():10.3f} |g|/loss {rows[-1]['rel_grad']:7.3f} "
                  f"cos(prev) {cos if cos is None else round(cos, 3)}  {dt:5.1f}s")
        out[str(gamma)] = rows
    return out


def random_skew(N, rng):
    T = np.triu(rng.standard_normal((N, N)), 1)
    S = T - T.T
    return S / np.linalg.norm(S)


def part_sens(cfg="N8_set1", n_dir=4, eps=1e-8):
    fdn = variants(cfg)["init"]
    rng = np.random.default_rng(3)
    L, win = int(10 * FS), FS // 10
    out = {}
    for gamma in (fdn.gamma, 1.0):
        base = fdn.with_gamma(gamma)
        h0 = base.impulse_response(L)
        e0 = np.sqrt(np.add.reduceat(h0 ** 2, np.arange(0, L, win)))
        acc = np.zeros_like(e0)
        for _ in range(n_dir):
            S = random_skew(fdn.N, rng)
            hp = FDN(base.delays, base.U @ expm(eps * S), base.b, base.c, gamma).impulse_response(L)
            hm = FDN(base.delays, base.U @ expm(-eps * S), base.b, base.c, gamma).impulse_response(L)
            d = (hp - hm) / (2 * eps)
            acc += np.sqrt(np.add.reduceat(d ** 2, np.arange(0, L, win))) / e0
        t = (np.arange(len(e0)) + 0.5) * win / FS
        rel = acc / n_dir
        slope = np.polyfit(np.log(t[5:]), np.log(rel[5:]), 1)[0]
        print(f"sens gamma={gamma}: relative sensitivity {rel[0]:.1f} at {t[0]:.2f}s, {rel[-1]:.1f} at {t[-1]:.1f}s, "
              f"log-log slope {slope:.2f}")
        out[str(gamma)] = {"t": t.tolist(), "rel": rel.tolist(), "loglog_slope": float(slope)}
    return out


def part_slice(cfg="N8_set1", span=0.2, n=241):
    fdn = variants(cfg)["init"]
    S = random_skew(fdn.N, np.random.default_rng(11))
    ts = np.linspace(-span, span, n)
    cases = [(1.0, 0.1), (1.0, 1.0), (1.0, 10.0), (fdn.gamma, 10.0)]
    out = {"t": ts.tolist()}
    for gamma, sec in cases:
        L = int(sec * FS)
        vals = np.array([whiteness(FDN(fdn.delays, fdn.U @ expm(t * S), fdn.b, fdn.c, gamma).impulse_response(L))
                         for t in ts])
        n_min = int(((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:])).sum())
        print(f"slice gamma={gamma} L={sec}s: loss {vals.min():.3f}..{vals.max():.3f}, {n_min} local minima")
        out[f"{gamma}_{sec}"] = {"gamma": gamma, "seconds": sec, "loss": vals.tolist(), "local_minima": n_min}
    return out


GAUSS_L1L2 = float(np.sqrt(2 / np.pi))  # E|x| / sqrt(E x^2) of a Gaussian


def early_sparsity_t(h, start, stop, win=FS // 50, hop=FS // 100):
    """Mean over 20 ms Hann windows in [start, stop) of 1 - (l1 / l2) / its Gaussian value: 0 for Gaussian
    noise, towards 1 for a few isolated echoes. A smooth stand-in for 1 - normalized echo density."""
    w = torch.hann_window(win, periodic=False, dtype=h.dtype)
    seg = h[start:stop].unfold(0, win, hop)
    l1 = (seg.abs() * w).sum(1) / w.sum()
    l2 = torch.sqrt((seg ** 2 * w).sum(1) / w.sum() + 1e-30)
    return (1 - l1 / l2 / GAUSS_L1L2).mean()


def early_sparsity(fdn, stop_s=0.5):
    h = torch.tensor(fdn.impulse_response(int(stop_s * FS)))
    return float(early_sparsity_t(h, int(fdn.delays.min()), len(h)))


def early_spread_t(h, start, stop, win=FS // 50, hop=FS // 100):
    """Mean over 20 ms Hann windows in [start, stop) of log(K_noise / K), K = (sum w x^2)^2 / sum w^2 x^4 the
    effective number of echoes in a window and K_noise = (sum w)^2 / (3 sum w^2) its value for Gaussian noise:
    0 for noise, log(K_noise) = 5.4 for a lone echo. Scale-free per window, so going quiet buys nothing, and
    fourth-moment, so one loud echo among quiet ones costs almost as much as a lone echo (early_sparsity_t, a first
    moment, has nearly saturated by then). Start half a window before the first echo, or the taper hides it."""
    w = torch.hann_window(win, periodic=False, dtype=h.dtype)
    seg = h[start:stop].unfold(0, win, hop)
    k = (seg ** 2 * w).sum(1) ** 2 / ((seg ** 4 * w ** 2).sum(1) + 1e-300)
    return torch.log(w.sum() ** 2 / (3 * (w ** 2).sum()) / (k + 1e-12)).mean()


def arrival_times(delays, T):
    """Samples in [1, T) where some path arrives: the distinct sums of delays (any order, any repeats). Between them
    the output of the FDN is exactly zero."""
    s = np.zeros(T, bool)
    s[0] = True
    for t in range(1, T):
        s[t] = any(s[t - d] for d in delays if d <= t)
    return np.nonzero(s[1:])[0] + 1


def arrival_windows(delays, stop, n_min=8, min_len=FS // 50):
    """Windows [s, e, n] from the first arrival to stop, each min_len long or as long as it takes to hold n_min
    arrivals (n = how many it holds), the next starting halfway through the last. Early on N = 4 that is ~50 ms,
    on a dense IR just min_len."""
    a = arrival_times([int(d) for d in delays], 4 * stop)
    wins, s = [], int(a[0])
    while s < stop:
        i = np.searchsorted(a, s)
        e = max(s + min_len, int(a[min(i + n_min, len(a)) - 1]) + 1)
        wins.append((s, e, int(np.searchsorted(a, e) - i)))
        s += (e - s) // 2
    return wins


def arrival_spread_t(h, gamma, wins):
    """early_spread_t over windows holding a fixed number of arrivals instead of a fixed 20 ms, rectangular, decay
    taken out: mean of log(n / K), 0 when the n arrivals in a window are equally loud, log n for a lone echo. With
    20 ms windows an N = 4 transient holds 1-3 arrivals per window, so a lone echo scores K = 1 however loud it is."""
    hc = h * gamma ** -torch.arange(len(h), dtype=h.dtype)
    terms = []
    for s, e, n in wins:
        seg = hc[s:e]
        k = (seg ** 2).sum() ** 2 / ((seg ** 4).sum() + 1e-300)
        terms.append(torch.log(n / (k + 1e-12)))
    return torch.stack(terms).mean()


def buildup_t(h, gamma, wins, depth_db=6.0, rise_s=0.15, tail_start=FS // 2):
    """Mean over arrival windows of the squared distance (in 10 dB units) of each window's level (mean power, decay
    taken out, against the tail's) from a build-up ramp: -depth_db at the first arrival, rising linearly in dB to
    0 dB rise_s later and staying there. Loud early echoes and an overshoot after the rise both cost, so the
    response grows into its tail instead of starting at full level."""
    hc = h * gamma ** -torch.arange(len(h), dtype=h.dtype)
    p = hc ** 2 / (hc[tail_start:] ** 2).mean()
    t0 = wins[0][0]
    terms = []
    for s, e, _ in wins:
        target = -depth_db * max(0.0, 1 - ((s + e) / 2 - t0) / (rise_s * FS))
        terms.append(((10 * torch.log10(p[s:e].mean() + 1e-12) - target) / 10) ** 2)
    return torch.stack(terms).mean()


def direct_ceiling_t(h, gamma, b, c, delays, lo_db=10.0, hi_db=22.0, tail_start=FS // 2):
    """Mean over lines of the squared excess (in 10 dB units) of each direct echo b_i c_i (input -> line i -> output,
    the one path that never passes U) over a ceiling rising linearly with the delay, from lo_db above the tail's
    level for the shortest line to hi_db for the longest. The unmixed echoes then build up with delay instead of
    the shortest line leading the response, as DiffFDN's do."""
    hc = h * gamma ** -torch.arange(len(h), dtype=h.dtype)
    m = torch.tensor([float(d) for d in delays], dtype=h.dtype)
    ceil = lo_db + (hi_db - lo_db) * (m - m.min()) / (m.max() - m.min())
    db = 10 * torch.log10((b * c) ** 2 / (hc[tail_start:] ** 2).mean() + 1e-12)
    return ((torch.relu(db - ceil) / 10) ** 2).mean()


GAUSS_FLAT = 2 - 2 * GAUSS_L1L2  # E(|x| - 1)^2 of a unit Gaussian


def transient_flatness_t(h, gamma, start, stop, tail_start=FS // 2):
    """Mean over [start, stop) of (|h~| - 1)^2, h~ = h / gamma^n over the RMS of h / gamma^n from tail_start on:
    each early sample measured against the level of the IR's own tail, with the decay taken out (exact for a
    homogeneous FDN). Echoes that stick out and gaps between them both cost, whatever the scale of b and c.
    0 for a +-1 sequence, GAUSS_FLAT = 0.40 for Gaussian noise."""
    hc = h * gamma ** -torch.arange(len(h), dtype=h.dtype)
    x = hc[start:stop] / torch.sqrt((hc[tail_start:] ** 2).mean())
    return ((x.abs() - 1) ** 2).mean()


def level_band_t(h, gamma, start, stop, band_db=10.0, floor_db=-30.0, tail_start=FS // 2):
    """Mean over [start, stop) of the squared distance (in units of 10 dB) of each sample's level from a band of
    +-band_db around the tail's level, decay taken out: echoes above the band are pushed down, quieter samples up,
    and samples inside it cost nothing. The level is floored at floor_db, so a silent sample costs as much as an
    echo the same distance above the band (transient_flatness_t charges a silent sample 1 and a 10x echo 81, so
    training on it buys a quiet start and a burst just past its window). Samples no path reaches yet add a constant."""
    hc = h * gamma ** -torch.arange(len(h), dtype=h.dtype)
    db = 10 * torch.log10(hc[start:stop] ** 2 / (hc[tail_start:] ** 2).mean() + 10 ** (floor_db / 10))
    return (torch.relu(db.abs() - band_db) / 10).pow(2).mean()


def transient_stats(fdn, stop_s=0.1, peak_stop_s=0.5, sec=2.0):
    """Over [min(m), stop_s): transient flatness, level band (+-10 dB) and mean level, and the loudest sample up to
    peak_stop_s, all against the tail RMS with the decay taken out (levels in dB)."""
    h = torch.tensor(fdn.impulse_response(int(sec * FS)))
    start, stop, peak_stop = int(fdn.delays.min()), int(stop_s * FS), int(peak_stop_s * FS)
    hc = h * fdn.gamma ** -torch.arange(len(h), dtype=h.dtype)
    tail = (hc[FS // 2:] ** 2).mean()
    return {"flatness": float(transient_flatness_t(h, fdn.gamma, start, stop)),
            "level_band": float(level_band_t(h, fdn.gamma, start, stop)),
            "early_level_db": float(10 * torch.log10((hc[start:stop] ** 2).mean() / tail)),
            "early_peak_db": float(10 * torch.log10((hc[start:peak_stop] ** 2).max() / tail)),
            "spread": float(early_spread_t(h, max(0, start - FS // 100), peak_stop)),
            "aspread": float(arrival_spread_t(h, fdn.gamma, arrival_windows(fdn.delays, stop))),
            # level after the first echo over 0-25, 25-50, 50-100 and 100-200 ms: how the response builds up
            "buildup_db": [float(10 * torch.log10((hc[start + a * FS // 1000:start + b * FS // 1000] ** 2).mean() / tail))
                           for a, b in ((0, 25), (25, 50), (50, 100), (100, 200))],
            # direct echo b_i c_i of each line against the tail level, lines in order of delay
            "direct_db": [float(10 * np.log10((fdn.b[i] * fdn.c[i]) ** 2 / float(tail)))
                          for i in np.argsort(fdn.delays)]}


def decay_stats(fdn, t60=2.5):
    """Late T60 (EDC -5..-35 dB), early decay time and how front-loaded the energy is, at a playback T60."""
    h = fdn.with_gamma(gamma_from_t60(t60, FS)).impulse_response(int(4.5 * FS))
    e = np.cumsum((h ** 2)[::-1])[::-1]
    db = 10 * np.log10(e / e[0])
    t_at = lambda level: np.argmax(db <= level) / FS  # noqa: E731
    return {"t60_late": 2 * (t_at(-35) - t_at(-5)), "edt": 6 * t_at(-10),
            "energy_first_100ms": float((h[:FS // 10] ** 2).sum() / e[0]), "largest_tap": float((h ** 2).max() / e[0])}


def peakedness(U):
    """sum U_ij^4 / N: 1 for a permutation (lines chained into loops), 1/N for a Hadamard matrix."""
    return (U ** 4).sum() / U.shape[0]


def pair_gains(theta):
    """Gains on per-pair circles: lines (2k, 2k+1) get sqrt(2) (cos theta_k, sin theta_k). Every pair keeps the
    same power, so at least N/2 lines are tapped, and ||g||^2 = N as for b = c = 1."""
    return np.sqrt(2) * torch.stack([torch.cos(theta), torch.sin(theta)], 1).reshape(-1)


def part_train(configs=CONFIGS, steps=400, lr=0.01, sec=2.0, gamma=0.9999, mix_weights=(0.0, 1.0), unit_gains=False,
               density=0.0, density_stop=0.5, pairs="", flat=0.0, flat_stop=0.1, level=0.0, level_band=10.0,
               level_stop=0.1, spread=0.0, spread_stop=0.5, spread_lead=0.0, aspread=0.0, aspread_n=8,
               aspread_stop=0.1, build=0.0, build_db=6.0, build_rise=0.15, build_stop=0.3, direct=0.0,
               direct_lo=10.0, direct_hi=22.0):
    """flat, level, spread: weights of transient_flatness_t over [min(m), flat_stop), level_band_t (+-level_band dB)
    over [min(m), level_stop) and early_spread_t over [min(m) - spread_lead, spread_stop). Windows starting at min(m)
    put the first echo on the zero of the Hann taper, so the loss cannot see it; a lead of half a window (0.01 s)
    centres it. aspread: weight of arrival_spread_t, windows of >= aspread_n arrivals up to aspread_stop. build:
    weight of buildup_t (ramp from -build_db to 0 dB over build_rise s) over arrival windows up to build_stop.
    direct: weight of direct_ceiling_t (direct echoes under direct_lo .. direct_hi dB, shortest to longest line). pairs: "c" or "bc" puts those gain vectors on per-pair circles (random phases, the other vector = 1);
    "quad" puts c on them and b on the same circles 90 degrees ahead."""
    torch.set_num_threads(4)
    outdir = ROOT / "runs" / "autocorr"
    outdir.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    out = {}
    for cfg in configs:
        init = variants(cfg)["init"]
        U0, L = torch.tensor(init.U), int(sec * FS)
        ones = torch.ones(init.N, dtype=torch.float64)
        awins = arrival_windows(init.delays, int(aspread_stop * FS), aspread_n) if aspread else None
        bwins = arrival_windows(init.delays, int(build_stop * FS), aspread_n) if build else None
        for mix in mix_weights:
            T = torch.zeros(init.N, init.N, dtype=torch.float64, requires_grad=True)
            if pairs:
                assert init.N % 2 == 0
                th = {k: (2 * np.pi * torch.rand(init.N // 2, generator=gen, dtype=torch.float64)).requires_grad_()
                      for k in ("c" if pairs == "quad" else pairs)}
                params = [T, *th.values()]

                def gains():
                    if pairs == "quad":  # per pair b_i c_i = -/+ sin(2 theta): equal magnitudes, never above 1
                        return pair_gains(th["c"] + np.pi / 2), pair_gains(th["c"])
                    return tuple(pair_gains(th[k]) if k in th else ones for k in "bc")
            else:
                b = ones if unit_gains else torch.tensor(init.b, requires_grad=True)
                c = ones if unit_gains else torch.tensor(init.c, requires_grad=True)
                params = [T] if unit_gains else [T, b, c]

                def gains():
                    return b, c
            opt = torch.optim.Adam(params, lr=lr)
            curve, t0 = [], time.perf_counter()
            for _ in range(steps):
                opt.zero_grad()
                U = rotated(U0, T)
                b, c = gains()
                h = unrolled_ir(U, b, c, init.delays, gamma, L)
                loss = whiteness_t(h) + mix * peakedness(U)
                if density:
                    loss = loss + density * early_sparsity_t(h, int(init.delays.min()), int(density_stop * FS))
                if flat:
                    loss = loss + flat * transient_flatness_t(h, gamma, int(init.delays.min()), int(flat_stop * FS))
                if spread:
                    loss = loss + spread * early_spread_t(h, max(0, int(init.delays.min()) - int(spread_lead * FS)),
                                                          int(spread_stop * FS))
                if aspread:
                    loss = loss + aspread * arrival_spread_t(h, gamma, awins)
                if build:
                    loss = loss + build * buildup_t(h, gamma, bwins, build_db, build_rise)
                if direct:
                    loss = loss + direct * direct_ceiling_t(h, gamma, b, c, init.delays, direct_lo, direct_hi)
                if level:
                    loss = loss + level * level_band_t(h, gamma, int(init.delays.min()), int(level_stop * FS), level_band)
                loss.backward()
                opt.step()
                curve.append(loss.item())
            dt = time.perf_counter() - t0
            b, c = gains()
            # evaluate at the training decay of the reference runs, like every other variant
            fdn = FDN(init.delays, rotated(U0, T).detach().numpy(), b.detach().numpy(), c.detach().numpy(),
                      init.gamma, meta={"objective": f"whiteness + {mix:g} * peakedness(U) + {density:g} * early_sparsity(0-"
                                         f"{density_stop:g} s) + {flat:g} * transient_flatness(0-{flat_stop:g} s) + {level:g} * "
                                         f"level_band(+-{level_band:g} dB, 0-{level_stop:g} s) + {spread:g} * "
                                         f"early_spread(0-{spread_stop:g} s, lead {spread_lead:g} s) + {aspread:g} * "
                                         f"arrival_spread(>= {aspread_n} arrivals, 0-{aspread_stop:g} s) + {build:g} * "
                                         f"buildup(-{build_db:g} dB over {build_rise:g} s, 0-{build_stop:g} s) + {direct:g} * "
                                         f"direct_ceiling({direct_lo:g}..{direct_hi:g} dB), "
                                         f"{sec:g} s window, gamma {gamma}",
                            "config": cfg})
            _, res = modal_decomposition(fdn)
            met = all_metrics(fdn, res)
            met["white_full"] = whiteness(fdn.impulse_response(int(2.5 * FS)))
            met["peakedness"] = float(peakedness(fdn.U))
            met["early_sparsity"] = early_sparsity(fdn)
            met.update(decay_stats(fdn))
            met.update(transient_stats(fdn))
            name = (f"{cfg}_white_mix{mix:g}" + (f"_{pairs}pairs" if pairs else "_unitgains" if unit_gains else "")
                    + (f"_dens{density:g}" if density else "") + (f"_flat{flat:g}" if flat else "")
                    + (f"_level{level:g}b{level_band:g}" if level else "")
                    + (f"_spread{spread:g}" + (f"s{spread_stop:g}" if spread_stop != 0.5 else "")
                       + (f"l{spread_lead:g}" if spread_lead else "") if spread else "")
                    + (f"_aspread{aspread:g}n{aspread_n}" if aspread else "")
                    + (f"_build{build:g}d{build_db:g}r{1000 * build_rise:g}" if build else "")
                    + (f"_direct{direct:g}l{direct_lo:g}h{direct_hi:g}" if direct else ""))
            (outdir / f"{name}.json").write_text(json.dumps(fdn.to_json()), encoding="utf-8")
            out[name] = {"config": cfg, "mix_weight": mix, "density_weight": density, "unit_gains": unit_gains,
                         "pairs": pairs, "flat_weight": flat, "level_weight": level, "level_band": level_band,
                         "spread_weight": spread, "spread_stop": spread_stop, "spread_lead": spread_lead,
                         "aspread_weight": aspread, "aspread_n": aspread_n, "aspread_stop": aspread_stop,
                         "build_weight": build, "build_db": build_db, "build_rise": build_rise, "build_stop": build_stop,
                         "direct_weight": direct, "direct_lo": direct_lo, "direct_hi": direct_hi,
                         "curve": curve, "time_s": dt, "metrics": met}
            print(f"train {name:28s} loss {curve[0]:.3f} -> {curve[-1]:.3f} in {dt:5.1f}s | sigma {met['std_db']:.2f} "
                  f"p99 {met['p99_db']:+.2f} max {met['max_db']:+.2f} diag {met['diag_energy']:.2f} "
                  f"peaked {met['peakedness']:.2f} NED 0.1-0.5 {met['ned_mean_100_500ms']:.2f} "
                  f"sparse {met['early_sparsity']:.2f} flat {met['flatness']:.2f} band {met['level_band']:.2f} early level {met['early_level_db']:+.1f} "
                  f"peak {met['early_peak_db']:.1f} dB spread {met['spread']:.2f} aspread {met['aspread']:.2f} "
                  f"build-up {' '.join(f'{v:+.1f}' for v in met['buildup_db'])} dB "
                  f"direct {' '.join(f'{v:+.0f}' for v in met['direct_db'])} dB "
                  f"tap {100 * met['largest_tap']:.1f}% EDT {met['edt']:.2f}s")
    return out


def part_train_unit():
    """Same, with b = c = 1 fixed (as in the Householder / Haar baselines): only U is trained."""
    return part_train(mix_weights=(1.0,), unit_gains=True)


def part_train_density(configs=CONFIGS, weights=(1.0, 3.0)):
    """Whiteness + mixing + early density, with b = c = 1 fixed and with trained gains."""
    out = {}
    for unit in (True, False):
        for wd in weights:
            out.update(part_train(configs, mix_weights=(1.0,), unit_gains=unit, density=wd))
    return out


def part_train_pairs(configs=CONFIGS):
    """Whiteness + mixing + density 3 with gains on per-pair circles: c only (b = 1), and both b and c."""
    out = {}
    for pairs in ("c", "bc"):
        out.update(part_train(configs, mix_weights=(1.0,), density=3.0, pairs=pairs))
    return out


def part_train_quad(configs=CONFIGS):
    """b and c on the same per-pair circles 90 degrees apart, so a line with a large c gets a small b."""
    return part_train(configs, mix_weights=(1.0,), density=3.0, pairs="quad")


def part_train_flat(weight, density="3"):
    """Free gains with the first 100 ms trained towards the tail's level, sample by sample, on
    top of whiteness + mixing (+ density)."""
    return part_train(mix_weights=(1.0,), density=float(density), flat=float(weight))


def part_train_level(weight, band="10", density="3", configs="all"):
    """Free gains with every sample of the first 100 ms driven into a +-band dB range around the tail's level, on
    top of whiteness + mixing (+ density). configs: "all" or comma-separated names."""
    return part_train(CONFIGS if configs == "all" else configs.split(","), mix_weights=(1.0,), density=float(density),
                      level=float(weight), level_band=float(band))


def part_train_spread(weight, density="3", configs="all", stop="0.5", lead="0.01"):
    """Free gains with the effective number of echoes per 20 ms window up to stop seconds pushed towards noise: the
    arriving energy shared evenly, loud echoes down and quiet ones up, on top of whiteness + mixing (+ density).
    The windows start lead seconds before the first echo (runs from before the fix used lead 0)."""
    return part_train(CONFIGS if configs == "all" else configs.split(","), mix_weights=(1.0,), density=float(density),
                      spread=float(weight), spread_stop=float(stop), spread_lead=float(lead))


def part_train_aspread(weight, density="3", configs="all", n="8"):
    """Free gains with the spread measured over windows of at least n arrivals (20 ms once the IR is dense enough)
    in the first 100 ms, on top of whiteness + mixing (+ density)."""
    return part_train(CONFIGS if configs == "all" else configs.split(","), mix_weights=(1.0,), density=float(density),
                      aspread=float(weight), aspread_n=int(n))


def part_train_build(weight, depth="6", rise="0.15", aspread="0.5", density="3", configs="all"):
    """Free gains with the early envelope held to a build-up ramp (-depth dB at the first echo to 0 dB after rise s),
    on top of whiteness + mixing + density + arrival-window spread."""
    return part_train(CONFIGS if configs == "all" else configs.split(","), mix_weights=(1.0,), density=float(density),
                      aspread=float(aspread), build=float(weight), build_db=float(depth), build_rise=float(rise))


def part_train_direct(weight, lo="10", hi="22", build="30", aspread="0.5", density="3", configs="all"):
    """W18's objective (density + arrival spread + build-up ramp) plus a ceiling on the direct echoes that rises with
    the delay, so the shortest lines cannot lead the response (as in DiffFDN's N6 gains)."""
    return part_train(CONFIGS if configs == "all" else configs.split(","), mix_weights=(1.0,), density=float(density),
                      aspread=float(aspread), build=float(build), direct=float(weight), direct_lo=float(lo),
                      direct_hi=float(hi))


def part_decay(t60=2.5):
    """EDT vs late T60 and how front-loaded the energy is, at the previewer's expDE playback decay."""
    out = {}
    for cfg in CONFIGS:
        ref = variants(cfg)["optimized"]
        rows = {"DiffFDN": ref, "Haar b=c=1": baseline("random", ref.delays)}
        for path in sorted((ROOT / "runs" / "autocorr").glob(f"{cfg}_white_mix1*.json")):
            rows[path.stem.removeprefix(f"{cfg}_")] = FDN.from_json(json.loads(path.read_text(encoding="utf-8")))
        for name, f in rows.items():
            r = decay_stats(f, t60)
            r["early_sparsity"] = early_sparsity(f)
            out[f"{cfg}|{name}"] = r
            print(f"decay {cfg} {name:28s} T60(-5..-35) {r['t60_late']:.2f}s EDT {r['edt']:.2f}s "
                  f"energy<100ms {100 * r['energy_first_100ms']:4.1f}% largest tap {100 * r['largest_tap']:5.2f}% "
                  f"early sparsity {r['early_sparsity']:.2f}")
    return out


# ------------------------------------------------------------------ figures
def figures(d):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    if "sens" in d:
        for gamma, col, lab in (("1.0", COLORS["optimized"], "lossless (γ = 1)"),
                                ("0.9999", COLORS["random"], "γ = 0.9999 (training)")):
            s = d["sens"][gamma]
            ax.loglog(s["t"], s["rel"], color=col, lw=2, label=f"{lab}, slope {s['loglog_slope']:.2f}")
        ax.set_xlabel("time in the IR (s)")
        ax.set_ylabel("‖∂h/∂θ‖ / ‖h‖ per 100 ms window")
        ax.set_title("IR sensitivity to a rotation of U grows with time", loc="left")
        ax.legend(loc="upper left")

    ax = axes[1]
    if "slice" in d:
        t = np.array(d["slice"]["t"])
        for (key, col) in zip(("1.0_0.1", "1.0_1.0", "1.0_10.0"), RAMP):
            s = d["slice"][key]
            v = np.array(s["loss"])
            ax.plot(t, v / v[len(v) // 2], color=col, lw=1.5,
                    label=f"lossless, {s['seconds']:g} s window")
        s = d["slice"]["0.9999_10.0"]
        v = np.array(s["loss"])
        ax.plot(t, v / v[len(v) // 2], color=COLORS["random"], lw=1.5, ls=(0, (4, 2)),
                label="γ = 0.9999, 10 s window")
        ax.set_xlabel("rotation angle along a random direction (rad)")
        ax.set_ylabel("whiteness / value at 0")
        ax.set_title("Loss along one path stays smooth at every length", loc="left")
        ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[2]
    if "limit" in d:
        for name in ("init", "optimized"):
            rows = d["limit"][name]
            sec = [r["seconds"] for r in rows]
            ax.loglog(sec, [r["white"] + 1 for r in rows], color=COLORS[name], lw=2, marker="o", ms=4,
                      label=f"{'RO init' if name == 'init' else 'DiffFDN'}: measured")
            ax.loglog(sec, [r["predicted"] + 1 for r in rows], color=COLORS[name], lw=1.2, ls=(0, (4, 2)),
                      label=f"{'RO init' if name == 'init' else 'DiffFDN'}: (2L/3)·Σ|ρ|⁴/(Σ|ρ|²)²")
        ax.axhline(2, color=MUTED, lw=1, ls=":")
        ax.text(3, 1.6, "Gaussian noise ≈ 2", color=INK2, fontsize=8)
        ax.set_xlabel("window length L (s), lossless")
        ax.set_ylabel("whiteness + 1")
        ax.set_title("Long windows: whiteness → modal 4th moment", loc="left")
        ax.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIG / "autocorr_study.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parts = sys.argv[1:] or ["metrics", "limit", "grad", "sens", "slice"]
    lock = OUT.with_suffix(".lock")
    for p in parts:  # "name:arg:arg" passes arguments
        name, *args = p.split(":")
        result = globals()[f"part_{name}"](*args)
        while True:  # parallel runs: read-modify-write under a lock file so they keep each other's parts
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL)
                break
            except FileExistsError:
                time.sleep(0.05)
        try:
            data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
            old = data.get(p)
            # merge runs into an existing part, so a rerun with new defaults (new run names) keeps the old runs
            data[p] = {**old, **result} if isinstance(old, dict) and isinstance(result, dict) else result
            OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
        finally:
            os.close(fd)
            os.remove(lock)
    if {"metrics", "limit", "grad", "sens", "slice"} & {p.split(":")[0] for p in parts}:
        figures(data)
