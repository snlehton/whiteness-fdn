"""Paper Fig. 4: temporal loss ||h||_2/||h||_1 for feedback matrices interpolated between I and U,
A_beta = expm(beta * logm(U)), evaluated over many frequency-sampling sizes M ~ U[0.8K, K].

    python scripts/fig4_temporal_vs_beta.py runs/reference/N8_set1
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import expm, logm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fdnopt.fdn import load_reference_mat  # noqa: E402

K = 480000


@torch.no_grad()
def temporal_loss(U, b, c, delays, gamma, M, dev="cuda"):
    """Reference temporal loss for one frequency-sampling size M (same masking/zero padding as the solver)."""
    z = torch.polar(torch.ones(M, dtype=torch.float64), torch.linspace(0, 1, M, dtype=torch.float64) * np.pi).to(dev)
    m = torch.tensor(delays, dtype=torch.float64, device=dev)
    A = torch.tensor(U * gamma ** np.asarray(delays)[None, :], dtype=torch.complex128, device=dev)
    P = torch.diag_embed(z[:, None] ** m) - A
    x = torch.linalg.solve(P, torch.tensor(b, dtype=torch.complex128, device=dev).expand(M, -1))
    H = torch.zeros(K, dtype=torch.complex128, device=dev)
    H[:M] = x @ torch.tensor(c, dtype=torch.complex128, device=dev)
    h = torch.fft.irfft(H, norm="ortho")
    return (torch.linalg.norm(h, 2) / torch.linalg.norm(h, 1)).item()


def main(run):
    run = Path(run).resolve()
    f = load_reference_mat(run / "parameters_init.mat")
    L = np.real(logm(f.U))
    rng = np.random.default_rng(0)
    Ms = rng.integers(int(0.8 * K), K, 32)
    betas = [0, 0.25, 0.5, 0.75, 1.0]
    res = {}
    for beta in betas:
        Ub = expm(beta * L)
        res[beta] = [temporal_loss(Ub, f.b, f.c, f.delays, f.gamma, int(M)) for M in Ms]
        print(f"beta={beta}: median {np.median(res[beta]) * 1e3:.3f}e-3")
    out = ROOT / "docs" / "figures" / f"fig4_temporal_vs_beta_{run.name}.png"
    fig, ax = plt.subplots(figsize=(4.6, 2.7), dpi=150)
    ax.boxplot([np.array(res[b]) * 1e3 for b in betas], tick_labels=[str(b) for b in betas],
               medianprops={"color": "#2a78d6", "linewidth": 2}, flierprops={"markersize": 3})
    ax.set_xlabel("β  (0 = identity / parallel combs, 1 = initial U)")
    ax.set_ylabel("L_temporal × 10³")
    ax.set_title(f"Temporal loss vs. matrix density · {run.name}", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    (run / "fig4_temporal_vs_beta.json").write_text(json.dumps({str(b): v for b, v in res.items()}))
    print(out)


if __name__ == "__main__":
    import evaluate  # noqa: F401  (shared matplotlib style)
    evaluate.style()
    main(sys.argv[1])
