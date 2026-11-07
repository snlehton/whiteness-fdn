"""Modal-excitation statistics of Haar-random orthogonal FDNs (many draws) for each Table 1 delay set,
to judge DiffFDN against a typical random matrix rather than the near-identity reference init.

    python scripts/baseline_stats.py [draws]
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fdnopt.fdn import baseline  # noqa: E402
from fdnopt.metrics import mixing_stats, modal_excitation_stats  # noqa: E402
from fdnopt.modal import lossless_pole_angles, polish, residues  # noqa: E402

SETS = {
    "N4_set1": [1499, 1889, 2381, 2999], "N4_set2": [797, 839, 2381, 2999],
    "N6_set1": [997, 1153, 1327, 1559, 1801, 2099], "N6_set2": [887, 911, 941, 1699, 1951, 2053],
    "N8_set1": [809, 877, 937, 1049, 1151, 1249, 1373, 1499], "N8_set2": [241, 263, 281, 293, 1193, 1319, 1453, 1597],
}
KEYS = ["std_db", "p99_db", "max_db", "frac_above_6db"]


def main(draws=20):
    out = {}
    for name, delays in SETS.items():
        rows = {"unit": [], "gauss": []}
        diag = []
        for seed in range(draws):
            f = baseline("random", delays, 0.9999, seed=seed)
            poles = polish(f, f.gamma * np.exp(1j * lossless_pole_angles(f.delays, f.U)))
            rows["unit"].append(modal_excitation_stats(residues(f, poles)))
            rng = np.random.default_rng(1000 + seed)
            g = replace(f, b=rng.standard_normal(f.N), c=rng.standard_normal(f.N))
            rows["gauss"].append(modal_excitation_stats(residues(g, poles)))
            diag.append(mixing_stats(f)["diag_energy"])
        out[name] = {k: {m: [float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))] for m in KEYS}
                     for k, v in rows.items()}
        out[name]["diag_energy"] = [float(np.mean(diag)), float(np.std(diag))]
        u, gs = out[name]["unit"], out[name]["gauss"]
        print(f"{name}: unit gains σ {u['std_db'][0]:.2f}±{u['std_db'][1]:.2f}  p99 {u['p99_db'][0]:+.2f}±{u['p99_db'][1]:.2f}  "
              f"max {u['max_db'][0]:+.2f} | gauss gains σ {gs['std_db'][0]:.2f}±{gs['std_db'][1]:.2f}  "
              f"p99 {gs['p99_db'][0]:+.2f}  max {gs['max_db'][0]:+.2f} | diag {out[name]['diag_energy'][0]:.2f}", flush=True)
    (ROOT / "runs" / "haar_baseline_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
