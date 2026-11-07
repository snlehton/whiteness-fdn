"""Markdown tables for docs/reproduction.md from runs/*/eval.json, runs/haar_baseline_stats.json and
loss_components.json / fig4 outputs. Prints to stdout.

    python scripts/report_tables.py > runs/report_tables.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["N4_set1", "N4_set2", "N6_set1", "N6_set2", "N8_set1", "N8_set2"]
AUTHORS = ["pretrained00", "pretrained01", "pretrained02", "pretrained03", "pretrained04"]


def load_eval(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def arrow(a, b, fmt="{:.2f}", signed=False):
    f = (lambda v: ("+" if v >= 0 else "") + fmt.format(v)) if signed else fmt.format
    return f"{f(a)} → **{f(b)}**"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    haar_path = ROOT / "runs" / "haar_baseline_stats.json"
    haar = json.loads(haar_path.read_text()) if haar_path.exists() else {}

    print("### Modal excitation, init → optimized (dB relative to the mean of 20·log10|ρ|)\n")
    print("| Configuration | M | σ | p99 | loudest mode | modes > +6 dB | Haar random σ (20 draws) | Haar p99 | Householder σ / p99 |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = [(c, ROOT / "runs" / "reference" / c / "eval.json", "ours") for c in CONFIGS]
    rows += [(a, ROOT / "external" / "diff-fdn-colorless-dafx23" / "demo" / a / "eval.json", "authors") for a in AUTHORS]
    for name, path, who in rows:
        e = load_eval(path)
        if not e:
            continue
        i, o, h = e["metrics"]["init"], e["metrics"]["optimized"], e["metrics"]["householder"]
        key = next((c for c in CONFIGS if e["label"].startswith(c.replace("_set", " set #"))), None)
        hz = haar.get(key, {}).get("unit")
        hs = f"{hz['std_db'][0]:.2f} ± {hz['std_db'][1]:.2f}" if hz else "–"
        hp = f"+{hz['p99_db'][0]:.2f} ± {hz['p99_db'][1]:.2f}" if hz else "–"
        print(f"| {e['label']} | {sum(e['delays'])} | {arrow(i['std_db'], o['std_db'])} | {arrow(i['p99_db'], o['p99_db'], signed=True)} | "
              f"{arrow(i['max_db'], o['max_db'], signed=True)} | {100 * i['frac_above_6db']:.1f}% → **{100 * o['frac_above_6db']:.1f}%** | "
              f"{hs} | {hp} | {h['std_db']:.2f} / +{h['p99_db']:.2f} |")

    print("\n### Mixing and density, init → optimized\n")
    print("| Configuration | ‖diag U‖²/N (dense ≈ 1/N) | ‖h‖₂/‖h‖₁ ×10³ | NED 0.1–0.5 s | NED 0.5–1 s | Householder NED 0.1–0.5 s | Haar draw NED 0.1–0.5 s |")
    print("|---|---|---|---|---|---|---|")
    for name, path, who in rows:
        e = load_eval(path)
        if not e:
            continue
        i, o, h = e["metrics"]["init"], e["metrics"]["optimized"], e["metrics"]["householder"]
        r = e["metrics"].get("random")
        rn = f"{r['ned_mean_100_500ms']:.2f}" if r else "–"
        print(f"| {e['label']} | {arrow(i['diag_energy'], o['diag_energy'])} (1/N = {i['diag_energy_dense_ref']:.2f}) | "
              f"{arrow(1e3 * i['sparsity_ir'], 1e3 * o['sparsity_ir'])} | {arrow(i['ned_mean_100_500ms'], o['ned_mean_100_500ms'])} | "
              f"{arrow(i['ned_mean_500_1000ms'], o['ned_mean_500_1000ms'])} | {h['ned_mean_100_500ms']:.2f} | {rn} |")

    print("\n### Loss components per epoch (fixed evaluation batch of 8 sizes M)\n")
    print("| Configuration | L_spectral init → epoch 1 → final | α·L_temporal init → epoch 1 → final |")
    print("|---|---|---|")
    for c in CONFIGS:
        p = ROOT / "runs" / "reference" / c / "loss_components.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s, t, a = d["spectral"], d["temporal"], d["alpha"]
        print(f"| {c} | {s[0]:.3f} → {s[1]:.3f} → {s[-1]:.3f} | {a * t[0]:.3f} → {a * t[1]:.3f} → {a * t[-1]:.3f} |")

    for c in CONFIGS:
        p = ROOT / "runs" / "reference" / c / "fig4_temporal_vs_beta.json"
        if p.exists():
            d = json.loads(p.read_text())
            import numpy as np
            print(f"\n### Temporal loss vs β ({c}, median over 32 sizes M, ×10³)\n")
            print("| β | " + " | ".join(d.keys()) + " |")
            print("|---" * (len(d) + 1) + "|")
            print("| L_temporal ×10³ | " + " | ".join(f"{1e3 * np.median(v):.3f}" for v in d.values()) + " |")


if __name__ == "__main__":
    main()
