"""Objective evaluation of trained DiffFDN runs (paper Sec. 4) + export for the web previewer.

For every run directory holding parameters_init.mat / parameters.mat / args.txt (reference solver
output) this computes the modal decomposition and coloration metrics of
    init        - random orthogonal start of the optimisation (paper condition "RO")
    optimized   - DiffFDN result
    householder - Householder matrix with unit gains (paper condition "HH")
    random      - Haar-random orthogonal matrix with unit gains
    custom      - (web only) alternative training results for the same delays (ALTERNATIVES), selectable in the
                  previewer; the first one present is the default
and writes <run>/eval.json, docs/figures/*.png and web/data/*.json.

    python scripts/evaluate.py                # all runs under runs/reference + authors' pretrained sets
    python scripts/evaluate.py runs/foo/bar   # specific runs
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fdnopt.fdn import FDN, baseline, load_reference_mat, read_args_txt  # noqa: E402
from fdnopt.metrics import all_metrics  # noqa: E402
from fdnopt.modal import cached_modal  # noqa: E402

FIG = ROOT / "docs" / "figures"
WEB = ROOT / "web" / "data"
CACHE = ROOT / "runs" / "cache"

# categorical identity, fixed across all figures and the web viewer (pairwise-validated, see web/style.css)
COLORS = {"optimized": "#2a78d6", "init": "#e87ba4", "householder": "#008300", "random": "#eda100"}
LABELS = {"optimized": "DiffFDN (optimized)", "init": "RO (init)", "householder": "HH (unit gains)",
          "random": "Haar random (unit gains)"}
VARIANTS = ["init", "optimized", "householder", "random"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_SPECTRAL, C_TEMPORAL = "#6250d6", "#008300"

# alternative training results (scripts/autocorr_study.py, runs/autocorr/<config>_<suffix>.json) offered in the
# previewer's custom slot of each reference run: (experiment id, file suffix, label). Ids number the listenable
# variants in the order they were made; the menu lists the paper's final models first, then the rest newest
# first, and preselects the first one present.
ALTERNATIVES = [
    # the paper's final models first, so each configuration opens on its own (W19 for N = 4 and 6, W14 for N = 8)
    ("W19", "white_mix1_dens3_aspread0.5n8_build30d6r150_direct3l10h22", "W18 + direct-echo ceiling (+10 dB shortest line to +22 dB longest), weight 3 (paper's final model for N = 4 and 6)"),
    ("W14", "white_mix1_dens3_spread0.5s0.1l0.01", "Whiteness + mixing + density 3 + early spread (20 ms windows), weight 0.5, trained gains (paper's final model for N = 8)"),
    ("W20", "white_mix1_dens3_aspread0.5n8_build30d6r150_direct10l10h22", "W18 + direct-echo ceiling (+10 dB shortest line to +22 dB longest), weight 10"),
    ("W18", "white_mix1_dens3_aspread0.5n8_build30d6r150", "W15 + build-up ramp (−6 dB to tail level over 150 ms), weight 30"),
    ("W17", "white_mix1_dens3_aspread0.5n8_build10d6r150", "W15 + build-up ramp (−6 dB to tail level over 150 ms), weight 10"),
    ("W16", "white_mix1_dens3_aspread1n8", "Whiteness + mixing + density 3 + arrival-window spread, weight 1, trained gains"),
    ("W15", "white_mix1_dens3_aspread0.5n8", "Whiteness + mixing + density 3 + arrival-window spread, weight 0.5, trained gains"),
    ("W13", "white_mix1_spread0.5s0.1l0.01", "Whiteness + mixing + early spread (20 ms windows), weight 0.5, trained gains"),
    ("W12", "white_mix1_spread0.5s0.1", "Whiteness + mixing + early spread (20 ms windows, first echo unseen), weight 0.5, trained gains"),
    ("W11", "white_mix1_dens3_flat3", "Whiteness + mixing + density 3 + per-sample level target (quiet start), trained gains"),
    ("W10", "white_mix1_quadpairs_dens3", "Whiteness + mixing + density 3, b and c on pair circles, 90° apart"),
    ("W09", "white_mix1_bcpairs_dens3", "Whiteness + mixing + density 3, b and c on pair circles"),
    ("W08", "white_mix1_cpairs_dens3", "Whiteness + mixing + density 3, b = 1, c on pair circles"),
    ("W07", "white_mix1_dens3", "Whiteness + mixing + density 3, trained gains"),
    ("W06", "white_mix1_dens1", "Whiteness + mixing + density 1, trained gains"),
    ("W05", "white_mix1_unitgains_dens3", "Whiteness + mixing + density 3, b = c = 1"),
    ("W04", "white_mix1_unitgains_dens1", "Whiteness + mixing + density 1, b = c = 1"),
    ("W03", "white_mix1_unitgains", "Whiteness + mixing, b = c = 1"),
    ("W02", "white_mix1", "Whiteness + mixing, trained gains"),
    ("W01", "white_mix0", "Whiteness only, trained gains"),
]

DELAY_SETS = {
    (1499, 1889, 2381, 2999): "N4 set #1", (797, 839, 2381, 2999): "N4 set #2",
    (997, 1153, 1327, 1559, 1801, 2099): "N6 set #1", (887, 911, 941, 1699, 1951, 2053): "N6 set #2",
    (809, 877, 937, 1049, 1151, 1249, 1373, 1499): "N8 set #1",
    (241, 263, 281, 293, 1193, 1319, 1453, 1597): "N8 set #2",
}


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
    })


def discover(paths):
    if paths:
        return [Path(p).resolve() for p in paths]
    runs = sorted(p.parent for p in (ROOT / "runs" / "reference").glob("*/parameters.mat"))
    runs += sorted(p.parent for p in (ROOT / "external" / "diff-fdn-colorless-dafx23" / "demo").glob("pretrained*/parameters.mat"))
    return runs


def run_id(run: Path) -> tuple[str, str, str]:
    """(id, group, label)"""
    args = read_args_txt(run / "args.txt")
    dset = DELAY_SETS.get(tuple(int(round(d)) for d in args["delays"]), f"N{len(args['delays'])} custom")
    if "pretrained" in run.name:
        return f"authors_{run.name}", "authors", f"{dset} · authors' {run.name} (α={args['alpha']}, {args['max_epochs']} ep)"
    rel = run.relative_to(ROOT / "runs")
    return "_".join(rel.parts), rel.parts[0], f"{dset} · {rel.parts[0]}/{rel.name}"


def evaluate_run(run: Path):
    args = read_args_txt(run / "args.txt")
    delays = np.rint(args["delays"]).astype(int)
    gamma = float(args["gain_per_sample"])
    fdns = {
        "init": load_reference_mat(run / "parameters_init.mat", delays, gamma),
        "optimized": load_reference_mat(run / "parameters.mat", delays, gamma),
        "householder": baseline("householder", delays, gamma),
        "random": baseline("random", delays, gamma, seed=0),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    key = "-".join(map(str, delays))
    modal, metrics = {}, {}
    for name, f in fdns.items():
        cache = CACHE / f"{name}_{key}_{gamma}.npz" if name in ("householder", "random") else run / f"modal_{name}.npz"
        poles, res = cached_modal(f, cache)
        modal[name] = (poles, res)
        metrics[name] = all_metrics(f, res)
    alts = []
    for wid, suffix, label in ALTERNATIVES:
        long = f"{wid} · {label}"
        path = ROOT / "runs" / "autocorr" / f"{run.name}_{suffix}.json"
        if not path.exists():
            continue
        f = FDN.from_json(json.loads(path.read_text(encoding="utf-8"))).with_gamma(gamma)
        poles, res = cached_modal(f, run / f"modal_{suffix}.npz")
        alts.append({"id": suffix, "short": "Whiteness", "long": long, "fdn": f, "modal": (poles, res),
                     "metrics": all_metrics(f, res)})
    labels = {}
    if alts:  # the preselected alternative fills the custom slot
        fdns["custom"], modal["custom"], metrics["custom"] = alts[0]["fdn"], alts[0]["modal"], alts[0]["metrics"]
        labels["custom"] = {"short": alts[0]["short"], "long": alts[0]["long"]}
    rid, group, label = run_id(run)
    out = {"id": rid, "group": group, "label": label, "run_dir": str(run.relative_to(ROOT)),
           "delays": delays.tolist(), "gamma_train": gamma, "args": args, "metrics": metrics, "labels": labels}
    (run / "eval.json").write_text(json.dumps(out, indent=1, default=str))
    return out, fdns, modal, alts


def fig_modal_hist(out, modal, path):
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=150)
    bins = np.arange(-30, 30.5, 0.5)
    dashes = {"householder": (0, (4, 2)), "random": (0, (1.5, 1.5))}
    for name in VARIANTS:
        db = 20 * np.log10(np.abs(modal[name][1]) + 1e-300)
        db = db[db > np.median(db) - 100.0]   # drop unexcited modes (see modal_excitation_stats)
        db -= db.mean()
        cnt, _ = np.histogram(db, bins)
        ax.stairs(cnt, bins, color=COLORS[name], linewidth=2 if name not in dashes else 1.4,
                  linestyle=dashes.get(name, "-"),
                  label=f"{LABELS[name]} · σ={db.std():.1f} dB, p99={np.percentile(db, 99):+.1f} dB")
    ax.set_xlim(-20, 26)
    ax.set_xlabel("Modal excitation 20·log10|ρᵢ| relative to mean (dB)")
    ax.set_ylabel("Modes per 0.5 dB")
    ax.set_title(out["label"], loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_losses(out, run, path):
    comp_path = run / "loss_components.json"
    if not comp_path.exists():
        return None
    comp = json.loads(comp_path.read_text())
    ep = np.array(comp["epoch"]) + 1   # model_init -> 0, model_e0 -> 1 (after first epoch)
    fig, ax = plt.subplots(figsize=(4.6, 2.7), dpi=150)
    ax.plot(ep, comp["spectral"], color=C_SPECTRAL, lw=2, marker="o", ms=3, label="L_spectral")
    ax.plot(ep, np.array(comp["temporal"]) * comp["alpha"], color=C_TEMPORAL, lw=2, marker="o", ms=3,
            label=f"α·L_temporal (α={comp['alpha']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(out["label"], loc="left", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return comp


def web_fdn(f):
    return {"U": np.round(f.U, 8).tolist(), "b": np.round(f.b, 8).tolist(), "c": np.round(f.c, 8).tolist()}


def web_modal(poles, res):
    return {"angle": np.round(np.angle(poles), 6).tolist(), "db": np.round(20 * np.log10(np.abs(res) + 1e-300), 3).tolist()}


def export_web(out, fdns, modal, run, alts=()):
    WEB.mkdir(parents=True, exist_ok=True)
    losses = {}
    if (run / "losses.mat").exists():
        L = sio.loadmat(run / "losses.mat")
        losses = {"train": np.ravel(L["train"]).tolist(), "valid": np.ravel(L["valid"]).tolist()}
    if (run / "loss_components.json").exists():
        losses["components"] = json.loads((run / "loss_components.json").read_text())
    data = {
        "id": out["id"], "group": out["group"], "label": out["label"], "delays": out["delays"],
        "fs": 48000, "gamma_train": out["gamma_train"],
        "variants": {k: web_fdn(f) for k, f in fdns.items()},
        "modal": {k: web_modal(*pr) for k, pr in modal.items()},
        "metrics": out["metrics"], "losses": losses, "args": {k: str(v) for k, v in out["args"].items()},
        "labels": out.get("labels", {}),
        "alternatives": [{"id": a["id"], "short": a["short"], "long": a["long"], **web_fdn(a["fdn"]),
                          "modal": web_modal(*a["modal"]), "metrics": a["metrics"]} for a in alts],
    }
    (WEB / f"{out['id']}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def write_index():
    items = []
    for p in sorted(WEB.glob("*.json")):
        if p.name == "index.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        items.append({"id": d["id"], "group": d["group"], "label": d["label"], "N": len(d["delays"]), "file": p.name})
    (WEB / "index.json").write_text(json.dumps(items, indent=1), encoding="utf-8")


def summary_table(outs):
    rows = ["| Run | variant | σ (dB) | p99 (dB) | max (dB) | >+6 dB | ‖diag U‖²/N | ‖h‖₂/‖h‖₁ ×1e3 | NED 0.1–0.5 s | NED 0.5–1 s |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for o in outs:
        for v in VARIANTS:
            m = o["metrics"][v]
            rows.append(f"| {o['label']} | {LABELS[v]} | {m['std_db']:.2f} | {m['p99_db']:+.2f} | {m['max_db']:+.2f} | "
                        f"{100 * m['frac_above_6db']:.1f}% | {m['diag_energy']:.2f} | {1e3 * m['sparsity_ir']:.2f} | "
                        f"{m['ned_mean_100_500ms']:.2f} | {m['ned_mean_500_1000ms']:.2f} |")
    return "\n".join(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    outs = []
    for run in discover(a.runs):
        out, fdns, modal, alts = evaluate_run(run)
        fig_modal_hist(out, modal, FIG / f"modal_{out['id']}.png")
        fig_losses(out, run, FIG / f"loss_{out['id']}.png")
        export_web(out, fdns, modal, run, alts)
        outs.append(out)
        m0, m1 = out["metrics"]["init"], out["metrics"]["optimized"]
        print(f"{out['label']:60s} σ {m0['std_db']:.2f}->{m1['std_db']:.2f} dB  p99 {m0['p99_db']:+.2f}->{m1['p99_db']:+.2f}  "
              f"max {m0['max_db']:+.2f}->{m1['max_db']:+.2f}")
    write_index()
    table = summary_table(outs)
    (ROOT / "runs" / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
