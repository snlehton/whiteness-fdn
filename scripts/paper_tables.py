"""The paper's results (Section 7, Figure 3) from the shipped parameters: DiffFDN [1] against the final objectives
of Section 6, per configuration. Modal measures at gamma = 0.9999; largest echo, early peak and EDT at a playback
T60 of 2.5 s; onset = level of the first 25 ms after the first echo against the tail, decay removed.

    python scripts/paper_tables.py [configs...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autocorr_study import CACHE, CONFIGS, decay_stats, transient_stats, variants  # noqa: E402
from fdnopt.fdn import FDN  # noqa: E402
from fdnopt.metrics import all_metrics  # noqa: E402
from fdnopt.modal import cached_modal  # noqa: E402

FINAL = {"N4": "dens3_aspread0.5n8_build30d6r150_direct3l10h22", "N6": "dens3_aspread0.5n8_build30d6r150_direct3l10h22",
         "N8": "dens3_spread0.5s0.1l0.01"}


def measures(fdn, tag):
    CACHE.mkdir(parents=True, exist_ok=True)
    _, res = cached_modal(fdn, CACHE / f"paper_{tag}.npz")
    m = {**all_metrics(fdn, res), **decay_stats(fdn), **transient_stats(fdn)}
    return (f"{m['std_db']:5.2f} {m['p99_db']:+6.2f} {m['max_db']:+6.2f} {100 * m['largest_tap']:5.1f}% "
            f"{m['early_peak_db']:5.1f} {m['buildup_db'][0]:+6.1f} {m['ned_mean_100_500ms']:5.2f} {m['edt']:5.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{'config':9s} {'model':8s} {'σ dB':>5s} {'p99':>6s} {'max':>6s} {'echo':>6s} {'peak':>5s} {'onset':>6s} "
          f"{'NED':>5s} {'EDT':>5s}")
    for cfg in sys.argv[1:] or CONFIGS:
        ours = FDN.from_json(json.loads((ROOT / "runs" / "autocorr" / f"{cfg}_white_mix1_{FINAL[cfg[:2]]}.json")
                                        .read_text(encoding="utf-8")))
        print(f"{cfg:9s} {'DiffFDN':8s} {measures(variants(cfg)['optimized'], f'{cfg}_difffdn')}")
        print(f"{cfg:9s} {'ours':8s} {measures(ours, f'{cfg}_final')}")
