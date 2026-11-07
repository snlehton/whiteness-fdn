"""Train the paper's final objectives (Section 6) from the init of DiffFDN [1], with b and c free:

  N = 4, 6   whiteness + mixing + 3 * density + 0.5 * arrival-window spread + 30 * build-up + 3 * direct-echo ceiling
  N = 8      whiteness + mixing + 3 * density + 0.5 * early spread (20 ms windows over the first 100 ms, 10 ms lead)

Writes runs/autocorr/<config>_white_mix1_<suffix>.json (the parameters) and prints the metrics. Training is
deterministic, so this reproduces the shipped parameter files.

    python scripts/train_final.py [N4_set1 N4_set2 N6_set1 N6_set2 N8_set1 N8_set2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autocorr_study as study  # noqa: E402


def train(cfg):
    if cfg.startswith("N8"):
        return study.part_train([cfg], mix_weights=(1.0,), density=3.0, spread=0.5, spread_stop=0.1, spread_lead=0.01)
    return study.part_train([cfg], mix_weights=(1.0,), density=3.0, aspread=0.5, build=30.0, direct=3.0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for cfg in sys.argv[1:] or study.CONFIGS:
        train(cfg)
