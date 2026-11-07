"""Per-epoch spectral / temporal loss components (paper Fig. 5) from saved checkpoints.

Uses the reference DiffFDN, losses and Dataset classes unchanged; evaluates every model_*.pt in a run
directory on a fixed batch of frequency-sampling sizes M ~ U[0.8K, K].

    python scripts/loss_curves.py runs/reference/N4_set1 [more run dirs...]
"""
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "diff-fdn-colorless-dafx23"))
sys.path.insert(0, str(ROOT))

from dataloader import Dataset  # noqa: E402
from losses import asy_p_loss, sparsity_loss  # noqa: E402
from models import DiffFDN  # noqa: E402

from fdnopt.fdn import read_args_txt  # noqa: E402


def epoch_key(p: Path):
    m = re.search(r"model_e(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def components(run_dir: Path, n_eval=8, batch=4):
    args = read_args_txt(run_dir / "args.txt")
    torch.manual_seed(1234)
    data = Dataset(n_eval, args["min_nfft"], args["max_nfft"], "cuda")
    net = DiffFDN(args["delays"], args["gain_per_sample"], "cuda")
    spec, temp = asy_p_loss(), sparsity_loss()
    ckpts = sorted(run_dir.glob("model_*.pt"), key=epoch_key)
    out = {"epoch": [], "spectral": [], "temporal": [], "alpha": args["alpha"]}
    for ck in ckpts:
        net.load_state_dict(torch.load(ck, map_location="cuda"))
        ls, lt = 0.0, 0.0
        for i in range(0, n_eval, batch):
            x, y = data.input[i:i + batch], data.labels[i:i + batch]
            H, h, _ = net(x)
            ls += spec(H, y).item()
            lt += temp(torch.abs(h), None).item()
        nb = n_eval // batch
        out["epoch"].append(epoch_key(ck))
        out["spectral"].append(ls / nb)
        out["temporal"].append(lt / nb)
        torch.cuda.empty_cache()
    (run_dir / "loss_components.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    for d in sys.argv[1:]:
        o = components(Path(d))
        print(d, " ".join(f"e{e}:{s:.3f}/{a * o['alpha']:.3f}" for e, s, a in zip(o["epoch"], o["spectral"], o["temporal"])))
