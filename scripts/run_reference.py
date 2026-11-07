"""Run the authors' unmodified DAFx23 solver (external/diff-fdn-colorless-dafx23/solver.py).

The reference code targets Python 3.10 / torch 2.0 / torchaudio. This wrapper only papers over
environment differences, it does not touch the optimisation itself:
  * torchaudio is not installed here; a stub module provides torchaudio.save() via soundfile.

Usage (same CLI as solver.py):
    python scripts/run_reference.py --train_dir runs/reference/N8_set1 --delays 809 877 937 ...
"""
import os
import runpy
import sys
import types
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "external" / "diff-fdn-colorless-dafx23"


def _torchaudio_save(filename, tensor, samplerate, bits_per_sample=32, channels_first=True):
    data = tensor.detach().cpu().numpy()
    if channels_first:
        data = data.T
    sf.write(filename, data, samplerate, subtype="FLOAT" if bits_per_sample == 32 else "PCM_16")


torchaudio = types.ModuleType("torchaudio")
torchaudio.save = _torchaudio_save
sys.modules["torchaudio"] = torchaudio

if __name__ == "__main__":
    # train_dir is resolved against the caller's cwd before we chdir into the reference repo
    argv = sys.argv[1:]
    if "--train_dir" in argv:
        i = argv.index("--train_dir") + 1
        argv[i] = str(Path(argv[i]).resolve())
    sys.path.insert(0, str(REF))
    os.chdir(REF)
    sys.argv = [str(REF / "solver.py")] + argv
    runpy.run_path(str(REF / "solver.py"), run_name="__main__")
