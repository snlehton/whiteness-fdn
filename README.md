# Whiteness training for colorless feedback delay networks

Code, trained parameters and paper for *Whiteness Training for Colorless Feedback Delay Networks*
(Sampsa Lehtonen, 2026): [paper (PDF)](docs/2026_WhitenessFDN_lehtonen.pdf) · [listen in the browser](https://snlehton.github.io/whiteness-fdn/).

A homogeneous feedback delay network (FDN) is trained on the whiteness of its unrolled impulse response,
plus three terms that shape its first 100 ms. On the six configurations of DiffFDN [1], it comes within
−1.1 to +0.5 dB of DiffFDN's modal-excitation spread, with a smaller largest echo in every configuration.
Training takes 16–86 s per configuration on a CPU.

*AI disclosure: the experiments, analysis code and text were produced with Claude (Anthropic), an AI assistant,
under the author's direction.*

## Setup

Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install torch numpy scipy matplotlib soundfile tqdm
```

The commands below run from the repository root (prefix them with `uv run` when using uv). Training runs on the CPU.

## Reproduce the paper

Results table and Figure 3 numbers, from the shipped parameters:

```bash
python scripts/paper_tables.py
```

Retrain the final models of Section 6. Training is deterministic, so this rewrites the shipped files unchanged:

```bash
python scripts/train_final.py N6_set1
```

Every experiment in the paper is a part of `scripts/autocorr_study.py`. For example, the whiteness-only and
mixing runs of Section 3, and the gain variants of Sections 4 and 5:

```bash
python scripts/autocorr_study.py train train_unit train_density train_pairs train_quad
```

Parts that take arguments are written `name:arg:arg`, as listed in the script's docstring:
`train_spread`, `train_aspread`, `train_build` and `train_direct`.

## Listen

The previewer is online at **https://snlehton.github.io/whiteness-fdn/**. Each configuration opens on the paper's
final model (W19 for N = 4 and 6, W14 for N = 8), and the other variants are in the "Whiteness variant" menu.
`.github/workflows/pages.yml` publishes `web/` there on every push that changes it.

To run it locally, or after changing the models, re-export the data and serve the page:

```bash
python scripts/evaluate.py
```

```bash
python scripts/serve.py
```

Then open http://127.0.0.1:8765. The first command exports `web/data` (a few minutes, mostly modal
decompositions). Pick a configuration, then a trained variant under "Whiteness variant", and switch against
DiffFDN, its init, Householder, Haar-random and a noise reference with the keys 1–8 while audio loops. Any JSON
with `{delays, U, b, c}` can be dropped onto the page to audition it.

## Layout

| Path | What |
|---|---|
| `fdnopt/` | NumPy FDN (parameters, exact impulse response, transfer function, baselines), modal decomposition by eigenphase counting, metrics |
| `scripts/autocorr_study.py` | The method: unrolled differentiable FDN in PyTorch, the whiteness loss and every counterweight and transient term, and all training runs |
| `scripts/train_final.py` | The paper's final objectives for chosen configurations |
| `scripts/paper_tables.py` | The paper's results from the shipped parameters |
| `scripts/evaluate.py`, `scripts/serve.py`, `web/` | Metrics, figures and the listening previewer |
| `scripts/run_reference.py`, `scripts/reproduce_dafx23.sh`, `scripts/loss_curves.py`, `scripts/fig4_temporal_vs_beta.py`, `scripts/report_tables.py`, `scripts/baseline_stats.py` | Reproduction of DiffFDN [1] (Section 2 and `docs/reproduction.md`) |
| `runs/autocorr/` | Trained parameters of every run, `<config>_white_mix1_<variant>.json` |
| `runs/autocorr_study.json` | Metrics and loss curves of every run |
| `runs/reference/<config>/` | DiffFDN init and trained parameters, from the authors' code |
| `docs/` | The paper and the notes on reproducing DiffFDN |

## Trained parameters

Each file in `runs/autocorr/` holds `{delays, U, b, c, gamma, fs, meta}`, with `meta.objective` naming the loss and
its weights. The configurations are the six of [1]: `N4_set1`, `N4_set2`, `N6_set1`, `N6_set2`, `N8_set1` and
`N8_set2`. The variants listed in the previewer, with the ids it shows:

| Id | File suffix | Variant (all: whiteness + Σ U⁴/N mixing, unless noted) |
|---|---|---|
| W01 | `white_mix0` | whiteness only, trained gains |
| W02 | `white_mix1` | + mixing, trained gains |
| W03 | `white_mix1_unitgains` | + mixing, b = c = 1 |
| W04 | `white_mix1_unitgains_dens1` | + density 1, b = c = 1 |
| W05 | `white_mix1_unitgains_dens3` | + density 3, b = c = 1 |
| W06 | `white_mix1_dens1` | + density 1, trained gains |
| W07 | `white_mix1_dens3` | + density 3, trained gains |
| W08 | `white_mix1_cpairs_dens3` | density 3, b = 1, c on pair circles |
| W09 | `white_mix1_bcpairs_dens3` | density 3, b and c on pair circles |
| W10 | `white_mix1_quadpairs_dens3` | density 3, b and c on pair circles 90° apart (Section 5) |
| W11 | `white_mix1_dens3_flat3` | density 3 + per-sample level target |
| W12 | `white_mix1_spread0.5s0.1` | early spread 0.5, 20 ms windows, first echo not covered |
| W13 | `white_mix1_spread0.5s0.1l0.01` | early spread 0.5, 20 ms windows |
| W14 | `white_mix1_dens3_spread0.5s0.1l0.01` | density 3 + early spread 0.5: **final model for N = 8** |
| W15 | `white_mix1_dens3_aspread0.5n8` | density 3 + arrival-window spread 0.5 |
| W16 | `white_mix1_dens3_aspread1n8` | density 3 + arrival-window spread 1 |
| W17 | `white_mix1_dens3_aspread0.5n8_build10d6r150` | W15 + build-up ramp, weight 10 |
| W18 | `white_mix1_dens3_aspread0.5n8_build30d6r150` | W15 + build-up ramp, weight 30 |
| W19 | `white_mix1_dens3_aspread0.5n8_build30d6r150_direct3l10h22` | W18 + direct-echo ceiling, weight 3: **final model for N = 4 and 6** |
| W20 | `white_mix1_dens3_aspread0.5n8_build30d6r150_direct10l10h22` | W18 + direct-echo ceiling, weight 10 |

Not every variant exists for every configuration; the other files in the folder are intermediate runs.

## DiffFDN reference

`runs/reference/<config>/` holds the initial and trained parameters (`parameters_init.mat`, `parameters.mat`)
and metrics from running the authors' DAFx23 code unmodified ([gdalsanto/diff-fdn-colorless](https://github.com/gdalsanto/diff-fdn-colorless),
branch `dafx23`, MIT License, © 2023 Gloria Dal Santo). Our training starts from the same init. To rerun DiffFDN, clone
that branch into `external/diff-fdn-colorless-dafx23` and run `scripts/reproduce_dafx23.sh`; a GPU is recommended.
See [docs/reproduction.md](docs/reproduction.md).

## References

1. G. Dal Santo, K. Prawda, S. J. Schlecht and V. Välimäki, "Differentiable feedback delay network for colorless
   reverberation," in *Proc. 26th Int. Conf. Digital Audio Effects (DAFx23)*, Copenhagen, 2023, pp. 244–251.

## License

MIT, see [LICENSE](LICENSE). The DiffFDN reference parameters in `runs/reference/` come from running the authors'
MIT-licensed code (© 2023 Gloria Dal Santo).

## Citation

```bibtex
@misc{lehtonen2026whiteness,
  author       = {Sampsa Lehtonen},
  title        = {Whiteness Training for Colorless Feedback Delay Networks},
  year         = {2026},
  howpublished = {\url{https://github.com/snlehton/whiteness-fdn}}
}
```
