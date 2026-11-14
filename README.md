# AWF: Projected Adversarial Water-Filling Foundation Model

This repository provides the learned Adversarial Water-Filling (AWF) solver for the manuscript **Adversarial Water-Filling: Theory, Algorithms, and a Wireless Foundation Model**. It includes the model, online problem generator, deterministic projections, a Mirror-Prox baseline, and scripts for training and evaluation.

## Overview

AWF allocates transmit power across channels against an interference allocation under power budgets and linear constraints. The learned solver combines a permutation-compatible channel encoder, constraint-aware bipartite message passing, and a 64-iteration primal-dual extragradient rollout. The released model applies Euclidean simplex projection to both the transmit allocation `p` and interference allocation `n` at each half-step and corrected step. The dual variables enforce the linear constraints through their updates.

The implementation supports 16QAM, 64QAM, and 256QAM through modulation-specific MMSE and mutual-information lookup tables. Training-style instances are generated online; the evaluation scripts can vary channel size, modulation, and constraint structure.

## Repository Structure

```text
.
├── checkpoints/
│   └── awf_projected_n_seed123_final.pt  # Pretrained projected-n model and lookup tables
├── mercury/
│   ├── checkpoint.py                     # Checkpoint loader
│   ├── data.py                           # Instance generation and batching
│   ├── model.py                          # Learned AWF model and rollout
│   ├── optimization.py                   # Projected Mirror-Prox baseline
│   ├── metrics.py                        # Objective and solution-quality metrics
│   ├── training.py                       # Training loop and curriculum
│   └── ...                               # Encoders, ablations, and supporting modules
├── scripts/
│   ├── smoke_test.py                     # Small checkpoint check
│   ├── evaluate_awf_vs_mirror_prox.py    # Paired quality evaluation
│   ├── train_projected_awf.py            # Projected-n training entry point
│   ├── weighted_objective_demo.py        # Weighted-objective example
│   └── weighted_size_pattern.py          # Weight-pattern and size diagnostic
├── tests/
├── requirements.txt
└── LICENSE
```

The `mercury` package also contains Direct-GNN, Graph-Transformer AWF, and ablation model definitions. The commands below use the projected-n AWF checkpoint.

## Installation

Use Python 3.10 or newer. In the repository root, install the dependencies in your chosen Python environment:

```bash
python -m pip install -r requirements.txt
```

The required packages are PyTorch and NumPy. CUDA is optional; the scripts use CUDA when available and otherwise run on CPU. The bundled checkpoint was trained with Python 3.11 and FP32.

## Pretrained Checkpoint

The pretrained model and its fixed MMSE/mutual-information lookup tables are stored in `checkpoints/awf_projected_n_seed123_final.pt`. The checkpoint specifies `n_update_mode="projected"`, 64 learned iterations, a 16-step feasibility warm-up, and the learned block-scalar stepsize upper bounds `alpha_p_max=0.08`, `alpha_n_max=0.60`, and `alpha_mu_max=0.90`.

Load the model from the repository root:

```python
from mercury.checkpoint import load_awf_checkpoint

model, metadata, tables, table_gammas = load_awf_checkpoint(
    "checkpoints/awf_projected_n_seed123_final.pt",
    device="cpu",
)
```

On a CUDA-capable GPU, load the same checkpoint with:

```python
import torch
from mercury.checkpoint import load_awf_checkpoint

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in this PyTorch environment")

model, metadata, tables, table_gammas = load_awf_checkpoint(
    "checkpoints/awf_projected_n_seed123_final.pt",
    device="cuda",
)
```

The loader restores the saved weights and stepsize bounds and places the model in evaluation mode.

## Quick Start

Check the checkpoint on one newly generated, 16-channel instance:

```bash
python -B scripts/smoke_test.py --device cpu --instances 1 --m 16 --output outputs/smoke_test.json
```

The check reports `J`, inequality violation, `KKT_p`, `KKT_n`, `R_stat`, budget errors, and whether the final state is finite.

Evaluate the frozen AWF model and Mirror-Prox on the same generated instance:

```bash
python -B scripts/evaluate_awf_vs_mirror_prox.py --device cpu --sizes 16 --instances 1 --mp-iterations 20 --output-dir outputs/baseline_smoke
```

This short run checks the evaluation path. The paired evaluation script writes per-instance CSV and aggregate JSON files to the selected output directory and refuses to overwrite an existing directory. Choose a new `--output-dir` for each run.

## Evaluation Options

The paired evaluation script accepts channel sizes through `--sizes`, modulation through `--modulation {16QAM,64QAM,256QAM}`, and constraints through `--constraint {sparse,group,prefix,dense}`. For example:

```bash
python -B scripts/evaluate_awf_vs_mirror_prox.py --sizes 32 64 128 --instances 10 --modulation 16QAM --constraint sparse --mp-iterations 3000 --output-dir outputs/size_check
python -B scripts/evaluate_awf_vs_mirror_prox.py --sizes 64 --instances 10 --modulation 256QAM --constraint sparse --mp-iterations 3000 --output-dir outputs/modulation_check
```

Use `--constraint group`, `prefix`, or `dense` with a distinct output directory to inspect other constraint families. Each command creates seeded instances with the existing generator. It does not load a saved paper test set.

The AWF evaluation uses `T=64` and 16 feasibility steps. Mirror-Prox uses 80 feasibility steps, fixed stepsizes `(alpha_p, alpha_n, alpha_mu)=(0.01, 0.03, 0.05)`, and exactly the requested `--mp-iterations`. This script evaluates final solution quality; it does not perform matched-runtime timing.

## Metrics

Let `m` be the number of valid channels, `K` the number of valid linear constraints, `gamma_i = beta_i p_i / (sigma_i + n_i)`, and `z = (p, n, mu)`. Both methods use the same implementations in `mercury/metrics.py`:

- `J = (1/m) sum_i I_i(gamma_i)`: average mutual information.
- `inequality = (1/K) sum_j [(A p)_j - p_hat_j]_+`: average positive linear-constraint violation.
- `KKT_p`: mean absolute transmit-side residual over the active coordinates selected by the existing KKT diagnostic.
- `KKT_n`: interference-side active-coordinate water-level residual plus its inactive-coordinate inequality residual.
- `R_stat`: projected first-order residual of the complete `(p, n, mu)` state.

The stationarity metric uses the fixed evaluation stepsize `eta_eval=0.01`:

```text
R_stat(z) = ||z - Pi_X(z + eta_eval * G(z))||_2
            / (eta_eval * sqrt(2m + K)).
```

Here `Pi_X` projects `p` and `n` onto their respective nonnegative budget simplexes and clips `mu` at zero. `G` is the AWF ascent/descent field used by the existing metric implementation. The evaluation stepsize is independent of the model's learned stepsizes. `J` measures utility, while the residuals describe first-order solution quality.

## Training a New Model

The training entry point uses online-generated sparse-constraint instances, 16QAM and 64QAM, a curriculum over channel sizes, batch size 16, AdamW with learning rate `1e-4`, and the projected-n rollout. By default it runs one epoch of 50 steps as a lightweight training example:

```bash
python -B scripts/train_projected_awf.py --seed 123 --epochs 1 --steps-per-epoch 50 --output outputs/new_awf.pt --report outputs/new_training_report.json
```

The bundled checkpoint supplies fixed lookup tables to this script; the model parameters are trained anew. The script refuses to overwrite an existing checkpoint or report. A short training run is a functional example and is not intended to reproduce a publication-scale training schedule.

## Weighted-Objective Example

The optional weight vector changes the objective to `J_w = (1/m) sum_i w_i I_i(gamma_i)`. Its weights enter the analytical gradients and the corresponding evaluation metrics. To run the supplied diagnostics:

```bash
python -B scripts/weighted_objective_demo.py
python -B scripts/weighted_size_pattern.py --instances-per-size 10
```

For a paired AWF/Mirror-Prox evaluation of a weighted objective, add `--weight-pattern random_positive`, `grouped`, or `wide_two_level` to `scripts/evaluate_awf_vs_mirror_prox.py`.

## Reproducibility Notes

Instance generation and training are stochastic. Record the seed, instance count, device, and evaluation settings when comparing runs. The examples create new instances; they do not reproduce the paper's predefined test instances. Generated files go under `outputs/`, which is ignored by Git. No numerical experiment results are bundled with this repository.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
