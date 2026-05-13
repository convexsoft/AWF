# AWF: Adversarial Water-Filling Foundation Model

This repository provides the core implementation for **Adversarial Water-Filling (AWF)** and the AWF wireless foundation model used in the manuscript:

**Adversarial Water-Filling: Theory, Algorithms, and a Wireless Foundation Model**

The code implements a domain-specific learned solver for mercury/water-filling AWF problems with discrete constellations, spatial linear constraints, and adversarial interference. The model combines permutation-invariant channel set encoding, constraint-aware graph message passing, and learned primal-dual update dynamics.

## Overview

Adversarial water-filling models a minimax resource-allocation problem in which a transmitter allocates power across channels while an adversary allocates interference power. This repository focuses on the discrete-constellation mercury/water-filling setting, where the derivative of the mutual-information function is represented using precomputed MMSE interpolation tables.

The implementation includes:

- QAM constellation simulation for MMSE and mutual-information tables.
- Online generation of AWF problem instances.
- Sparse, group, prefix, and dense constraint structures.
- A Perceiver-style channel set encoder.
- A bipartite GNN module for constraint-aware message passing.
- Learned primal-dual extragradient-style rollout dynamics.
- Mirror-Prox / projected primal-dual baselines.
- Evaluation scripts for size, modulation, and constraint generalization.

## Repository Structure

```text
AWF/
├── train.py                  # Main training and evaluation driver
├── eval.py                   # Checkpoint evaluation script
├── requirements.txt          # Python dependencies
├── awf_model_final.pt        # Model weight
├── LICENSE                   # Apache-2.0 license
└── mercury/
    ├── constants.py          # Modulation definitions and train/test groups
    ├── data.py               # AWF instance generation and constraint matrices
    ├── evaluation.py         # Evaluation utilities
    ├── losses.py             # Training loss and residual objectives
    ├── metrics.py            # Objective, feasibility, and KKT metrics
    ├── model.py              # AWF foundation model architecture
    ├── optimization.py       # Best-response and Mirror-Prox style baselines
    ├── qam.py                # QAM constellation and MMSE/I table construction
    ├── training.py           # Training loop and curriculum
    └── utils.py              # Projection, interpolation, and helper utilities
```

## Installation

Clone the repository:

```bash
git clone https://github.com/convexsoft/AWF.git
cd AWF
```

Create and activate a Python environment:

```bash
conda create -n awf python=3.10
conda activate awf
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
torch
numpy
matplotlib
```

A CUDA-capable GPU is recommended for training and evaluation. The scripts can also run on CPU, but runtime will be longer.

## Quick Start

Run the main training script:

```bash
python train.py
```

The script will:

1. set the random seed;
2. select CUDA if available;
3. build logarithmically spaced SNR grids;
4. simulate MMSE and mutual-information tables for the supported QAM constellations;
5. train the AWF foundation model on online-generated instances;
6. save a timestamped model checkpoint;
7. run baseline checks and generalization evaluations.

The public script is configured as a lightweight reproducibility entry point. For larger paper-scale experiments, increase the number of epochs, training steps, evaluation batches, and repeated runs as needed.

## Evaluating a Checkpoint

After training, evaluate a saved checkpoint using:

```bash
python eval.py --checkpoint path/to/checkpoint.pt
```

For example:

```bash
python eval.py --checkpoint awf_model_XXXX_final.pt
```

The evaluation script loads the model, reconstructs the MMSE/I tables, and reports:

- modulation-format generalization;
- held-out 256QAM generalization;
- constraint-structure generalization over sparse, group, prefix, and dense constraints.

A device can be specified explicitly:

```bash
python eval.py --checkpoint path/to/checkpoint.pt --device cuda
```

or

```bash
python eval.py --checkpoint path/to/checkpoint.pt --device cpu
```

## Problem Generation

The code generates AWF instances online. Each instance contains:

- channel gains `beta`;
- noise powers `sigma`;
- transmit budget `P`;
- adversarial interference budget `N`;
- linear constraint matrix `A`;
- constraint threshold vector `p_hat`;
- modulation-dependent distribution token;
- modulation ID.

The default budget mode is:

```text
per_channel_fixed
```

In this mode, per-channel budgets are sampled and scaled with the number of channels:

```text
P = m * P_bar,
N = m * N_bar.
```

This setting keeps the per-channel resource scale comparable across different problem sizes.

## Constraint Structures

The implementation supports several linear-constraint families:

```text
sparse
group
prefix
dense
```

The default training setup uses sparse random nonnegative constraints. Generalization evaluation additionally tests group, prefix, and dense structures.

In the instance generator, each row of `A` is normalized to unit sum. The threshold vector `p_hat` is generated by sampling a feasible power allocation and adding positive random slack to ensure a nonempty feasible constraint set.

## Modulation Settings

The supported modulations are:

```text
16QAM
64QAM
256QAM
```

The default training modulations are:

```text
16QAM
64QAM
```

The default test groups are:

```text
mod16      -> 16QAM
mod64      -> 64QAM
mixed      -> 16QAM and 64QAM
heldout256 -> 256QAM
```

Thus, 256QAM is used as a held-out modulation format to evaluate distribution generalization.

## MMSE and Mutual-Information Tables

The mercury/water-filling formulation uses the I-MMSE relationship. The code constructs interpolation tables by Monte Carlo simulation over a logarithmically spaced SNR grid.

By default:

- table SNR grid: approximately `[-10, 30]` dB with 128 points;
- distribution-token SNR grid: approximately `[-10, 30]` dB with 32 points;
- Monte Carlo samples per SNR point: `15000`.

These tables are generated at runtime by `mercury/qam.py`.

## Model Architecture

The AWF foundation model is implemented in `mercury/model.py`.

The main components are:

- `DistTokenMLP`: embeds the modulation-dependent distribution token;
- `PerceiverSetEncoder`: encodes variable-size unordered channel sets;
- `BipartiteMP`: performs message passing between channel nodes and constraint nodes;
- `Heads`: predicts initial variables and learned step sizes;
- `MercuryFoundationSolver`: combines set encoding, message passing, and learned primal-dual rollout.

The model outputs:

- transmit power allocation `p`;
- adversarial interference allocation `n`;
- dual variables associated with the linear constraints.

## Training Configuration

The public `train.py` script uses:

```text
seed = 123
budget_mode = per_channel_fixed
optimizer = Adam
learning_rate = 1e-4
batch_size = 16
epochs = 1
steps_per_epoch = 100
```

It also uses a curriculum over channel dimensions:

```text
Phase 1: m in [32, 96]
Phase 2: m in [32, 96] and [128, 256]
Phase 3: m in [32, 96], [128, 256], and [384, 512]
```

The script saves a final checkpoint named like:

```text
awf_model_<timestamp>_final.pt
```

For paper-scale training, use a longer schedule and more repeated evaluations.

## Baselines and Metrics

Baseline solvers and best-response utilities are implemented in `mercury/optimization.py`.

The main evaluation metrics are:

- `J`: normalized mutual-information objective;
- `InEq`: average inequality violation of `Ap <= p_hat`;
- `KKT_p`: transmit-side stationarity residual;
- `KKT_n`: interference-side stationarity residual;
- `runtime_ms`: runtime in milliseconds.

Mirror-Prox style projected primal-dual iterations are used as the main iterative baseline.

## Reproducibility Notes

The code uses stochastic components, including:

- Monte Carlo construction of MMSE/mutual-information tables;
- online random generation of AWF instances;
- random channel gains, noise powers, budgets, and constraints;
- stochastic model training;
- hardware-dependent GPU execution behavior.

Exact bitwise reproduction is therefore not expected. Results should be compared using averaged metrics over repeated runs.

For runtime measurement, use warm-up runs and CUDA synchronization when timing GPU execution. Runtime may vary depending on GPU temperature, power limits, background processes, memory pressure, and CUDA scheduling.

A typical reproducibility workflow is:

```bash
python train.py
python eval.py --checkpoint path/to/checkpoint.pt
```

For more stable results, run multiple repetitions and report the mean and standard deviation.

## Code Availability

This repository contains the core implementation and scripts needed to reproduce the main experimental workflow in the manuscript. Additional internal scripts for large-scale hyperparameter sweeps and unpublished extensions are not included in this public release.


## License

This project is released under the Apache License 2.0. See `LICENSE` for details.
