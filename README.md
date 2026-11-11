# SCOPE code

Research implementation of **Stopping by Certified Option Prices of Evidence (SCOPE)**, a method for allocating retrieval reads across a workload of retrieval-augmented question-answering queries. The code fits query-specific belief dynamics, plans stopping policies, and calibrates answer admission and resource use.

## What is included

- `code/fit22.py`: trains the verifier and belief transition model, then writes policy curves.
- `code/main22.py`: certifies and evaluates saved curves on calibration/test splits.
- `code/fpi.py`, `code/lattice16.py`, `code/bpm20.py`: model and planning components.
- `code/cert14.py`: certification routines.
- Remaining Python files are dependencies of these entry points.

This repository contains source code only. Manuscript files, figures, tables, logs, trajectories, model weights, and experiment results are excluded. You must supply your own trajectory data and any required reader model outputs.

## Environment

Python 3.10 or newer is recommended. Install NumPy, SciPy, PyTorch, and scikit-learn in a virtual environment. A CUDA-capable PyTorch build is needed for the training stage on GPU. The scripts also use standard-library modules.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install numpy scipy scikit-learn torch
```

## Run

Run from the repository root. The stage-one input is a trajectory directory containing the feature and optional hidden-state arrays expected by `code/main15.py` (`<dataset>_feat.npz` and, when available, `<dataset>_h<layer>.npy`). The precise array schema is defined in `load_unit` there.

```bash
python code/fit22.py --traj /path/to/trajectories --ds mix3 --out /path/to/cache --seed 0
python code/main22.py --cache /path/to/cache --ds mix3 --out /path/to/results --seed 0 --parts main
```

Use `python code/fit22.py --help` and `python code/main22.py --help` for all options. The entry points expect precomputed trajectories; this snapshot does not include dataset acquisition or trajectory generation. Results will depend on the supplied data, model outputs, seeds, and hardware.
