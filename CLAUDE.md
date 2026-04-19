# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GBGCL (Granular Ball Graph Contrastive Learning) — A PyTorch implementation exploring granule ball-based graph representation learning. Built on SGRL (NeurIPS 2024) principles with an extended granule diffusion module.

## Running Experiments

### Direct training (from `src/` directory)
```bash
cd src
python train.py --dataset_name CS --use_gb --gb_quity detach --gb_sim dot --gb_alpha 0.7 --num_epochs 700 --trials 5
```

### Hyperparameter sweep (from project root)
```bash
# Stage-A: coarse search (150 epochs, 1 trial)
$env:SWEEP_STAGE="A"; $env:SWEEP_WORKERS="2"; python tools/sweepX.py

# Stage-B: fine-tuning (700 epochs, 5 trials)
$env:SWEEP_STAGE="B"; $env:SWEEP_WORKERS="1"; python tools/sweepX.py
```

### Check experiment status / analyze results
```bash
python scripts/experiments_status.py
python tools/analyze_results.py   # outputs: analysis/overall_topk.csv
```

### Quick start scripts (in `scripts/`)
```bash
# CS dataset
python scripts/run_cs          # runs command listed in the file

# Other datasets: run_photo, run_computers, run_physics
```

## Key Arguments

| Argument | Purpose | Values |
|----------|---------|--------|
| `--use_gb` | Enable granule diffusion module | flag |
| `--gb_quity` | Granule quality metric | homo, detach, edges, deg |
| `--gb_sim` | Similarity metric | dot, cos, per |
| `--gb_alpha` | Fusion coefficient (node + ball) | 0.3–0.7 |
| `--gb_K` | Diffusion steps | 3, 5, 10, 20 |
| `--results_dir` | Output directory for CSVs | default: results |
| `--trials` | Number of training trials | default: 5 |
| `--device` | cuda or cpu | default: cuda |

## Architecture

### Core Training Flow (`src/train.py`)
1. GCN encoder → online + target networks (BYOL-style EMA)
2. Build granule balls from graph topology + embeddings
3. Ball graph construction → K-step diffusion → write back to nodes
4. Ball-level losses: scatter loss (RSM) + ball InfoNCE
5. Node-level BYOL loss between online/target
6. Final evaluation via Logistic Regression on learned embeddings

### Key Modules

| File | Role |
|------|------|
| `src/models.py` | Conv (GCN encoder), Online (predictor + EMA target), Target |
| `src/granular.py` | Granule ball clustering via quality-based graph partitioning |
| `src/gb_utils.py` | Ball diffusion, ball-level losses (scatter, InfoNCE), ball matching |
| `src/data.py` | Dataset loading for CS, Physics, Photo, Computers, etc. |
| `src/train.py` | Main training loop, CSV result logging to `results/` |

### Sweep Tool (`tools/sweepX.py`)
- Reads `FILTERS` dict to determine which (quity, sim, alpha) combos to test per dataset
- Reads `KS`, `BETAS`, `BALL_LOSS_W`, etc. for full hyperparameter space
- Stage-A: 150 epochs, 1 trial per combo → pick Top-K
- Stage-B: 700 epochs, 5 trials → fine-tune selected combos
- `already_done()` checks `results/<dataset>_summary.csv` for resume

## Project Structure

```
GBGCL/
├── src/                    # Core training code (train.py must be run from here)
│   ├── data.py             # Dataset loader
│   ├── models.py           # Online/Target/Conv modules
│   ├── train.py            # Main training loop
│   ├── granular.py         # Granule ball clustering
│   └── gb_utils.py         # Ball diffusion & losses
├── tools/
│   ├── sweepX.py           # Hyperparameter sweep (Stage-A/B)
│   ├── analyze_results.py  # Result analysis → analysis/overall_topk.csv
│   └── visualize_granules.py
├── scripts/
│   ├── experiments_status.py   # Check experiment progress
│   └── run_cs / run_photo / run_computers / run_physics  # Quick start commands
├── results/                # CSV outputs: <dataset>_summary.csv
├── analysis/               # Aggregated result analysis CS/Photo/Physics/Computers
├── logs/                   # Training logs
├── figures/                # Framework figures (final/, archive/, pptx/)
├── docs/                   # Project docs, weekly reports, defense materials
├── env.yaml                # Conda environment
└── CLAUDE.md
```

## Environment

Python 3.9.7, PyTorch 2.1.0, torch-geometric 2.5.3. See `env.yaml` for full dependencies.

## Common Patterns

- `results/<dataset>_summary.csv` is the canonical result file — sweepX and analyze_results both read/write here
- When adding new sweep parameters, update both `build_cmd()` in sweepX.py and the argparse in train.py
- Granule rebuilding interval controlled by `--gb_rebuild_every` (default 100 in Stage-B, 50 in Stage-A)
