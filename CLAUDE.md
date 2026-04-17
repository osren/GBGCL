# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGRL (Representation Scattering) - Official PyTorch implementation of the paper "Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering" (NeurIPS 2024).

## Running Experiments

Reproduce paper results with dataset-specific scripts:
```bash
bash run_cs          # CS dataset
bash run_photo       # Photo dataset
bash run_computers  # Computers dataset
bash run_physics    # Physics dataset
```

Or run directly with Python:
```bash
python train.py --dataset_name CS --log_dir ./logs/log_cs.txt --e1_lr 0.001 --e2_lr 0.001 --num_epochs 700 --hidden_dim 1024 --num_hop 1 --trials 20
```

Key arguments:
- `--use_gb` : Enable granule diffusion module
- `--gb_quity` : Granule quality metric (homo, detach, edges, deg)
- `--gb_sim` : Similarity metric (dot, cos, per)
- `--trials` : Number of training trials
- `--device` : cuda or cpu

## Architecture

### Core Model (models.py)
- **Conv**: GCN-based encoder with projection head
- **Online**: Online encoder + predictor + momentum-updated target encoder integration
- **Target**: Momentum-updated target encoder (BYOL-style)

### Granule Module
- **granular.py**: Granular ball clustering algorithm (quality-based graph partitioning)
- **gb_utils.py**: Granule diffusion on ball graphs, ball-level losses (scatter, InfoNCE)

### Training Flow
1. Node embeddings via GCN (online + target networks)
2. Build granules from graph topology + embeddings
3. Ball graph construction + K-step diffusion
4. Write diffused embeddings back to nodes
5. Ball-level losses: scatter loss (RSM) + ball InfoNCE
6. Node-level BYOL loss between online/target

### Data Loading (data.py)
Supports: Cora, CiteSeer, PubMed, dblp, Photo, Computers, CS, Physics, Wiki, PPI, WebKB, WikipediaNetwork

## Key Files

| File | Purpose |
|------|---------|
| train.py | Main training loop with CSV logging |
| models.py | Online/Target/Conv modules |
| granular.py | Granule ball clustering algorithm |
| gb_utils.py | Granule diffusion, ball losses |
| data.py | Dataset loader |

## Environment

Python 3.9.7, PyTorch 2.1.0, torch-geometric 2.5.3. See `env.yaml` for full dependencies.
