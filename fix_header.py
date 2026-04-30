#!/usr/bin/env python3
# Fix sweepX.py - replace garbled Chinese comments with English

with open('/Users/didi/Desktop/GBGCL/tools/sweepX.py', 'rb') as f:
    raw = f.read()

# Create clean English header
header = b'''# -*- coding: utf-8 -*-
# sweepX.py
# Automatic hyperparameter sweep: Stage-A (coarse) / Stage-B (fine-tune) + multi-GPU + dataset parallelization
import os, csv, shlex, itertools, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# =========================
# Configuration (modify here)
# =========================

# Datasets supported by data.py / train.py (--dataset_name)
#DATASETS = ["Computers", "Photo", "CS", "Physics"]
DATASETS = ["Physics", "Computers", "Photo", "CS"]
# Hyperparameter grid: Stage-A uses full grid, Stage-B uses FILTERS subset
QUITY  = ["homo", "detach", "edges"]
SIMS   = ["dot", "cos"]
ALPHAS = [0.7, 0.5, 0.3]

# Ball diffusion hyperparameters (full grid)
BETAS   = [0.1, 0.2, 0.3]          # grid [0.1, 0.2, 0.3]
KS      = [3, 5, 10, 20]            # grid [5, 10, 20]
W_MODES = ["topo", "center", "topo+center"] # grid ["topo", "center", "topo+center"]
KNNs    = [5, 10, 20]            # grid [5, 10, 20]

# Ball Loss hyperparameters (full grid)
BALL_LOSS_W      = [0.05]     # grid [0.0, 0.02, 0.05]
BALL_ANGLE_THR   = [15.0, 25.0]     # grid [10.0, 15.0, 25.0]
BALL_UNIFORM_TAU = [0.1]      # grid [0.05, 0.1]
BALL_INFO_W      = [0.02]     # grid [0.0, 0.01, 0.02]
BALL_INFO_TEMP   = [0.2]      # grid [0.1, 0.2, 0.3]

# Filter presets for dataset-specific Top-K selection
# Supports two formats:
# 1) 3-tuple (quity, sim, alpha) with full grid search defaults
# 2) 12-tuple (all hyperparameters fixed, no grid search)
FILTERS = {
'''

# Find FILTERS location and keep everything after it
idx = raw.find(b'FILTERS = {')
# Find where to continue: after the opening line
rest = raw[idx:]

# Keep FILTERS and everything after it
raw_new = header + rest

with open('/Users/didi/Desktop/GBGCL/tools/sweepX.py', 'wb') as f:
    f.write(raw_new)

print('Done - header fixed')