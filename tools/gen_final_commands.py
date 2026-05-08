#!/usr/bin/env python3
"""Generate nohup commands for final training with best parameters from overall_topk.csv"""

import pandas as pd
import os

# Read the topk results
df = pd.read_csv('/Users/didi/Desktop/GBGCL/analysis/overall_topk.csv')

# Select best parameters for each dataset (take first row per dataset with num_trials >= 5, or highest mean)
best_params = {}
for dataset in ['CS', 'Computers', 'Photo', 'Physics']:
    dataset_df = df[df['dataset'] == dataset]
    # Prefer entries with trials >= 5 for more reliable results
    valid_trials = dataset_df[dataset_df['num_trials'] >= 5]
    if len(valid_trials) > 0:
        best = valid_trials.iloc[0]
    else:
        best = dataset_df.iloc[0]
    best_params[dataset] = best

# Print selected parameters
print("=" * 60)
print("Selected Best Parameters for Each Dataset:")
print("=" * 60)
for dataset, row in best_params.items():
    print(f"\n{dataset}: mean={row['mean']:.6f}, std={row['std']:.6f}")
    print(f"  quity={row['gb_quity']}, sim={row['gb_sim']}, alpha={row['gb_alpha']}")
    print(f"  beta={row['gb_beta']}, K={int(row['gb_K'])}, knn={int(row['gb_knn'])}")
    print(f"  angle={int(row['ball_angle_thresh'])}, ball_loss_w={row['ball_loss_weight']}")

# Generate nohup commands
commands = []
for dataset, row in best_params.items():
    cmd = (
        f"cd /Users/didi/Desktop/GBGCL/src && nohup python train.py "
        f"--dataset_name {dataset} "
        f"--use_gb "
        f"--gb_quality {row['gb_quity']} "
        f"--gb_sim {row['gb_sim']} "
        f"--gb_alpha {row['gb_alpha']} "
        f"--gb_beta {row['gb_beta']} "
        f"--gb_K {int(row['gb_K'])} "
        f"--gb_knn {int(row['gb_knn'])} "
        f"--ball_loss_weight {row['ball_loss_weight']} "
        f"--ball_angle_thresh {int(row['ball_angle_thresh'])} "
        f"--ball_uniform_tau {row['ball_uniform_tau']} "
        f"--ball_infonce_weight {row['ball_infonce_weight']} "
        f"--ball_infonce_temp {row['ball_infonce_temp']} "
        f"--gb_w_mode {row['gb_w_mode']} "
        f"--num_epochs 700 "
        f"--trials 20 "
        f"--results_dir ../results_final "
        f"> ../logs/{dataset}_final_20trials.log 2>&1 &"
    )
    commands.append((dataset, cmd))
    print(f"\n[didi@MacBook-Pro] ~/GBGCL$ {cmd}")

# Save commands to a shell script
script_path = '/Users/didi/Desktop/GBGCL/run_final_20trials.sh'
with open(script_path, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("# Final training with best parameters from overall_topk.csv\n")
    f.write("# 700 epochs, 20 trials\n\n")
    for dataset, cmd in commands:
        f.write(f"# {dataset}\n{cmd}\n\n")

os.chmod(script_path, 0o755)
print(f"\n{'=' * 60}")
print(f"Script saved to: {script_path}")
print("=" * 60)