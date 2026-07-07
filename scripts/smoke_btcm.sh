#!/usr/bin/env bash
cd /home/ubuntu/workplace/tc2/GBGCL-main/src
python train.py --dataset_name Photo --use_gb --gb_btcm --gb_ball_emb_dim 1024 --gb_quity detach --gb_sim dot --gb_alpha 0.7 --num_epochs 50 --trials 1 --device cuda --results_dir results/btcm_smoke