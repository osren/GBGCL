#!/bin/bash
###
 # @Author: tancheng
 # @Date: 2026-05-08 11:43:10
 # @LastEditors: tancheng
 # @LastEditTime: 2026-05-08 11:44:42
 # @FilePath: /GBGCL/run_final_20trials.sh
 # @Description: 
### 
# Final training with best parameters from overall_topk.csv
# 700 epochs, 20 trials

# CS
nohup python src/train.py --dataset_name CS --use_gb --gb_quality detach --gb_sim dot --gb_alpha 0.3 --gb_beta 0.3 --gb_K 5 --gb_knn 5 --ball_loss_weight 0.05 --ball_angle_thresh 25 --ball_uniform_tau 0.1 --ball_infonce_weight 0.02 --ball_infonce_temp 0.2 --gb_w_mode center --num_epochs 700 --trials 20 --results_dir ../results_final > ../logs/CS_final_20trials.log 2>&1 &

# Computers
nohup python src/train.py --dataset_name Computers --use_gb --gb_quality homo --gb_sim dot --gb_alpha 0.7 --gb_beta 0.2 --gb_K 10 --gb_knn 10 --ball_loss_weight 0.05 --ball_angle_thresh 15 --ball_uniform_tau 0.1 --ball_infonce_weight 0.02 --ball_infonce_temp 0.2 --gb_w_mode topo+center --num_epochs 700 --trials 20 --results_dir ../results_final > ../logs/Computers_final_20trials.log 2>&1 &

# Photo
nohup python src/train.py --dataset_name Photo --use_gb --gb_quality homo --gb_sim cos --gb_alpha 0.3 --gb_beta 0.2 --gb_K 10 --gb_knn 10 --ball_loss_weight 0.05 --ball_angle_thresh 15 --ball_uniform_tau 0.1 --ball_infonce_weight 0.02 --ball_infonce_temp 0.2 --gb_w_mode topo+center --num_epochs 700 --trials 20 --results_dir ../results_final > ../logs/Photo_final_20trials.log 2>&1 &

# Physics
nohup python src/train.py --dataset_name Physics --use_gb --gb_quality detach --gb_sim dot --gb_alpha 0.3 --gb_beta 0.2 --gb_K 10 --gb_knn 10 --ball_loss_weight 0.05 --ball_angle_thresh 15 --ball_uniform_tau 0.1 --ball_infonce_weight 0.02 --ball_infonce_temp 0.2 --gb_w_mode topo+center --num_epochs 700 --trials 20 --results_dir ../results_final > ../logs/Physics_final_20trials.log 2>&1 &

