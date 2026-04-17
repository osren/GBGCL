#!/bin/bash
#SBATCH -J SGRLGB
#SBATCH -p your_partition
#SBATCH -N 1
#SBATCH -c 6                   # 每个任务占用 CPU 核
#SBATCH --gres=gpu:0           # 如需GPU改成:1
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm_%A_%a.out
#SBATCH -e logs/slurm_%A_%a.err
#SBATCH --array=0-31           # 与 joblist 行数一致（0~31）

set -e
mkdir -p log_multi results

# 建议放到大内存盘，避免 joblib 报 "No space left on device"
export JOBLIB_TEMP_FOLDER=/dev/shm
export SWEEP_STAGE=A     # A 粗筛 / B 精训
export PYTHONUNBUFFERED=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" joblist.txt)
read DATASET QUITY SIM ALPHA STAGE <<< "$LINE"

echo "[TASK] #$SLURM_ARRAY_TASK_ID  $DATASET | $QUITY-$SIM-$ALPHA | stage=$STAGE"

# 粗筛参数
if [ "$STAGE" = "A" ]; then
  NUM_EPOCHS=150; TRIALS=1; GB_REBUILD_EVERY=50
else
  NUM_EPOCHS=700; TRIALS=5; GB_REBUILD_EVERY=100
fi

python train_V5.py \
  --dataset_name "$DATASET" \
  --log_dir "./log_multi/${DATASET}_${QUITY}_${SIM}_${ALPHA}_${STAGE}.log" \
  --e1_lr 1e-4 --e2_lr 1e-4 \
  --num_epochs $NUM_EPOCHS \
  --hidden_dim 1024 --num_hop 1 --num_layers 1 \
  --momentum 0.99 --seed 66666 \
  --trials $TRIALS --log_every 50 --imp_thresh 1.0 \
  --gb_rebuild_every $GB_REBUILD_EVERY \
  --use_gb --gb_quity "$QUITY" --gb_sim "$SIM" --gb_alpha $ALPHA
