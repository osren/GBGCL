# 运行命令速查

> 训练须在 `src/` 下执行，或从根目录 `python src/train.py`（sweepX 已适配）。

## 环境

```bash
# 见 env.yaml — Python 3.9.7, PyTorch 2.1.0, PyG 2.5.3
```

## 1. SGRL 复现（无粒球）

```bash
cd src
python train.py --dataset_name CS --num_epochs 700 --trials 5 --device cuda
python train.py --dataset_name Photo --num_epochs 700 --trials 5 --device cuda
python train.py --dataset_name Physics --num_epochs 700 --trials 5 --device cuda
python train.py --dataset_name Computers --num_epochs 700 --trials 5 --device cuda
```

## 2. GBGCL 默认粒球

```bash
cd src
python train.py --dataset_name CS --use_gb --gb_quity detach --gb_sim dot --gb_alpha 0.7 \
  --num_epochs 700 --trials 5 --device cuda
```

## 3. 增量扩散验证

```bash
cd src
python train.py --dataset_name Photo --use_gb --gb_quity homo --gb_sim cos --gb_alpha 0.3 \
  --gb_incremental --gb_rebuild_every 10 --num_epochs 50 --trials 3 --device cuda
```

## 4. Option 开关

```bash
# Option B: Target 分支粒球
python train.py --dataset_name Photo --use_gb --gb_target_enhance --num_epochs 700 --trials 5

# Option A2: 多 quity 投票
python train.py --dataset_name Photo --use_gb --gb_ensemble --num_epochs 50 --trials 3

# Option A v1: 特征拼接
python train.py --dataset_name Photo --use_gb --gb_feature_concat --num_epochs 700 --trials 5
```

## 5. 超参扫参

```powershell
# Stage-A: 150 epoch, 1 trial
$env:SWEEP_STAGE="A"; $env:SWEEP_WORKERS="2"; python tools/sweepX.py

# Stage-B: 700 epoch, 5 trials
$env:SWEEP_STAGE="B"; $env:SWEEP_WORKERS="1"; python tools/sweepX.py
```

## 6. 结果分析

```bash
python tools/analyze_results.py    # → analysis/overall_topk.csv
python scripts/experiments_status.py
```

## 7. 脚本快捷方式

```bash
python scripts/run_cs
python scripts/run_photo
python scripts/run_computers
python scripts/run_physics
```

## 8. 服务器 / SLURM

见 `docs/服务器实验指南.md`、`scripts/slurm_sweep_array.sh`。
