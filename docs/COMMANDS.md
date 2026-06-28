# 运行命令速查

> 训练须在 `src/` 下执行，或从根目录 `python src/train.py`（sweepX 已适配）。  
> **分阶段 + nohup**：见 [`scripts/phases/README.md`](../../scripts/phases/README.md)。

## 环境

```bash
# 见 env.yaml — Python 3.9.7, PyTorch 2.1.0, PyG 2.5.3
conda activate gbgcl
```

---

## 推荐：分阶段脚本（服务器 nohup）

在项目根目录执行：

```bash
# 后台启动某一阶段
bash scripts/phases/run_phase_nohup.sh 0      # 阶段0 SGRL 复现
bash scripts/phases/run_phase_nohup.sh 1      # 阶段1 增量扩散
bash scripts/phases/run_phase_nohup.sh 3a    # Stage-A 扫参
bash scripts/phases/run_phase_nohup.sh 4      # 汇总分析

# 前台调试（不 nohup）
bash scripts/phases/run_phase_nohup.sh 0 --foreground

# 查看进度
tail -f logs/nohup/phase0_*.out
cat logs/phases/journal.csv
```

| 阶段 | 含义 |
|------|------|
| `0` | 四数据集 SGRL 复现（无粒球，700×5） |
| `1` | E7 增量扩散 vs 对照（Photo+Computers，50×3） |
| `2` | G-SGRL BTCM/BRSM（未实现则 SKIP） |
| `3a` / `3b` | sweepX Stage-A / Stage-B |
| `4` | analyze_results + 实验状态 |

可选隔离结果目录：

```bash
export RESULTS_DIR=/path/to/GBGCL/results/phase0_sgrl
bash scripts/phases/run_phase_nohup.sh 0
```

---

## 结果怎么记录

| 层级 | 路径 | 用途 |
|------|------|------|
| **主结果** | `results/<Dataset>_summary.csv` | 每 trial 一行 `clf_mean`，论文对照 |
| **训练 log** | `logs/phases/phaseN/<Dataset>_<tag>.log` | debug 指标、loss |
| **nohup 输出** | `logs/nohup/phaseN_*.out` | 整段 stdout |
| **运行台账** | `logs/phases/journal.csv` | 阶段/数据集/成功失败 |
| **人工笔记** | `docs/EXPERIMENTS.md` | 阶段结束后抄关键 ACC |
| **聚合** | `analysis/overall_topk.csv` | `phase4` 或 `analyze_results.py` |

阶段 1 请在 log 中检查：`[h_DEBUG]`、`[Incremental]`、`h_z_diff`。

---

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

```bash
# Linux
export SWEEP_STAGE=A SWEEP_WORKERS=2
python tools/sweepX.py

export SWEEP_STAGE=B SWEEP_WORKERS=1
python tools/sweepX.py
```

```powershell
# Windows PowerShell
$env:SWEEP_STAGE="A"; $env:SWEEP_WORKERS="2"; python tools/sweepX.py
$env:SWEEP_STAGE="B"; $env:SWEEP_WORKERS="1"; python tools/sweepX.py
```

## 6. 结果分析

```bash
python tools/analyze_results.py    # → analysis/overall_topk.csv
python scripts/experiments_status.py
bash scripts/experiments_status.sh
```

## 7. 脚本快捷方式（旧）

```bash
# 历史脚本，参数可能与当前默认不一致，优先用 phases/
python scripts/run_cs
```

## 8. 服务器 / SLURM

见 `docs/服务器实验指南.md`、`scripts/slurm_sweep_array.sh`。
