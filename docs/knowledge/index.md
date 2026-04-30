# Knowledge Index

> GBGCL 项目知识库入口

## 项目概述

**GBGCL** (Granular Ball Graph Contrastive Learning) — 基于 Granular Ball 的图表示学习框架，基于 SGRL (NeurIPS 2024) 扩展粒球扩散模块。

## 知识目录

### 快速入口

| 目录 | 说明 |
|------|------|
| [knowledge/architecture.md](architecture.md) | 技术栈与架构 |
| [knowledge/datasets.md](datasets.md) | 支持的数据集 |
| [knowledge/modules.md](modules.md) | 核心模块说明 |
| [knowledge/api.md](api.md) | 命令行接口 |

### 专题知识

| 目录 | 说明 |
|------|------|
| `prd/` | 需求文档 |
| `experiments/` | 实验记录 |

## 常用命令

```bash
# 训练单个配置
cd src && python train.py --dataset_name CS --use_gb --gb_quity detach --gb_sim dot --gb_alpha 0.7

# Stage-A 粗筛
SWEEP_STAGE=A SWEEP_WORKERS=2 python tools/sweepX.py

# Stage-B 精筛
SWEEP_STAGE=B SWEEP_WORKERS=1 python tools/sweepX.py
```

## 核心文件

```
src/
├── train.py      # 主训练脚本
├── models.py    # Online/Target/Conv 模块
├── granular.py # 粒球聚类
├── gb_utils.py # 球扩散与损失函数
└── data.py    # 数据集加载
```