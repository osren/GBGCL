# GBGCL Architecture

> 技术栈与架构概览

## 项目信息

- **项目名称**: GBGCL (Granular Ball Graph Contrastive Learning)
- **研究方向**: 图表示学习 + 粒球扩散
- **核心框架**: PyTorch + PyTorch Geometric

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| Python | 3.9.7 | - |
| PyTorch | 2.1.0 | - |
| torch-geometric | 2.5.3 | - |
| sklearn | - | - |
| rich | - | 终端输出 |

## 核心架构

```
训练流程:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ GCN Encoder │ -> │ Online Net  │ -> │ Target Net  │
│  (EMA)     │    │            │    │   (EMA)    │
└─────────────┘    └─────────────┘    └─────────────┘
                        │                   │
                        v                   v
                 ┌─────────────────────────────┐
                 │    粒球扩散模块        │
                 │ 1. 构建粒球 (K-means)  │
                 │ 2. 球图 K步扩散         │
                 │ 3. 回写到节点          │
                 └─────────────────────────────┘
                        │                   │
                        v                   v
                 ┌─────────────┐    ┌─────────────┐
                 │ 球级散射损失 │    │ 球级InfoNCE│
                 │  (RSM)     │    │           │
                 └─────────────┘    └─────────────┘
                        │                   │
                        v                   v
                 ┌─────────────────────────────┐
                 │    BYOL 节点对比损失       │
                 └─────────────────────────────┘
```

## 关键超参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `gb_quity` | 粒球质量度量 | homo, detach, edges, deg |
| `gb_sim` | 相似度度量 | dot, cos, per |
| `gb_alpha` | 融合系数 | 0.3-0.7 |
| `gb_K` | 扩散步数 | 3, 5, 10, 20 |
| `gb_rebuild_every` | 粒球重建间隔 | 50, 100 |

## 依赖关系

```
train.py (入口)
├── models.py (Conv, Online, Target)
├── data.py (load_dataset)
├── gb_utils.py
│   ├── granule_diffuse_and_write
│   ├── build_granules
│   ├── ball_scatter_loss
│   ├── ball_infonce
│   └── hungarian_matching
└── granular.py (build_granules, compute_ball_centers)
```

<!-- TODO: 确认完整依赖图 -->