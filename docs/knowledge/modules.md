# Core Modules

> GBGCL 核心模块说明

## 模块文件

### train.py

主训练脚本，包含:
- `train_online()`: 单epoch在线训练
- `train()`: 主训练循环
- `fit_logistic_regression()`: 评估
- `main()`: 入口

### models.py

模型定义:

| 类 | 说明 |
|----|------|
| `Conv` | GCN 编码器 |
| `Online` | 在线网络（含预测头） |
| `Target` | 目标网络（EMA） |

### granular.py

粒球构建:

| 函数 | 说明 |
|------|------|
| `build_granules()` | K-means聚类构建粒球 |
| `compute_ball_centers()` | 计算球心 |

### gb_utils.py

工具函数:

| 函数 | 说明 |
|------|------|
| `granule_diffuse_and_write()` | K步扩散+回写 |
| `ball_scatter_loss()` | 球级散射损失(RSM) |
| `ball_infonce()` | 球级InfoNCE |
| `hungarian_matching()` | 匈牙利匹配 |
| `jaccard_between_balls()` | Jaccard相似度 |

### data.py

数据加载:

```python
load_dataset(dataset_name, dataset_dir)
```

## 训练流程

```
1. 加载数据
2. 初始化Online/Target网络
3. 每epoch:
   a. Online前向 -> 获取h, h_pred, h_target
   b. 定时重建粒球
   c. 计算球级损失 (scatter + InfoNCE)
   d. 计算BYOL节点损失
   e. 反向传播更新Online
   f. EMA更新Target
4. 评估: Logistic Regression分类精度