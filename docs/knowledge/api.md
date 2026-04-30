# Command Line API

> train.py 命令行参数

## 常用参数

```bash
python train.py --dataset_name CS --use_gb --gb_quity detach --gb_sim dot --gb_alpha 0.7
```

## 数据集参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_name` | 数据集名称 | Computers |
| `--data_dir` | 数据集目录 | ../../datasets |
| `--results_dir` | 结果输出目录 | results |

## 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_epochs` | 训练轮数 | 700 |
| `--trials` | 训练次数 | 3 |
| `--hidden_dim` | 隐藏层维度 | 1024 |
| `--seed` | 随机种子 | 66666 |
| `--device` | 设备 | cuda |

## Granule Ball 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_gb` | 启用粒球模块 | (flag) |
| `--gb_quity` | 质量度量 | detach |
| `--gb_sim` | 相似度度量 | dot |
| `--gb_alpha` | 融合系数 | 0.6 |
| `--gb_beta` | beta参数 | 0.2 |
| `--gb_K` | 扩散步数 | 10 |
| `--gb_rebuild_every` | 粒球重建间隔 | 50 |

## 损失函数参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ball_loss_weight` | 球级散射权重 | 0.05 |
| `--ball_infonce_weight` | 球级InfoNCE权重 | 0.02 |
| `--ball_infonce_temp` | InfoNCE温度 | 0.2 |

## sweepX.py 参数

通过环境变量设置:

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `SWEEP_STAGE` | 搜索阶段 (A/B) | A |
| `SWEEP_WORKERS` | 并行数 | 2 |

## 返回值

训练完成后在 `results/<dataset>_summary.csv` 中记录:

- `trial`: 试验编号
- `dataset`: 数据集名
- `clf_mean`: 分类精度均值
- `clf_var`: 分类精度方差