# Knowledge Index

> 开发参考。研究策略与基线见上级目录 `docs/ROADMAP.md`、`docs/BASELINES.md`。

## 文档导航

| 文档 | 说明 |
|------|------|
| [../ROADMAP.md](../ROADMAP.md) | 研究路线（优先阅读） |
| [../BASELINES.md](../BASELINES.md) | 可引用基线 |
| [../EXPERIMENTS.md](../EXPERIMENTS.md) | 实验记录 |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | 架构精简版 |
| [../COMMANDS.md](../COMMANDS.md) | 命令速查 |
| [architecture.md](architecture.md) | 模块级技术栈 |
| [datasets.md](datasets.md) | 数据集 |
| [modules.md](modules.md) | 核心模块 |
| [api.md](api.md) | CLI 参数 |

## 常用命令

```bash
cd src && python train.py --dataset_name CS --use_gb --gb_quity detach --gb_sim dot --gb_alpha 0.7
SWEEP_STAGE=A python tools/sweepX.py
```

## 核心代码

```
src/train.py, models.py, granular.py, gb_utils.py, data.py
tools/sweepX.py, analyze_results.py
```
