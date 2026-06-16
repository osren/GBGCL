# Datasets

> GBGCL 支持的数据集

## 已支持数据集

| 数据集 | 类型 | 节点 | 边 | 特征维 | 类 | SGRL 基线 |
|--------|------|------|-----|--------|-----|-----------|
| CS | Coauthor | 18,333 | 81,894 | 6,805 | 15 | 94.15% |
| Photo | Amazon | 7,650 | 119,081 | 745 | 8 | 93.95% |
| Computers | Amazon | 13,752 | 245,861 | 767 | 10 | 90.23% |
| Physics | Coauthor | 34,493 | 247,962 | 8,415 | 5 | 96.23% |
| Cora | 引文网络 | ~2.7k | ~5k | ~7 | 经典基准 |
| CiteSeer | 引文网络 | ~3.3k | ~4.6k | ~6 | 经典基准 |
| PubMed | 引文网络 | ~19.7k | ~44k | ~3 | 医学论文 |

## 数据集来源

使用 `torch_geometric.datasets` 加载:

- Planetoid: Cora, CiteSeer, PubMed
- Amazon: Photo, Computers
- Coauthor: CS, Physics

## 使用方式

```python
from data import load_dataset
data = load_dataset('CS', './datasets')
```

基线来源：SGRL NeurIPS 2024 Table 1。详见 `docs/BASELINES.md`。