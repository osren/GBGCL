# Datasets

> GBGCL 支持的数据集

## 已支持数据集

| 数据集 | 类型 | 节点数 | 边数 | 类别数 | 说明 |
|--------|------|--------|------|------|--------|------|
| CS | 引文网络 | ~3k | ~9k | ~15 | 计算机科学论文 |
| Photo | 引文网络 | ~7k | ~119k | ~8 | 照片分类 |
| Computers | 引文网络 | ~13k | ~245k | ~10 | 计算机领域 |
| Physics | 引文网络 | ~34k | ~178k | ~5 | 物理学论文 |
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

<!-- TODO: 确认各数据集的具体节点数、边数、类别数 -->