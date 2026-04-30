# Option A 操作指南

> 自适应质量函数选择 - 根据图统计自动选择 quity（homo/detach/edges）

---

## 快速测试

```bash
cd /Users/didi/Desktop/GBGCL/src
python train.py --dataset_name Photo --use_gb --gb_quality auto --num_epochs 10 --trials 1
```

预期输出：
```
[Auto] Selected quity: detach
```

---

## 完整训练

```bash
python train.py --dataset_name Photo --use_gb --gb_quality auto --num_epochs 700 --trials 5
```

---

## 对比测试（自动 vs 手动）

```bash
# Option A 自动选择
cd src
python train.py --dataset_name Photo --use_gb --gb_quality auto --num_epochs 700 --trials 5

# 手动指定（对照）
python train.py --dataset_name Photo --use_gb --gb_quality detach --num_epochs 700 --trials 5
python train.py --dataset_name Photo --use_gb --gb_quality homo --num_epochs 700 --trials 5
```

---

## 批量测试（sweepX）

修改 `tools/sweepX.py` 中的 FILTERS：

```python
FILTERS = {
    "Photo": [
        ("auto", "dot", 0.3),  # auto 自动选择
    ],
    "Computers": [
        ("auto", "dot", 0.7),
    ],
    "Physics": [
        ("auto", "dot", 0.3),
    ],
}
```

运行：
```bash
cd /Users/didi/Desktop/GBGCL

# Stage-A 粗筛
SWEEP_STAGE=A SWEEP_WORKERS=2 python tools/sweepX.py

# Stage-B 精训
SWEEP_STAGE=B SWEEP_WORKERS=1 python tools/sweepX.py
```

---

## 自动选择逻辑

| 输入条件 | 选择结果 | 说明 |
|---------|---------|------|
| 同质率 > 0.6（有标签） | `homo` | 边两端多为同类节点 |
| 度变异系数 > 0.8 | `detach` | 异质图，度分布差异大 |
| 低度节点比例 > 30% | `edges` | 噪声图，存在大量低度节点 |
| 其他 | `detach` | 默认 |

**统计指标计算：**
- 同质率：统计边两端标签相同的比例
- 度变异系数：degree_std / avg_degree
- 低度节点比例：度数 < 平均度数 × 0.5 的节点比例

---

## 各数据集预期选择

| 数据集 | 预期选择 | 原因 |
|--------|---------|------|
| CS | `detach` | 同质率中等 |
| Photo | `edges` | 稀疏图，噪声边较多 |
| Computers | `detach` | 异质性较强 |
| Physics | `detach` 或 `homo` | 同质性较高（需验证） |

---

## 回退

如需回退到修改前状态：

1. **train.py**: `--gb_quity` 移除 `'auto'` 选项
2. **gb_utils.py**: `build_granules()` 移除 `labels` 参数
3. **sweepX.py**: QUITY 移除 `'auto'`
4. **granular.py**: 删除 `auto_quality()` 方法