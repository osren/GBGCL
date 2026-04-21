# 粒球质量（Granular Ball Quity）计算文档

> 本文档详细说明 GBGCL 中粒球质量（Quity）的计算公式、数学含义和分裂条件。
> 代码位置：`src/granular.py`

---

## 一、概述

粒球（Granular Ball）是图中的语义一致的节点簇，同时满足：
1. **拓扑连通性**：同一粒球内的节点在图上相互连通
2. **语义相似性**：同一粒球内的节点嵌入彼此接近
3. **自适应粒度**：粒球大小由数据分布自动决定，无需预设 K

粒球质量（Quity）用于衡量一个子图是否适合作为一个独立的粒球，是决定是否继续分裂的关键指标。

---

## 二、质量函数定义

### 2.1 `quity` (detach) — 默认质量函数

**代码位置**：`granular.py:32-40`

```python
def quity(self, adj_s, z_detach):
    num_edges = torch.sum(adj_s) // 2
    x = torch.matmul(torch.t(z_detach).double(), adj_s.double().to(z_detach.device))
    x = torch.matmul(x, z_detach.double())
    x = torch.trace(x)
    degree_s = adj_s.sum(dim=1)
    y = torch.matmul(torch.t(z_detach).double(), degree_s.double().to(z_detach.device))
    y = (y ** 2).sum() / (2 * num_edges + 1e-9)
    return ((x - y) / (2 * num_edges + 1e-9))
```

**数学公式**：

$$\text{quality}_{\text{detach}} = \frac{\text{Tr}(Z^T A Z) - \frac{\|d\|^2}{2|E|}}{2|E|}$$

其中：
- $Z \in \mathbb{R}^{N \times d}$：节点嵌入矩阵
- $A \in \mathbb{R}^{N \times N}$：邻接矩阵
- $d \in \mathbb{R}^N$：度向量，$d_i = \sum_j A_{ij}$
- $|E|$：子图中的边数

**各项含义**：

| 项 | 含义 |
|----|------|
| $\text{Tr}(Z^T A Z) = \sum_{i,j} A_{ij} \cdot \langle z_i, z_j \rangle$ | 嵌入在边上的加权总和，反映节点与邻居的语义关联程度 |
| $\frac{\|d\|^2}{2|E|} = \frac{\sum_i d_i^2}{2\|E\|}$ | 随机期望值（如果边随机分布时的期望加权总和） |
| $2\|E\|$ | 归一化因子 |

**物理意义**：

该公式衡量子图的嵌入紧密程度**超过随机期望的程度**：
- 值越大 → 子图内部嵌入一致性越高、结构越紧密
- 值越小 → 子图可能包含语义不一致的节点

---

### 2.2 `quity_homo` (homo) — 同质性质量函数

**代码位置**：`granular.py:47-54`

```python
def quity_homo(self, adj_s, z_detach):
    sim_matrix = torch.mm(z_detach, z_detach.T)
    sim_matrix.fill_diagonal_(0)
    weighted = adj_s.to(sim_matrix.device) * sim_matrix
    total_similarity = weighted.sum()
    num_edges = adj_s.sum() / 2
    return total_similarity / (num_edges + 1e-9)
```

**数学公式**：

$$\text{quality}_{\text{homo}} = \frac{\sum_{i,j} A_{ij} \cdot \text{sim}(z_i, z_j)}{|E|}$$

其中 $\text{sim}(z_i, z_j) = \frac{z_i^T z_j}{\|z_i\| \|z_j\|}$ 是余弦相似度。

**各项含义**：

| 项 | 含义 |
|----|------|
| $\sum_{i,j} A_{ij} \cdot \text{sim}(z_i, z_j)$ | 所有边上两端节点相似度的加权总和 |
| $\|E\|$ | 归一化因子（边数） |

**物理意义**：

衡量"**物以类聚**"的程度：
- 如果同质性高，说明边连接的都是语义相似的节点
- 适合用于社区结构明显的数据集（如 CS、Physics）

---

### 2.3 `quity_edges` (edges) — 边密度质量函数

**代码位置**：`granular.py:56-60`

```python
def quity_edges(self, adj_s):
    sub_edges = adj_s.sum() // 2
    degree_sum = adj_s.sum(dim=1).sum().item()
    m_exp = (degree_sum ** 2) / (4 * max(1, self.total_edges))
    return (sub_edges / max(1, self.total_edges)) - (m_exp / max(1, self.total_edges))
```

**数学公式**：

$$\text{quality}_{\text{edges}} = \frac{|E_s| - \frac{d_{sum}^2}{4|E_{total}|}}{|E_{total}|}$$

其中：
- $|E_s|$：子图的边数
- $d_{sum} = \sum_i d_i$：子图中所有节点的度之和
- $|E_{total}|$：原始图的边数（全局）

**各项含义**：

| 项 | 含义 |
|----|------|
| $\|E_s\|$ | 子图的实际边数 |
| $\frac{d_{sum}^2}{4\|E_{total}\|}$ | 如果边在节点间随机分布的期望边数 |

**物理意义**：

衡量子图的边密度**超过随机期望的程度**：
- 值越大 → 子图内部连接比随机情况更紧密（存在社区结构）
- 适合用于噪声边较多的数据集（如 Computers、Photo）

---

### 2.4 `quity_degree` (deg) — 度数质量函数

**代码位置**：`granular.py:42-45`

```python
def quity_degree(self, adj_s):
    total_edges = adj_s.sum() // 2
    num_nodes = adj_s.shape[0]
    return total_edges / max(1, num_nodes)
```

**数学公式**：

$$\text{quality}_{\text{deg}} = \frac{|E_s|}{N_s}$$

其中 $N_s$ 是子图的节点数。

**物理意义**：

最简单直观的连通性度量——**子图的平均度**：
- 值越大 → 子图越稠密
- 适合快速测试或简单场景

---

## 三、分裂条件

### 3.1 分裂算法流程

代码位置：`granular.py:127-174`

```
图 G (N 个节点)
    ↓
1. 初始化：按度数排序，取前 √N 个节点作为初始球心
    ↓
2. 对每个初始球执行 BFS 分裂：
    ↓
    ┌──────────────────────────────────────┐
    │  用 BFS 将节点分成两组（两个候选子球）   │
    │  计算两个子球的质量 qa, qb            │
    │                                       │
    │  if quality_parent > (qa+qb)/2.5:    │
    │      停止分裂（保留原球）              │
    │  else:                               │
    │      递归分裂两个子球                 │
    └──────────────────────────────────────┘
    ↓
3. 输出：粒球集合 {GB_1, GB_2, ..., GB_B}
```

### 3.2 分裂判断公式

**代码位置**：`granular.py:169`

```python
if quality_f > (qa + qb) / 2.5:
    # 不分裂，保留原粒球
    split_GB_list.append(list(graph.nodes()))
else:
    # 分裂成两个子粒球
    split_bfs(sub_a, ...)
    split_bfs(sub_b, ...)
```

**数学公式**：

$$\text{quality}_{parent} > \frac{\text{quality}_a + \text{quality}_b}{2.5}$$

### 3.3 分裂条件解读

| 条件 | 含义 |
|------|------|
| $\text{quality}_{parent}$ | 父粒球的质量 |
| $\frac{\text{quality}_a + \text{quality}_b}{2.5}$ | 两个子球平均质量的 40%（1/2.5） |

**物理意义**：

- **分裂阈值**：只有当父球质量超过子球平均质量的 **2.5 倍** 时，才停止分裂
- **反向思考**：如果分裂后质量下降超过 60%（即子球质量不足父球的 40%），则分裂得不偿失
- 这是一个**贪心策略**：保证每次分裂都有收益

**参数调节**：

如果需要更细粒度的粒球，可以减小分母（从 2.5 改为 2.0 或 1.5）：
```python
# 更细粒度（更多小粒球）
if quality_f > (qa + qb) / 1.5:

# 更粗粒度（更少大粒球）
if quality_f > (qa + qb) / 3.0:
```

---

## 四、质量函数选择指南

### 4.1 各数据集适用性

| 数据集 | 推荐质量函数 | 原因 |
|--------|-------------|------|
| **CS** | `detach` 或 `homo` | 学术合作网络，社区结构清晰，边都是同领域合作关系 |
| **Physics** | `detach` 或 `homo` | 同样是学术网络，社区结构明显 |
| **Photo** | `edges` | 亚马逊商品图，跨品类噪声边较多，需要边密度判别 |
| **Computers** | `edges` | 同上，商品共购关系复杂，噪声边多 |

### 4.2 在 sweepX.py 中使用

```python
# 修改 QUITY 列表以测试不同质量函数
QUITY = ["homo", "detach", "edges", "deg"]
```

或在训练时指定：
```bash
python train.py --gb_quity detach   # 默认
python train.py --gb_quity homo     # 同质性
python train.py --gb_quity edges    # 边密度
python train.py --gb_quity deg      # 度数
```

---

## 五、公式符号汇总表

| 符号 | 含义 | 维度 |
|------|------|------|
| $Z$ | 节点嵌入矩阵 | $N \times d$ |
| $A$ | 邻接矩阵 | $N \times N$ |
| $d$ | 度向量 | $N$ |
| $\|E\|$ 或 $|E_s|$ | 边数（子图） | scalar |
| $N$ 或 $N_s$ | 节点数（子图） | scalar |
| $\text{sim}(z_i, z_j)$ | 余弦相似度 | scalar |
| $\text{Tr}(X)$ | 矩阵trace（对角线之和） | scalar |

---

*文档更新时间：2026-04-21*