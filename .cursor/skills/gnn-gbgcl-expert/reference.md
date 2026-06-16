# 可核验参考文献与基线数据

> 仅收录有公开 PDF/表格的数字。更新时请附 URL 或论文表号。

## 1. 本项目直接对标：SGRL

**论文**：He et al., *Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering*, NeurIPS 2024.  
**OpenReview**：https://openreview.net/forum?id=R8SolCx62K  
**代码**：https://github.com/hedongxiao-tju/SGRL

**Table 1 — 节点分类 F1（线性评估，X+A）**：

| 数据集 | SGRL | BGRL | GRACE | DGI | Raw Features |
|--------|------|------|-------|-----|--------------|
| WikiCS | 79.40±0.10 | 76.86±0.74 | 77.97±0.63 | 75.35±0.14 | 71.98±0.00 |
| Computers | **90.23±0.03** | 89.69±0.37 | 86.50±0.33 | 83.95±0.47 | 73.81±0.00 |
| Photo | **93.95±0.03** | 93.07±0.38 | 92.46±0.18 | 91.61±0.22 | 78.53±0.00 |
| Coauthor-CS | **94.15±0.04** | 92.59±0.14 | 92.17±0.04 | 92.15±0.63 | 90.37±0.00 |
| Coauthor-Physics | **96.23±0.01** | 95.48±0.08 | OOM | 94.51±0.52 | 93.58±0.00 |

**Table 2 消融（SGRL 全文）**：

| 变体 | Computers | Photo | Co.CS | Co.Physics |
|------|-----------|-------|-------|------------|
| w/o RSM & TCM (=BGRL级) | 89.69 | 93.07 | 92.59 | 95.48 |
| w/o TCM | 89.54 | 93.58 | 94.08 | 96.19 |
| w/o EMA | 90.03 | 93.92 | 93.89 | 96.16 |
| **Full SGRL** | **90.23** | **93.95** | **94.15** | **96.23** |

**解读**：TCM 对 Computers 贡献约 **+0.69%**；Physics 上各组件边际极小（~0.01–0.04%），接近饱和。

## 2. 近年相邻 GCL（同数据集，仅供参考）

| 方法 | 出处 | 备注 |
|------|------|------|
| GraphECL | Xiao et al. 2024 | AAAI 相关工作表含 CS/Physics ~94–96% 档 |
| FUG | NeurIPS 2024 | Photo ~93.1%, Computers ~88.4%, CS ~92.9%, Physics ~95.6%（跨域重建设定，非直接可比） |
| AS-GCL | arXiv:2502.13525 | 2025，多数据集 SOTA 宣称需核对协议 |

**注意**：2024 论文 *Classic GNNs are Strong Baselines* (arXiv:2412.06173) 指出在 Coauthor/Amazon 上**调优 MLP 可接近多数 GNN**，高维特征数据集上无结构方法的头寸有限——解释「难超基线」的外部因素。

## 3. 粒球 + 图学习（机制参考，非节点分类同设定）

| 工作 | 出处 | 与 GBGCL 关系 |
|------|------|----------------|
| GBGC | IJCAI 2025 | 粒球图粗化，多粒度超节点 |
| SGBGC | AAAI 2024 | 监督粒球划分用于可扩展 GNN |
| MGCN-FLC | arXiv:2603.26729 | 粒球拓扑构建 + 特征增强 |
| GBC 综述 | Xia et al. 2022–2024 | 粒球分裂、纯度、多粒度原理 |

粒球在图中主要用于**粗化/拓扑构造**，GBGCL 的创新在于接入 **GCL 训练环**——该交叉点文献仍少，不宜声称已有 SOTA 先例。

## 4. 数据集（Shchur et al. 2018 Planetoid/Amazon/Coauthor）

| 数据集 | 节点 | 边 | 特征维 | 类 | 同质性高 |
|--------|------|-----|--------|-----|----------|
| Photo | 7,650 | 119,081 | 745 | 8 | 是 |
| Computers | 13,752 | 245,861 | 767 | 10 | 是 |
| CS | 18,333 | 81,894 | 6,805 | 15 | 是 |
| Physics | 34,493 | 247,962 | 8,415 | 5 | 是 |

## 5. 检索建议（人工核验用）

- OpenReview / arXiv：方法名 + dataset 名 + "node classification"
- Papers with Code：OpenCodePapers 各数据集 benchmark 页（注意是否同 splits）
- 禁止：无出处的「业界普遍」「通常能达到」类表述
