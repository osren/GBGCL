# 基线对照表（可引用）

> 所有数字附出处。本项目结果来自 `results/<dataset>_summary.csv`（2026-06-16 统计）。

## 1. 主对标：SGRL (NeurIPS 2024)

**文献**：He, D. et al. *Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering.* NeurIPS 2024. [OpenReview](https://openreview.net/forum?id=R8SolCx62K)

**设置**：节点分类，线性评估（冻结 encoder + Logistic Regression），输入 X+A，hidden=1024，700 epochs（与 `train.py` 一致）。

### Table 1 — SGRL vs 经典 GCL

| 数据集 | SGRL | BGRL | GRACE | DGI | Raw X |
|--------|------|------|-------|-----|-------|
| WikiCS | 79.40±0.10 | 76.86±0.74 | 77.97±0.63 | 75.35±0.14 | 71.98 |
| **Computers** | **90.23±0.03** | 89.69±0.37 | 86.50±0.33 | 83.95±0.47 | 73.81 |
| **Photo** | **93.95±0.03** | 93.07±0.38 | 92.46±0.18 | 91.61±0.22 | 78.53 |
| **Co.CS** | **94.15±0.04** | 92.59±0.14 | 92.17±0.04 | 92.15±0.63 | 90.37 |
| **Co.Physics** | **96.23±0.01** | 95.48±0.08 | — OOM | 94.51±0.52 | 93.58 |

### SGRL 消融（论文 Table 2）

| 变体 | Computers | Photo | Co.CS | Co.Physics |
|------|-----------|-------|-------|------------|
| ≈ BGRL (w/o RSM&TCM) | 89.69 | 93.07 | 92.59 | 95.48 |
| w/o TCM | 89.54 | 93.58 | 94.08 | 96.19 |
| w/o EMA | 90.03 | 93.92 | 93.89 | 96.16 |
| **Full** | **90.23** | **93.95** | **94.15** | **96.23** |

**启示**：
- **Computers**：TCM 约 +0.69%（90.23 vs 89.54）→ 结构约束仍有空间，粒球应服务 TCM。
- **Physics**：各组件边际 ≤0.04% → 超基线极难，机制分析价值 > 刷点。
- **Photo**：w/o EMA 已达 93.92 → 与 Full 差 0.03%。

---

## 2. 本项目 GBGCL 结果

**协议**：`--use_gb`，`clf_mean` 为 logistic regression 精度；与 SGRL 同评估链（见 `train.py`）。

### 2.1 700 epoch 组合均值（5 trials，同超参）

| 数据集 | 最佳组合 (quity,sim,α) | 均值 ACC | vs SGRL |
|--------|------------------------|----------|---------|
| CS | detach, dot, 0.3 | 93.83% | -0.32% |
| Photo | homo, cos, 0.3 | 93.87% | -0.08% |
| Physics | detach, dot, 0.3 | 96.21% | -0.02% |
| Computers | homo, dot, 0.7 | 89.97% | -0.26% |

### 2.2 单 trial 最高（含 150ep Stage-A）

| 数据集 | 最高 ACC | 配置 | epochs |
|--------|----------|------|--------|
| CS | 94.28%* | detach,dot,0.3 (Stage-A) | 150 |
| Photo | 93.94% | homo,cos,0.3 | 700 |
| Physics | 96.24% | detach,dot,0.3 | 700 |
| Computers | 90.06% | homo,dot,0.7 | 700 |

\*150 epoch 粗筛值，**不可与 SGRL 700ep 直接比**；700ep CS max ≈ 93.90%。

### 2.3 无粒球 BYOL（本项目 Photo，见 EXPERIMENTS.md）

| 设置 | ACC |
|------|-----|
| 无 `--use_gb` | ~93.50% (700ep, 5 trial 汇报) |

粒球默认配置在 Photo 上**未优于**纯 BYOL。

---

## 3. 近年相邻工作（非直接可比，定上限参考）

| 方法 | 出处 | Computers | Photo | CS | Physics |
|------|------|-----------|-------|-----|---------|
| SGRL | NeurIPS'24 | 90.23 | 93.95 | 94.15 | 96.23 |
| GraphECL | 2024 | — | — | ~94.1 | ~96.0 |
| FUG rebuild | NeurIPS'24 | 88.4 | 93.1 | 92.9 | 95.6 |

FUG/GraphECL 设定（跨图重建、不同 split）与 SGRL **不可混比**；仅说明 93–96% 档竞争激烈。

### 监督 GNN 上限（OpenCodePapers 汇总，不同协议）

- Coauthor Physics：GCN 调优可达 **~97.5%**（*Classic GNNs are Strong Baselines*, 2024）— **半监督/全监督**，非自监督线性探针。
- Amazon Computers：GAT **~94.1%** 同上半监督语境。

自监督线性探针下，**SGRL 已接近该设定下的实用上限**。

---

## 4. 使用本表的原则

1. 写论文/答辩：主表只放 **SGRL Table 1 + 本项目同协议**。
2. 讨论「提升空间」：引用 **SGRL 消融** 而非监督 GNN SOTA。
3. 更新数字：改 CSV 后重跑 `tools/analyze_results.py`，并更新本节日期。
