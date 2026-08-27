---
title: "GBGCL 论文中文简述"
subtitle: "供内容核对用"
author: "Cheng Tan | 生成日期：2026-08-25"
geometry: margin=2.2cm
documentclass: ctexart
header-includes:
  - \usepackage{booktabs}
  - \usepackage{array}
---

本文档为 `论文写作/main.tex` 的中文内容摘要，便于核对标题、方法描述、实验数字与结论是否与代码及实验记录一致。对应英文稿约 5 页（1–4 页正文 + 第 5 页参考文献，共 20 条，均为 2021–2025 年文献）。

# 一、基本信息

| 项目 | 内容 |
|------|------|
| 英文标题 | GBGCL: Granular Ball Graph Contrastive Learning with Ball-Aware Topology Encoding |
| 中文意译 | 基于粒球感知拓扑编码的粒球图对比学习 |
| 作者 | 谭程（*），重庆邮电大学，重庆 400065 |
| 邮箱 | 1529924810@qq.com |
| 关键词 | 图对比学习、表示散射、粒球计算、自监督学习 |

# 二、摘要（核心主张）

1. **背景**：SGRL（NeurIPS 2024）通过 RSM（表示散射）与 TCM（拓扑约束）在无负样本、弱增广设定下取得强性能，但 TCM 仅在**节点邻接**上传播，**中观社区结构**利用不足。
2. **方法**：提出 **GBGCL**——构建质量引导粒球、在粒球图上扩散，并通过 **BTCM**（Ball-aware Topology Convolution Module）将粒球嵌入注入**双编码器**（online + target），而非仅作并行辅助分支。
3. **结果**：在 SGRL 协议下，四个数据集上 **full GBGCL 均略超 SGRL（+0.05~0.09 个百分点）**；消融表明增益来自 **BTCM**，而非并行 write-back alone。

# 三、Introduction（动机与贡献）

## 3.1 问题链

- GNN 需大量标签；GCL 利用图结构自监督学习节点表示。
- SGRL 用 RSM + TCM + EMA 自举，在 CS、Computers 等同质图 benchmark 上达到 SOTA。
- **局限**：TCM 仅做 $\hat{A}^k$ 节点级聚合，**中观尺度**（介于单节点与全图之间的社区）未进入主编码器。
- 粒球计算支持多粒度划分；但若只做并行扩散 + write-back，测试时仍用 concat(H_o, H_pr)，粒球信号**几乎进不了下游探针**；全局散射（分散）与粒球内扩散（凝聚）还可能**相互抵消**。

## 3.2 方法概述（Figure 1 六阶段）

- **(a)** 输入图 (X,A)
- **(b)** 双 SGRL 编码器（online f_θ + EMA target f_φ）
- **(c)** 质量引导粒球划分，节点分配 β(i)
- **(d)** 粒球图上 K 步消息传递得 H_ball
- **(e)** 节点 write-back + **BTCM 注入**
- **(f)** RSM / TCM / 对齐损失；评估用 concat(H_o, H_pr)

## 3.3 两条贡献

- 将 GBGCL 形式化为 SGRL 的多粒度扩展；**BTCM = BallConv 替换标准 GCN**，对称用于双编码器。
- 四数据集实验：full GBGCL 超 SGRL 0.05–0.09 pp；消融证明 **BTCM 是决定性组件**。

# 四、Preliminaries（预备知识）

- 图 G=(V,E)，特征 X，邻接 A，度归一化邻接 $\hat{A}$。
- GCL：无标签训练编码器 f，线性探针评估下游任务。
- SGRL 骨架：online 参数 θ + target 参数 φ（EMA）；target 做 RSM，online 做 TCM + predictor 对齐。
- **评估协议**：concat(online 嵌入, predictor 嵌入) + 逻辑回归。

# 五、Methodology（方法细节）

## 5.1 RSM + TCM 骨干（与 SGRL 一致）

- **RSM**：target 嵌入 L2 归一化后，最小化到全局中心 c 的距离（鼓励散射）。
- **TCM**：H_online^topo = $\hat{A}^k$ H_online + H_online。
- **对齐损失** L_align：predictor 输出与 target 余弦相似度最大化。
- EMA：φ ← τφ + (1-τ)θ。

## 5.2 粒球扩散模块

- **划分**：Granular 模块按质量（homo / detach / edges）递归分裂为球 B，中心为球内节点嵌入均值。
- **粒球图**：亲和矩阵 W（topo + center，可选 kNN）；K 步扩散得 Z_ball。
- **Write-back**：z_i = α h_i + (1-α) z_ball[β(i)]。
- 每 T_rebuild epoch 重建粒球；支持 gb_incremental 增量刷新中心。
- **粒球级损失**：匈牙利匹配后的 ball scatter + InfoNCE（权重 0.05 / 0.02）。

## 5.3 BTCM（核心创新）

- **动机**：无 BTCM 时 Amazon Photo 上 cos(h, z_new)≈0.9996，h_z_diff 仅 2–4%。
- **Node–ball lookup**：B_i = H_ball[β(i)]；重建间隔间缓存 prev_H_ball。
- **BallConv**：替换 GCN，边消息 m_ij = PReLU(BN(W_m [h_i || h_j || B_i || B_j]))，均值聚合；双编码器共享 B；默认 d_b=1024。
- **变体**：Full GBGCL = SGRL + 粒球扩散 + 粒球损失 + BTCM；w/o BTCM = 仅并行分支。

## 5.4 总目标

L_online = L_align + λ_s L_ball-s + λ_n L_ball-nce；target 最小化 L_RSM。BTCM 使梯度经 BallConv 直接影响探针嵌入。

# 六、Experiment（实验）

## 6.1 设置

| 项目 | 内容 |
|------|------|
| 数据集 | Amazon-Computers、Amazon-Photo、Coauthor-CS、Coauthor-Physics |
| 对比方法 | Raw、DGI、GRACE、BGRL、MVGRL（经典数值引 SGRL）；另提及 GraphMAE、CARL-G、GSTBench |
| 协议 | 1 层编码器，hidden 1024，Adam lr 1e-3，700 epoch，5 trials，seed 66666，冻结嵌入 + 逻辑回归 |
| 超参搜索 | Stage-A 150 ep + Stage-B 700 ep 网格搜索 |
| 最优超参 | CS: detach,dot,α=0.3；Photo: homo,cos,0.3；Physics: detach,dot,0.3；Computers: homo,dot,0.7 |

## 6.2 主表结果（Table 1，%）

| 方法 | Computers | Photo | Co.CS | Co.Physics |
|------|-----------|-------|-------|------------|
| SGRL | 90.23±0.03 | 93.95±0.03 | 94.15±0.04 | 96.23±0.01 |
| GBGCL w/o BTCM | 89.97±0.08 | 93.87±0.07 | 93.83±0.06 | 96.21±0.03 |
| **GBGCL (full)** | **90.31±0.06** | **94.02±0.05** | **94.22±0.05** | **96.28±0.02** |

**解读**：Full 四数据集最优，较 SGRL +0.05~0.09 pp；w/o BTCM 略低于 SGRL；Computers 提升最大（+0.08 pp）。

## 6.3 机制诊断（Photo）

| 指标 | w/o BTCM | + BTCM |
|------|----------|--------|
| cos(h, z_new) | ≈ 0.9996 | ≈ 0.992 |
| h_z_diff | 2–4% | 8–12% |
| 粒球纯度 | ~90.7% | ~90.7% |

## 6.4 消融（Photo）

| 变体 | Acc. (%) |
|------|----------|
| SGRL-equiv.（无 use_gb） | 93.50 |
| GBGCL w/o BTCM | 93.87 |
| + feature concat | 93.38 |
| + online residual | 92.86 |
| + target enhance | 93.54 |
| + larger ball loss | 93.53 |
| **GBGCL full (+BTCM)** | **94.02** |

## 6.5 讨论与未来工作

- 增益 modest 但一致，符合同质图 benchmark 接近天花板的情况。
- 未来：**BRSM**（粒球级散射）、粒球邻居 InfoNCE 正样本。

# 七、Conclusion（结论）

GBGCL 在 SGRL 的 RSM/TCM 之上，用粒球图扩散、粒球级损失与 **BTCM 注入双编码器**，在四 benchmark 上稳定略超 SGRL；消融将改进归因于 BTCM，而非并行 write-back alone。

# 八、建议重点核对项

1. **数值**：主表与消融是否与 `results/` CSV 一致。
2. **w/o BTCM vs. SGRL**：论文主张 w/o BTCM 略低于 SGRL。
3. **机制指标**：cos、h_z_diff、粒球纯度是否有日志支撑。
4. **超参**：detach/homo、dot/cos、α 是否与 sweep 最优一致。
5. **评估协议**：concat(H_o, H_pr) + 逻辑回归是否与 `src/train.py` 一致。
6. **作者信息**：单位、邮箱是否为定稿版本。
7. **参考文献**：20 条均为 2021–2025；Table 1 经典基线引 SGRL，DGI/GRACE/MVGRL 无单独 cite。

---

源文件：`F:\GBGCL\修改意见\GBGCL论文中文简述.md`  
对应英文稿：`F:\GBGCL\论文写作\main.tex`
