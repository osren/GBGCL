# GBGCL 研究路线图

> **最后更新**：2026-06-16  
> **愿景**：在 SGRL 散射机制上引入**多粒度粒球扩散**，形成可论证的「拓扑约束 + 多粒度散射」，稳定超越 SGRL 基线。

---

## 1. 我们在做什么

| 层 | 内容 |
|----|------|
| 基座 | SGRL：Target 上 **RSM**（全局 center-away 散射）+ Online 上 **TCM**（拓扑聚合预测 Target）+ EMA |
| 扩展 | GBGCL：粒球划分 → 球图 K 步扩散 → `z_new` 回写 + 球级 Scatter/InfoNCE |
| 假设 | 粒球提供**介于节点与全图之间的中粒度拓扑**，可加强 TCM 式结构约束并补充 RSM 未覆盖的局部语义 |

**SGRL 论文要点**（He et al., NeurIPS 2024, `output/sgrl.../sgrl.md`）：
- 三种 GCL 框架隐含共同机制：**representation scattering**
- SGRL 显式 RSM + 分离式 TCM，避免负样本与增广开销
- 全数据集 SOTA 在 Table 1；消融显示 **Computers 上 TCM 贡献最大**

---

## 2. 真实基线与我们的位置

详见 [`BASELINES.md`](BASELINES.md)。摘要（SGRL Table 1 vs 本项目 `results/*_summary.csv`，2026-06 统计）：

| 数据集 | SGRL 基线 | 我们 700ep 组合均值 | 我们单 trial 最高 | 差距 | 实验覆盖 |
|--------|-----------|---------------------|-------------------|------|----------|
| CS | 94.15±0.04 | 93.83 (detach,dot,0.3) | 93.90 (700ep) | **-0.3%** | 335 行 ✅ |
| Photo | 93.95±0.03 | 93.87 (homo,cos,0.3) | 93.94 | **-0.01%** | 335 行 ✅ |
| Physics | 96.23±0.01 | 96.21 (detach,dot,0.3) | 96.24 | ~0 | **17 行** ⚠️ |
| Computers | 90.23±0.03 | 89.97 (homo,dot,0.7) | 90.06 | **-0.17%** | **69 行** ⚠️ |

**结论**：
- 并非「CS 已稳定超基线」——当前 CSV 最高 700ep 约 **93.9%**，低于 SGRL **94.15%**（20 trial 专项跑 max 94.06% 仍略低）。
- Photo/Physics **已触及 SGRL 天花板**；Computers **有 TCM 级改进空间**（SGRL 消融 TCM +0.69%）但未发挥。
- Physics/Computers **扫参严重不足**，不能得出「粒球无效」的结论。

---

## 3. 为什么提效小、难超基线（机制 + 外部）

### 3.1 外部：基线已强、数据集同质性高

- SGRL 在五个基准已是 2024 GCL 强线；Physics 上 Full vs w/o TCM 仅差 **0.04%**。
- Coauthor/Amazon 高维特征、高同质性 → 线性探针下 **MLP 可接近 GNN**（*Classic GNNs are Strong Baselines*, 2024）。
- **可提升空间有限**：Photo/Physics 上 0.1–0.3% 可能已是合理天花板，除非改评估或架构。

### 3.2 内部：粒球未进入 SGRL 主路径

当前实现（见 [`ARCHITECTURE.md`](ARCHITECTURE.md)）问题链：

```
问题 A：粒球模块与 TCM/RSM 并联而非融合
  → 球级 loss 权重 0.05，对主 BYOL 梯度影响弱

问题 B：重建间隔内嵌入演化、粒球结构滞后（部分由 --gb_incremental 缓解）
  → cos_sim(h, z_new) ≈ 0.9996，扩散几乎不改变方向

问题 C：散射 vs 平滑的目标冲突
  → RSM：推离全局中心；球内扩散：拉近同球节点
  → 未在「球心散射 + 球内约束」分层解耦

问题 D：评估用原始 or_embeds+pr_embeds，非 z_new
  → 训练时粒球对下游嵌入贡献间接
```

### 3.3 已试方案为何失败

见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。Option A（拼接/残差）在 Photo 上**弱于纯 BYOL**；Option B/C 仅 +0.03–0.04%。说明：**旁路拼接/小权重 loss 不够**，需要改 SGRL 双路信息流。

---

## 4. 核心障碍 → 解决思路

| # | 障碍 | 解决思路 | 验证信号 |
|---|------|----------|----------|
| O1 | 粒球不在 TCM 路径 | **BTCM**：用球扩散 `z_ball` 替代或加性融合 `H_topology = Â^k H + γ·z_ball` | Computers ACC ↑；`h_z_diff` ↑ 且 cos_sim 仍 <0.99 |
| O2 | 散射仅在节点级 | **BRSM**：对球心 `H_ball` 做 center-away loss（类比 SGRL Eq.4） | 球间分离度↑，下游 ACC↑ |
| O3 | 结构/嵌入不同步 | **增量扩散** `--gb_incremental` + 每 N epoch 重建 | 非重建 epoch 的 `h_z_diff` 稳定 >3% |
| O4 | Physics/Computers 未扫参 | 跑完 `sweepX` Stage-A→B（FILTERS 已扩充） | CSV 行数与 CS 同级 |
| O5 | 球级 loss 过弱 | 球级权重与 SGRL 主 loss **同量级调度**（非一次调到 0.3） | loss 曲线：ball_loss 与 alignment 同阶 |

---

## 5. 推荐技术路线：G-SGRL（粒球增强散射图学习）

**阶段 0 — 现状固化（1–2 天）**
- [ ] 固定 SGRL 复现：四数据集无 `--use_gb`，确认与论文 Table 1 一致（±0.2%）
- [ ] 记录 BYOL vs SGRL vs GBGCL 三列基线表写入 `EXPERIMENTS.md`

**阶段 1 — 机制验证（3–5 天）**
- [ ] `--gb_incremental` + `gb_rebuild_every=10` on Photo，50 epoch × 3 trials
- [ ] 对比 `cos_sim`, `h_z_diff`, ball_purity 日志（`docs/debug_plan` 指标）
- [ ] 若 `h_z_diff` 仍 <2%：增大 `gb_beta` 或减小 `gb_alpha`（见改进方案 Phase 1）

**阶段 2 — 架构融合（1–2 周，核心）**

```
                    ┌─ RSM on H_target (保留 SGRL)
                    │
  Target Encoder ───┤
                    └─ BRSM on H_ball_target (新增，球心散射)

  Online Encoder ───┬─ TCM: Â^k H_online (保留)
                    └─ BTCM: fuse(z_ball_online) → Predictor → align H_target

  粒球：每 N epoch 重建 GB_node_list；每 epoch incremental 更新球心+扩散
```

实现要点（`train.py` / `gb_utils.py`）：
1. `granule_diffuse_and_write` 输出进入 **Online 拓扑分支**（仿 TCM Eq.5），而非仅辅助 loss
2. 球心 `H_ball` 上增加 `L_scatter_ball = -mean(||h̃_b - c_ball||²)`，与节点级 RSM 分工
3. Target 分支可选 `gb_target_enhance`（Option B）与 BTCM **二选一**先验证，避免堆叠

**阶段 3 — 系统扫参（与阶段 2 并行）**
- Physics + Computers：Stage-A 全网格 → Top-3 → Stage-B 700ep×5 trials
- 重点：`gb_alpha∈{0.3,0.5,0.7}`, `gb_quity∈{detach,edges,homo}`, `gb_K∈{3,5,10}`

**阶段 4 — 论文叙事**
- 贡献点：**多粒度散射**（节点 RSM + 球 BRSM）+ **粒球拓扑约束**（BTCM 扩展 TCM）
- 对标：SGRL Table 1 + 消融；粒球侧引用 GBGC/SGBGC（粗化/多粒度）

---

## 6. 里程碑与停止条件

| 里程碑 | 成功标准 | 失败则 |
|--------|----------|--------|
| M1 复现 SGRL | 四集与 Table 1 ±0.2% | 先修训练/评估协议 |
| M2 增量扩散 | Photo `h_z_diff`≥3% 且 ACC 不降 | 改 w_mode/beta 或重建频率 |
| M3 BTCM 原型 | Computers ≥90.23% 5-trial 均值 | 回退并只保留 BRSM |
| M4 全面超基线 | 至少 2/4 数据集稳定超 SGRL | 收缩贡献到「Computers + 机制分析」 |

**停止条件（假设「粒球+SGRL」无效）**：
- M2+M3 完成后，四集均值仍 ≤ SGRL，且 `h_z_diff` 无法拉高 → 转向「粒球仅用于粗化预训练」或换数据集（OGB 等）。

---

## 7. 文档索引

| 文档 | 用途 |
|------|------|
| [BASELINES.md](BASELINES.md) | 可引用基线表 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 已做实验与结论 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 代码与数据流 |
| [COMMANDS.md](COMMANDS.md) | 运行命令 |
| [archive/](archive/) | 历史文档（勿作现状依据） |
| `.cursor/skills/gnn-gbgcl-expert/` | AI 专家背景 skill |

---

## 8. 立即行动（本周）

1. **跑 SGRL 纯基线**（无 `--use_gb`）四数据集，核对 Table 1  
2. **Photo 上验证** `--gb_incremental --gb_rebuild_every 10`，盯 debug 日志  
3. **启动 Physics/Computers Stage-A** sweep（`SWEEP_STAGE=A`）  
4. **开 BTCM 分支**（小 PR）：在 `train_online` 将 `z_new` 注入 topology 聚合路径
