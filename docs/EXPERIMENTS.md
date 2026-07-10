# 实验记录（已尝试方案）

> 汇总自周报、`archive/` 历史文档与 `results/*.csv`。新实验请在文末追加一行。

## 1. 实验协议（统一）

- Encoder：1-layer GCN, hidden=1024, num_hop=1
- 训练：700 epochs（扫参 Stage-A 为 150），trials=5（汇报用均值）
- 评估：`clf_mean`，Logistic Regression on `(or_embeds + pr_embeds)`
- 设备：CUDA；种子 66666

---

## 2. 方案总览

| ID | 方案 | 关键 flags | Photo ACC | vs BYOL | 结论 |
|----|------|------------|-----------|---------|------|
| E0 | 纯 BYOL | 无 `--use_gb` | ~93.50% | baseline | 当前 Photo 最强参考 |
| E1 | 粒球 Baseline | `--use_gb --gb_quity homo` | ~93.44% | -0.06% | 粒球默认有害 |
| E2 | Option A v1 特征融合 | `--gb_feature_concat` | ~93.38% | -0.12% | 拼接引入噪声 |
| E3 | Option A v2 残差 | `--gb_residual_online` | ~92.86% | -0.64% | 明显变差 |
| E4 | Option B Target 增强 | `--gb_target_enhance` | ~93.54% | +0.04% | 微弱正向 |
| E5 | Option C 损失权重↑ | ball_loss_weight 等 | ~93.53% | +0.03% | 微弱正向 |
| E6 | Option A2 Ensemble | `--gb_ensemble` | 50ep×3 | 权重≈0.33 各 quity | 未选出差异 quity |
| E7 | 增量扩散 | `--gb_incremental` | **待跑** | — | 代码已合入 2026-05-19 |
| E8 | 系统扫参 Stage-A/B | `sweepX.py` | 见 CSV | — | CS/Photo 充分；Physics/Computers 不足 |

*Photo E0–E6 来源：周报 2026-05-12；精度为汇报约数。*

---

## 3. 四数据集扫参状态

| 数据集 | CSV 行数 | 700ep 5-trial 组合数 | 状态 |
|--------|----------|----------------------|------|
| CS | 335 | 多组 | 完成度高 |
| Photo | 335 | 多组 | 完成度高 |
| Physics | 17 | 2 | **严重不足** |
| Computers | 69 | 少量 | **不足** |

---

## 4. 诊断性指标（Photo，粒球 Baseline）

| 指标 | 观测值 | 含义 |
|------|--------|------|
| cos_sim(h, z_new) | ~0.9996 | 扩散几乎不改变方向 |
| ball_purity | ~90.7% | 划分质量尚可 |
| h_z_diff | 2–4% | 回写幅度过小 |

→ 问题不在「球分得不好」，而在「扩散信息未进入主表示学习路径」。

---

## 5. 专项跑（results_final / analysis_final）

| 数据集 | 配置 | trials | max | 备注 |
|--------|------|--------|-----|------|
| CS | detach,dot,0.3, K=5,center | 20 | 94.06% | 仍 < SGRL 94.15% |

---

## 6. Phase R：Photo / Computers 复跑验证（2026-07-08）

服务器侧 `scripts/phases/phaseR_repeat_validation.sh`（PID 555977），4 组 × 700ep × 5trial，总耗时 1h44min：

| 组别 | 数据集 | 配置 | 5 trials mean | vs SGRL | Δ vs baseline |
|------|--------|------|---------------|---------|---------------|
| R-A | Photo | Phase 0 baseline（`--use_gb` 不开，SGRL 等价） | 0.9391 ± 0.0007 | -0.04pp | — |
| R-B | Photo | Stage-B Top-1（quity=detach, sim=dot, α=0.3, K=10） | 0.9380 ± 0.0008 | -0.15pp | **-0.11pp** |
| R-C | Computers | Phase 0 baseline | 0.9004 ± 0.0002 | -0.19pp | — |
| R-D | Computers | Stage-B Top-1（quity=homo, sim=dot, α=0.7） | 0.8998 ± 0.0005 | -0.25pp | **-0.06pp** |

**关键结论**：

1. 之前 "Photo -0.20pp vs SGRL" 的 finding **不准确** —— Phase 0 baseline 自身就 -0.04pp，**SGRL 的 93.95% 本身就是上限附近**
2. Stage-B Top-1 vs baseline 仅差 **-0.11pp（Photo）** 和 **-0.06pp（Computers）**，在 SGRL 标准差 ±0.5pp 范围内
3. **粒球扩散（GBGCL）对 Photo/Computers 没有显著增益**，基本就是 SGRL 噪声水平
4. BTCM 才是 GBGCL 论文真正的差异化方向，需要把球扩散嵌入 InfoNCE 才能拉开差距
5. CS / Physics 还没跑过 Phase R，本周内补跑（CS 94.15% / Physics 96.23% 是 SGRL 最强数据集，需单独评估强基线上是否退化）

结果文件：`results/phaseR/{Photo,Computers}_summary.csv`；分析脚本：`tools/analyze_phaseR.py`。

---

## 7. 失败假设登记

| 假设 | 结果 |
|------|------|
| 调 alpha/beta/K 即可超基线 | Photo/Computers 宽扫后仍持平或略低 |
| Online 残差吸收粒球信息 | Option A v2 显著变差 |
| 多 quity 投票自动选最优 | Photo 上三 quity 权重接近 |
| 仅增大球级 loss | +0.03%，不足 |
| GBGCL-Stage-B 在 Photo 上比 SGRL 退化 -0.20pp | Phase R 否定：baseline 自身就 -0.04pp，Stage-B Top-1 仅再降 -0.11pp（噪声内） |

---

## 8. 待跑实验（与 ROADMAP 对齐）

```bash
# E7 增量扩散快速验证
cd src
python train.py --dataset_name Photo --use_gb --gb_quity homo --gb_sim cos --gb_alpha 0.3 \
  --gb_incremental --gb_rebuild_every 10 --num_epochs 50 --trials 3 --device cuda

# SGRL 复现（无粒球）
python train.py --dataset_name CS --num_epochs 700 --trials 5 --device cuda

# Stage-A sweep
cd ..
$env:SWEEP_STAGE="A"; python tools/sweepX.py
```

---

## 9. 变更日志

| 日期 | 事件 |
|------|------|
| 2026-05-19 | 增量扩散 `incremental_diffuse_and_write` 合入 |
| 2026-05-12 | Option A/B/C 系统对比完成 |
| 2026-05-09 | Option A2 Ensemble 实现 |
| 2026-04 | Phase1 sweepX 搜索空间扩充 |
| 2026-07-08 | Phase R Photo/Computers 完成；BTCM scaffold 落地（Photo 50ep 0.8897 smoke） |
