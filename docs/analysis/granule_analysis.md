# 粒球模块训练分析报告

**日期：2026年5月12日**

---

## 1. 实验配置

```bash
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda
```

---

## 2. 关键指标数据

### 2.1 粒球构建指标 (Photo_metrics.csv)

| epoch | num_balls | avg_size | h_norm | z_new_norm | h_z_diff | selected_quality |
|-------|-----------|----------|--------|------------|----------|------------------|
| 0 | 226 | 33.8 | 0.0617 | 0.0584 | 0.2268 | homo |
| 10 | 224 | 34.2 | 0.7035 | 0.7035 | 0.0292 | homo |
| 20 | 224 | 34.2 | 0.9013 | 0.9030 | 0.0324 | homo |
| 30 | 224 | 34.2 | 0.8780 | 0.8798 | 0.0365 | homo |
| 40 | 224 | 34.2 | 0.7119 | 0.7122 | 0.0428 | homo |

### 2.2 训练损失指标 (Photo_train.csv)

观察发现：
- `loss_ball_scatter` 和 `loss_ball_infonce` 只在 epoch 0, 10, 20, 30, 40（有值）
- 其他 epoch 全为 0.0000

这说明**粒球损失只在 rebuild 的 epoch 生效**。

---

## 3. 关键发现

### 发现1：粒球只在 rebuild 时生效

```python
# gb_rebuild_every = 10 时
# epoch 0, 10, 20, 30, 40: 粒球损失有值
# epoch 1-9, 11-19, ...: 粒球损失为 0
```

**影响**：粒球模块在 90% 的训练时间内不参与，无法影响训练过程。

### 发现2：h 和 z_new 差异很小（除 epoch 0 外）

| epoch | h_z_diff |
|-------|----------|
| 0 | 22.68% |
| 10 | 2.92% |
| 20 | 3.24% |
| 30 | 3.65% |
| 40 | 4.28% |

**分析**：后续重建时，粒球增强后的嵌入 z_new 与原始嵌入 h 几乎相同，说明：
1. 嵌入已经训练得比较好，粒球扩散带来的改变很小
2. alpha=0.6 使得 60% 保留原始信息，40% 使用新信息
3. 两者融合后差异被压缩

### 发现3：ACC 结果

| 配置 | ACC |
|------|-----|
| BYOL (无粒球) | 0.9350 |
| 加粒球 | 0.9338 |

**结论**：当前粒球模块没有带来正向提升。

---

## 4. 问题分析

### 问题1：时间维度的不连续
- 粒球只在特定 epoch 构建
- 中间 epoch 完全不使用粒球信息

### 问题2：信息维度的高度相似
- h 和 z_new 的差异只有 2-4%
- 粒球扩散没有带来足够的新信息

### 问题3：损失权重不对等
- BYOL loss (1.0) 主导
- ball_scatter (0.05) 和 ball_infonce (0.02) 权重太小

---

## 5. 建议的改进方向

### 方向1：每 epoch 都做粒球扩散（而不是只重建）

```python
# 不是只在 rebuild 时做，而是每 epoch 都做
for epoch in range(num_epochs):
    z_new = granule_diffuse_and_write(h, ...)  # 每 epoch 都执行
```

### 方向2：增大信息差异

调整 alpha 或 diffusion 参数，使 z_new 与 h 差异更大：
```python
alpha = 0.3  # 更激进的融合
```

### 方向3：在线粒球更新

不是离线构建，而是每 epoch 根据当前嵌入动态调整粒球：
```python
# 每 epoch 都重新构建粒球
GB_nodes = build_granules(h, edge_index, quity)
```

---

## 6. 需要更多指标

为了进一步分析，建议增加：

| 指标 | 用途 |
|------|------|
| cos_sim(h, z_new) | 余弦相似度 |
| ball_purity | 粒球纯度 |
| gradient_norm | 梯度范数 |
| epoch_time | 每 epoch 耗时 |

---

## 7. 结论

当前粒球模块的核心问题是**没有真正融入训练流程**，只是"锦上添花"式附加，而不是端到端优化。

需要从根本上重新设计训练流程，让粒球信息每 epoch 都参与训练。