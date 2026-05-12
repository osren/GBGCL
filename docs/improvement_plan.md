# 粒球模块改进方案分析

**日期：2026年5月12日**

---

## 问题诊断

当前粒球扩散存在以下问题：

| 指标 | 值 | 分析 |
|------|-----|------|
| cos_sim(h, z_new) | 0.9996 | 几乎同方向，只有长度变化 |
| ball_purity | 90.7% | 粒球划分已经很好了 |
| h_z_diff | 2-4% | 差异太小 |

**核心问题**：粒球扩散只是在**已有方向的微小调整**，没有引入新的语义信息。

---

## 改进方案

### 方案1：增大 beta 扩散系数

#### 原理
当前 beta=0.2，控制 K 步扩散的混合强度。

公式：```H^{t+1} = (1 - β)H^t + β D^{-1}W H^t```

- beta 越小：保留越多原始球心信息
- beta 越大：更多跨球信息混合

#### 预期效果
- 更大的 beta 会让信息在球之间更强的混合
- 可能产生新的方向（不一定完全在原方向上）

#### 实施
```bash
# 测试不同 beta 值
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_beta 0.5 --num_epochs 50 --trials 1 --device cuda
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_beta 0.8 --num_epochs 50 --trials 1 --device cuda
```

---

### 方案2：降低 alpha 融合系数

#### 原理
当前 alpha=0.6，控制回写时的混合比例。

公式：```z_new = α * node_embed + (1-α) * HK```

- alpha 越大：更多保留原始节点信息
- alpha 越小：更多使用扩散后的粒球信息

#### 预期效果
- 更小的 alpha 会让 z_new 包含更多粒球扩散的改变
- 可能引入更大的方向变化

#### 实施
```bash
# 测试不同 alpha 值
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_alpha 0.3 --num_epochs 50 --trials 1 --device cuda
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_alpha 0.1 --num_epochs 50 --trials 1 --device cuda
```

---

### 方案3：改变 w_mode 权重模式

#### 原理
当前使用 `topo+center`（拓扑+语义混合）。

可选模式：
- `topo`：纯拓扑权重
- `center`：纯语义（球心相似度）权重
- `topo+center`：混合

#### 预期效果
- 不同模式会产生不同的球图结构
- 可能改变信息传播的方式

#### 实施
```bash
# 测试不同 w_mode
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_w_mode topo --num_epochs 50 --trials 1 --device cuda
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_w_mode center --num_epochs 50 --trials 1 --device cuda
```

---

### 方案4：每 epoch 都做粒球扩散（不重建）

#### 原理
当前只在 rebuild 时构建粒球，导致 90% 的时间粒球不参与训练。

修改为：每个 epoch 都用当前嵌入构建新粒球，但不重建，只是用新嵌入更新。

#### 预期效果
- 粒球信息每 epoch 都参与训练
- 持续的指导信息进入训练

#### 实施（需要代码修改）
```python
# 不是只在 epoch % gb_rebuild_every == 0 时执行
# 而是每个 epoch 都执行，但保留之前的结构
```

---

## 方案对比

| 方案 | 改动程度 | 实现难度 | 预期收益 | 风险 |
|------|----------|---------|---------|------|
| 方案1：beta | 参数 | 低 | 中 | 低 |
| 方案2：alpha | 参数 | 低 | 中 | 低 |
| 方案3：w_mode | 参数 | 低 | 低 | 不确定 |
| 方案4：每epoch扩散 | 代码 | 中 | 高 | 中 |

---

## 推荐执行顺序

### 快速验证（第1轮）

先测试方案1和2，这些只是命令行参数改动：

```bash
# === 方案1：beta 测试 ===
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_beta 0.5 --gb_alpha 0.3 --num_epochs 50 --trials 1 --device cuda > logs/beta_05.log 2>&1 &

nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_beta 0.8 --gb_alpha 0.3 --num_epochs 50 --trials 1 --device cuda > logs/beta_08.log 2>&1 &

# === 方案2：alpha 测试 ===
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_alpha 0.3 --num_epochs 50 --trials 1 --device cuda > logs/alpha_03.log 2>&1 &

nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_alpha 0.1 --num_epochs 50 --trials 1 --device cuda > logs/alpha_01.log 2>&1 &
```

### 汇总结果

```bash
# 运行汇总脚本
bash scripts/summary.sh
```

---

## 预期目标

通过这些改进，期望达到：

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| cos_sim | 0.9996 | < 0.98 |
| h_z_diff | 2-4% | > 10% |
| ACC | ~0.934 | > 0.935 |

---

## 下一步行动

1. 先测试方案1（beta）和方案2（alpha）
2. 根据结果选择最佳组合
3. 如无效再考虑方案4（每epoch扩散）