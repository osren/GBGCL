# 完整渐进式改进方案

**核心结论：调参数效果很弱，需要从架构层面重新设计迭代式传播机制**

---

## 当前架构问题

```
epoch 0:  构建粒球 → 扩散 → 回写 → 完成（只在此时生效）
epoch 1:  重新计算 h（原始 GCN 输出），不使用上一步的扩散结果 ❌
epoch 2:  重新计算 h（原始 GCN 输出），不使用扩散结果 ❌
...
```

**问题本质**：粒球扩散结果只是一个"一次性过滤器"，没有累积效应。

---

## 渐进式改进方案（按优先级）

### Phase 1: 快速验证（2-3天）

**目标**：确认 Target 分支增强是否有效

#### 1.1 Option B: Target 分支增强

```python
# train.py - train_online() 中修改
# 当前代码（约79-92行）：
with torch.no_grad():
    z_new, gb_sizes, H_ball, GB_node_list, selected_quality = granule_diffuse_and_write(
        node_embed=h, edge_index=data.edge_index, ...
    )

# 修改为：也对 Target 输出做粒球扩散
with torch.no_grad():
    z_target, _, _, _ = granule_diffuse_and_write(
        node_embed=h_target, edge_index=data.edge_index,
        quity=args.gb_quity, sim=args.gb_sim,
        ...
    )
    h_target = z_target  # 增强 Target
```

**验证命令**：
```bash
# 需要修改代码后运行
```

---

### Phase 2: 核心架构改进（1周）

**目标**：实现迭代式粒球传播

#### 2.1 Option A-v1: 特征拼接（推荐）

将粒球增强特征与原始特征拼接，让下一轮 GCN 使用：

```python
# models.py - Conv 类修改
def __init__(self, input_dim, hidden_dim, proj_dim, activation, num_layers, ..., use_gb_feature=False):
    self.use_gb_feature = use_gb_feature
    # 原始输入维度
    self.feature_proj = nn.Linear(input_dim, input_dim)

def forward(self, x, edge_index, gb_enhanced=None):
    # 如果有粒球增强特征，拼接
    if self.use_gb_feature and gb_enhanced is not None:
        x = torch.cat([x, gb_enhanced], dim=-1)

    z = x
    for conv in self.layers:
        z = conv(z, edge_index)
        z = self.activation(z)

    return z, self.projection_head(z)
```

**train.py 修改**：
- 维护一个全局的 gb_enhanced 特征
- 每个 epoch 结束时更新 gb_enhanced
- 下一个 epoch 传入 GCN

#### 2.2 Option A-v2: 输出残差连接（简化版）

不修改输入层，在 GCN 输出后加残差：

```python
# 在 train.py 的 train_online() 中
h, h_pred, h_target = online(data.x, data.edge_index)

if args.use_gb:
    with torch.no_grad():
        z_new, ... = granule_diffuse_and_write(h, ...)
        # 将扩散结果作为残差加到 h 上
        h = h + z_new * 0.1  # 小权重，避免破坏原始表示
```

---

### Phase 3: 高阶优化（2周）

#### 3.1 增加粒球重建频率

```python
# 当前：gb_rebuild_every = 50 或 100
# 修改为：
parser.add_argument('--gb_rebuild_every', type=int, default=10)  # 每 10 个 epoch 重建
```

#### 3.2 多轮扩散（迭代细粒球）

与单次扩散不同，多轮扩散在每轮扩散后重新构建粒球：

```python
def multi_round_diffusion(node_embed, edge_index, num_rounds=3):
    z = node_embed
    for r in range(num_rounds):
        z, ... = granule_diffuse_and_write(z, ...)
        # 下一轮使用更新后的嵌入
    return z
```

---

## 执行计划

| 阶段 | 任务 | 预期效果 | 时间 |
|------|------|----------|------|
| Phase 1 | Option B - Target 增强 | 确认简单改进是否有效 | 2-3天 |
| Phase 2 | Option A - 特征拼接 | 实现迭代式传播 | 1周 |
| Phase 3 | 增加重建频率 | 保持粒球更新 | 1周 |

---

## 推荐执行顺序

```
Step 1: 先做 Phase 1（Option B）
       ↓ 验证是否有效
Step 2: 再做 Phase 2（核心架构）
       ↓ 真正解决迭代问题
Step 3: 最后做 Phase 3（频率优化）
```

---

## 关键代码片段

### train.py 需要修改的位置（约79-106行）

```python
# 当前：
if args.use_gb and (epoch % args.gb_rebuild_every == 0):
    z_new, gb_sizes, H_ball, GB_node_list, selected_quality = granule_diffuse_and_write(
        node_embed=h, edge_index=data.edge_index, ...
    )
    h_pred = online.predictor(z_new)  # 只用这一次 ❌

# 修改为：
if args.use_gb:
    if epoch % args.gb_rebuild_every == 0:
        z_new, ... = granule_diffuse_and_write(...)  # 定期重建

    # Option A: 使用累积的扩散结果
    if hasattr(online, 'gb_history') and online.gb_history is not None:
        h = h + online.gb_history * 0.1  # 残差连接

    h_pred = online.predictor(h)
```

---

## 评估标准

每做一个改进，需要验证：

1. **不下降**：改进后 ACC >= 改进前
2. **稳定性**：std 不显著增加
3. **边际收益**：每个改进的增量效果 > 0.1%

如果连续 3 个改进都无法带来正向收益，说明整体方向有问题，需要重新审视。