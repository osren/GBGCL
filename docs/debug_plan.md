# 方案4 增量扩散调试计划

## 问题描述

增量扩散模式（`--gb_incremental`）测试结果：
- **ACC 下降**：0.9353 → 0.9345
- **h_norm 完全不变**：每个 epoch 都是 `[0.0299, 0.0515, 0.0818, 0.0398, 0.0415]`
- **H_ball 完全不变**：每个 epoch 都是 `[0.0645, 0.0595, 0.0651, 0.0552, 0.0554]`

## 核心观察

| 指标 | 正常模式（epoch 0） | 增量模式（epoch 1-49） |
|------|---------------------|------------------------|
| h_norm | 变化 | **完全不变** |
| H_ball | 变化 | **完全不变** |
| cos_sim | 0.9840 | **保持 0.9840** |

正常模式下 epoch 0 的 h 和 H_ball 也没有变化，但这可能是因为 epoch 0 嵌入未训练。

## 调试任务

### 测试1：正常模式（不带 --gb_incremental）

```bash
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 20 --trials 1 --device cuda
```

观察输出中的 `[h_DEBUG]`，看 h_norm 是否随 epoch 变化。

### 测试2：增量模式

```bash
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_incremental --num_epochs 20 --trials 1 --device cuda
```

观察 `[h_DEBUG]` 和 `[INCR]`，看：
- h_norm 是否变化
- h_mean, h_std 是否有变化

## 预期结果

| 模式 | h_norm 变化 | 说明 |
|------|-------------|------|
| 正常模式 | **应该变化** | 因为每次调用 online() 都会更新权重 |
| 增量模式 | **应该变化** | 如果不变，说明 Online 网络没有更新 |

## 如果两种模式下 h_norm 都不变

说明 epoch 0 的输出本身就被固定了（可能是权重初始化问题）。

## 如果正常模式下 h_norm 变化，但增量模式下不变

说明 `online()` 函数调用有问题，可能是：
1. `use_incremental=True` 时跳过了 online() 调用
2. `prev_GB_node_list` 传递有问题

---

## 服务器命令汇总

```bash
# 测试1：正常模式
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 20 --trials 1 --device cuda

# 测试2：增量模式
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --gb_incremental --num_epochs 20 --trials 1 --device cuda
```

## 下一步

根据测试结果决定：
- **如果正常模式 h_norm 变化** → 问题在增量模式的逻辑
- **如果两种模式 h_norm都不变** → 问题在 online() 本身，需要检查 epoch 0 输出