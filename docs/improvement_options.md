# Option A: 迭代式粒球传播

## 核心问题

当前粒球只在 epoch 0 构建一次，后续不更新；粒球扩散结果不继承到下一个 epoch。

## 改进方案

将粒球扩散结果写入节点特征，形成累积效应，使下一轮训练能够继承扩散信息。

## 实现方式

### 方式 1: 特征拼接（推荐）
在 GCN 输入侧拼接原始特征和粒球增强特征：
```python
# models.py 修改 - 在 Conv 的 forward 中
h_enhanced = concat([original_x, gb_enhanced_x])
```

### 方式 2: 残差连接
在 GCN 输出后添加粒球增强的残差连接：
```python
h_final = h_gcn + gb_enhanced * weight
```

## 命令示例

```bash
# 需要修改 models.py 后再运行
python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda
```

---

# Option B: Target 分支增强

## 核心问题

Target Encoder 没有使用粒球增强，两个分支都没有粒球信息。

## 改进方案

对 Target Encoder 的输出也进行粒球扩散，使双分支都受益于粒球增强。

## 实现方式

```python
# train.py - train_online() 中修改
with torch.no_grad():
    h_target = target(x, edge_index)
    if args.use_gb:
        z_target, ... = granule_diffuse_and_write(h_target, ...)
        h_target = z_target
```

## 命令示例

```bash
# 等待代码修改后运行
```

---

# Option C: 增大球级损失权重

## 核心问题

球级损失权重太小（ball_loss_weight=0.05, ball_infonce_weight=0.02），BYOL 损失主导，粒球模块作用有限。

## 改进方案

直接将 ball_loss_weight 从 0.05 提升到 0.2-0.3，使粒球模块在训练中发挥更重要作用。

## 命令示例

```bash
cd /Users/didi/Desktop/GBGCL

# 测试 ball_loss_weight = 0.3
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --ball_loss_weight 0.3 --ball_infonce_weight 0.1 --device cuda > logs/option_c_03.log 2>&1 &

# 测试 ball_loss_weight = 0.5
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --ball_loss_weight 0.5 --ball_infonce_weight 0.2 --device cuda > logs/option_c_05.log 2>&1 &

# 测试 ball_loss_weight = 0.8 (激进)
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --ball_loss_weight 0.8 --ball_infonce_weight 0.3 --device cuda > logs/option_c_08.log 2>&1 &
```

## 对比测试（不变）

```bash
# 对比：原始参数
nohup python src/train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda > logs/option_c_original.log 2>&1 &

# 对比：无粒球 (baseline)
nohup python src/train.py --dataset_name Photo --num_epochs 50 --trials 1 --device cuda > logs/baseline.log 2>&1 &
```

---

# 推荐执行顺序

1. **先执行 Option C** - 只需改参数，风险最低，立刻可验证
2. **再看 Option B** - 需要改 train.py，低风险
3. **最后考虑 Option A** - 需要改 models.py，侵入性较大