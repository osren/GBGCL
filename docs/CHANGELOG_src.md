# GBGCL 代码修改记录 (Changelog)

> 用于记录对 `src/` 核心代码的修改，以便追溯和回退。
> 每次完整修改后更新此文档。

---

## 2026-05-11 | option-a-iterative-propagation

### 修改概述
实现 Online 分支残差连接（Option A），将粒球扩散结果累积并加到下一轮训练，实现迭代式传播。

### 修改文件

#### 1. `src/train.py`

| 修改位置 | 修改内容 |
|---------|---------|
| 约74-93行 | `train_online()` 新增 `gb_accum` 参数和残差连接逻辑 |
| 约233行 | 在每个 trial 开始时初始化 `gb_accum = None` |
| 约277-291行 | 调用 `train_online()` 并更新累积变量 |
| 约176-182行 | 返回 `z_for_accum` |
| 约372-375行 | 新增 argparse `--gb_residual_online` 和 `--gb_residual_weight` 参数 |

```python
# 新增参数
parser.add_argument('--gb_residual_online', action='store_true',
                    help='Enable residual connection for Online branch (Option A)')
parser.add_argument('--gb_residual_weight', type=float, default=0.1)

# 训练循环中
gb_accum = None  # 初始化
online_loss, sim_mean, z_for_accum = train_online(..., gb_accum)
if z_for_accum is not None:
    gb_accum = 0.9 * gb_accum + 0.1 * z_for_accum  # EMA 累积
```

---

### 回退指南

如需回退到修改前状态，执行以下操作：

1. **train.py**: 移除 `gb_accum` 参数和残差逻辑，移除返回值中的 `z_for_accum`，删除新增的 argparse 参数

---

### 使用方法

```bash
cd src

# 测试 Option A: 残差连接
python train.py --dataset_name Photo --use_gb --gb_quity homo --gb_residual_online --gb_residual_weight 0.1 --num_epochs 50 --trials 1 --gb_rebuild_every 10 --device cuda
```

---

## 2026-05-11 | option-b-target-enhance

### 修改概述
实现 Target 分支粒球增强（Option B），对 Target Encoder 的输出也进行粒球扩散，使双分支都受益于粒球增强。

### 修改文件

#### 1. `src/train.py`

| 修改位置 | 修改内容 |
|---------|---------|
| 约121-134行 | 修改 InfoNCE 条件判断，增加 `gb_target_enhance` 选项 |
| 约134-142行 | 对 Target 输出也做粒球扩散增强 |
| 约333-334行 | 新增 argparse `--gb_target_enhance` 参数 |

```python
# 新增参数
parser.add_argument('--gb_target_enhance', action='store_true',
                    help='Enable granule diffusion on Target branch')

# 修改条件判断
if args.ball_infonce_weight > 0 or getattr(args, 'gb_target_enhance', False):
    # 对 Target 输出也做粒球扩散
    if getattr(args, 'gb_target_enhance', False):
        z_target, ... = granule_diffuse_and_write(h_target, ...)
        h_target = z_target
```

---

### 回退指南

如需回退到修改前状态，执行以下操作：

1. **train.py**: 恢复原来的条件判断，移除 Target 增强代码块，删除 `--gb_target_enhance` 参数

---

### 使用方法

```bash
cd src

# 测试 Target 增强
python train.py --dataset_name Photo --use_gb --gb_quity homo --gb_target_enhance --num_epochs 50 --trials 1 --device cuda
```

---

## 2026-05-07 | option-a2-ensemble-voting

### 修改概述
实现多 quity 并行构建 + 自适应权重投票机制（Option A2），解决静态决策无法根据训练效果动态调整的问题。

### 修改文件

#### 1. `src/gb_utils.py`

| 修改位置 | 修改内容 |
|---------|---------|
| 第228-285行 | 新增 `_evaluate_ball_quality()` - 基于球间分离度 + 球内紧凑度的质量评估 |
| 第288-337行 | 新增 `build_granules_ensemble()` - 多 quity 并行构建 + softmax 权重投票 |
| 第189-223行 | 修改 `granule_diffuse_and_write()` - 新增 `use_ensemble`、`ensemble_quities`、`ensemble_temp`、`select` 参数，返回 `selected_quality` |

```python
# 质量评估
def _evaluate_ball_quality(node_embed, edge_index, GB_node_list, quity):
    # 分离度：不同球心之间的相似度（越低越好）
    # 紧凑度：球内节点到球心的距离（越低越好）
    # score = -separation - compactness * 0.1

# 投票构建
def build_granules_ensemble(node_embed, edge_index, quities, sim, temp=1.0):
    # 并行构建多种 quity 的粒球
    # 基于质量得分计算 softmax 权重
    # 返回最佳 quity 和权重字典
```

#### 2. `src/train.py`

| 修改位置 | 修改内容 |
|---------|---------|
| argparse (约304-313行) | 新增 `--gb_ensemble`、`--gb_ensemble_quities`、`--gb_ensemble_temp`、`--gb_ensemble_select` 参数 |
| train_online() 调用 (约82-91行) | 修改 `granule_diffuse_and_write()` 调用，传入 ensemble 相关参数 |

```python
# 新增参数
parser.add_argument('--gb_ensemble', action='store_true')
parser.add_argument('--gb_ensemble_quities', type=str, default='homo,detach,edges')
parser.add_argument('--gb_ensemble_temp', type=float, default=1.0)
parser.add_argument('--gb_ensemble_select', type=str, default='hard', choices=['hard', 'soft'])

# 调用时传入
z_new, gb_sizes, H_ball, GB_node_list, selected_quality = granule_diffuse_and_write(
    ...,
    use_ensemble=args.gb_ensemble,
    ensemble_quities=args.gb_ensemble_quities.split(','),
    ensemble_temp=args.gb_ensemble_temp,
    select=args.gb_ensemble_select
)
```

---

### 回退指南

如需回退到修改前状态，执行以下操作：

1. **gb_utils.py**: 删除 `_evaluate_ball_quality()`、`build_granules_ensemble()`，恢复 `granule_diffuse_and_write()` 原签名（移除 ensemble 参数）
2. **train.py**: 删除新增的 argparse 参数，恢复 `train_online()` 中的 `granule_diffuse_and_write()` 调用

---

### 使用方法

```bash
cd src

# 测试模式（短训练）
python train.py --dataset_name Photo --use_gb --gb_ensemble --gb_ensemble_quities homo,detach,edges --num_epochs 50 --trials 1

# 对比基线
python train.py --dataset_name Photo --use_gb --gb_quity homo --num_epochs 50 --trials 1
python train.py --dataset_name Photo --use_gb --gb_quity detach --num_epochs 50 --trials 1
python train.py --dataset_name Photo --use_gb --gb_quity edges --num_epochs 50 --trials 1

# 完整训练
python train.py --dataset_name Photo --use_gb --gb_ensemble --num_epochs 700 --trials 5
```

预期输出：
```
[Ensemble] quity=homo, weight=0.2xxx
[Ensemble] quity=detach, weight=0.6xxx
[Ensemble] quity=edges, weight=0.2xxx
[Ensemble] Selected: detach
```

---

## 2026-04-30 | option-a-auto-quality

### 修改概述
实现自适应质量函数选择（Option A），根据图统计自动选择 quity（homo/detach/edges）。

### 修改文件

#### 1. `src/granular.py`

| 修改位置 | 修改内容 |
|---------|---------|
| 第20-88行 | 新增 `auto_quality(edge_index, labels)` 静态方法 |

```python
@staticmethod
def auto_quality(edge_index: torch.Tensor, labels: torch.Tensor = None) -> str:
    """根据图统计自动选择最优 quity"""
    # 有标签时：计算同质率 > 0.6 → homo
    # 无标签时：用度分布推断
    # - degree_cv > 0.8 → detach（异质图）
    # - low_degree_ratio > 0.3 → edges（噪声图）
    # - 否则 → detach
```

#### 2. `src/gb_utils.py`

| 修改位置 | 修改内容 |
|---------|---------|
| 第8-14行 | 新增 `get_auto_quality()` 包装函数 |
| build_granules() | 支持 `quity='auto'` 参数，传入 `labels` |

```python
# quity='auto' 时自动选择
if quity == "auto":
    quity = get_auto_quality(edge_index, labels)
```

#### 3. `src/train.py`

| 修改位置 | 修改内容 |
|---------|---------|
| argparse (约302行) | `--gb_quity` 添加 `'auto'` 选项 |
| build_granules 调用 (约119行) | 传入 `labels=data.y` |

```python
# 新增参数选项
parser.add_argument('--gb_quity', ..., choices=[..., 'auto'])

# 调用时传入 labels
GB2_node_list, _, _ = build_granules(
    h_target, data.edge_index,
    quity=args.gb_quity, sim=args.gb_sim,
    labels=data.y)
```

#### 4. `tools/sweepX.py`

| 修改位置 | 修改内容 |
|---------|---------|
| 第16行 | QUITY 添加 `'auto'` 选项便于批量测试 |

---

### 回退指南

如需回退到修改前状态，执行以下操作：

1. **granular.py**: 删除 `auto_quality()` 方法
2. **gb_utils.py**: 删除 `get_auto_quality()`，恢复 `build_granules()` 原签名
3. **train.py**: 移除 `'auto'` 选项，恢复 `build_granules()` 调用
4. **sweepX.py**: 从 QUITY 删除 `'auto'`

---

### 使用方法

```bash
cd src
python train.py --dataset_name Photo --use_gb --gb_quality auto --num_epochs 700 --trials 5
```

应输出：`[Auto] Selected quity: detach`（根据图统计自动选择）

---

## 2026-04-17 | phase1-sweep-updates

### 修改概述
扩展 sweepX.py 超参数搜索空间，修复 Windows 兼容性和结果目录不一致问题。

### 修改文件

#### 1. `src/train.py`

| 修改位置 | 修改内容 |
|---------|---------|
| argparse 参数区 (约286行) | 新增 `--results_dir` 参数，默认值 `'results'` |
| run() 函数 (约163行) | `os.makedirs('results', ...)` → `os.makedirs(args.results_dir, ...)` |
| run() 函数 (约174行) | `csv_path = f"results/..."` → `csv_path = os.path.join(args.results_dir, f"...")` |

```python
# 新增参数
parser.add_argument('--results_dir', type=str, default='results')

# 修改目录创建
os.makedirs(args.results_dir, exist_ok=True)

# 修改 CSV 路径
csv_path = os.path.join(args.results_dir, f"{args.dataset_name}_summary.csv")
```

---

### 回退指南

如需回退到修改前状态，执行以下操作：

1. **回退 train.py**:
   ```bash
   # 移除 --results_dir 参数
   # 恢复 os.makedirs('results', ...)
   # 恢复 csv_path = f"results/..."
   ```

---

### 相关文件同步修改

- `tools/sweepX.py`: RESULTS_DIR 改为 "results"，添加 --results_dir 参数传递
- `tools/analyze_results.py`: RESULTS_DIR 改为 "results"

---

*End of changelog*