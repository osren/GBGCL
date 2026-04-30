# GBGCL 代码修改记录 (Changelog)

> 用于记录对 `src/` 核心代码的修改，以便追溯和回退。
> 每次完整修改后更新此文档。

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