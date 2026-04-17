# GBGCL 代码修改记录 (Changelog)

> 用于记录对 `src/` 核心代码的修改，以便追溯和回退。
> 每次完整修改后更新此文档。

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