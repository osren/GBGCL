# GBGCL Phase 1 代码修改计划

> 作者：谭成  
> 日期：2026-04-15  
> 目标：补完 Physics / Computers 系统实验，找到 Photo / Computers 超越 SGRL 基线的配置，验证 CS 最优配置可复现性。

---

## 一、当前实验状态诊断

| 数据集 | CSV 行数 | 精训 Trials | 当前最优 ACC | SGRL 基线 | 差距 | 状态 |
|--------|---------|------------|------------|---------|------|------|
| CS | 充分 | 5 trial | **94.28%** | 93.95% | +0.33% | ✅ 正向，已完成 |
| Photo | 充分 | Stage-A 为主 | ~93.87% | 93.95% | -0.08% | ⚠️ 仍未超基线 |
| **Physics** | 仅 10+行 | 5 trial（2 组合） | **96.23%** | 96.23% | ~0% | ⚠️ 组合太少，未扫参 |
| **Computers** | 约 100 行 | Stage-A 1 trial | ~90.23% | 90.23% | ~0% | ❌ 未精训，未稳定 |

**核心问题**：  
- Physics / Computers 缺乏系统的超参扫描；只有默认配置的 5 trial，无法说明哪组参数最优。  
- Photo / Computers 的 Stage-A 粗筛分数与基线持平或略低，需要在更宽的搜索空间中寻找突破点。  
- `sweepX.py` 当前 `DATASETS = ["Physics"]`，且 `FILTERS` 的候选组合仍偏少，`ALPHAS` 只有 `[0.7, 0.3]`，漏掉了 `alpha=0.5` 这一重要中间值。

---

## 二、修改任务清单

### 任务 T1 — `sweepX.py`：扩充搜索空间 & 切换数据集

**文件**：`F:\GBGCL\tools\sweepX.py`

#### T1-1 修改 `DATASETS`（同时跑四个数据集）

```python
# 旧
DATASETS = ["Physics"]

# 新（Phase 1 需要补全这四个）
DATASETS = ["Physics", "Computers", "Photo", "CS"]
```

> 说明：CS 已有结果，断点续跑机制（`already_done`）会自动跳过已完成的组合，无需担心重复。

---

#### T1-2 在 `ALPHAS` 中补充 `0.5`

```python
# 旧
ALPHAS = [0.7, 0.3]

# 新（增加中间值）
ALPHAS = [0.7, 0.5, 0.3]
```

> 说明：`alpha` 控制节点原始表示与球扩散表示的融合比例。当前两端极值均未能让 Photo/Computers 超基线，`0.5` 是最自然的折中，理论上可以平衡局部 vs 全局信息。

---

#### T1-3 扩充 `FILTERS`（增加 Physics / Computers / Photo 的候选组合）

```python
FILTERS = {
    "Computers": [
        ("detach", "dot", 0.7),
        ("homo",   "dot", 0.7),
        ("detach", "dot", 0.5),   # ← 新增 alpha=0.5
        ("homo",   "cos", 0.5),   # ← 新增
        ("detach", "cos", 0.3),   # ← 新增：cos 相似度在 Computers 上未充分测试
    ],
    "Photo": [
        ("detach", "dot", 0.3),
        ("homo",   "cos", 0.3),
        ("detach", "dot", 0.5),   # ← 新增
        ("homo",   "dot", 0.5),   # ← 新增
        ("detach", "cos", 0.5),   # ← 新增
    ],
    "CS": [
        ("detach", "dot", 0.7),
        ("detach", "dot", 0.3),
    ],
    "Physics": [
        ("homo",   "cos", 0.3),
        ("detach", "dot", 0.3),
        ("homo",   "cos", 0.5),   # ← 新增
        ("detach", "dot", 0.5),   # ← 新增
        ("homo",   "dot", 0.3),   # ← 新增：dot 相似度在 Physics 上未充分测试
    ],
}
```

---

#### T1-4 扩充 `KS`（增加 K=3 的快速扩散选项）

```python
# 旧
KS = [5, 10, 20]

# 新（K=3 对大图更省内存，也可能避免过度平滑）
KS = [3, 5, 10, 20]
```

> 说明：Physics（34,493 节点）和 Computers（13,752 节点）是中大型图，K=3 的轻量扩散值得尝试。

---

#### T1-5 调整 Stage-B 精训参数（保证 5 trial 充分性）

```python
# 旧（Stage-B）
NUM_EPOCHS = 700
TRIALS = 5
GB_REBUILD_EVERY = 100

# 新（不变，但增加注释说明为何这样设置）
# Stage-B 700 epoch 已足够收敛；TRIALS=5 满足统计置信度要求
# 仅对需要精训的数据集启用 Stage-B（通过环境变量 SWEEP_STAGE=B 控制）
```

> 建议运行：`$env:SWEEP_STAGE="B"; python tools/sweepX.py`

---

### 任务 T2 — `sweepX.py`：Windows 兼容性修复

**背景**：当前 `build_cmd` 生成的命令使用 Linux 风格 `CUDA_VISIBLE_DEVICES=0 python ...`，在 Windows PowerShell 下会报错。

#### T2-1 修复 `build_cmd` 以兼容 Windows

```python
def build_cmd(dataset: str, p: dict, stage_tag: str) -> str:
    tag = "_".join([dataset, p["gb_quity"], p["gb_sim"], str(p["gb_alpha"]), stage_tag])
    log_path = os.path.join(LOG_DIR, f"{tag}.log")

    # 检测是否为 Windows
    import platform
    is_windows = platform.system() == "Windows"

    base_args = [
        "python", "src/train.py",       # ← 注意加上 src/ 前缀
        "--dataset_name", shlex.quote(dataset),
        "--log_dir",      shlex.quote(log_path),
        "--e1_lr",        str(E1_LR),
        "--e2_lr",        str(E2_LR),
        "--num_epochs",   str(NUM_EPOCHS),
        "--hidden_dim",   str(HIDDEN_DIM),
        "--num_hop",      str(NUM_HOP),
        "--num_layers",   str(NUM_LAYERS),
        "--momentum",     str(MOMENTUM),
        "--seed",         str(SEED),
        "--trials",       str(TRIALS),
        "--log_every",    str(LOG_EVERY),
        "--gb_rebuild_every", str(GB_REBUILD_EVERY),
        "--device",       DEVICE,
        "--use_gb",
        "--gb_quity",     p["gb_quity"],
        "--gb_sim",       p["gb_sim"],
        "--gb_alpha",     str(p["gb_alpha"]),
        "--gb_beta",      str(p["gb_beta"]),
        "--gb_K",         str(p["gb_K"]),
        "--gb_w_mode",    p["gb_w_mode"],
        "--gb_knn",       str(p["gb_knn"]),
        "--ball_loss_weight",   str(p["ball_loss_weight"]),
        "--ball_angle_thresh",  str(p["ball_angle_thresh"]),
        "--ball_uniform_tau",   str(p["ball_uniform_tau"]),
        "--ball_infonce_weight",str(p["ball_infonce_weight"]),
        "--ball_infonce_temp",  str(p["ball_infonce_temp"]),
    ]

    if is_windows:
        # Windows：通过 set 环境变量方式传 CUDA_VISIBLE_DEVICES
        cmd = f"set CUDA_VISIBLE_DEVICES=0 && {' '.join(base_args)}"
    else:
        cmd = f"CUDA_VISIBLE_DEVICES=0 {' '.join(base_args)}"

    return cmd
```

> ⚠️ **注意**：同时把 `python train.py` 改为 `python src/train.py`，因为 sweepX.py 位于 `tools/` 目录，train.py 在 `src/` 下。或在运行时 `cd F:\GBGCL\src` 后执行。

---

### 任务 T3 — `train.py`：修复结果目录不一致问题

**问题**：`train.py` 写结果到 `results/<DATASET>_summary.csv`，而 `sweepX.py` 读结果时用 `RESULTS_DIR = "results_CUDA"`，两者不一致，导致断点续跑失效。

#### T3-1 统一结果目录

**方案 A（推荐）**：在 `train.py` 中增加 `--results_dir` 参数，默认为 `results`：

```python
# 在 argparse 中新增
parser.add_argument('--results_dir', type=str, default='results')

# 在 run() 中将：
csv_path = f"results/{args.dataset_name}_summary.csv"
# 改为：
os.makedirs(args.results_dir, exist_ok=True)
csv_path = os.path.join(args.results_dir, f"{args.dataset_name}_summary.csv")
```

**在 `sweepX.py` 的 `build_cmd` 中同步传入参数**：

```python
"--results_dir", RESULTS_DIR,   # 与 RESULTS_DIR 保持一致
```

**在 `sweepX.py` 的 `result_csv_paths` 中对应修改**：

```python
def result_csv_paths(dataset_name: str) -> List[str]:
    names = ALIASES.get(dataset_name, [dataset_name])
    return [os.path.join(RESULTS_DIR, f"{n}_summary.csv") for n in names]
    # RESULTS_DIR 保持 "results_CUDA" 或统一为 "results"，两处必须一致
```

> 推荐将 `sweepX.py` 中的 `RESULTS_DIR = "results_CUDA"` 统一改为 `RESULTS_DIR = "results"` 以匹配现有 CSV 文件。

---

### 任务 T4 — `sweepX.py`：工作目录修复

**问题**：`sweepX.py` 调用 `subprocess.call(cmd, shell=True)` 时，CWD 默认是 `tools/`，但 `train.py` 在 `src/` 下，数据路径 `../../datasets` 也是相对于 `src/` 的。

#### T4-1 在 `run_one` 中显式指定 CWD

```python
def run_one(dataset: str, p: dict, stage_tag: str) -> int:
    print(f"[RUN] {dataset} | {p} | stage={stage_tag}")
    cmd = build_cmd(dataset, p, stage_tag)
    # 显式切换到 src 目录执行
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    return subprocess.call(cmd, shell=True, cwd=src_dir)
```

---

### 任务 T5 — `analyze_results.py`：对齐结果目录

**文件**：`F:\GBGCL\tools\analyze_results.py`

当前脚本中 `RESULTS_DIR = "results"` 与 `sweepX.py` 的 `"results_CUDA"` 不一致，建议统一：

```python
# 统一为
RESULTS_DIR = "results"   # 与 train.py 实际写入目录一致
```

---

## 三、修改后的 Phase 1 实验运行方式

### Step 1：Stage-A 粗筛（快速，约 2-4 小时）

```bash
# 在 F:\GBGCL 目录下执行
cd F:\GBGCL
$env:SWEEP_STAGE = "A"
$env:SWEEP_WORKERS = "2"
python tools/sweepX.py
```

粗筛完成后，运行分析脚本找出 Top-3 配置：

```bash
python tools/analyze_results.py
# 查看 analysis/overall_topk.csv
```

### Step 2：Stage-B 精训（基于 Stage-A Top 结果）

根据分析结果，手动更新 `sweepX.py` 中 `FILTERS` 的内容为粗筛最优组合，然后：

```bash
$env:SWEEP_STAGE = "B"
$env:SWEEP_WORKERS = "1"   # 精训建议单进程，避免 GPU 竞争
python tools/sweepX.py
```

### Step 3：验证 CS 最优配置（5 trial 精训）

CS 已有结果，确认最优组合是 `(detach, dot, 0.7)` 或 `(detach, dot, 0.3)`：

```bash
# 直接在 src/ 下单独运行，验证 CS 精训结果可复现
cd F:\GBGCL\src
python train.py \
  --dataset_name CS \
  --use_gb \
  --gb_quity detach \
  --gb_sim dot \
  --gb_alpha 0.7 \
  --num_epochs 700 \
  --trials 5
```

---

## 四、预期结果（Phase 1 完成后）

| 数据集 | 预期策略 | 目标 ACC |
|--------|---------|---------|
| CS | 复现已有最优 | ≥ 94.28%（稳定 5 trial） |
| Photo | 扩充 alpha + cos 搜索 | ≥ 93.95%（超基线） |
| Computers | 扩充候选，含 alpha=0.5 | ≥ 90.23%（超基线） |
| Physics | 5 trial + 新候选组合 | ≥ 96.23%（超或持平） |

---

## 五、修改文件总览

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `tools/sweepX.py` | DATASETS、ALPHAS、FILTERS 扩充；Windows 兼容；CWD 修复；RESULTS_DIR 统一 | 🔴 最高 |
| `src/train.py` | 增加 `--results_dir` 参数，统一结果写入路径 | 🟠 高 |
| `tools/analyze_results.py` | 统一 RESULTS_DIR | 🟡 中 |

---

## 六、风险提示

1. **GPU 显存**：Physics（34K 节点）精训时如出现 OOM，需将 `SWEEP_WORKERS` 降为 1，或减小 `HIDDEN_DIM` 至 512。
2. **断点续跑前提**：`already_done()` 依赖 CSV 文件已存在且列名一致，修改 `--results_dir` 后需确保新旧路径统一。
3. **Windows subprocess**：`shell=True` 在 Windows 下使用 cmd.exe，确保 `&&` 语法兼容（PowerShell 用 `;` 分隔）。如有问题，改为 `subprocess.run(base_args, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, cwd=src_dir)`。

---

*本计划由 AI 助手根据代码审查和实验数据分析自动生成，如有疑问请对照源代码进一步确认。*
