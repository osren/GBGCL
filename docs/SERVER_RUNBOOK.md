# 服务器执行手册（推送清单 + 分步操作）

> **最后更新**：2026-06-16  
> 在 Linux GPU 服务器上按阶段跑实验。本地改代码 → 推送到 Git → 服务器 `git pull` → `nohup` 执行。

---

## 一、需要同步到服务器的内容

### 方式 A：推荐 — Git 全量同步

服务器已有仓库时，**每次本地 push 后**在服务器执行：

```bash
cd /path/to/GBGCL
git pull origin main
```

会拉取以下**运行必需**目录（与当前 `main` 一致即可）：

| 类别 | 路径 | 是否必须 |
|------|------|----------|
| 训练代码 | `src/train.py`, `src/models.py`, `src/granular.py`, `src/gb_utils.py`, `src/data.py` | ✅ 必须 |
| 扫参工具 | `tools/sweepX.py`, `tools/analyze_results.py` | ✅ 阶段 3 必须 |
| **分阶段脚本** | `scripts/phases/` 整个目录 | ✅ 必须 |
| 状态脚本 | `scripts/experiments_status.py`, `scripts/experiments_status.sh` | 建议 |
| 环境定义 | `env.yaml` | 首次建环境必须 |
| 参考文档 | `docs/BASELINES.md`, `docs/ROADMAP.md`, `docs/COMMANDS.md` | 可选 |

**不要从本地上传/覆盖**（在服务器生成）：

- `results/` — 实验 CSV
- `logs/` — 训练与 nohup 日志
- `analysis/` — 聚合结果
- `datasets/` — 数据集（需在服务器单独准备，见下文）

### 方式 B：仅 rsync 增量（无 Git 时）

在**本地**项目根目录执行（改 `USER`、`HOST`、`REMOTE`）：

```bash
REMOTE=/path/to/GBGCL
HOST=your_server

rsync -avz --relative \
  src/train.py src/models.py src/granular.py src/gb_utils.py src/data.py \
  tools/sweepX.py tools/analyze_results.py \
  scripts/phases scripts/experiments_status.py scripts/experiments_status.sh \
  env.yaml docs/SERVER_RUNBOOK.md docs/BASELINES.md \
  ./ ${USER}@${HOST}:${REMOTE}/
```

### 方式 C：最小文件清单（手动 scp）

若只更新分阶段脚本：

```
GBGCL/scripts/phases/common.sh
GBGCL/scripts/phases/run_phase_nohup.sh
GBGCL/scripts/phases/phase0_sgrl_baseline.sh
GBGCL/scripts/phases/phase1_incremental_diffusion.sh
GBGCL/scripts/phases/phase2_gsgrl.sh
GBGCL/scripts/phases/phase3_sweep_stage_a.sh
GBGCL/scripts/phases/phase3_sweep_stage_b.sh
GBGCL/scripts/phases/phase4_analyze.sh
GBGCL/scripts/phases/README.md
```

---

## 二、服务器首次准备（只做一次）

### Step 1：克隆或确认目录

```bash
cd /path/to/your/work
git clone https://github.com/osren/GBGCL.git
cd GBGCL
```

### Step 2：Python 环境

```bash
conda env create -f env.yaml   # 已存在则跳过
conda activate gbgcl            # 环境名以 env.yaml 为准
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 3：数据集（必须，不在 Git 里）

```bash
ls datasets/
# 应能看到 CS/ Computers/ Photo/ Physics/ 等
```

若无数据：从本机 `scp -r datasets/ user@host:/path/to/GBGCL/datasets/` 或按 `src/data.py` 让 PyG 自动下载到 `datasets/`。

### Step 4：脚本可执行权限

```bash
chmod +x scripts/phases/*.sh
```

### Step 5：冒烟测试（约 1 分钟）

```bash
cd /path/to/GBGCL/src
python train.py --dataset_name Photo --num_epochs 1 --trials 1 --device cuda
```

无报错即可进入分阶段实验。

---

## 三、分阶段执行步骤（核心）

**所有阶段均在项目根目录 `GBGCL/` 下执行**，使用 `nohup` 后台跑。

### 总流程图

```
git pull
   ↓
阶段 0：SGRL 复现（无粒球）
   ↓
阶段 1：增量扩散验证
   ↓
阶段 3a：Stage-A 扫参
   ↓
阶段 4：分析 → 看 overall_topk.csv
   ↓
（按需改 tools/sweepX.py FILTERS）
   ↓
阶段 3b：Stage-B 精训
   ↓
阶段 4：再分析 → 更新 docs/EXPERIMENTS.md
```

阶段 2（BTCM/BRSM）在代码实现后再跑，当前会自动 SKIP。

---

### 阶段 0：SGRL 基线复现

**目的**：四数据集无粒球，对标 SGRL Table 1。

```bash
cd /path/to/GBGCL
git pull origin main
conda activate gbgcl

# 可选：结果单独目录，避免与旧实验混在一起
# export RESULTS_DIR=/path/to/GBGCL/results/phase0_sgrl

bash scripts/phases/run_phase_nohup.sh 0
```

| 项 | 说明 |
|----|------|
| 内容 | CS / Photo / Physics / Computers，700 epoch × 5 trials，无 `--use_gb` |
| 耗时 | 每数据集数小时，四集串行 |
| 监控 | `tail -f logs/nohup/phase0_*.out` |
| PID | `cat logs/nohup/phase0.pid` → `ps -p <pid>` |

**完成后检查**：

```bash
wc -l results/*_summary.csv
grep -l "use_gb" results/CS_summary.csv  # 阶段0 行应 use_gb=0 或无粒球配置
tail logs/phases/journal.csv
```

对照 `docs/BASELINES.md`：CS 94.15%、Photo 93.95%、Physics 96.23%、Computers 90.23%。

---

### 阶段 1：增量扩散（E7）

**目的**：验证 `--gb_incremental` 是否拉高 `h_z_diff`。

```bash
bash scripts/phases/run_phase_nohup.sh 1
```

| 项 | 说明 |
|----|------|
| 内容 | Photo + Computers；incremental vs 非 incremental，50ep×3 |
| 日志 | `logs/phases/phase1/Photo_incremental_50x3.log` 等 |
| 关键指标 | 搜 `[h_DEBUG]`、`[Incremental]`、`h_z_diff` |

```bash
grep -E 'h_DEBUG|Incremental|h_z_diff' logs/phases/phase1/*.log | tail -30
```

---

### 阶段 3a：超参 Stage-A 粗筛

**目的**：补全 Physics / Computers 扫参。

```bash
export SWEEP_WORKERS=2    # 按 GPU 数量调整，OOM 则改为 1
bash scripts/phases/run_phase_nohup.sh 3a
```

| 项 | 说明 |
|----|------|
| 内容 | `sweepX.py`，150 epoch × 1 trial |
| 日志 | `logs/phases/phase3a/sweep_stage_a.log` |
| 断点续跑 | 已写入 `results/*_summary.csv` 的组合会自动跳过 |

---

### 阶段 4：结果分析

```bash
bash scripts/phases/run_phase_nohup.sh 4
```

输出：

- `analysis/overall_topk.csv`
- `logs/phases/phase4/analyze.log`

```bash
head -15 analysis/overall_topk.csv
```

根据 Top-K **决定是否修改** `tools/sweepX.py` 里的 `FILTERS` 再跑 Stage-B。

---

### 阶段 3b：超参 Stage-B 精训

```bash
export SWEEP_WORKERS=1    # 精训建议单进程
bash scripts/phases/run_phase_nohup.sh 3b
```

| 项 | 说明 |
|----|------|
| 内容 | 700 epoch × 5 trials，仅 `FILTERS` 中配置 |
| 耗时 | 数天级别 |

结束后再次执行阶段 4。

---

### 阶段 2：G-SGRL（暂缓）

```bash
bash scripts/phases/run_phase_nohup.sh 2
```

当前 `--gb_btcm` 未实现时会 **SKIP** 并记入 journal。实现 BTCM/BRSM 后 `git pull` 再跑。

---

## 四、结果记录（写到哪里）

| 自动产出 | 路径 | 你要做的事 |
|----------|------|------------|
| 主结果 CSV | `results/<Dataset>_summary.csv` | 论文/对标用 `clf_mean` |
| 阶段训练 log | `logs/phases/phaseN/*.log` | 查 debug 指标 |
| nohup 总输出 | `logs/nohup/phaseN_*.out` | 排错 |
| 运行台账 | `logs/phases/journal.csv` | 看 OK/FAIL |
| Top-K 汇总 | `analysis/overall_topk.csv` | 阶段 4 后查看 |

**建议人工记录**（拉回本地后更新）：

1. 打开 `docs/EXPERIMENTS.md`，追加表格：阶段、日期、`RESULTS_DIR`、四集 ACC、与 SGRL 差值。
2. 重大结论同步 `docs/BASELINES.md` 的「本项目结果」小节。

**journal 快速查看**：

```bash
column -t -s, logs/phases/journal.csv | tail -20
```

---

## 五、实验结果拉回本地

在**本地**执行：

```bash
REMOTE=/path/to/GBGCL
HOST=your_server

rsync -avz ${USER}@${HOST}:${REMOTE}/results/ ./results/
rsync -avz ${USER}@${HOST}:${REMOTE}/analysis/ ./analysis/
rsync -avz ${USER}@${HOST}:${REMOTE}/logs/phases/ ./logs/phases/
rsync -avz ${USER}@${HOST}:${REMOTE}/logs/nohup/ ./logs/nohup/
```

然后在本地：

```bash
python tools/analyze_results.py
```

---

## 六、常用运维命令

```bash
# 查看后台是否在跑
ps -p $(cat logs/nohup/phase0.pid)

# 停止某一阶段
kill $(cat logs/nohup/phase0.pid)

# 前台调试（不 nohup，适合冒烟）
bash scripts/phases/run_phase_nohup.sh 0 --foreground

# GPU 占用
nvidia-smi

# 实验状态汇总
bash scripts/experiments_status.sh
```

---

## 七、故障排查

| 现象 | 处理 |
|------|------|
| `python: command not found` | `conda activate gbgcl` |
| 找不到 datasets | 检查 `datasets/` 或 `src/data.py` 的 `data_dir` |
| CUDA OOM | `export SWEEP_WORKERS=1`；Physics 可考虑减小 `hidden_dim` |
| 脚本 Permission denied | `chmod +x scripts/phases/*.sh` |
| sweep 无新任务 | 组合已在 CSV 中 → 正常跳过；或检查 `FILTERS` |
| phase0 ACC 与论文差很多 | 先确认无 `--use_gb`、700 epoch、同评估协议 |

---

## 八、与本仓库文档的关系

| 文档 | 用途 |
|------|------|
| 本文 | 服务器推送 + 逐步执行 |
| [COMMANDS.md](COMMANDS.md) | 单条命令速查 |
| [BASELINES.md](BASELINES.md) | SGRL 对标数字 |
| [ROADMAP.md](ROADMAP.md) | 研究路线与里程碑 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 实验记录（手动更新） |
| `scripts/phases/README.md` | 脚本参数说明 |

---

## 九、一键命令备忘（复制用）

```bash
# === 每次实验前 ===
cd /path/to/GBGCL && git pull && conda activate gbgcl

# === 按顺序 nohup ===
bash scripts/phases/run_phase_nohup.sh 0
# 等阶段0结束后再：
bash scripts/phases/run_phase_nohup.sh 1
bash scripts/phases/run_phase_nohup.sh 3a
bash scripts/phases/run_phase_nohup.sh 4
# 改 FILTERS 后：
bash scripts/phases/run_phase_nohup.sh 3b
bash scripts/phases/run_phase_nohup.sh 4
```
