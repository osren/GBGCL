# 分阶段实验脚本

在 **Linux 服务器**上用 `nohup` 按 ROADMAP 阶段跑实验。Windows 本地请用 `docs/COMMANDS.md` 中的单条命令或 WSL。

## 快速开始

```bash
cd /path/to/GBGCL
conda activate gbgcl   # 或你的环境名

# 后台启动阶段 0（SGRL 四数据集复现）
bash scripts/phases/run_phase_nohup.sh 0

# 查看输出
tail -f logs/nohup/phase0_*.out

# 查看是否在跑
cat logs/nohup/phase0.pid
ps -p $(cat logs/nohup/phase0.pid)
```

## 阶段对照

| 阶段 | 脚本 | 内容 | 典型耗时 |
|------|------|------|----------|
| **0** | `phase0_sgrl_baseline.sh` | 无 `--use_gb`，四集 700ep×5 | 数小时×4 |
| **1** | `phase1_incremental_diffusion.sh` | E7 增量 vs 对照，Photo+Computers 50ep×3 | ~1–2h |
| **2** | `phase2_gsgrl.sh` | BTCM/BRSM（代码未实现时自动 SKIP） | — |
| **3a** | `phase3_sweep_stage_a.sh` | `sweepX` Stage-A | 数小时 |
| **3b** | `phase3_sweep_stage_b.sh` | `sweepX` Stage-B | 数天 |
| **4** | `phase4_analyze.sh` | `analyze_results` + 状态汇总 | 分钟级 |

推荐顺序：**0 → 1 → 3a → 4 →（改 FILTERS）→ 3b → 4**

## nohup 启动器

```bash
bash scripts/phases/run_phase_nohup.sh <0|1|2|3a|3b|4>           # 后台
bash scripts/phases/run_phase_nohup.sh 0 --foreground             # 前台（调试）
```

可选环境变量：

```bash
export RESULTS_DIR=/path/to/GBGCL/results/phase0_sgrl   # 隔离某阶段 CSV
export DEVICE=cuda
export SWEEP_WORKERS=2
bash scripts/phases/run_phase_nohup.sh 3a
```

## 结果记录（三层）

### 1. 自动：训练 CSV（主结果）

`train.py` 每次 trial 追加一行到：

```
results/<Dataset>_summary.csv          # 默认
# 或 RESULTS_DIR 指定目录
```

字段含 `clf_mean`、`gb_quity`、`use_gb` 等，**论文/对照用此文件**。

### 2. 自动：阶段日志

```
logs/phases/phase0/CS_sgrl_700x5.log     # 每数据集训练 log（含 debug 指标）
logs/nohup/phase0_20260616_120000.out    # nohup 整段 stdout
```

增量扩散请在该 log 中搜：`[h_DEBUG]`、`[Incremental]`、`h_z_diff`。

### 3. 自动：运行台账 journal

```
logs/phases/journal.csv
```

列：`timestamp, phase, dataset, tag, results_dir, log_file, status, note`

每次 `run_train` 结束写入 OK/FAIL。

### 4. 手动：实验笔记（建议）

跑完一阶段后，把关键数字抄进 `docs/EXPERIMENTS.md`：

- 阶段编号、日期、`RESULTS_DIR`
- 各数据集 ACC（均值/最高）
- 与 `docs/BASELINES.md` 中 SGRL 的差值
- 失败或 SKIP 原因

阶段 4 会生成 `analysis/overall_topk.csv`，可一并引用。

## 常用检查命令

```bash
# 台账
column -t -s, logs/phases/journal.csv | tail -20

# 某阶段日志
ls logs/phases/phase1/

# 结果行数
wc -l results/*_summary.csv

# 汇总（阶段 4 或手动）
python tools/analyze_results.py
bash scripts/experiments_status.sh
```

## 停止任务

```bash
kill $(cat logs/nohup/phase0.pid)
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `common.sh` | 路径、`run_train`、journal |
| `run_phase_nohup.sh` | nohup 入口 |
| `phase*.sh` | 各阶段具体命令 |
