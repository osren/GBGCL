# GBGCL 超参数搜索实验指南

> 本文档说明如何使用 sweepX.py 进行超参数搜索，包括运行方式、结果分析和后续步骤。

---

## 一、实验流程概览

```
Stage-A (粗筛) → 分析结果 → Stage-B (精筛) → 最终结果
```

| 阶段 | 用途 | Epochs | Trials | 时长 |
|------|------|--------|--------|------|
| Stage-A | 快速探索最优超参组合 | 150 | 1 | ~1-2 天 |
| Stage-B | 精训 Top 配置，稳定结果 | 700 | 5 | ~1-2 天 |

---

## 二、运行实验

### 2.1 准备工作

```bash
# 登陆 
{
    username: ubuntu;
    password: 123123;
}

# 激活环境
conda activate CoBF

# 确认数据集存在
ls datasets/
# 应看到: CS/  Computers/  Photo/  Physics/
```

### 2.2 运行 Stage-A 粗筛

**方式一：使用脚本（推荐）**
```bash
# 在项目根目录运行
bash run_sweep.sh
```

**方式二：直接运行**
```bash
# 可选：指定并行数
export SWEEP_STAGE=A
export SWEEP_WORKERS=2   # 默认 2

# 运行
python tools/sweepX.py
```

**方式三：后台运行（服务器）**
```bash
nohup bash run_sweep.sh > sweep_output.log 2>&1 &
```

### 2.3 运行 Stage-B 精筛

当 Stage-A 完成后，分析结果并更新 FILTERS，然后运行：

```bash
export SWEEP_STAGE=B
export SWEEP_WORKERS=1
python tools/sweepX.py
```

---

## 三、日志与结果位置

### 3.1 日志文件

| 类型 | 位置 | 说明 |
|------|------|------|
| sweepX 运行日志 | `./sweepX_YYYYMMDD_HHMMSS.log` | 任务提交记录 |
| 错误日志 | `./sweepX_errors_YYYYMMDD_HHMMSS.log` | 仅 `[ERR` 行 |
| 子任务输出 | `log_CUDA/<dataset>_<quity>_<sim>_<alpha>_<stage>.out` | 每个组合的输出 |

### 3.2 结果文件

| 类型 | 位置 | 说明 |
|------|------|------|
| 原始结果 | `results/<dataset>_summary.csv` | 所有 trial 的原始数据 |
| Top 配置 | `analysis/overall_topk.csv` | 各数据集最优配置汇总 |

---

## 四、分析结果

### 4.1 实时查看进度

```bash
# 统计已完成多少行
wc -l results/*_summary.csv

# 查看分析结果
python tools/analyze_results.py
cat analysis/overall_topk.csv
```

### 4.2 analyze_results.py 输出格式

```
dataset,gb_quity,gb_sim,gb_alpha,gb_K,mean_acc,std_acc,trials
Physics,homo,cos,0.3,10,96.23,0.12,5
Photo,detach,dot,0.5,5,94.01,0.15,5
Computers,detach,dot,0.7,5,90.45,0.18,5
CS,detach,dot,0.7,5,94.28,0.10,5
```

### 4.3 手动查看单个数据集

```bash
# 查看 CS 结果
cat results/CS_summary.csv | head -20

# 按准确率排序
cat results/CS_summary.csv | sort -t',' -k6 -rn | head -10
```

---

## 五、预期结果

### 5.1 成功标准

| 数据集 | 基线 ACC | 目标 ACC | 说明 |
|--------|---------|---------|------|
| CS | 93.95% | ≥ 94.0% | 需稳定复现 |
| Photo | 93.95% | ≥ 94.0% | 寻找突破配置 |
| Computers | 90.23% | ≥ 90.5% | 需精训验证 |
| Physics | 96.23% | ≥ 96.2% | 需更多组合测试 |

### 5.2 最终交付物

1. **`results/<dataset>_summary.csv`** - 每个数据集的完整实验记录
2. **`analysis/overall_topk.csv`** - 各数据集最优配置及准确率
3. **`log_CUDA/`** - 所有实验的详细日志

### 5.3 如何判断完成

- 出现 `[INFO] No More new to Run.Done.`
- `results/` 下各数据集 CSV 行数稳定
- `analysis/overall_topk.csv` 有数据

---

## 六、后续方向

### 6.1 短期优化

| 方向 | 说明 |
|------|------|
| **扩大搜索空间** | 增加 alpha=0.5, K=3, 更多 quity/sim 组合 |
| **更多 trials** | 将 Stage-B 的 trials 从 5 增至 10 |
| **学习率调优** | 当前固定 1e-5，可尝试不同 lr 组合 |

### 6.2 中期探索

| 方向 | 说明 |
|------|------|
| **Granule 重建频率** | 当前 50/100，探索不同重建间隔 |
| **损失函数权重** | 调整 ball_loss_weight, ball_infonce_weight |
| **图神经网络层数** | 当前 1 层，尝试 2-3 层 |

### 6.3 长期研究

| 方向 | 说明 |
|------|------|
| **新数据集** | 扩展到 Cora, CiteSeer, PubMed 等 |
| **消融实验** | 验证各模块（scatter loss, ball InfoNCE）的贡献 |
| **可视化** | 绘制 t-SNE/UMAP 可视化，分析 granule 分布 |

---

## 七、常见问题

### Q1: 如何断点续跑？
直接重新运行 `python tools/sweepX.py`，已完成的任务会自动跳过。

### Q2: 如何删除某个数据集的结果重新开始？
```bash
rm results/CS_summary.csv
python tools/sweepX.py
```

### Q3: OOM 内存不足怎么办？
```bash
# 方式一：减少并行数
export SWEEP_WORKERS=1

# 方式二：减小隐藏层维度（在 sweepX.py 中）
HIDDEN_DIM = 512  # 从 1024 改为 512
```

### Q4: 如何查看某个具体配置的日志？
```bash
# 根据参数组合找到对应的 out 文件
ls log_CUDA/ | grep CS_detach_dot_0.7

# 查看日志
cat log_CUDA/CS_detach_dot_0.7_A.out | tail -50
```

---

*文档更新时间: 2026-04-20*