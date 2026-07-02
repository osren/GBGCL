# 下一步实验手册 — 2026-07-02

> 衔接 [`RESULTS_2026-07-02.md`](RESULTS_2026-07-02.md) 与 [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md)。
> 当前进度：Phase 0–4 已完成，Phase 2 (BTCM/BRSM) 阻塞。本手册定义 **四个连续阶段**，在 1–2 周内把粒球从「并行可选模块」推进到「与 TCM 融合的核心组件」。

---

## 一、路线图总览

```
[已完] Phase 0 ──► Phase 3a ──► Phase 3b ──► Phase 4
                                                  │
                          ▼                        ▼
       ┌─────────── Stage W ──────────┐    （Top-K 已写入 overall_topk.csv）
       │ ball_loss_weight 敏感性      │
       │ 50ep × 3 trials, 4 个权重    │
       └────────────────┬────────────┘
                        ▼
       ┌─────────── Stage R ──────────┐
       │ Photo/Computers 复跑         │   并行
       │ Phase0 与 Stage-B Top 各 5 试 │
       └────────────────┬────────────┘
                        ▼
       ┌─────────── Stage BTCM ───────┐
       │ 实现 --gb_btcm               │   ←─ 架构级修复
       │ Photo 烟测 → Photo/CS 50ep×3 │
       └────────────────┬────────────┘
                        ▼
       ┌─────────── Stage 2' ─────────┐
       │ BRSM + Phase 2 重跑          │
       │ BTCM+BRSM 全量 700ep × 5     │
       └────────────────┬────────────┘
                        ▼
                  更新 BASELINES.md / EXPERIMENTS.md
```

**关键判定**：Stage W / R 跑完后再决定是否启动 BTCM（B 之前是诊断，A 才是修复）。

---

## 二、Stage W：ball_loss_weight 敏感性扫参

**目的**：验证 `RESULTS_2026-07-02.md` 的假设 — `ball_loss_weight=0.05` 是 Photo/Physics/Computers 收益受限的瓶颈。每个组合 50ep × 3 trials，约 4 小时（2 卡并行）。

### 2.1 设计

固定 `Stage-B Top-1` 的其他超参（见 `analysis/overall_topk.csv`），仅扫权重：

| 数据集 | quity | sim | alpha | beta | K | w_mode | ball_loss_weight |
|--------|-------|-----|-------|------|---|--------|------------------|
| Photo | detach | dot | 0.3 | 0.2 | 10 | topo+center | **{0.05, 0.1, 0.2, 0.5, 1.0}** |
| Computers | homo | dot | 0.7 | 0.2 | 10 | topo+center | **{0.05, 0.1, 0.2, 0.5, 1.0}** |

> Photo 的 0.05 已在 Stage-B 跑过，可直接复用 `Photo_summary.csv` 里 `ball_loss_weight=0.05` 的行做基线（节省 1 个组合）。

### 2.2 命令

```bash
cd /path/to/GBGCL
git pull origin main
conda activate gbgcl

# 隔离结果目录
export RESULTS_DIR="${PWD}/results/phaseW_ball_weight"

# Stage W（前台调试用 --foreground）
bash scripts/phases/run_phase_nohup.sh W
# 等阶段 W 全部完成后：
python tools/analyze_results.py    # 读 results/phaseW_ball_weight/
```

### 2.3 判读规则

- 若某 (dataset, weight) 的 `clf_mean` 比现有 Stage-B Top-1 高 ≥ 0.3%：写入 `analysis/ball_weight_winners.csv`，**立即**更新 `tools/sweepX.py` 的 `FILTERS` 重跑 Stage-B（见 §二.4）
- 若所有权重结果差异 ≤ 0.1%：确认权重不是瓶颈，按计划推进到 Stage BTCM
- 若出现 NaN/Inf（`ball_loss_weight ≥ 0.5` 概率）：记录发散阈值，仍推进 BTCM

### 2.4 重跑 Stage-B（条件触发）

若 §2.3 判出 winner，按 `docs/COMMANDS.md` 的 Stage-B 命令把最优权重注入 `FILTERS`：

```python
# tools/sweepX.py 局部补丁
FILTERS = {
    "Photo": [
        ("detach", "dot", 0.3, 0.2, 10.0, "topo+center", 10, 0.2, 0.5, 1.0, ...),
        # ↑ 把原 0.05 改为 winner
    ],
}
```

然后执行 `bash scripts/phases/run_phase_nohup.sh 3b` + `4`。

---

## 三、Stage R：Photo/Computers 复跑

**目的**：分离 `-0.20%` 是真退化还是 Stage-B 5-trial 噪声。每组 5 trials，**等于** Stage-B 统计强度。

### 3.1 设计

| 组 | 数据集 | 配置来源 | 命令 | trials |
|----|--------|----------|------|--------|
| R-A | Photo | Phase 0（无粒球） | `train.py` 默认 | 5 |
| R-B | Photo | Stage-B Top-1（overwall_topk.csv 里的 detach/dot/0.3/0.2/10） | 完整旗标 | 5 |
| R-C | Computers | Phase 0（无粒球） | `train.py` 默认 | 5 |
| R-D | Computers | Stage-B Top-1（homo/dot/0.7/0.2/10） | 完整旗标 | 5 |

每组 700 epoch × 5 trials，与 Stage-B 主力配置完全可比。预计 16–20 小时（2 卡并行 = 8–10 小时）。

### 3.2 命令

```bash
export RESULTS_DIR="${PWD}/results/phaseR_repeat"
bash scripts/phases/run_phase_nohup.sh R
```

### 3.3 判读

- 若 R-A 与 R-B 的 5-trial `clf_mean` 差的 |Δ| ≤ 0.3%：-0.20% 属噪声，Stage-B Top-1 可信
- 若 R-B 比 R-A 稳定高 ≥ 0.5%：Stage-B Top-1 提取 best-of-best 的偏置过强，需要考虑随机化重选
- 写出 `analysis/repeat_validation.csv`：`dataset, config_tag, trial_1..trial_5, mean, std`

### 3.4 不与 Stage W 互斥

Stage W 与 Stage R 可 **并行** 跑在不同 GPU 上（Stage W 用 `--device cuda:1`，Stage R 用 `:0`）。互不阻塞。

---

## 四、Stage BTCM：球扩散融合进 TCM（架构修复）

**目的**：解决 CLAUDE.md 指出的根因 — ball diffusion 与 SGRL TCM/RSM **并行**而非**融合**。当前 `train.py:187` 的 `h_pred = online.predictor(z_new)` 用的是 `z_new = h + α·ball_emb`（后融合），评测时 `gb_feature_concat` 默认 False 又退回到 `or` embedding。

BTCM（Ball-aware TCM）要把 ball 信息 **写进** `Conv.forward` 的消息传递函数本身，使 `z_new` 是 TCM 一步内的产物而非后置 `α·ball_emb`。

### 4.1 架构动机

```
[当前]  h → Conv(x, edge_index) → h
        然后：z_new = h + α·ball_emb        ← 后融合

[BTCM]  h, ball_emb → BallTCM(...) → z_new  ← 一步内融合
        其中每条消息  m_{ij} = σ(W·[h_i || h_j || ball(i) || ball(j)])
```

### 4.2 代码改动清单（最小集合）

> **不要**直接编辑 `models.py`/`train.py` 直到 §4.3 设计 review 完成。以下为占位 diff。

**A. `src/models.py` — 新增 `BallConv.forward`**

```python
class BallConv(torch.nn.Module):
    """GCN conv + ball-emb concatenation into message function."""
    def __init__(self, in_dim, out_dim, ball_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim * 2 + ball_dim * 2, out_dim)  # src+dst+ball_src+ball_dst
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.act = torch.nn.PReLU()
    def forward(self, x, edge_index, ball_feat):
        # ball_feat: [N, ball_dim]，节点→所属球的嵌入
        src, dst = edge_index[0], edge_index[1]
        m = torch.cat([x[src], x[dst], ball_feat[src], ball_feat[dst]], dim=-1)
        out = self.lin(m)
        # 按 dst 聚合
        out = scatter_mean(out, dst, dim=0, dim_size=x.size(0))
        return self.act(self.bn(out))
```

**B. `src/models.py` — `Online.forward` 增加 `gb_ball_feat` 通路**

```python
def forward(self, x, edge_index, gb_feature=None, gb_ball_feat=None):
    if gb_ball_feat is not None and getattr(self, 'btcm_enabled', False):
        h = self.conv1.btcm(x, edge_index, gb_ball_feat)   # 跳进 BallConv
        h = self.conv2.btcm(h, edge_index, gb_ball_feat)
    else:
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
    return h
```

**C. `src/train.py` — argparse 加旗标**

```python
parser.add_argument('--gb_btcm', action='store_true',
                    help='Fuse ball-emb into TCM message passing (Stage 2)')
parser.add_argument('--gb_ball_emb_dim', type=int, default=64)
```

并在 `train_online` 调用 `online(..., gb_ball_feat=ball_tensor)`。`ball_tensor` 由 `gb_utils.py` 的新函数 `build_ball_tensor(H_ball, GB_node_list)` 提供（每个节点取其所属球的 embedding，索引可广播）。

**D. `src/gb_utils.py` — 加 `build_ball_tensor`**

```python
def build_ball_tensor(H_ball, GB_node_list, device):
    """Return tensor [N, ball_dim]：每个节点查其所在球的 embedding。"""
    node2ball = torch.zeros(H_ball.size(0), dtype=torch.long, device=device)
    for bid, nlist in enumerate(GB_node_list):
        if len(nlist) > 0:
            node2ball[torch.as_tensor(nlist, device=device)] = bid
    return H_ball[node2ball]   # [N, D]
```

### 4.3 review 检查项（PR 前必看）

| 检查项 | 期望 |
|--------|------|
| BallConv 参数对齐 | `ball_dim = args.gb_ball_emb_dim` 与 H_ball.size(1) 一致 |
| 不修改 SGRL 骨干 | `--gb_btcm=False` 时输出与现状 byte-for-byte 相等（pytest 已有 baseline） |
| EMA 同步 | target 网络仍接收 `gb_ball_feat`，否则 BYOL 端不一致 |
| 数值稳定 | 与 SGRL 同 LR/Init；首次跑做 `--num_epochs 1` 烟测 |

### 4.4 阶段运行

```bash
# (1) 烟测
cd src && python train.py --dataset_name Photo --num_epochs 1 --trials 1 \
  --use_gb --gb_btcm --device cuda

# (2) Phase B nohup（前台调试 --foreground）
export RESULTS_DIR="${PWD}/results/phaseB_btcm"
bash scripts/phases/run_phase_nohup.sh B
```

`phaseB_btcm.sh` 已自带 SKIP 守卫：未检测到 `--gb_btcm` 时不执行，与 `phase2_gsgrl.sh` 一致。

### 4.5 判读

- `h_z_diff` 首次 50 epoch > 0.3（当前 0.05–0.1）：融合真的在写入
- Photo `clf_mean` 比 Phase 0 基线高 ≥ 0.5%：BTCM 生效
- 否则检查 §4.3 表格

---

## 五、Stage 2'：BRSM 与 Phase 2 重跑

**前置**：§四 BTCM 5-trial 至少一集稳定正收益。

### 5.1 BRSM（Ball Scattering Module）

把 `gb_utils.ball_scatter_loss`（train.py:231 当前已 + `ball_loss_weight` 系数）升级为分多尺度散射：

```python
def ball_scattering_loss(H_ball, scales=(1, 2, 4)):
    losses = []
    for s in scales:
        # ball graph 上下 s 阶邻接上传播
        H_s = H_ball @ (normalize_ball_adj(s) @ H_ball.T).T
        losses.append(- simclr(H_s, H_ball))
    return torch.stack(losses).mean()
```

预估实现 + 单 GPU 烟测 1–2 天。

### 5.2 Phase 2 重跑

```bash
export SWEEP_STAGE="2"
export RESULTS_DIR="${PWD}/results/phase2_gsgrl"
bash scripts/phases/run_phase_nohup.sh 2
```

`phase2_gsgrl.sh` 检测到 `--gb_btcm` 会自动启用 BTCM/BRSM 路径（见仓库内文件）。

### 5.3 收尾更新

跑完 Phase 4 后：

- `docs/EXPERIMENTS.md` 追加 2026-07-2X 行：BTCM vs GB-vs-Base diff
- `docs/BASELINES.md`「本项目结果」小节刷新四数据集数字
- `analysis/overall_topk.csv` 与 `results/<ds>_summary.csv` 同步

---

## 六、一键命令备忘

```bash
# ============= 服务器环境（每次会话前） =============
cd /path/to/GBGCL && git pull && conda activate gbgcl

# ============= A. 诊断（W + R，并行） =============
# 卡 0：
export RESULTS_DIR="${PWD}/results/phaseW_ball_weight"
CUDA_VISIBLE_DEVICES=0 bash scripts/phases/run_phase_nohup.sh W &
# 卡 1：
export RESULTS_DIR="${PWD}/results/phaseR_repeat"
CUDA_VISIBLE_DEVICES=1 bash scripts/phases/run_phase_nohup.sh R &

# W + R 都完后：
python tools/analyze_results.py

# ============= B. BTCM 修复（实现 PR merge 后） =============
export RESULTS_DIR="${PWD}/results/phaseB_btcm"
bash scripts/phases/run_phase_nohup.sh B      # 50ep×3 烟测
# 通过后调整 FILTERS，重跑 Phase 3b
bash scripts/phases/run_phase_nohup.sh 3b

# ============= C. Phase 2' =============
export RESULTS_DIR="${PWD}/results/phase2_gsgrl"
bash scripts/phases/run_phase_nohup.sh 2
bash scripts/phases/run_phase_nohup.sh 4

# ============= D. 拉回本地 =============
rsync -avz ${USER}@${HOST}:${REMOTE}/{results,analysis,logs}/ ./{results,analysis,logs}/
```

---

## 七、风险与回滚

| 风险 | 触发条件 | 回滚 / 处理 |
|------|----------|-------------|
| Stage W 高权重发散 | ball_loss_weight ≥ 0.5 出现 NaN | 记入 `logs/phases/phaseW/`，跳过该组合继续跑 |
| Stage R 揭示 -0.20% 真实 | R-A vs R-B 差稳定 ≥ 0.5% | 改用 R-A（Phase 0 参数）作为 SGRL 对标，重写 `docs/BASELINES.md` |
| BTCM 与 SGRL 不等价 | `--gb_btcm=False` 复现失败 | 走 `git bisect` 范围化回滚；保留 `ball_loss_weight=0` 路径 |
| BTCM 评测仍 ≤ 基线 | 5-trial 提升 < 0.1% | 改回后融合方案，转向更大 K（20）和 BRSM 同时开 |
| 总时长超 14 天 | | 砍 BRSM，仅交付 BTCM 单点修复 |

---

## 八、检查清单（PR review 模板）

每个新阶段 PR 提交前：

- [ ] 与 Stage-B Top-1 数字可比（同一 `tools/analyze_results.py` 路径）
- [ ] journal.csv 写入 `phaseW` / `phaseR` / `phaseB` 标签
- [ ] `docs/EXPERIMENTS.md` 当周表格追加一行
- [ ] 若改 `src/*`：加 pytest 或 1-epoch 烟测脚本 `tests/test_<name>.py`
- [ ] 阶段脚本能在 `bash scripts/phases/run_phase_nohup.sh X --foreground` 下前台跑通

---

## 九、关联文档

| 文档 | 用途 |
|------|------|
| [`RESULTS_2026-07-02.md`](RESULTS_2026-07-02.md) | 当前结果与问题（本文输入） |
| [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md) | 服务器推送 + 已有阶段 (0/1/3a/3b/4) |
| [`COMMANDS.md`](COMMANDS.md) | 单条训练/扫参命令速查 |
| [`ROADMAP.md`](ROADMAP.md) | BTCM/BRSM 在 G-SGRL 总路线中的位置 |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | 实验记录（手动追加表格） |
| [`BASELINES.md`](BASELINES.md) | SGRL 论文基线数字 |
