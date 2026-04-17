# sweep.py
# 自动并行扫参：Stage-A(粗筛)/Stage-B(精训) + 断点续跑 + 数据集别名兼容
import os, sys, csv, shlex, itertools, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# =========================
# 配置区（按需修改）
# =========================

# 数据集（与你 data.py / train_V5.py 的 --dataset_name 对齐）
DATASETS = ["Computers", "Photo", "CS", "Physics"]  # 如需用 Co.CS/Co.Physics 也可改

# 搜索空间（A阶段跑全量；B阶段通常精简为空间或用 FILTERS）
QUITY  = ["homo", "detach"]
SIMS   = ["dot", "cos"]
ALPHAS = [0.3, 0.7]

# （可选）按数据集定制Top-K候选（设置后只跑这里列出的组合）
# 例如：A阶段跑全量；A结束→运行 analyze_results.py 选出Top-K填到此处→B阶段仅跑这些
FILTERS = {
    "Computers": [("detach","dot",0.7), ("homo","dot",0.7)],
    "Photo":     [("detach","dot",0.3), ("homo","cos",0.3)],
    "CS":        [("detach","dot",0.7), ("detach","dot",0.3)],
    "Physics":   [("homo","cos",0.3),   ("detach","dot",0.3)],
}

# 阶段设置：环境变量优先（默认A）
STAGE = os.environ.get("SWEEP_STAGE", "A").upper()
if STAGE == "A":         # 粗筛：快
    NUM_EPOCHS = 150
    TRIALS = 1
    GB_REBUILD_EVERY = 50
else:                    # 精训：稳
    NUM_EPOCHS = 700
    TRIALS = 5
    GB_REBUILD_EVERY = 100

# 训练公共超参（与你的 train_V5.py 保持一致）
E1_LR = 1e-4
E2_LR = 1e-4
HIDDEN_DIM = 1024
NUM_HOP = 1
NUM_LAYERS = 1
MOMENTUM = 0.99
SEED = 66666
LOG_EVERY = 50
IMP_THRESH = 1.0

# 并发度（环境变量优先）
MAX_WORKERS = int(os.environ.get("SWEEP_WORKERS", "2"))

# sklearn / joblib 的临时目录（避免 /tmp 爆盘）
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/dev/shm")  # Windows 可改到大盘路径

RESULTS_DIR = "results"
LOG_DIR = "log_multi"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# 工具函数
# =========================

ALIASES = {
    # 结果CSV可能出现的文件前缀别名；already_done() 会同时检查
    "Computers": ["Computers"],
    "Photo":     ["Photo"],
    "CS":        ["CS", "Co.CS", "CoCS"],
    "Physics":   ["Physics", "Co.Physics", "CoPhysics"],
}

def result_csv_paths(dataset_name: str) -> List[str]:
    """给定规范数据集名，返回可能的结果CSV路径列表"""
    names = ALIASES.get(dataset_name, [dataset_name])
    return [os.path.join(RESULTS_DIR, f"{n}_summary.csv") for n in names]

def already_done(dataset: str, quity: str, sim: str, alpha: float) -> bool:
    """断点续跑：检查 results/*_summary.csv 是否已有该组合的记录"""
    for path in result_csv_paths(dataset):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("use_gb") != "1":
                        continue
                    if r.get("gb_quity") != quity:   # 字段名与 train_V5.py 对齐
                        continue
                    if r.get("gb_sim") != sim:
                        continue
                    try:
                        if float(r.get("gb_alpha", "nan")) == float(alpha):
                            return True
                    except Exception:
                        continue
        except Exception:
            continue
    return False

def build_cmd(dataset: str, quity: str, sim: str, alpha: float, stage_tag: str) -> str:
    log_path = os.path.join(LOG_DIR, f"{dataset}_{quity}_{sim}_{alpha}_{stage_tag}.log")
    cmd = f"""
    python train_V5.py \
      --dataset_name {shlex.quote(dataset)} \
      --log_dir {shlex.quote(log_path)} \
      --e1_lr {E1_LR} \
      --e2_lr {E2_LR} \
      --num_epochs {NUM_EPOCHS} \
      --hidden_dim {HIDDEN_DIM} \
      --num_hop {NUM_HOP} \
      --num_layers {NUM_LAYERS} \
      --momentum {MOMENTUM} \
      --seed {SEED} \
      --trials {TRIALS} \
      --log_every {LOG_EVERY} \
      --imp_thresh {IMP_THRESH} \
      --gb_rebuild_every {GB_REBUILD_EVERY} \
      --use_gb \
      --gb_quity {quity} \
      --gb_sim {sim} \
      --gb_alpha {alpha}
    """
    return cmd

def run_one(dataset: str, quity: str, sim: str, alpha: float, stage_tag: str) -> int:
    print(f"[RUN] {dataset} | {quity}-{sim}-{alpha} | stage={stage_tag}")
    cmd = build_cmd(dataset, quity, sim, alpha, stage_tag)
    return subprocess.call(cmd, shell=True)

# =========================
# 主流程
# =========================

def main():
    stage_tag = STAGE
    jobs: List[Tuple[str,str,str,float,str]] = []

    # 生成任务
    for ds in DATASETS:
        # 若该数据集在 FILTERS 中定义了候选，则只跑候选；否则跑全量空间
        combos = FILTERS.get(ds, list(itertools.product(QUITY, SIMS, ALPHAS)))
        for (q, s, a) in combos:
            if already_done(ds, q, s, a):
                print(f"[SKIP] {ds} {q}-{s}-{a} 已存在结果，跳过。")
                continue
            jobs.append((ds, q, s, float(a), stage_tag))

    if not jobs:
        print("[INFO] 没有新的组合需要运行。Done.")
        return

    print(f"[INFO] 总任务数: {len(jobs)} | 并发度: {MAX_WORKERS} | stage={stage_tag}")
    print(f"[INFO] JOBLIB_TEMP_FOLDER={os.environ.get('JOBLIB_TEMP_FOLDER')}")

    # 并行执行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, *job): job for job in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                code = fut.result()
                print(f"[DONE] {job} → return {code}")
            except Exception as e:
                print(f"[ERR ] {job} → {e}")

if __name__ == "__main__":
    main()
