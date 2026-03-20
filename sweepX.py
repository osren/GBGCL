# -*- coding: utf-8 -*-
# sweepX.py
# 自动并行扫参：Stage-A(粗筛)/Stage-B(精训) + 断点续跑 + 数据集别名兼容（适配 train.py CUDA+CSV 版）
import os, csv, shlex, itertools, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# =========================
# 配置区（按需修改）
# =========================

# 数据集（与你 data.py / train.py 的 --dataset_name 对齐）
#DATASETS = ["Computers", "Photo", "CS", "Physics"]
DATASETS = ["Physics"]
# ——基础粒球搜索空间（A 阶段可全量；B 阶段一般用 FILTERS）——
QUITY  = ["homo", "detach"]
SIMS   = ["dot", "cos"]
ALPHAS = [0.7,0.3]

# ——粒球扩散参数搜索空间——
BETAS   = [0.1, 0.2, 0.3]          # 可改 [0.1, 0.2, 0.3]
KS      = [5, 10, 20]            # 可改 [5, 10, 20]
W_MODES = ["topo", "center", "topo+center"] # 可改 ["topo", "center", "topo+center"]
KNNs    = [5, 10, 20]            # 可改 [5, 10, 20]

# ——球级 Loss 搜索空间——
BALL_LOSS_W      = [0.05]     # 可改 [0.0, 0.02, 0.05]
BALL_ANGLE_THR   = [15.0, 25.0]     # 可改 [10.0, 15.0, 25.0]
BALL_UNIFORM_TAU = [0.1]      # 可改 [0.05, 0.1]
BALL_INFO_W      = [0.02]     # 可改 [0.0, 0.01, 0.02]
BALL_INFO_TEMP   = [0.2]      # 可改 [0.1, 0.2, 0.3]

# （可选）按数据集定制 Top-K 候选
# 支持两种写法：
# 1) 只给 (quity, sim, alpha) 三元组（其余参数走上面的默认值）
# 2) 给完整版 12 元组（会完全覆盖上面的默认值）
FILTERS = {
    "Computers": [
        ("detach","dot",0.7),
        ("homo","dot",0.7),
        # 完整元组示例（优先级更高）：
        # ("detach","dot",0.7, 0.2,10,"topo+center",10, 0.05,15.0,0.1, 0.02,0.2),
    ],
    "Photo": [
        ("detach","dot",0.3),
        ("homo","cos",0.3),
    ],
    "CS": [
        ("detach","dot",0.7),
        ("detach","dot",0.3),
    ],
    "Physics": [
        ("homo","cos",0.3),
        ("detach","dot",0.3),
    ],
}

# 阶段设置：环境变量优先（默认 A）
STAGE = os.environ.get("SWEEP_STAGE", "A").upper()
if STAGE == "A":         # 粗筛：快
    NUM_EPOCHS = 150
    TRIALS = 1
    GB_REBUILD_EVERY = 50
else:                    # 精训：稳
    NUM_EPOCHS = 700
    TRIALS = 5
    GB_REBUILD_EVERY = 100

# 训练公共超参（与你的 train.py 保持一致）
E1_LR = 1e-5
E2_LR = 1e-5
HIDDEN_DIM = 1024
NUM_HOP = 1
NUM_LAYERS = 1
MOMENTUM = 0.99
SEED = 66666
LOG_EVERY = 50

# 设备（可用环境变量覆盖）
DEVICE = os.environ.get("SWEEP_DEVICE", "cuda")

# 并发度（环境变量优先）
MAX_WORKERS = int(os.environ.get("SWEEP_WORKERS", "2"))

# sklearn / joblib 的临时目录（避免 /tmp 爆盘）
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/dev/shm")

# 与 train.py 对齐：结果写在 results/
RESULTS_DIR = "results_CUDA"
LOG_DIR = "log_CUDA"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# 工具函数
# =========================

ALIASES = {
    "Computers": ["Computers"],
    "Photo":     ["Photo"],
    "CS":        ["CS", "Co.CS", "CoCS"],
    "Physics":   ["Physics", "Co.Physics", "CoPhysics"],
}

def result_csv_paths(dataset_name: str) -> List[str]:
    names = ALIASES.get(dataset_name, [dataset_name])
    return [os.path.join(RESULTS_DIR, f"{n}_summary.csv") for n in names]

def _row_match(r: dict, kv: dict) -> bool:
    
    for k, v in kv.items():
        if k not in r: 
            return False
        rv = r[k]
        try:
            if isinstance(v, float):
                if float(rv) != float(v): return False
            else:
                if str(rv) != str(v): return False
        except Exception:
            if str(rv) != str(v): return False
    return True

def already_done(dataset: str, params: dict) -> bool:
    
    for path in result_csv_paths(dataset):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("use_gb") != "1":
                        continue
                    if _row_match(r, params):
                        return True
        except Exception:
            continue
    return False

def build_cmd(dataset: str, p: dict, stage_tag: str) -> str:
    
    tag = "_".join([dataset, p["gb_quity"], p["gb_sim"], str(p["gb_alpha"]), stage_tag])
    log_path = os.path.join(LOG_DIR, f"{tag}.log")

    cmd = f"""
    CUDA_VISIBLE_DEVICES=0 python train.py \
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
      --gb_rebuild_every {GB_REBUILD_EVERY} \
      --device {DEVICE} \
      --use_gb \
      --gb_quity {p['gb_quity']} \
      --gb_sim {p['gb_sim']} \
      --gb_alpha {p['gb_alpha']} \
      --gb_beta {p['gb_beta']} \
      --gb_K {p['gb_K']} \
      --gb_w_mode {p['gb_w_mode']} \
      --gb_knn {p['gb_knn']} \
      --ball_loss_weight {p['ball_loss_weight']} \
      --ball_angle_thresh {p['ball_angle_thresh']} \
      --ball_uniform_tau {p['ball_uniform_tau']} \
      --ball_infonce_weight {p['ball_infonce_weight']} \
      --ball_infonce_temp {p['ball_infonce_temp']}
    """
    return " ".join(line.strip() for line in cmd.splitlines() if line.strip())

def run_one(dataset: str, p: dict, stage_tag: str) -> int:
    print(f"[RUN] {dataset} | {p} | stage={stage_tag}")
    cmd = build_cmd(dataset, p, stage_tag)
    return subprocess.call(cmd, shell=True)

# =========================
# 主流程
# =========================

def main():
    stage_tag = STAGE
    jobs: List[Tuple[str, dict, str]] = []

    for ds in DATASETS:
        # 若 FILTERS 给了候选：既支持三元组也支持完整 12 元组
        if ds in FILTERS and len(FILTERS[ds]) > 0:
            for tpl in FILTERS[ds]:
                if len(tpl) == 3:
                    q, s, a = tpl
                    # 其余参数走默认空间（单值的话就一个组合）
                    for (beta, K, wm, knn, blw, ang, tau, biw, bit) in itertools.product(
                        BETAS, KS, W_MODES, KNNs, BALL_LOSS_W, BALL_ANGLE_THR, BALL_UNIFORM_TAU, BALL_INFO_W, BALL_INFO_TEMP
                    ):
                        p = dict(gb_quity=q, gb_sim=s, gb_alpha=float(a),
                                 gb_beta=float(beta), gb_K=int(K), gb_w_mode=wm, gb_knn=int(knn),
                                 ball_loss_weight=float(blw), ball_angle_thresh=float(ang), ball_uniform_tau=float(tau),
                                 ball_infonce_weight=float(biw), ball_infonce_temp=float(bit))
                        if already_done(ds, p):
                            print(f"[SKIP] {ds} {p} Already have,Pass")
                            continue
                        jobs.append((ds, p, stage_tag))
                else:
                    # 完整 12 元组
                    (q, s, a, beta, K, wm, knn, blw, ang, tau, biw, bit) = tpl
                    p = dict(gb_quity=q, gb_sim=s, gb_alpha=float(a),
                             gb_beta=float(beta), gb_K=int(K), gb_w_mode=wm, gb_knn=int(knn),
                             ball_loss_weight=float(blw), ball_angle_thresh=float(ang), ball_uniform_tau=float(tau),
                             ball_infonce_weight=float(biw), ball_infonce_temp=float(bit))
                    if already_done(ds, p):
                        print(f"[SKIP] {ds} {p} Already have,Pass")
                        continue
                    jobs.append((ds, p, stage_tag))
        else:
            # 否则跑全量笛卡尔积
            for (q, s, a, beta, K, wm, knn, blw, ang, tau, biw, bit) in itertools.product(
                QUITY, SIMS, ALPHAS, BETAS, KS, W_MODES, KNNs, BALL_LOSS_W, BALL_ANGLE_THR, BALL_UNIFORM_TAU, BALL_INFO_W, BALL_INFO_TEMP
            ):
                p = dict(gb_quity=q, gb_sim=s, gb_alpha=float(a),
                         gb_beta=float(beta), gb_K=int(K), gb_w_mode=wm, gb_knn=int(knn),
                         ball_loss_weight=float(blw), ball_angle_thresh=float(ang), ball_uniform_tau=float(tau),
                         ball_infonce_weight=float(biw), ball_infonce_temp=float(bit))
                if already_done(ds, p):
                    print(f"[SKIP] {ds} {p} Already have,Pass")
                    continue
                jobs.append((ds, p, stage_tag))

    if not jobs:
        print("[INFO] No More new to Run.Done.")
        return

    print(f"[INFO] Total_jobs: {len(jobs)} | Worker_online: {MAX_WORKERS} | stage={stage_tag}")
    print(f"[INFO] JOBLIB_TEMP_FOLDER={os.environ.get('JOBLIB_TEMP_FOLDER')}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, *job): job for job in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                code = fut.result()
                print(f"[DONE] {job} -> return {code}")
            except Exception as e:
                print(f"[ERR ] {job} -> {e}")

if __name__ == "__main__":
    main()
