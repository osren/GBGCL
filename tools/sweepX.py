# -*- coding: utf-8 -*-
# sweepX.py
# Automatic hyperparameter sweep: Stage-A (coarse) / Stage-B (fine-tune) + multi-GPU + dataset parallelization
import os, csv, shlex, itertools, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# =========================
# Configuration (modify here)
# =========================

# Datasets supported by data.py / train.py (--dataset_name)
#DATASETS = ["Computers", "Photo", "CS", "Physics"]
DATASETS = ["Physics", "Computers", "Photo", "CS"]
# Hyperparameter grid: Stage-A uses full grid, Stage-B uses FILTERS subset
QUITY  = ["homo", "detach", "edges"]
SIMS   = ["dot", "cos"]
ALPHAS = [0.7, 0.5, 0.3]

# Ball diffusion hyperparameters (full grid)
BETAS   = [0.1, 0.2, 0.3]          # grid [0.1, 0.2, 0.3]
KS      = [3, 5, 10, 20]            # grid [5, 10, 20]
W_MODES = ["topo", "center", "topo+center"] # grid ["topo", "center", "topo+center"]
KNNs    = [5, 10, 20]            # grid [5, 10, 20]

# Ball Loss hyperparameters (full grid)
BALL_LOSS_W      = [0.05]     # grid [0.0, 0.02, 0.05]
BALL_ANGLE_THR   = [15.0, 25.0]     # grid [10.0, 15.0, 25.0]
BALL_UNIFORM_TAU = [0.1]      # grid [0.05, 0.1]
BALL_INFO_W      = [0.02]     # grid [0.0, 0.01, 0.02]
BALL_INFO_TEMP   = [0.2]      # grid [0.1, 0.2, 0.3]

# Filter presets for dataset-specific Top-K selection
# Supports two formats:
# 1) 3-tuple (quity, sim, alpha) with full grid search defaults
# 2) 12-tuple (all hyperparameters fixed, no grid search)
FILTERS = {
    "Photo": [
        ("detach","cos",0.3),    # P1: cos similarity
        ("edges","cos",0.3),     # P2: edges quality + cos
        ("homo","cos",0.5),     # P3: homo + alpha=0.5
        ("detach","dot",0.5),   # P4: alpha=0.5 intermediate
        ("edges","dot",0.3),     # P5: edges + dot
    ],
    "Computers": [
        ("edges","cos",0.3),   # P1: edges + cos
        ("homo","cos",0.5),    # P2: homo + cos + alpha=0.5
        ("detach","cos",0.5),   # P3: detach + cos + alpha=0.5
        ("edges","dot",0.5),   # P4: edges + dot + alpha=0.5
        ("detach","cos",0.3),   # P5: detach + cos baseline
        ("detach","dot",0.7),    # baseline: original best config
        ("homo","dot",0.7),     # baseline
    ],
    "Physics": [
        ("detach","cos",0.3),   # P1: detach + cos
        ("edges","dot",0.3),    # P2: edges + dot
        ("homo","cos",0.3),    # baseline
        ("detach","dot",0.3),   # baseline
    ],
    "CS": [
        # CS already beats baseline, resume from checkpoint
        ("detach","dot",0.7),
        ("detach","dot",0.3),
    ],
}

# Stage configuration (controlled by environment, default A)
STAGE = os.environ.get("SWEEP_STAGE", "A").upper()
if STAGE == "A":         # coarse search
    NUM_EPOCHS = 150
    TRIALS = 1
    GB_REBUILD_EVERY = 50
else:                    # fine-tune
    NUM_EPOCHS = 700
    TRIALS = 5
    GB_REBUILD_EVERY = 100

# Training hyperparameters (same as train.py defaults)
E1_LR = 1e-5
E2_LR = 1e-5
HIDDEN_DIM = 1024
NUM_HOP = 1
NUM_LAYERS = 1
MOMENTUM = 0.99
SEED = 66666
LOG_EVERY = 50

# Device configuration (cuda or cpu)
DEVICE = os.environ.get("SWEEP_DEVICE", "cuda")

# Concurrency (number of parallel workers)
MAX_WORKERS = int(os.environ.get("SWEEP_WORKERS", "2"))

# sklearn / joblib temp folder to avoid /tmp issues
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/dev/shm")

# train.py output: writes to results/
RESULTS_DIR = "results"
LOG_DIR = "log_CUDA"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# Helper functions
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

    # Detect Windows platform
    import platform
    is_windows = platform.system() == "Windows"

    # Build base arguments
    base_args = [
        "python", "src/train.py",
        "--dataset_name", shlex.quote(dataset),
        "--log_dir", shlex.quote(log_path),
        "--e1_lr", str(E1_LR),
        "--e2_lr", str(E2_LR),
        "--num_epochs", str(NUM_EPOCHS),
        "--hidden_dim", str(HIDDEN_DIM),
        "--num_hop", str(NUM_HOP),
        "--num_layers", str(NUM_LAYERS),
        "--momentum", str(MOMENTUM),
        "--seed", str(SEED),
        "--trials", str(TRIALS),
        "--log_every", str(LOG_EVERY),
        "--gb_rebuild_every", str(GB_REBUILD_EVERY),
        "--device", DEVICE,
        "--results_dir", RESULTS_DIR,
        "--use_gb",
        "--gb_quity", p["gb_quity"],
        "--gb_sim", p["gb_sim"],
        "--gb_alpha", str(p["gb_alpha"]),
        "--gb_beta", str(p["gb_beta"]),
        "--gb_K", str(p["gb_K"]),
        "--gb_w_mode", p["gb_w_mode"],
        "--gb_knn", str(p["gb_knn"]),
        "--ball_loss_weight", str(p["ball_loss_weight"]),
        "--ball_angle_thresh", str(p["ball_angle_thresh"]),
        "--ball_uniform_tau", str(p["ball_uniform_tau"]),
        "--ball_infonce_weight", str(p["ball_infonce_weight"]),
        "--ball_infonce_temp", str(p["ball_infonce_temp"]),
    ]

    if is_windows:
        # Windows: use set command for environment variable
        cmd = f"set CUDA_VISIBLE_DEVICES=0 && {' '.join(base_args)}"
    else:
        cmd = f"CUDA_VISIBLE_DEVICES=0 {' '.join(base_args)}"

    return cmd

def run_one(dataset: str, p: dict, stage_tag: str) -> int:
    print(f"[RUN] {dataset} | {p} | stage={stage_tag}")
    cmd = build_cmd(dataset, p, stage_tag)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = "_".join([dataset, p["gb_quity"], p["gb_sim"], str(p["gb_alpha"]), stage_tag])
    stdout_stderr_file = os.path.join(LOG_DIR, f"{tag}.out")

    import platform
    is_windows = platform.system() == "Windows"

    with open(stdout_stderr_file, "w", encoding="utf-8") as f:
        if is_windows:
            # Windows: start new process group via start command
            detached_cmd = f'start "" cmd /c "{cmd} > {stdout_stderr_file} 2>&1"'
            subprocess.Popen(detached_cmd, shell=True, cwd=src_dir,
                            creationflags=0x00000010)  # CREATE_NEW_PROCESS_GROUP
        else:
            # Unix: nohup + redirect stdin from /dev/null
            detached_cmd = f"{cmd} > {stdout_stderr_file} 2>&1"
            subprocess.Popen(["nohup", "bash", "-c", detached_cmd],
                             cwd=src_dir, stdin=subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT)
    print(f"[BG]   log: {stdout_stderr_file}")
    return 0

# =========================
# Main entry point
# =========================

def main():
    stage_tag = STAGE
    jobs: List[Tuple[str, dict, str]] = []

    for ds in DATASETS:
        # From FILTERS dict or full grid if not in FILTERS
        if ds in FILTERS and len(FILTERS[ds]) > 0:
            for tpl in FILTERS[ds]:
                if len(tpl) == 3:
                    q, s, a = tpl
                    # Full grid over remaining hyperparameters
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
                    # 12-tuple: all params fixed
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
            # Full grid search
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