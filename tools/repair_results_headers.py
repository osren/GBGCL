# -*- coding: utf-8 -*-
import os, csv, shutil, glob
RESULTS_DIR = "results"
BACKUP_DIR  = os.path.join(RESULTS_DIR, "_backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

# 规范的“长头部”——与你当前 train.py 写入的列一致
CANONICAL_HEADER = [
    'trial', 'dataset', 'best_online_loss', 'best_target_loss',
    'clf_mean', 'clf_var', 'num_epochs', 'hidden_dim',
    'use_gb', 'gb_quity', 'gb_sim', 'gb_alpha',
    'gb_beta', 'gb_K', 'gb_w_mode', 'gb_knn',
    'gb_rebuild_every',
    'ball_loss_weight', 'ball_angle_thresh', 'ball_uniform_tau',
    'ball_infonce_weight', 'ball_infonce_temp',
    'seed'
]

def repair_one(csv_path: str):
    # 1) 读整个文件（raw reader），拿到每行的列数，找出最大列数
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        rows = list(rdr)

    if not rows:
        print(f"[SKIP] empty file: {csv_path}")
        return

    orig_header = rows[0]
    data_rows   = rows[1:]
    max_cols = max(len(r) for r in rows)

    # 2) 如果原 header 列数 < 文件里最长行 → 需要修复
    if len(orig_header) >= max_cols and len(orig_header) >= len(CANONICAL_HEADER):
        print(f"[OK  ] header already long: {csv_path}")
        return

    # 3) 建一个“目标头部”
    #    规则：以 CANONICAL_HEADER 为主；如文件中行长度比规范更长，就在末尾填充“extra_i”临时列名保证不会丢数据
    target_header = CANONICAL_HEADER[:]
    if max_cols > len(target_header):
        for i in range(len(target_header), max_cols):
            target_header.append(f"extra_{i-len(CANONICAL_HEADER)+1}")

    # 4) 备份原文件
    shutil.copy2(csv_path, os.path.join(BACKUP_DIR, os.path.basename(csv_path)))

    # 5) 写回：头部换成 target_header；每行用空串补齐到相同列数
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(target_header)
        for r in rows[1:]:
            if len(r) < len(target_header):
                r = r + [""] * (len(target_header) - len(r))
            elif len(r) > len(target_header):
                r = r[:len(target_header)]
            w.writerow(r)
    os.replace(tmp_path, csv_path)
    print(f"[FIX ] header -> {len(orig_header)} -> {len(target_header)} | {csv_path}")

def main():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv")))
    if not paths:
        print(f"[INFO] no csv in {RESULTS_DIR}")
        return
    for p in paths:
        try:
            repair_one(p)
        except Exception as e:
            print(f"[ERR ] {p}: {e}")

if __name__ == "__main__":
    main()
