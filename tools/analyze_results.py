'''
# analyze_results.py - 分析训练结果，生成汇总报告
# 用法: python tools/analyze_results.py --results_dir results_final --out_dir analysis_final
'''
import os, glob, csv, numpy as np
import argparse
from collections import defaultdict
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
parser.add_argument("--out_dir", type=str, default="analysis", help="Output directory")
parser.add_argument("--topk", type=int, default=3, help="Top-K records per dataset")
args = parser.parse_args()

RESULTS_DIR = args.results_dir
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# 聚合字段 - 相同配置组合的key
KEY_FIELDS = [
    "gb_quity", "gb_sim", "gb_alpha",
    "gb_beta", "gb_K", "gb_w_mode", "gb_knn",
    "ball_loss_weight", "ball_angle_thresh", "ball_uniform_tau",
    "ball_infonce_weight", "ball_infonce_temp"
]

def load_dataset_csv(dataset):
    path = os.path.join(RESULTS_DIR, f"{dataset}_summary.csv")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def build_key(row, fields):
    """用指定字段构建配置key"""
    key = []
    for k in fields:
        v = row.get(k, "")
        try:
            v = float(v)
        except:
            pass
        key.append((k, v))
    return tuple(key)

def main(topk=3):
    # 查找所有数据集
    datasets = []
    for f in glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv")):
        datasets.append(os.path.basename(f).replace("_summary.csv", ""))
    datasets = sorted(set(datasets))
    print(f"[INFO] Found datasets: {datasets}")

    overall_rows = []

    for ds in datasets:
        rows = load_dataset_csv(ds)
        if not rows:
            print(f"[WARN] no results for {ds}")
            continue

        # 只统计 use_gb=1 的实验
        rows = [r for r in rows if r.get("use_gb", "0") == "1"]
        if not rows:
            print(f"[WARN] no GB results for {ds}")
            continue

        # 动态获取可用的列
        present_cols = set(rows[0].keys()) & set(KEY_FIELDS)
        use_fields = [k for k in KEY_FIELDS if k in present_cols]
        print(f"[INFO] {ds}: {len(rows)} rows, fields: {use_fields}")

        # 聚合同一配置的结果
        bucket = defaultdict(list)
        for r in rows:
            k = build_key(r, use_fields)
            bucket[k].append(float(r["clf_mean"]))

        # 计算统计量
        aggs = []
        for k, vals in bucket.items():
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            mx = float(np.max(vals))
            k_dict = dict(k)
            aggs.append((mu, sd, mx, k_dict, len(vals)))

        # 排序：均值降序，方差升序
        aggs.sort(key=lambda x: (-x[0], x[1]))

        # 输出数据集汇总（覆盖模式）
        out_path = os.path.join(OUT_DIR, f"{ds}_gb_summary.csv")
        header = ["dataset", "timestamp", "mean", "std", "max", "num_trials", "stat_type"] + use_fields
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for (mu, sd, mx, kd, n) in aggs:
                row = [ds, datetime.now().strftime("%Y-%m-%d %H:%M"), f"{mu:.6f}", f"{sd:.6f}", f"{mx:.6f}", n, "mean"]
                row += [kd.get(k, "") for k in use_fields]
                w.writerow(row)
        print(f"[SAVE] {out_path} | Top-{topk}: mean={aggs[0][0]:.4f}+-{aggs[0][1]:.4f}, max={aggs[0][2]:.4f} (n={aggs[0][4]})")

        # 记录 overall
        for (mu, sd, mx, kd, n) in aggs[:topk]:
            row = [ds, f"{mu:.6f}", f"{sd:.6f}", f"{mx:.6f}", n, "mean"]
            row += [kd.get(k, "") for k in use_fields]
            overall_rows.append(row)

    # 输出总表（覆盖模式）
    if not overall_rows:
        print("[WARN] no overall rows")
        return

    overall_path = os.path.join(OUT_DIR, "overall_topk.csv")
    header = ["dataset", "mean", "std", "max", "num_trials", "stat_type"] + use_fields
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(overall_rows)

    print(f"[SAVE] {overall_path}")
    print("\n=== Best Results (Mean) ===")
    for row in overall_rows[:len(datasets)]:
        print(f"  {row[0]}: {row[1]}+-{row[2]} (max={row[3]})")

    print("\n=== Best Max per Dataset ===")
    for ds in datasets:
        ds_rows = [r for r in overall_rows if r[0] == ds]
        if ds_rows:
            max_row = max(ds_rows, key=lambda r: float(r[3]))
            print(f"  {ds}: {max_row[3]}")

if __name__ == "__main__":
    main(topk=args.topk)