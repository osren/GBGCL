'''
# analyze_results.py
import os, glob, csv, numpy as np
from collections import defaultdict

RESULTS_DIR = "results"
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset_csv(dataset):
    path = os.path.join(RESULTS_DIR, f"{dataset}_summary.csv")
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def key_of(row):
    return (row["gb_quity"], row["gb_sim"], float(row["gb_alpha"]))

def main(topk=3):
    datasets = []
    for f in glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv")):
        datasets.append(os.path.basename(f).replace("_summary.csv",""))
    datasets = sorted(set(datasets))

    overall_rows = []

    for ds in datasets:
        rows = load_dataset_csv(ds)
        if not rows:
            print(f"[WARN] no results for {ds}"); continue

        # 聚合同一组合的多个 trial（已经是trial级均值，这里再聚合成组合级统计）
        bucket = defaultdict(list)
        for r in rows:
            if r.get("use_gb","0") != "1":   # 只看 GB 实验
                continue
            k = key_of(r)
            bucket[k].append(float(r["clf_mean"]))

        aggs = []
        for k, vals in bucket.items():
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            q, s, a = k
            aggs.append((mu, sd, q, s, a, len(vals)))

        aggs.sort(key=lambda x: (-x[0], x[1]))  # 均值降序，方差升序

        # 写出每个数据集的汇总
        out_path = os.path.join(OUT_DIR, f"{ds}_gb_summary.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "gb_quity", "gb_sim", "gb_alpha", "mean", "std", "num_trials"])
            for (mu, sd, q, s, a, n) in aggs:
                w.writerow([ds, q, s, a, f"{mu:.4f}", f"{sd:.4f}", n])
        print(f"[SAVE] {out_path} | Top-{topk}:", aggs[:topk])

        # 记录 overall
        for (mu, sd, q, s, a, n) in aggs[:topk]:
            overall_rows.append([ds, q, s, a, f"{mu:.4f}", f"{sd:.4f}", n])

    # 合并总表
    overall_path = os.path.join(OUT_DIR, "overall_topk.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "gb_quity", "gb_sim", "gb_alpha", "mean", "std", "num_trials"])
        w.writerows(overall_rows)
    print(f"[SAVE] {overall_path}")

if __name__ == "__main__":
    main(topk=3)
'''
# analyze_results.py
import os, glob, csv, numpy as np
from collections import defaultdict

RESULTS_DIR = "results"    # 与 train.py / sweepX.py 对齐
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# 想更粗的聚合，就删减下面的字段
KEY_FIELDS = [
    "gb_quity", "gb_sim", "gb_alpha",
    "gb_beta", "gb_K", "gb_w_mode", "gb_knn",
    "ball_loss_weight", "ball_angle_thresh", "ball_uniform_tau",
    "ball_infonce_weight", "ball_infonce_temp"
]

def load_dataset_csv(dataset):
    path = os.path.join(RESULTS_DIR, f"{dataset}_summary.csv")
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def build_key(row, present_cols):
    key = []
    for k in KEY_FIELDS:
        if k not in present_cols:
            continue
        v = row.get(k)
        # 尝试 float 化
        try:
            v = float(v)
        except Exception:
            pass
        key.append((k, v))
    return tuple(key)

def main(topk=3):
    datasets = []
    for f in glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv")):
        datasets.append(os.path.basename(f).replace("_summary.csv",""))
    datasets = sorted(set(datasets))

    overall_rows = []

    for ds in datasets:
        rows = load_dataset_csv(ds)
        if not rows:
            print(f"[WARN] no results for {ds}"); continue

        # 仅统计 use_gb=1 的试验
        rows = [r for r in rows if r.get("use_gb","0") == "1"]
        if not rows:
            print(f"[WARN] no GB results for {ds}"); continue

        # 动态确定可用的列
        present_cols = set(rows[0].keys())

        # 组合桶：key -> [acc...]
        bucket = defaultdict(list)
        for r in rows:
            k = build_key(r, present_cols)
            bucket[k].append(float(r["clf_mean"]))

        aggs = []
        for k, vals in bucket.items():
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            # 展开 key 供写出
            k_dict = dict(k)
            aggs.append((mu, sd, k_dict, len(vals)))

        aggs.sort(key=lambda x: (-x[0], x[1]))  # 均值降序，方差升序

        # 写出每个数据集的汇总
        out_path = os.path.join(OUT_DIR, f"{ds}_gb_summary.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["dataset", "mean", "std", "num_trials"] + [k for k in KEY_FIELDS if k in present_cols]
            w.writerow(header)
            for (mu, sd, kd, n) in aggs:
                row = [ds, f"{mu:.6f}", f"{sd:.6f}", n] + [kd.get(k, "") for k in KEY_FIELDS if k in present_cols]
                w.writerow(row)
        print(f"[SAVE] {out_path} | Top-{topk}:",
              [(a[0], a[2]) for a in aggs[:topk]])

        # 记录 overall（只取 Top-K）
        for (mu, sd, kd, n) in aggs[:topk]:
            overall_rows.append([ds, f"{mu:.6f}", f"{sd:.6f}", n] + [kd.get(k, "") for k in KEY_FIELDS if k in present_cols])

    # 合并总表
    overall_path = os.path.join(OUT_DIR, "overall_topk.csv")
    # 动态 header
    header = ["dataset", "mean", "std", "num_trials"]
    # 从已有 per-dataset 文件找一份 header
    if overall_rows:
        # 推断包含的参数集合（取 overall_rows 里最长的一行来推断）
        max_len = max(len(r) for r in overall_rows)
        # 目标列数 = 4 + len(有效 KEY_FIELDS)
        # 直接使用 analyze 的 KEY_FIELDS 顺序写出
        header += KEY_FIELDS
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(overall_rows)

    print(f"[SAVE] {overall_path}")

if __name__ == "__main__":
    main(topk=3)
