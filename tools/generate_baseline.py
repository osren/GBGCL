'''
# generate_baseline.py - 从分析结果生成与 Baseline 对比的 CSV
# 用法: python tools/generate_baseline.py --source_dir analysis_final
#       python tools/generate_baseline.py --source_dir results_final --method "GB+OptionC"
'''
import os, csv, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing overall_topk.csv")
parser.add_argument("--method", type=str, default=None, help="Method name (default: source_dir)")
parser.add_argument("--output", type=str, default="BaseLine.csv", help="Output CSV file")
args = parser.parse_args()

METHOD = args.method or args.source_dir.rstrip("/")

# 读取数据
source_path = os.path.join(args.source_dir, "overall_topk.csv")

# 如果没有 overall_topk.csv，尝试直接从 *_summary.csv 读取
if not os.path.exists(source_path):
    import glob
    print(f"[INFO] No overall_topk.csv, reading from *_summary.csv directly")
    dataset_best = {}
    for f in glob.glob(os.path.join(args.source_dir, "*_summary.csv")):
        ds = os.path.basename(f).replace("_summary.csv", "")
        with open(f) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [r for r in reader if r.get("use_gb", "0") == "1"]
            if rows:
                max_val = max(float(r["clf_mean"]) for r in rows)
                dataset_best[ds] = max_val * 100

print("[INFO] Best max per dataset:")
for ds, mx in dataset_best.items():
    print(f"  {ds}: {mx:.2f}")

# 读取或创建输出 CSV
output_path = args.output
header = ["Method"]

# 确定数据集列顺序
ds_order = ["CS", "Computers", "Photo", "Physics"]
for ds in ds_order:
    if ds in dataset_best:
        header.append(ds)

# 尝试读取现有数据
existing = {}
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing[r["Method"]] = r

# 写入
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # 写入现有数据
    for method, row in existing.items():
        if method != METHOD:  # 不覆盖新方法
            writer.writerow([row.get(h, "") for h in header])

    # 写入新方法
    new_row = [METHOD]
    for ds in header[1:]:
        val = dataset_best.get(ds, "")
        new_row.append(f"{val:.2f}" if val else "")
    writer.writerow(new_row)

print(f"[SAVE] {output_path}")
print(f"  {METHOD}: {new_row[1:]}")