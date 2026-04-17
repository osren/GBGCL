#!/usr/bin/env python3
"""
experiments_status.py - 实验现状整理脚本

生成 SGRL 项目实验状态的详细汇总报告。
"""

import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")

def print_header(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50 + "\n")

def load_all_results():
    """加载所有结果文件"""
    all_data = []
    for f in RESULTS_DIR.glob("*_summary.csv"):
        df = pd.read_csv(f)
        df['source_file'] = f.name
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def dataset_overview(df):
    """数据集概览"""
    print("【数据集概览】")
    print("-" * 40)

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        trials = subset['trial'].nunique()
        total = len(subset)
        print(f"  {dataset:12s}: {total:4d} 条记录, {trials:2d} 个 trials")

def best_results(df):
    """各数据集最佳结果"""
    print("【各数据集最佳结果 (clf_mean)】")
    print("-" * 40)

    for dataset in sorted(df['dataset'].unique()):
        subset = df[df['dataset'] == dataset]
        best_idx = subset['clf_mean'].idxmax()
        best = subset.loc[best_idx]

        print(f"\n  {dataset}:")
        print(f"    最佳准确率: {best['clf_mean']:.6f} (±{best['clf_var']:.2e})")
        print(f"    Trial: {best['trial']}")
        print(f"    配置: gb_quity={best['gb_quity']}, gb_sim={best['gb_sim']}, "
              f"gb_alpha={best['gb_alpha']}, gb_K={best['gb_K']}")

def config_distribution(df):
    """实验配置统计"""
    print("【实验配置分布】")
    print("-" * 40)

    # gb_quality
    print("\n  gb_quity (粒球质量):")
    for val, count in df['gb_quity'].value_counts().items():
        print(f"    {val:10s}: {count:4d} 次")

    # gb_sim
    print("\n  gb_sim (相似度):")
    for val, count in df['gb_sim'].value_counts().items():
        print(f"    {val:10s}: {count:4d} 次")

    # gb_alpha
    print("\n  gb_alpha 分布:")
    for val in sorted(df['gb_alpha'].unique()):
        count = (df['gb_alpha'] == val).sum()
        print(f"    {val:.1f}: {count:4d} 次")

    # gb_K
    print("\n  gb_K 分布:")
    for val in sorted(df['gb_K'].unique()):
        count = (df['gb_K'] == val).sum()
        print(f"    K={int(val):2d}: {count:4d} 次")

def log_statistics():
    """日志文件统计"""
    print("【日志文件统计】")
    print("-" * 40)

    cuda_logs = list(LOGS_DIR.glob("log_CUDA/*.log"))
    train_logs = list(LOGS_DIR.glob("*.log"))
    granular_counts = list(LOGS_DIR.glob("granular_count/*.txt"))

    print(f"  CUDA日志:    {len(cuda_logs):3d} 个")
    print(f"  训练日志:    {len(train_logs):3d} 个")
    print(f"  粒球计数:    {len(granular_counts):3d} 个")

def missing_experiments(df):
    """分析缺失的实验配置"""
    print("【缺失的实验配置建议】")
    print("-" * 40)

    # 完整配置网格
    all_qualities = ['homo', 'detach']
    all_sims = ['dot', 'cos']
    all_alphas = [0.3, 0.7]
    all_trials = range(1, 21)

    datasets = df['dataset'].unique()

    print("\n  已有的完整实验组合数:")
    for dataset in sorted(datasets):
        subset = df[df['dataset'] == dataset]
        configs = subset.groupby(['gb_quity', 'gb_sim', 'gb_alpha']).ngroups
        print(f"    {dataset}: {configs}")

def main():
    print_header("SGRL 实验现状汇总")

    # 加载数据
    df = load_all_results()

    if df.empty:
        print("  未找到实验结果文件!")
        return

    # 生成报告
    dataset_overview(df)
    best_results(df)
    config_distribution(df)
    log_statistics()
    missing_experiments(df)

    print("\n" + "=" * 50)
    print(f"  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

if __name__ == "__main__":
    main()
