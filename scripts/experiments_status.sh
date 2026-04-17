#!/bin/bash
# ============================================================
# experiments_status.sh - 实验现状整理脚本
# ============================================================
# 生成项目实验状态的汇总报告
#
# 输出内容:
#   1. 实验数据集概览
#   2. 各数据集最佳结果
#   3. 实验配置统计
#   4. 缺失的实验组合
# ============================================================

RESULTS_DIR="results"
LOGS_DIR="logs"

echo "=========================================="
echo "  SGRL 实验现状汇总"
echo "=========================================="
echo ""

# 1. 数据集概览
echo "【数据集概览】"
echo "-------------------------------------------"
for f in "$RESULTS_DIR"/*_summary.csv; do
    if [[ -f "$f" ]]; then
        dataset=$(basename "$f" | sed 's/_summary.csv//')
        trials=$(tail -n +2 "$f" | cut -d',' -f1 | sort -u | wc -l)
        total=$(tail -n +2 "$f" | wc -l)
        echo "  $dataset: $total 条记录, $trials 个 trials"
    fi
done
echo ""

# 2. 各数据集最佳结果
echo "【各数据集最佳结果 (clf_mean)】"
echo "-------------------------------------------"
for f in "$RESULTS_DIR"/*_summary.csv; do
    if [[ -f "$f" ]]; then
        dataset=$(basename "$f" | sed 's/_summary.csv//')
        best=$(tail -n +2 "$f" | sort -t',' -k5 -rn | head -1)
        best_acc=$(echo "$best" | cut -d',' -f5)
        best_trial=$(echo "$best" | cut -d',' -f1)
        best_config=$(echo "$best" | cut -d',' -f9- | sed 's/,/ /g')
        printf "  %-12s: %.4f (trial=%s, config: %s)\n" "$dataset" "$best_acc" "$best_trial" "$best_config"
    fi
done
echo ""

# 3. 实验配置统计
echo "【实验配置分布】"
echo "-------------------------------------------"
echo "  gb_quity (粒球质量):"
tail -n +2 "$RESULTS_DIR"/*.csv 2>/dev/null | cut -d',' -f9 | sort | uniq -c | while read count val; do
    printf "    %-10s: %s 次\n" "$val" "$count"
done

echo "  gb_sim (相似度):"
tail -n +2 "$RESULTS_DIR"/*.csv 2>/dev/null | cut -d',' -f10 | sort | uniq -c | while read count val; do
    printf "    %-10s: %s 次\n" "$val" "$count"
done

echo "  gb_alpha:"
tail -n +2 "$RESULTS_DIR"/*.csv 2>/dev/null | cut -d',' -f11 | sort -u | tr '\n' ' '
echo ""
echo ""

# 4. 日志文件统计
echo "【日志文件统计】"
echo "-------------------------------------------"
echo "  CUDA日志:    $(find "$LOGS_DIR"/log_CUDA -name "*.log" 2>/dev/null | wc -l) 个"
echo "  训练日志:    $(find "$LOGS_DIR" -maxdepth 1 -name "*.log" 2>/dev/null | wc -l) 个"
echo "  粒球计数:    $(find "$LOGS_DIR"/granular_count -name "*.txt" 2>/dev/null | wc -l) 个"
echo ""

# 5. 未完成/待实验配置
echo "【建议补充的实验】"
echo "-------------------------------------------"
echo "  提示: 检查 results/ 中缺失的配置组合"
echo ""

echo "=========================================="
echo "  报告生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
