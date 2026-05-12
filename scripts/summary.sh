#!/bin/bash
# GBGCL 结果汇总脚本 - 直接读取日志文件

echo "========================================="
echo "       Photo 数据集实验结果汇总"
echo "========================================="

LOG_DIR="logs"

echo ""
echo "=== 1. BYOL (无粒球) ==="
if [ -f "$LOG_DIR/byol_baseline.log" ]; then
    echo "均值:"
    grep "TRIAL" $LOG_DIR/byol_baseline.log | awk '{print $5}' | awk '{sum+=$1} END {print sum/NR}'
    echo "最大值:"
    grep "TRIAL" $LOG_DIR/byol_baseline.log | awk '{print $5}' | sort -n | tail -1
    echo "trial次数:"
    grep -c "TRIAL" $LOG_DIR/byol_baseline.log
else
    echo "文件不存在: $LOG_DIR/byol_baseline.log"
fi

echo ""
echo "=== 2. Baseline (有粒球，无残差) ==="
if [ -f "$LOG_DIR/baseline_700_5.log" ]; then
    echo "均值:"
    grep "TRIAL" $LOG_DIR/baseline_700_5.log | awk '{print $5}' | awk '{sum+=$1} END {print sum/NR}'
    echo "最大值:"
    grep "TRIAL" $LOG_DIR/baseline_700_5.log | awk '{print $5}' | sort -n | tail -1
    echo "trial次数:"
    grep -c "TRIAL" $LOG_DIR/baseline_700_5.log
else
    echo "文件不存在: $LOG_DIR/baseline_700_5.log"
fi

echo ""
echo "=== 3. Option A (有粒球+残差) ==="
if [ -f "$LOG_DIR/option_a_700_5.log" ]; then
    echo "均值:"
    grep "TRIAL" $LOG_DIR/option_a_700_5.log | awk '{print $5}' | awk '{sum+=$1} END {print sum/NR}'
    echo "最大值:"
    grep "TRIAL" $LOG_DIR/option_a_700_5.log | awk '{print $5}' | sort -n | tail -1
    echo "trial次数:"
    grep -c "TRIAL" $LOG_DIR/option_a_700_5.log
else
    echo "文件不存在: $LOG_DIR/option_a_700_5.log"
fi

echo ""
echo "========================================="