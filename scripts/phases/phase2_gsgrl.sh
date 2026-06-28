#!/usr/bin/env bash
# 阶段 2：G-SGRL 架构融合（BTCM / BRSM）
# 代码尚未实现时本脚本仅打印说明；实现后取消注释并填写命令。
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phase2"

log_info "=== Phase 2: G-SGRL (BTCM / BRSM) ==="

# 检测 --gb_btcm 是否已加入 train.py
if grep -q 'gb_btcm' "${SRC_DIR}/train.py" 2>/dev/null; then
  log_info "检测到 --gb_btcm，可在此添加 BTCM 实验命令"
  # 示例（实现后启用）:
  # run_train "${PHASE}" Computers "btcm_50x3" \
  #   --use_gb --gb_btcm --gb_incremental --gb_rebuild_every 10 \
  #   --num_epochs 50 --trials 3 ...
else
  log_info "SKIP: --gb_btcm 尚未实现。请先完成 ROADMAP 阶段 2 代码，再运行本脚本。"
  record_journal "${PHASE}" "ALL" "placeholder" "N/A" "SKIP" "BTCM/BRSM not implemented"
  exit 0
fi
