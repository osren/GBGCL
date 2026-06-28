#!/usr/bin/env bash
# 阶段 1：增量扩散机制验证（E7）
# Photo + Computers，50 epoch × 3 trials；对比 incremental vs baseline_gb
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phase1"
# 可选隔离：RESULTS_DIR="${PROJECT_ROOT}/results/phase1_incremental"

log_info "=== Phase 1: incremental diffusion (E7) ==="

for ds in Photo Computers; do
  # 1a. 增量扩散
  run_train "${PHASE}" "${ds}" "incremental_50x3" \
    --use_gb \
    --gb_quity homo \
    --gb_sim cos \
    --gb_alpha 0.3 \
    --gb_incremental \
    --num_epochs 50 \
    --trials 3 \
    --gb_rebuild_every 10

  # 1b. 对照：同配置但不增量
  run_train "${PHASE}" "${ds}" "no_incremental_50x3" \
    --use_gb \
    --gb_quity homo \
    --gb_sim cos \
    --gb_alpha 0.3 \
    --gb_rebuild_every 10 \
    --num_epochs 50 \
    --trials 3
done

log_info "=== Phase 1 complete. 查看 log 中 [h_DEBUG] [Incremental] 与 h_z_diff ==="
