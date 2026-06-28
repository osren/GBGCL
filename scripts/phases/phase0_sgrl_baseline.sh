#!/usr/bin/env bash
# 阶段 0：SGRL 复现（无粒球，700 epoch × 5 trials）
# 对标 SGRL NeurIPS 2024 Table 1
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phase0"
# 可选：RESULTS_DIR="${PROJECT_ROOT}/results/phase0_sgrl" ./phase0_sgrl_baseline.sh

DATASETS=(CS Photo Physics Computers)
EPOCHS=700
TRIALS=5

log_info "=== Phase 0: SGRL baseline (no --use_gb) ==="

for ds in "${DATASETS[@]}"; do
  run_train "${PHASE}" "${ds}" "sgrl_700x5" \
    --num_epochs "${EPOCHS}" \
    --trials "${TRIALS}" \
    --num_hop 1 \
    --hidden_dim 1024 \
    --gb_rebuild_every 100
done

log_info "=== Phase 0 complete. Check ${RESULTS_DIR}/*_summary.csv and ${JOURNAL} ==="
