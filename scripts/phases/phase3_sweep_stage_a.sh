#!/usr/bin/env bash
# 阶段 3a：超参扫参 Stage-A（150 epoch × 1 trial）
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phase3a"
export SWEEP_STAGE=A
export SWEEP_WORKERS="${SWEEP_WORKERS:-2}"
export SWEEP_DEVICE="${DEVICE}"

LOG_FILE="${LOGS_DIR}/${PHASE}/sweep_stage_a.log"
mkdir -p "${LOGS_DIR}/${PHASE}"

log_info "=== Phase 3a: sweepX Stage-A ==="
log_info "SWEEP_WORKERS=${SWEEP_WORKERS} LOG=${LOG_FILE}"

cd "${PROJECT_ROOT}"
export RESULTS_DIR  # sweepX 使用 results/

if python tools/sweepX.py 2>&1 | tee "${LOG_FILE}"; then
  record_journal "${PHASE}" "ALL" "sweep_stage_a" "${LOG_FILE}" "OK"
else
  record_journal "${PHASE}" "ALL" "sweep_stage_a" "${LOG_FILE}" "FAIL"
  exit 1
fi

log_info "=== Phase 3a complete. Run phase4_analyze.sh next ==="
