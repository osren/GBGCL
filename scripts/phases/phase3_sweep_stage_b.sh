#!/usr/bin/env bash
# 阶段 3b：超参扫参 Stage-B（700 epoch × 5 trials，精训 FILTERS 子集）
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phase3b"
export SWEEP_STAGE=B
export SWEEP_WORKERS="${SWEEP_WORKERS:-1}"
export SWEEP_DEVICE="${DEVICE}"

LOG_FILE="${LOGS_DIR}/${PHASE}/sweep_stage_b.log"
mkdir -p "${LOGS_DIR}/${PHASE}"

log_info "=== Phase 3b: sweepX Stage-B ==="
log_info "建议先运行 phase4_analyze.sh 确认 Stage-A Top-K，必要时更新 tools/sweepX.py FILTERS"
log_info "SWEEP_WORKERS=${SWEEP_WORKERS} LOG=${LOG_FILE}"

cd "${PROJECT_ROOT}"

if python tools/sweepX.py 2>&1 | tee "${LOG_FILE}"; then
  record_journal "${PHASE}" "ALL" "sweep_stage_b" "${LOG_FILE}" "OK"
else
  record_journal "${PHASE}" "ALL" "sweep_stage_b" "${LOG_FILE}" "FAIL"
  exit 1
fi

log_info "=== Phase 3b complete ==="
