#!/usr/bin/env bash
# 阶段 4：结果汇总与分析
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phase4"
LOG_FILE="${LOGS_DIR}/${PHASE}/analyze.log"
mkdir -p "${LOGS_DIR}/${PHASE}"

log_info "=== Phase 4: analyze & status ==="

cd "${PROJECT_ROOT}"

{
  echo "=== analyze_results.py ==="
  python tools/analyze_results.py
  echo ""
  echo "=== experiments_status.py ==="
  python scripts/experiments_status.py 2>/dev/null || bash scripts/experiments_status.sh 2>/dev/null || true
  echo ""
  echo "=== overall_topk.csv (head) ==="
  head -20 analysis/overall_topk.csv 2>/dev/null || echo "no overall_topk.csv"
} 2>&1 | tee "${LOG_FILE}"

record_journal "${PHASE}" "ALL" "analyze" "${LOG_FILE}" "OK"
log_info "=== Phase 4 complete. 将关键数字更新到 docs/EXPERIMENTS.md / BASELINES.md ==="
