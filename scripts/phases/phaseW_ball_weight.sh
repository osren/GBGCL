#!/usr/bin/env bash
# Stage W: ball_loss_weight 敏感性扫参
# 固定 Stage-B Top-1 的其他超参，仅扫 {0.05(基线), 0.1, 0.2, 0.5, 1.0}
# Photo: detach / dot / alpha=0.3 / beta=0.2 / K=10 / w_mode=topo+center
# Computers: homo  / dot / alpha=0.7 / beta=0.2 / K=10 / w_mode=topo+center
# 50 epoch × 3 trials, ~4h 双卡并行
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phaseW"
WEIGHTS=(0.05 0.1 0.2 0.5 1.0)
EPOCHS=50
TRIALS=3

log_info "=== Stage W: ball_loss_weight sensitivity ==="
log_info "RESULTS_DIR=${RESULTS_DIR}"

# Photo
for w in "${WEIGHTS[@]}"; do
  tag="photo_${w}_50x3"
  # 0.05 已存在则跳过（直接复用 Stage-B 行）
  if [[ "${w}" == "0.05" ]] && \
     [[ -f "${RESULTS_DIR}/Photo_summary.csv" ]] && \
     grep -q ",0.05,${EPOCHS}," "${RESULTS_DIR}/Photo_summary.csv"; then
    log_info "SKIP photo weight=0.05 (reusing Stage-B row)"
    record_journal "${PHASE}" "Photo" "${tag}" "reused" "OK" "from Stage-B"
    continue
  fi
  run_train "${PHASE}" "Photo" "${tag}" \
    --use_gb \
    --gb_quity detach \
    --gb_sim dot \
    --gb_alpha 0.3 \
    --gb_beta 0.2 \
    --gb_K 10 \
    --gb_w_mode topo+center \
    --ball_loss_weight "${w}" \
    --num_epochs "${EPOCHS}" \
    --trials "${TRIALS}" \
    --gb_rebuild_every 50 \
    || { log_info "FAIL Photo weight=${w}"; continue; }
done

# Computers
for w in "${WEIGHTS[@]}"; do
  tag="computers_${w}_50x3"
  if [[ "${w}" == "0.05" ]] && \
     [[ -f "${RESULTS_DIR}/Computers_summary.csv" ]] && \
     grep -q ",0.05,${EPOCHS}," "${RESULTS_DIR}/Computers_summary.csv"; then
    log_info "SKIP computers weight=0.05 (reusing Stage-B row)"
    record_journal "${PHASE}" "Computers" "${tag}" "reused" "OK" "from Stage-B"
    continue
  fi
  run_train "${PHASE}" "Computers" "${tag}" \
    --use_gb \
    --gb_quity homo \
    --gb_sim dot \
    --gb_alpha 0.7 \
    --gb_beta 0.2 \
    --gb_K 10 \
    --gb_w_mode topo+center \
    --ball_loss_weight "${w}" \
    --num_epochs "${EPOCHS}" \
    --trials "${TRIALS}" \
    --gb_rebuild_every 50 \
    || { log_info "FAIL Computers weight=${w}"; continue; }
done

log_info "=== Stage W complete. Run analyze_results then check analysis/ball_weight_winners.csv ==="
