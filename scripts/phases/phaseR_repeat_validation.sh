#!/usr/bin/env bash
# Stage R: Photo/Computers 复跑 — 区分 -0.20% 是真退化还是 Stage-B 5-trial 噪声
# 每组 700 epoch × 5 trials:
#   R-A Photo:     Phase 0 baseline (no --use_gb)
#   R-B Photo:     Stage-B Top-1
#   R-C Computers: Phase 0 baseline
#   R-D Computers: Stage-B Top-1
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phaseR"
EPOCHS=700
TRIALS=5

log_info "=== Stage R: Photo/Computers repeat validation ==="
log_info "RESULTS_DIR=${RESULTS_DIR}"

# R-A: Photo Phase 0 baseline
run_train "${PHASE}" "Photo" "phase0_baseline_700x5" \
  --num_epochs "${EPOCHS}" \
  --trials "${TRIALS}" \
  --gb_rebuild_every 100 \
  || log_info "FAIL R-A Photo baseline"

# R-B: Photo Stage-B Top-1
run_train "${PHASE}" "Photo" "stageB_top1_700x5" \
  --use_gb \
  --gb_quity detach \
  --gb_sim dot \
  --gb_alpha 0.3 \
  --gb_beta 0.2 \
  --gb_K 10 \
  --gb_w_mode topo+center \
  --ball_loss_weight 0.05 \
  --num_epochs "${EPOCHS}" \
  --trials "${TRIALS}" \
  --gb_rebuild_every 100 \
  || log_info "FAIL R-B Photo Top-1"

# R-C: Computers Phase 0 baseline
run_train "${PHASE}" "Computers" "phase0_baseline_700x5" \
  --num_epochs "${EPOCHS}" \
  --trials "${TRIALS}" \
  --gb_rebuild_every 100 \
  || log_info "FAIL R-C Computers baseline"

# R-D: Computers Stage-B Top-1
run_train "${PHASE}" "Computers" "stageB_top1_700x5" \
  --use_gb \
  --gb_quity homo \
  --gb_sim dot \
  --gb_alpha 0.7 \
  --gb_beta 0.2 \
  --gb_K 10 \
  --gb_w_mode topo+center \
  --ball_loss_weight 0.05 \
  --num_epochs "${EPOCHS}" \
  --trials "${TRIALS}" \
  --gb_rebuild_every 100 \
  || log_info "FAIL R-D Computers Top-1"

log_info "=== Stage R complete. Analyze: results/phaseR_repeat/Photo_summary.csv & Computers_summary.csv ==="
