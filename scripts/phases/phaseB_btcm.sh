#!/usr/bin/env bash
# Stage BTCM: 球扩散融合进 TCM 路径
# 与 phase2_gsgrl.sh 同样的 SKIP 守卫：未检测到 --gb_btcm 时记录 SKIP 后退出
# 实现后取消下方注释即可启用
set -euo pipefail
source "$(dirname "$0")/common.sh"

PHASE="phaseB"

log_info "=== Stage BTCM: ball-aware TCM (--gb_btcm) ==="

if grep -q 'gb_btcm' "${SRC_DIR}/train.py" 2>/dev/null; then
  log_info "--gb_btcm 已实现，开始 BTCM 实验"

  # (1) 烟测：Photo 1 epoch
  log_info "Step 1: smoke test (Photo, 1 epoch)"
  run_train "${PHASE}" "Photo" "smoke_1x1" \
    --use_gb --gb_btcm \
    --gb_quity detach --gb_sim dot --gb_alpha 0.3 --gb_beta 0.2 --gb_K 10 \
    --ball_loss_weight 0.05 \
    --num_epochs 1 --trials 1 \
    --device "${DEVICE}" \
    --gb_rebuild_every 50 \
    || { log_info "FAIL smoke test"; record_journal "${PHASE}" "Photo" "smoke_1x1" "N/A" "FAIL" "smoke broke"; exit 1; }

  # (2) Photo 50 epoch × 3 trials
  log_info "Step 2: Photo 50ep x 3 trials"
  run_train "${PHASE}" "Photo" "btcm_50x3" \
    --use_gb --gb_btcm \
    --gb_quity detach --gb_sim dot --gb_alpha 0.3 --gb_beta 0.2 --gb_K 10 \
    --ball_loss_weight 0.05 \
    --num_epochs 50 --trials 3 \
    --gb_rebuild_every 50 \
    || log_info "FAIL Photo btcm_50x3"

  # (3) Computers 50 epoch × 3 trials
  log_info "Step 3: Computers 50ep x 3 trials"
  run_train "${PHASE}" "Computers" "btcm_50x3" \
    --use_gb --gb_btcm \
    --gb_quity homo --gb_sim dot --gb_alpha 0.7 --gb_beta 0.2 --gb_K 10 \
    --ball_loss_weight 0.05 \
    --num_epochs 50 --trials 3 \
    --gb_rebuild_every 50 \
    || log_info "FAIL Computers btcm_50x3"

  # (4) CS 50 epoch × 3 trials
  log_info "Step 4: CS 50ep x 3 trials"
  run_train "${PHASE}" "CS" "btcm_50x3" \
    --use_gb --gb_btcm \
    --gb_quity detach --gb_sim dot --gb_alpha 0.3 --gb_beta 0.3 --gb_K 5 \
    --ball_loss_weight 0.05 \
    --num_epochs 50 --trials 3 \
    --gb_rebuild_every 50 \
    || log_info "FAIL CS btcm_50x3"

  log_info "=== Stage BTCM 50ep done. 验证 h_z_diff > 0.3 与 clf_mean 增益 ==="

else
  log_info "SKIP: --gb_btcm 尚未在 train.py 实现。请先完成 ROADMAP §四 BTCM 章节中的代码改动。"
  record_journal "${PHASE}" "ALL" "placeholder" "N/A" "SKIP" "BTCM not implemented"
  exit 0
fi
