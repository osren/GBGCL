#!/usr/bin/env bash
# 使用 nohup 后台启动某一阶段实验
#
# 用法:
#   bash scripts/phases/run_phase_nohup.sh 0          # 阶段 0，后台
#   bash scripts/phases/run_phase_nohup.sh 1          # 阶段 1
#   bash scripts/phases/run_phase_nohup.sh 3a         # Stage-A 扫参
#   bash scripts/phases/run_phase_nohup.sh 3b         # Stage-B 扫参
#   bash scripts/phases/run_phase_nohup.sh 4          # 分析
#   bash scripts/phases/run_phase_nohup.sh W          # 阶段 W: ball_loss_weight 敏感性
#   bash scripts/phases/run_phase_nohup.sh R          # 阶段 R: Photo/Computers 复跑
#   bash scripts/phases/run_phase_nohup.sh B          # 阶段 BTCM: --gb_btcm 烟测 + 50ep
#   bash scripts/phases/run_phase_nohup.sh 0 --foreground   # 前台同步跑
#
# 环境变量（可选）:
#   RESULTS_DIR=/path/to/results/phase0
#   DEVICE=cuda
#   SWEEP_WORKERS=2
set -euo pipefail

PHASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PHASE_SCRIPT_DIR}/../../" && pwd)"
NOHUP_DIR="${PROJECT_ROOT}/logs/nohup"
mkdir -p "${NOHUP_DIR}"

PHASE="${1:-}"
FOREGROUND="${2:-}"

if [[ -z "${PHASE}" ]]; then
  echo "Usage: $0 <phase> [--foreground]"
  echo "  phase: 0 | 1 | 2 | 3a | 3b | 4"
  exit 1
fi

case "${PHASE}" in
  0)   SCRIPT="${PHASE_SCRIPT_DIR}/phase0_sgrl_baseline.sh" ;;
  1)   SCRIPT="${PHASE_SCRIPT_DIR}/phase1_incremental_diffusion.sh" ;;
  2)   SCRIPT="${PHASE_SCRIPT_DIR}/phase2_gsgrl.sh" ;;
  3a)  SCRIPT="${PHASE_SCRIPT_DIR}/phase3_sweep_stage_a.sh" ;;
  3b)  SCRIPT="${PHASE_SCRIPT_DIR}/phase3_sweep_stage_b.sh" ;;
  4)   SCRIPT="${PHASE_SCRIPT_DIR}/phase4_analyze.sh" ;;
  W)   SCRIPT="${PHASE_SCRIPT_DIR}/phaseW_ball_weight.sh" ;;
  R)   SCRIPT="${PHASE_SCRIPT_DIR}/phaseR_repeat_validation.sh" ;;
  B)   SCRIPT="${PHASE_SCRIPT_DIR}/phaseB_btcm.sh" ;;
  *)
    echo "Unknown phase: ${PHASE}"
    exit 1
  ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT="${NOHUP_DIR}/phase${PHASE}_${STAMP}.out"
PID_FILE="${NOHUP_DIR}/phase${PHASE}.pid"

if [[ "${FOREGROUND}" == "--foreground" ]]; then
  echo "Running in foreground: ${SCRIPT}"
  bash "${SCRIPT}"
  exit 0
fi

nohup bash "${SCRIPT}" > "${OUT}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started phase ${PHASE} (PID=${PID})"
echo "  stdout/stderr: ${OUT}"
echo "  pid file:      ${PID_FILE}"
echo "  tail -f ${OUT}"
echo "  kill \$(cat ${PID_FILE})   # 如需停止"
