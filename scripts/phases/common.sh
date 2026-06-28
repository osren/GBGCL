#!/usr/bin/env bash
# 分阶段实验公共函数（被 scripts/phases/*.sh 引用）
set -euo pipefail

# scripts/phases/common.sh -> 项目根目录 GBGCL/
PHASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PHASE_SCRIPT_DIR}/../../" && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
LOGS_DIR="${PROJECT_ROOT}/logs/phases"
NOHUP_DIR="${PROJECT_ROOT}/logs/nohup"
JOURNAL="${LOGS_DIR}/journal.csv"

export PYTHONUNBUFFERED=1

mkdir -p "${LOGS_DIR}" "${NOHUP_DIR}"

# 可选：隔离某阶段结果到 results/phase0 等（默认写入 results/）
export RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets}"
export DEVICE="${DEVICE:-cuda}"
export SWEEP_WORKERS="${SWEEP_WORKERS:-2}"

log_info() { echo "[$(date '+%F %T')] $*"; }

# 记录到 journal.csv：phase,dataset,tag,results_dir,log_file,status,note
record_journal() {
  local phase="$1" dataset="$2" tag="$3" log_file="$4" status="$5" note="${6:-}"
  if [[ ! -f "${JOURNAL}" ]]; then
    echo "timestamp,phase,dataset,tag,results_dir,log_file,status,note" > "${JOURNAL}"
  fi
  echo "$(date '+%F %T'),${phase},${dataset},${tag},${RESULTS_DIR},${log_file},${status},${note}" >> "${JOURNAL}"
}

run_train() {
  # run_train <phase> <dataset> <tag> <extra args...>
  local phase="$1" dataset="$2" tag="$3"
  shift 3
  local log_file="${LOGS_DIR}/${phase}/${dataset}_${tag}.log"
  mkdir -p "${LOGS_DIR}/${phase}"

  log_info "START phase=${phase} dataset=${dataset} tag=${tag}"
  log_info "LOG=${log_file} RESULTS=${RESULTS_DIR} DATA=${DATA_DIR}"

  cd "${SRC_DIR}"
  if python train.py \
    --dataset_name "${dataset}" \
    --data_dir "${DATA_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --log_dir "${log_file}" \
    --device "${DEVICE}" \
    "$@"; then
    record_journal "${phase}" "${dataset}" "${tag}" "${log_file}" "OK"
    log_info "DONE phase=${phase} dataset=${dataset} tag=${tag}"
  else
    record_journal "${phase}" "${dataset}" "${tag}" "${log_file}" "FAIL"
    log_info "FAIL phase=${phase} dataset=${dataset} tag=${tag}"
    return 1
  fi
}
