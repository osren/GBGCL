#!/usr/bin/env bash
# 检查 PyG 离线数据集是否齐全（服务器无法访问 GitHub 时必须本地放置 raw npz）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${DATA_DIR:-${ROOT}/datasets}"

# PyG 期望: datasets/<Name>/raw/<file>.npz
declare -A RAW_FILES=(
  [CS]="ms_academic_cs.npz"
  [Physics]="ms_academic_phy.npz"
  [Photo]="amazon_electronics_photo.npz"
  [Computers]="amazon_electronics_computers.npz"
)

ok=0
fail=0

echo "DATA_DIR=${DATA}"
echo ""

for name in CS Physics Photo Computers; do
  f="${DATA}/${name}/raw/${RAW_FILES[$name]}"
  if [[ -f "$f" ]]; then
    echo "  OK   ${name}  ->  ${f}"
    ok=$((ok + 1))
  else
    echo "  MISS ${name}  ->  ${f}"
    fail=$((fail + 1))
  fi
done

echo ""
if [[ -d "${DATA}/cs" && ! -d "${DATA}/CS" ]]; then
  echo "WARN: 发现小写目录 datasets/cs ，Linux 上 PyG 需要 datasets/CS"
  echo "      可执行: mv ${DATA}/cs ${DATA}/CS"
  echo ""
fi

if [[ "$fail" -gt 0 ]]; then
  echo "缺少 ${fail} 个数据集。请在本机下载 npz 后 scp 到服务器，或："
  echo "  mkdir -p ${DATA}/CS/raw"
  echo "  # 将 ms_academic_cs.npz 放入 ${DATA}/CS/raw/"
  echo ""
  echo "下载地址（需能访问 GitHub 的机器）："
  echo "  https://github.com/shchur/gnn-benchmark/tree/master/data/npz"
  exit 1
fi

echo "All required datasets present."
