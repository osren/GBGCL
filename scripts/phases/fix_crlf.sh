#!/bin/sh
# 修复 scripts/phases 下 shell 脚本的 CRLF（Windows 换行）
# 用法: sh scripts/phases/fix_crlf.sh
cd "$(dirname "$0")/../../" || exit 1
for f in scripts/phases/*.sh; do
  sed -i 's/\r$//' "$f"
done
echo "CRLF fixed: scripts/phases/*.sh"
