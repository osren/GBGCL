#!/bin/bash
# ============================================================
# organize_project.sh - 通用项目结构整理脚本
# ============================================================
# 用法: ./organize_project.sh [--dry-run] [--config CONFIG_FILE]
#
# 默认分类规则:
#   src/          - 核心源代码 (.py 主逻辑文件)
#   scripts/      - 运行脚本 (run_*, *.sh)
#   tools/        - 工具脚本 (sweep, analyze, generate 等)
#   tests/        - 测试文件 (test_*.py, *_test.py)
#   docs/         - 文档 (README*, *.md, LICENSE)
#   logs/         - 日志文件 (*.log, nohup_*)
#   data/         - 数据文件 (.csv, .json, .npz, .pt, .pkl)
#   backup/       - 备份/原始版本 (*_or.py, *_origin.py, *.bak)
#   config/       - 配置文件 (.yaml, .yml, .json, .cfg, .ini)
#   results/      - 实验结果 (已存在的 results/)
#   notebooks/    - Jupyter notebooks (.ipynb)
#   notebooks/    - Jupyter notebooks (.ipynb)
#   lib/          - 第三方库/依赖 (vendor/, lib/)
#   tmp/          - 临时文件
#
# 选项:
#   --dry-run     - 仅显示将要执行的操作，不实际移动文件
#   --config      - 指定自定义配置文件
# ============================================================

set -e

DRY_RUN=false
CONFIG_FILE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 定义分类规则 (模式 -> 目标目录)
declare -A RULES=(
    # 核心源代码 - 主要的 .py 文件
    ["*.py"]="src"

    # 运行脚本
    ["run_*"]="scripts"
    ["*.sh"]="scripts"

    # 工具脚本
    ["sweep*.py"]="tools"
    ["analyze*.py"]="tools"
    ["gen_*.py"]="tools"
    ["repair*.py"]="tools"
    ["visualize*.py"]="tools"

    # 测试文件
    ["test_*.py"]="tests"
    ["*_test.py"]="tests"

    # 文档
    ["README*"]="docs"
    ["*.md"]="docs"
    ["LICENSE*"]="docs"
    ["CHANGELOG*"]="docs"

    # 日志
    ["*.log"]="logs"
    ["nohup_*"]="logs"

    # 备份/原始
    ["*_or.py"]="backup"
    ["*_origin.py"]="backup"
    ["*_bak.py"]="backup"
    ["*.bak"]="backup"

    # Jupyter notebooks
    ["*.ipynb"]="notebooks"
)

# 自定义配置文件格式示例:
# {
#   "src": ["train.py", "models.py", "core/*.py"],
#   "scripts": ["run_*.sh", "slurm*.sh"],
#   "exclude": ["important.py", "config.yaml"]
# }

# 执行移动
move_file() {
    local src="$1"
    local dest="$2"

    if [[ "$src" == "$dest" ]] || [[ -z "$dest" ]]; then
        return
    fi

    if [[ -d "$dest" ]]; then
        dest_dir="$dest"
    else
        dest_dir="$(dirname "$dest")"
    fi

    if [[ ! -d "$dest_dir" ]]; then
        mkdir -p "$dest_dir"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY-RUN] mv $src -> $dest"
    else
        mv "$src" "$dest"
        echo "  [MOVE] $src -> $dest"
    fi
}

# 主逻辑
echo "=========================================="
echo "项目结构整理"
echo "=========================================="
[[ "$DRY_RUN" == true ]] && echo "[DRY-RUN 模式 - 仅预览]"

# 1. 创建标准目录结构
STANDARD_DIRS="src scripts tools tests docs logs data backup config results notebooks topo"
for dir in $STANDARD_DIRS; do
    if [[ ! -d "$dir" ]] && [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$dir"
    fi
done

# 2. 扫描并分类文件
echo ""
echo "扫描项目文件..."

# 排除目录
EXCLUDE_DIRS=".git .claude __pycache__ node_modules .venv venv"

for item in *; do
    # 跳过排除的目录
    skip=false
    for exclude in $EXCLUDE_DIRS; do
        if [[ "$item" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    [[ "$skip" == true ]] && continue

    # 跳过标准目录
    for dir in $STANDARD_DIRS; do
        if [[ "$item" == "$dir" ]] || [[ -d "$item" && "$item" != "topo" ]]; then
            continue 2
        fi
    done

    # 匹配规则
    matched=false
    for pattern in "${!RULES[@]}"; do
        dest_dir="${RULES[$pattern]}"

        case "$pattern" in
            *.py|run_*|test_*.py|*_test.py|README*|*.md|LICENSE*|*.log|nohup_*|*_or.py|*_origin.py|*_bak.py|*.bak|*.sh|*.ipynb|sweep*.py|analyze*.py|gen_*.py|repair*.py|visualize*.py)
                if [[ "$item" == $pattern ]]; then
                    move_file "$item" "$dest_dir/"
                    matched=true
                    break
                fi
                ;;
        esac
    done

    # 未匹配的文件显示警告
    if [[ "$matched" == false ]] && [[ -f "$item" ]]; then
        echo "  [UNCLASSIFIED] $item"
    fi
done

echo ""
echo "=========================================="
echo "整理完成!"
echo "=========================================="
[[ "$DRY_RUN" == true ]] && echo "使用不带 --dry-run 参数运行以实际执行移动"
