---
description: "Open Harness V2 初始化入口。普通模式初始化服务仓库；Hub 模式初始化集中式文档仓库。详细步骤下沉到 bootstrap agents。"
---

## User Input

```text
$ARGUMENTS
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--hub` | 进入集中式文档库模式，仅生成系统视图层 | `/oh:init --hub` |
| `--knowledge <path>` | 导入外部知识库文档 | `/oh:init --knowledge ./knowledge-seed/` |

## 入口判断

解析 `$ARGUMENTS`：

- 含 `--hub` → 执行 **Hub 模式**
- 不含 `--hub` → 执行 **普通模式（服务仓库）**

## 执行编排

### Hub 模式

调用 Agent tool：

- 读取 `~/.claude/skills/openharness/agents/init-hub-bootstrap.md`（安装后重写为 `{install-dir}/commands/oh/agents/init-hub-bootstrap.md`）
- 传递上下文：
  - 当前仓库根目录
  - 原始参数：`$ARGUMENTS`

Hub 模式目标产物：

- `docs/knowledge/system/flow-seeds.yaml`
- `docs/knowledge/system/core-flows/*.md`
- `docs/knowledge/system/system-map.md`
- `docs/knowledge/index.md`
- `CLAUDE.md`

Hub 模式非目标：

- `docs/harness/`
- `docs/hooks/`
- `.claude/settings.json` hooks
- 服务仓库执行用 `docs/specs/` 骨架

### 普通模式（服务仓库）

调用 Agent tool：

- 读取 `~/.claude/skills/openharness/agents/init-service-bootstrap.md`（安装后重写为 `{install-dir}/commands/oh/agents/init-service-bootstrap.md`）
- 传递上下文：
  - 当前仓库根目录
  - 原始参数：`$ARGUMENTS`

普通模式目标产物：

- `docs/harness/`
- `docs/knowledge/`
- `docs/knowledge/service-meta.yaml`
- `docs/specs/`
- `docs/hooks/`
- `CLAUDE.md`

## 约束

- **命令文件只负责编排** — 详细生成流程统一下沉到 bootstrap agents
- **生成的 `docs/knowledge/` 必须支持后续命令按需加载** — 先读 `docs/knowledge/index.md` 建立完整知识地图，再进入相关索引或稳定入口，最后按需读取叶子文档
- **knowledge 子目录应优先生成索引入口** — 如 `prd/index.md`、`exp/index.md`、`api/index.md`、`database/index.md`；`service-boundary.md` 保持稳定入口语义
- **Hub 与普通模式职责分离** — 文档库只做视图和系统级方案，代码仓库才做约束和执行工作区
- **不捏造信息** — 无法确认的内容必须标 `<!-- TODO: 待确认 -->`
