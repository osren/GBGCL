---
description: "需求文档生成入口。普通模式：Phase 1 生成设计文档并自动 review，Phase 2 生成执行文档并自动 review。Hub 模式：生成跨服务方案母稿和 service-briefs。详细流程下沉到 phase agents。"
---

## User Input

```text
$ARGUMENTS
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `<name>` | feature 名称，用于创建目录（必填） | `early-repayment` |
| `<prd-path>` | 产品文档路径（Phase 1 或 Hub 模式必填） | `./docs/prd-draft.md` 或绝对路径 |
| `--continue` | 普通模式进入 Phase 2，生成执行文档 | `/oh:propose early-repayment --continue` |
| `--hub` | 进入集中式文档库模式，生成跨服务方案母稿 | `/oh:propose --hub credit-apply ./docs/proposal-source.md` |

## 入口判断

解析 `$ARGUMENTS`：

- 同时包含 `--hub` 与 `--continue` → **终止执行**
  - 输出：`⛔ Hub 模式当前不支持 --continue。请在服务仓库执行普通 /oh:propose <name> <service-brief-path>。`
- 含 `--hub` → 执行 **Hub 模式**
- 含 `--continue` 且不含 `--hub` → 执行 **普通模式 Phase 2**
- 其余情况 → 执行 **普通模式 Phase 1**

## 条件 Skills 编排

执行 `/oh:propose` 时，以下 skills 不是默认硬编排，而是按场景条件触发：

| Skill | 触发条件 | 使用效果 |
|------|---------|---------|
| `brainstorming` | 原始需求含糊、存在多个可行方案、涉及高风险架构取舍、用户明确要比较方案时 | 先产出 2-4 个可实现选项、tradeoff 和推荐路径，再进入 design 生成 |
| `writing-plans` | `design.md` 已基本确定，且需要把方案映射为可执行顺序、依赖和验证点时 | 将设计方向整理成执行计划骨架，再映射到 `tasks.md`、`verification.md` 的结构和顺序 |

## 执行编排

### Hub 模式

调用 Agent tool：

- 读取 `~/.claude/skills/openharness/agents/propose-hub-phase1.md`（安装后重写为 `{install-dir}/commands/oh/agents/propose-hub-phase1.md`）
- 传递上下文：
  - feature 名称
  - 产品文档路径
  - 当前仓库根目录

Hub 模式产物：

- `docs/specs/active/{name}/proposal.md`
- `docs/specs/active/{name}/design.md`
- `docs/specs/active/{name}/service-briefs/*.md`

### 普通模式 Phase 1

调用 Agent tool：

- 读取 `~/.claude/skills/openharness/agents/propose-service-design.md`（安装后重写为 `{install-dir}/commands/oh/agents/propose-service-design.md`）
- 传递上下文：
  - feature 名称
  - 产品文档路径
  - 当前仓库根目录

Phase 1 产物：
- `docs/specs/active/{name}/proposal.md`
- `docs/specs/active/{name}/spec.md`
- `docs/specs/active/{name}/design.md`

Phase 1 完成后：
- **自动执行 `/oh:review`**
- review scope 仅限：`proposal.md / spec.md / design.md`
- 不读取也不要求 `tasks.md / verification.md / code`

### 普通模式 Phase 2

调用 Agent tool：

- 读取 `~/.claude/skills/openharness/agents/propose-service-execution-prep.md`（安装后重写为 `{install-dir}/commands/oh/agents/propose-service-execution-prep.md`）
- 传递上下文：
  - feature 名称
  - 当前仓库根目录

Phase 2 产物：
- `docs/specs/active/{name}/tasks.md`
- `docs/specs/active/{name}/verification.md`

Phase 2 完成后：
- **自动执行 `/oh:review`**
- review scope 仅限：`tasks.md / verification.md` 及其依赖文档
- 不要求代码或 apply 结果

## 约束

- **命令文件只负责入口判断和编排** — 详细 phase 流程统一下沉到 agents
- **`/oh:propose` 也完整受 `docs/harness/` 约束体系约束** — 生成 proposal / spec / design / tasks / verification 时不得只按局部约束工作
- **`/oh:propose` 也必须按需加载该项目的 `docs/knowledge/` 知识库** — 不得跳过知识入口直接凭经验生成方案或执行文档
- **读取 harness 时必须全量加载** — 先读 `docs/harness/index.md`，再依次检查各子目录 `index.md`，并读取每个子目录下的全部叶子规则文件
- **读取 knowledge 时必须渐进式加载** — 先读 `docs/knowledge/index.md`，再按任务需要进入相关索引或稳定入口，最后按需读取叶子文档
- **harness 叶子规则文件不能跳过** — 每个子目录下的叶子文件必须全部读取，不得只读部分
- **阶段完成后自动 review** — 但 review 只能基于当前阶段已生成工件工作，不得把未生成工件当成阻断条件
- **Hub 模式不支持 `--continue`** — Hub 仓库不生成执行文档
- **Hub 模式不面向 `oh:apply`** — 代码实现必须回到各服务仓库
