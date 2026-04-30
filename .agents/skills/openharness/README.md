# OpenHarness

OpenHarness 是一套面向 Claude Code 的 AI Coding workflow skill。

它的目标不是“帮你多生成几份文档”，而是把 **约束、知识、规格、审查** 这四件事放进同一条研发链路里，让 Agent 真正按边界工作，而不是只靠即时对话和临场猜测。

## 它是怎么工作的

OpenHarness 的核心不是“让 Agent 直接开始写代码”，而是先帮项目建立一个适合长期协作的工作环境。

这套环境主要包括：

- `harness`：定义必须遵守的业务、架构和基础设施约束
- `knowledge`：沉淀系统现状、领域知识和项目上下文
- `specs`：管理需求文档、执行状态和验证材料
- `hooks`：补充动作后检查与 review checklist

在执行过程中，Agent 需要先从 `harness/index.md` 和 `knowledge/index.md` 进入，再按索引渐进式下钻到相关子目录和叶子文档，而不是跳过入口直接凭经验工作。

在具体需求上，OpenHarness 会先收敛 `proposal`（需求背景）、`spec`（行为契约）、`design`（技术方案），再补 `tasks`（执行清单）和 `verification`（验证方案），并在各阶段自动触发 `/oh:review`；需要时再执行独立 `/oh:verify`，最后归档并沉淀知识。

所以它更像一套完整的研发工作流，而不是一个单纯帮你写文档或写代码的 prompt 包。

## 安装

使用d-skill安装完openharness后，在 Claude Code 中先执行：

```text
/openharness
```

这一步会安装：

- `/oh:*` 命令
- OpenHarness 配套 skills

如果当前会话没有立即识别到 `/oh:*` 命令，重启 Claude Code 或新开一个会话即可。

## 快速开始

### 1. 初始化项目

```text
/oh:init
```

这一步会生成基础工作区：

- `CLAUDE.md`：项目入口与导航
- `docs/harness/`：约束层，告诉 Agent 什么不能违反
- `docs/knowledge/`：知识层，告诉 Agent 当前系统是什么样
- `docs/specs/`：需求工作区，存放 proposal/spec/design/tasks/verification
- `docs/hooks/`：审查机制，承载轻量 runtime guard、阶段门禁说明和 CR checklist

### 2. 启动第一个需求

```text
/oh:propose <feature-name> <prd-path>
```

Phase 1 会生成：

- `proposal.md`：说明为什么做、范围是什么、关键旅程是什么
- `spec.md`：定义系统必须满足的行为契约
- `design.md`：说明准备怎么做，并作为审批主稿

生成完成后会自动触发一次 `/oh:review`，用于检查需求和方案是否收敛。

### 3. 补齐执行文档

```text
/oh:propose <feature-name> --continue
```

Phase 2 会生成：

- `tasks.md`：执行任务清单、阶段状态和 checkpoint
- `verification.md`：测试方案、轻量交付门禁和证据门禁

生成完成后会再次自动触发 `/oh:review`，用于检查执行准备是否完整。

### 4. 实现、验收、归档

```text
/oh:apply
# 自动 /oh:review

/oh:verify   # 可选，独立验收
/oh:archive
```

其中：

- `/oh:apply`：按 design 与 tasks 落地实现
- `/oh:review`：负责阶段文档 review、Design/Hub 审批、apply 后综合 review
- `/oh:verify`：负责独立验收，只看契约和最终结果
- `/oh:archive`：归档需求，并沉淀 Proposal 与相关知识

## 工作流总览

```text
/openHarness
→ /oh:init
→ /oh:propose <name> <prd-path>
→ 自动 /oh:review（Phase 1）
→ /oh:propose <name> --continue
→ 自动 /oh:review（Phase 2）
→ /oh:apply
→ 自动 /oh:review（apply 后综合）
→ 可选 /oh:verify
→ /oh:archive
```

## 里面有什么

### 核心命令

- `/oh:init` — 初始化项目工作区
- `/oh:propose` — 两段式生成需求与执行文档
- `/oh:review` — 阶段 review / 审批 / apply 后综合 review
- `/oh:apply` — 执行实现，并自动触发 review
- `/oh:verify` — 独立验收
- `/oh:rollback` — 回退到 checkpoint
- `/oh:archive` — 归档并沉淀知识
- `/oh:status` — 输出项目状态报告

### 核心特点

- **不是只写 Spec**：而是把约束加载、知识加载、文档生成、审批、实现、review、verify、archive 串成完整链路
- **渐进式加载上下文**：先读 `index.md`，再按需下钻，避免上下文噪音和局部失真
- **review / verify 分层清晰**：review 负责开发侧综合检查，verify 负责独立验收
- **文档职责明确**：proposal / spec / design / tasks / verification 各自有清晰边界，不混写

## 为什么它和别的 workflow 不太一样

和 OpenSpec 相比，OpenHarness 不只关注 spec 本身，更强调运行期治理。

和 Superpower 相比，OpenHarness 不只强调开发闭环，更强调约束、审批、证据和可回溯性。

如果要一句话概括：

> **OpenHarness = 规格驱动工作流 + review/verify 分层 + 约束与知识治理**

## 适合什么场景

OpenHarness 更适合：

- 中后台/企业系统
- 金融、支付、交易、风控等高约束业务
- 多人协作、需要审批和验收闭环的团队
- 希望沉淀项目知识库，而不是只做一次性 AI coding 的团队

如果只是快速起一个 demo，它可能偏重。

但如果你想让 Agent 在真实项目中长期、稳定、可控地工作，它会更合适。
