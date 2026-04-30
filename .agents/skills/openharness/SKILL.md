---
name: openharness
description: "Open Harness V2 安装器。将 openharness 命令和配套 skills 安装到项目或全局 Claude Code 目录，使 /oh:init、/oh:propose 等命令以及 verification/debug/review/tdd/planning/parallel-dispatch skills 可用。当用户说 open harness、install openharness 时触发。"
triggers:
  - open harness
  - install openharness
---

# Open Harness V2 安装器

**Open Harness V2** 是面向企业级系统的 AI 辅助开发范式，核心理念："人类驾驭，Agent执行" — 约束前置、知识库驱动、审批控制、多Agent评审。

## 安装流程

### Step 1: 询问安装位置

使用 AskUserQuestion 询问用户：

> 请选择安装位置：
> 1. **项目级**：安装到当前项目的 `.claude/`（仅当前项目可用）
> 2. **全局**：安装到 `~/.claude/`（所有项目均可用）

根据用户选择，记 `{install-dir}` 为对应路径（项目级 → `.claude/`，全局 → `~/.claude/`）。

### Step 2: 确认 skill 路径

skill 的 base directory 已由系统注入（见文件顶部 `Base directory` 行），记为 `{skill-base}`。验证 namespaced commands 目录和内置 skills 目录存在：

```bash
ls {skill-base}/commands/oh/
ls {skill-base}/skills/
```

若不存在，提示用户检查 skill 安装是否完整。

### Step 3: 复制文件

```bash
mkdir -p {install-dir}/commands/oh
mkdir -p {install-dir}/skills

# 命令文件（namespaced 安装到 oh/）
cp {skill-base}/commands/oh/init.md      {install-dir}/commands/oh/init.md
cp {skill-base}/commands/oh/propose.md   {install-dir}/commands/oh/propose.md
cp {skill-base}/commands/oh/review.md    {install-dir}/commands/oh/review.md
cp {skill-base}/commands/oh/apply.md     {install-dir}/commands/oh/apply.md
cp {skill-base}/commands/oh/verify.md    {install-dir}/commands/oh/verify.md
cp {skill-base}/commands/oh/rollback.md  {install-dir}/commands/oh/rollback.md
cp {skill-base}/commands/oh/archive.md   {install-dir}/commands/oh/archive.md
cp {skill-base}/commands/oh/status.md    {install-dir}/commands/oh/status.md

# agents、templates、reference 整目录复制到同一命名空间根
cp -r {skill-base}/agents     {install-dir}/commands/oh/agents
cp -r {skill-base}/templates  {install-dir}/commands/oh/templates
cp -r {skill-base}/reference  {install-dir}/commands/oh/reference

# OpenHarness 内置 skills
cp -r {skill-base}/skills/verification-before-completion {install-dir}/skills/verification-before-completion
cp -r {skill-base}/skills/systematic-debugging          {install-dir}/skills/systematic-debugging
cp -r {skill-base}/skills/requesting-code-review        {install-dir}/skills/requesting-code-review
cp -r {skill-base}/skills/receiving-code-review         {install-dir}/skills/receiving-code-review
cp -r {skill-base}/skills/test-driven-development       {install-dir}/skills/test-driven-development
cp -r {skill-base}/skills/brainstorming                 {install-dir}/skills/brainstorming
cp -r {skill-base}/skills/writing-plans                 {install-dir}/skills/writing-plans
cp -r {skill-base}/skills/executing-plans               {install-dir}/skills/executing-plans
cp -r {skill-base}/skills/parallel-dispatch             {install-dir}/skills/parallel-dispatch

# 修正命令/agent 文档中的路径引用（skill 源路径 → 安装目标路径）
find {install-dir}/commands/oh -name "*.md" | xargs sed -i '' "s|~/.claude/skills/openharness/|{install-dir}/commands/oh/|g"
```

安装后的命令布局：
- `.claude/commands/oh/init.md` → `/oh:init`
- `.claude/commands/oh/propose.md` → `/oh:propose`
- 其余命令同理

安装后的 skills 布局：
- `.claude/skills/verification-before-completion/`
- `.claude/skills/systematic-debugging/`
- `.claude/skills/requesting-code-review/`
- `.claude/skills/receiving-code-review/`
- `.claude/skills/test-driven-development/`
- `.claude/skills/brainstorming/`
- `.claude/skills/writing-plans/`
- `.claude/skills/executing-plans/`
- `.claude/skills/parallel-dispatch/`

### Step 4: 输出安装报告

```
## Open Harness V2 安装完成

安装位置：{install-dir}/

已安装命令：
- /oh:init      → 项目初始化
- /oh:propose   → 需求文档生成（两段式）
- /oh:review    → 阶段文档 review / Design 审批 / apply 后综合 review
- /oh:apply     → 执行实现（完成后自动触发 /oh:review）
- /oh:verify    → 独立验收
- /oh:rollback  → 回退
- /oh:archive   → 归档
- /oh:status    → 状态报告

已安装 skills：
- verification-before-completion
- systematic-debugging
- requesting-code-review
- receiving-code-review
- test-driven-development
- brainstorming
- writing-plans
- executing-plans
- parallel-dispatch

### 快速开始

先判断你当前所在的仓库类型：

#### 1. 服务仓库（默认）
适用：单服务需求设计与实现。

1. 执行 `/oh:init` 初始化约束、知识库和工作区
2. 执行 `/oh:propose <feature-name> <prd-path>` 生成 Phase 1 proposal/spec/design，并自动 `/oh:review`
3. 按 review 结果调整后，完成 Design 审批
4. 执行 `/oh:propose <feature-name> --continue` 生成 Phase 2 tasks/verification，并自动 `/oh:review`
5. 执行 `/oh:apply` 进入实现，完成后自动 `/oh:review`
6. 按需执行 `/oh:verify` 做独立验收，再 `/oh:archive`

#### 2. Hub 仓库
适用：跨服务方案母稿、系统视图、service-briefs 分发。

1. 执行 `/oh:init --hub` 初始化系统视图层
2. 执行 `/oh:propose --hub <feature-name> <prd-path>` 生成跨服务方案母稿
3. 执行 `/oh:review` 完成 Hub 方案审批
4. 将 `service-briefs/*.md` 分发到各服务仓库
5. 在目标服务仓库继续执行普通 `/oh:propose <feature-name> <service-brief-path>`

> `/oh:propose` 是两段式流程：先 `<prd-path>` 生成设计文档，再 `--continue` 生成执行文档。

---

## 命令说明

安装后可用的命令：

| 命令 | 用法 | 说明 |
|------|------|------|
| /oh:init | `/oh:init [--knowledge <path>]` 或 `/oh:init --hub` | 普通模式：服务仓库初始化；Hub 模式：集中式文档库初始化系统视图层 |
| /oh:propose | `/oh:propose <name> <prd-path>`、`/oh:propose <name> --continue`、`/oh:propose --hub <name> <prd-path>` | 普通模式：先设计、后执行文档；Hub 模式：跨服务方案母稿 + service-briefs |
| /oh:review | `/oh:review [feature]` | 综合 review（Phase 1/2 文档 review、Design / Hub 审批、apply 后综合 review） |
| /oh:apply | `/oh:apply [feature]` | 执行实现（含审批检查，完成后自动触发 `/oh:review`） |
| /oh:verify | `/oh:verify [feature]` | 独立验收（独立视角验证 spec / design / verification / code 契约） |
| /oh:rollback | `/oh:rollback [feature] [--to <checkpoint>]` | 回退到指定 checkpoint |
| /oh:archive | `/oh:archive [feature]` | 归档需求（自动沉淀 Proposal，输出知识库更新建议） |
| /oh:status | `/oh:status` | 项目状态报告 |

## 内置 Skills

| Skill | 适用场景 | 与工作流的关系 |
|------|---------|--------------|
| `verification-before-completion` | 声称完成/修复/通过前 | 约束 apply / review / verify / archive 的证据要求 |
| `systematic-debugging` | 出现失败、回归、行为不一致 | 适用于 apply / review / verify 阶段的问题定位 |
| `requesting-code-review` | 送审前整理 review 上下文 | 适用于 /oh:review 前的自检和材料打包 |
| `receiving-code-review` | 收到 review / verify finding 后处理 | 适用于 review / verify 返回问题后的修复闭环 |
| `test-driven-development` | 选择 TDD 实现方式 | 适用于 /oh:apply --tdd 或 bugfix 的 RED/GREEN/REFACTOR |
| `brainstorming` | 方案未收敛、需要比较多个选项 | 适用于 propose 前、复杂改造前的方案探索 |
| `writing-plans` | 需要把方向变成可执行步骤 | 适用于 brainstorm 后、正式动手前的计划拆解 |
| `executing-plans` | 已有计划，需要按步骤推进 | 适用于 apply 或大改造过程中的进度执行 |
| `parallel-dispatch` | 可并行拆分的研究、实现或验证任务 | 适用于复杂任务中的多 agent 分工与侧车验证 |

## 工作流总览

```text
服务仓库：
/oh:init → /oh:propose <name> <prd-path> → 自动 /oh:review（Phase 1 文档）→ /oh:propose <name> --continue → 自动 /oh:review（Phase 2 文档）→ /oh:apply → 自动 /oh:review（apply 后综合）→ 可选 /oh:verify → /oh:archive

Hub 仓库：
/oh:init --hub → /oh:propose --hub <name> <prd-path> → /oh:review → service-briefs/*.md → 分发到服务仓库继续普通 propose
```

| 控制点 | 说明 |
|--------|------|
| `/oh:review` | `design.md` 必须审批通过后才能进入 `--continue` 和 `/oh:apply` |
| `/oh:apply` | 仅在服务仓库执行；Hub 模式不直接进入实现 |
| `/oh:archive` | 自动归档 Proposal，并输出知识库更新建议 |

## 项目目录结构（概览）

```text
project-root/
├── CLAUDE.md
└── docs/
    ├── harness/      # 约束层
    ├── knowledge/    # 知识库
    ├── specs/        # active / completed 工作区
    └── hooks/        # 审查机制
```

更完整的设计背景、目录 rationale 与范式说明，请参考 `reference/openHarnessV2.md`。
