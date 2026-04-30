---
description: "服务仓库 propose Phase 2 总控 agent。审批通过后生成 tasks、verification，并写入 ready-to-exec checkpoint。"
---

## 任务：执行服务仓库 propose Phase 2

### Step 1: 检查前置条件

检查 `docs/specs/active/{name}/design.md` 是否存在：

- 不存在 → 提示先执行 Phase 1（`/oh:propose {name} {prd-path}`）
- 存在且 `approval: pending` → **终止执行**，提示先完成 `/oh:review` 并获得 `approval: approved`
- 存在且 `approval: approved` → 继续执行

### Step 2: 读取已有文档、知识库与约束体系

并行读取：

- `docs/specs/active/{name}/proposal.md`
- `docs/specs/active/{name}/spec.md`
- `docs/specs/active/{name}/design.md`
- `docs/knowledge/index.md` — knowledge 总入口，先建立完整知识地图
- 再按执行文档生成需要进入相关知识索引或稳定入口（如 `project-context.md`、`architecture.md`、`domain-model.md`、`api/index.md`、`database/index.md`、`prd/index.md`、`exp/index.md`、`service-boundary.md`）
- `docs/harness/index.md` — harness 总入口，建立完整约束地图
- 依次检查所有子目录索引，并读取每个子目录下的全部叶子规则文件：
  - `docs/harness/invariants/index.md` 及全部叶子规则
  - `docs/harness/architecture/index.md` 及全部叶子规则
  - `docs/harness/infrastructure/index.md` 及全部叶子规则
  - `docs/harness/linters/index.md` 及全部叶子规则

在生成 `tasks.md / verification.md` 之前，若出现以下任一情况，先条件触发 `writing-plans`：

- `design.md` 已确定，但执行顺序仍不清晰
- 存在明显依赖链、阶段拆分或验证关口
- 需求规模较大，直接生成 `tasks.md` 容易失去主线

触发后先整理一份轻量执行计划骨架，至少包含：

- 目标结果
- 步骤顺序
- 依赖关系
- 每个大步骤的验证点
- 可能的分支点或阻塞条件

再把这份骨架映射进 `tasks.md` 的顺序与 `verification.md` 的验证结构。

### Step 3: 生成任务清单

基于 `design.md` 的详细设计与 `spec.md` 的行为契约，生成 `docs/specs/active/{name}/tasks.md`：

```markdown
---
feature: {name}
---
# {name} — 任务清单

| ID | 任务描述 | 优先级 | 状态 | 依赖 |
|----|---------|--------|------|------|
| T-001 | {任务} | P0 | [ ] 待执行 | — |
```

### Step 4: 生成验收验证文档

基于 `spec.md` + `tasks.md`，生成 `docs/specs/active/{name}/verification.md`，包含：

- 功能测试用例（GIVEN/WHEN/THEN，TC-001 格式）
- 边界测试用例（TC-B01 格式）
- 异常测试用例（TC-E01 格式）
- Invariants 检查清单
- 轻量交付门禁（进入 review / archive / 完成声明前的阻断条件）
- 轻量证据门禁（声称完成 / 测试通过 / finding 已修复所需的最小证据）

### Step 5: 更新 tasks checkpoint

在 `tasks.md` 的 `## Checkpoints` 区追加：

```markdown
| {YYYY-MM-DD HH:MM} | ready-to-exec | {HEAD 或 N/A} | 已生成执行清单与验证方案 | tasks.md / verification.md |
```

### Step 6: 输出报告并自动进入 review

```text
## /oh:propose Phase 2 完成 — {name}

### 已生成
- [x] docs/specs/active/{name}/tasks.md
- [x] docs/specs/active/{name}/verification.md
- [x] docs/specs/active/{name}/tasks.md（已追加 ready-to-exec checkpoint）

### 自动执行
- [x] 自动触发 /oh:review
- review scope: tasks.md / verification.md 及其依赖文档
- 不要求代码或 apply 结果

### 下一步
查看 review 报告，确认执行准备无误后执行 /oh:apply {name}
```

## 约束

- **Phase 2 必须在审批通过后执行** — `approval: approved` 之前不得生成 tasks.md / verification.md
- **约束引用必须真实** — verification.md 中引用的约束必须来自 `docs/harness/` 的实际索引与叶子规则，不得凭空编造或只基于局部目录判断
