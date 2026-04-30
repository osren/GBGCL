---
description: "归档已完成的需求。将 specs/active/{feature} 移入 specs/completed/，自动归档 Proposal 到 knowledge/prd/，扫描代码变更输出知识库更新建议（人工确认后执行），提示经验沉淀。"
---

## User Input

```text
$ARGUMENTS
```

## Feature 上下文解析

按以下统一规则定位目标 feature：

- 用户显式传入 `[feature]` → 直接使用该 feature
- 未传入 `[feature]` 且 `docs/specs/active/` 下只有一个活跃 feature → 直接使用该 feature
- 未传入 `[feature]` 且存在多个活跃 feature → 列出候选并让用户选择
- 未传入 `[feature]` 且无活跃 feature → 终止并提示用户先创建或指定 feature

## 前置检查

执行前检查以下条件，不满足则提示用户：

1. `docs/specs/active/{feature}/` 目录存在
2. `tasks.md` 中已记录执行完成状态或 `exec-done` checkpoint（已完成执行）
3. 综合 review 已通过（优先检查 `verification.md` / `design.md` 中的综合 review 结果；无法确认时询问用户）
4. `verification.md` 中的轻量交付门禁与证据门禁均已满足，不存在“失败 / 待验证 / 不满足”的关键项
5. 若存在 `verify-report.md`，其结果不能为“不通过”
6. 若最近一轮在处理 review / verify / human finding，必须能看到 finding 对账结果
7. 不存在仍待处理的 BLOCKER 问题

## 执行编排

调用 Agent tool：

- 读取 `~/.claude/skills/openharness/agents/archive-feature.md`（安装后重写为 `{install-dir}/commands/oh/agents/archive-feature.md`）
- 传递上下文：
  - feature 名称
  - 当前仓库根目录

归档阶段目标产物：

- `docs/specs/completed/{feature}/`
- `docs/knowledge/prd/{feature}-proposal.md`
- `docs/knowledge/prd/index.md`
- 按用户确认后执行的 `docs/knowledge/` 更新
- 可选的 `docs/knowledge/exp/{feature}-exp.md`

## 约束

- **命令文件只负责入口判断和编排** — 详细归档流程统一下沉到 agent
- **`/oh:archive` 也必须按需加载该项目的 `docs/knowledge/` 知识库** — 不得跳过知识入口直接更新 `knowledge/`
- **读取 knowledge 时必须渐进式加载** — 先读 `docs/knowledge/index.md`，再按归档需要进入相关索引或稳定入口，最后按需更新叶子文档
- **不自动修改 harness/** — 归档阶段只更新 `knowledge/`，不修改约束文件
- **只输出真实变更的建议** — 不编造不存在的变更
- **经验沉淀不强制** — 用户可选择跳过
- **经验内容基于事实** — 只总结 `tasks.md` / `verification.md` / `review` / `verify` 中有记录的内容
- **归档前必须检查 `verification.md` 中的轻量交付门禁与证据门禁** — 任一关键项为“失败”或“待验证”时不得归档
- **proposal.md 必须归档** — 这是归档流程中的默认动作，不需要额外确认
