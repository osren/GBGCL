---
feature: {feature-name}
execution-status: pending
current-checkpoint: docs-generated
last-updated: {YYYY-MM-DD}
---

# {feature-name} — 任务清单

## Execution Summary

- 当前阶段：{待执行 / 执行中 / 已完成}
- 当前重点：{当前主线任务}
- 风险或阻塞：{若无则填“无”}

## 任务列表

| ID | 任务描述 | 类型 | 优先级 | 状态 | 依赖 | 备注 |
|----|---------|------|--------|------|------|------|
| T-001 | {实现核心能力} | 业务层 | P0 | [ ] 待执行 | - | |
| T-002 | {补齐接口/数据层改动} | 实现 | P0 | [ ] 待执行 | T-001 | |
| T-003 | {补齐测试与验证脚本} | 测试 | P0 | [ ] 待执行 | T-001,T-002 | |
| T-004 | {完成回归与交付整理} | 交付 | P1 | [ ] 待执行 | T-003 | |

## Checkpoints

| 时间 | Checkpoint | git-head | 摘要 | 影响范围 |
|------|------------|----------|------|---------|
| {YYYY-MM-DD HH:MM} | docs-generated | {HEAD 或 N/A} | 已生成 tasks.md 初版 | tasks.md |

## 状态说明

- `[ ] 待执行` — 尚未开始
- `[~] 进行中` — 正在执行
- `[x] 已完成` — 执行完成
- `[!] 阻塞` — 遇到问题，需要处理
