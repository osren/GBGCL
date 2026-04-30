---
feature: {feature-name}
project-mode: {greenfield|brownfield}
source: docs/specs/active/{feature-name}/proposal.md
---

# {feature-name}

## Purpose

{一句话说明本 feature 为系统新增、修改或约束了什么能力}

## Scope

### In scope
- {本次 feature 明确覆盖的能力或行为}

### Out of scope
- {本次明确不处理的内容}

## Requirements

### Requirement: {核心能力一}
The system SHALL {能力要求}.

#### Scenario: {正常场景名称}
- **GIVEN** {前置条件}
- **WHEN** {触发动作}
- **THEN** {预期结果}
- **AND** {附加断言}

#### Scenario: {边界场景名称}
- **GIVEN** {边界或特殊前置条件}
- **WHEN** {触发动作}
- **THEN** {边界情况下的预期行为}

### Requirement: {核心能力二}
The system SHALL {能力要求}.

#### Scenario: {异常场景名称}
- **GIVEN** {导致异常的条件}
- **WHEN** {触发动作}
- **THEN** 系统应返回 {错误码} "{错误信息}"
- **AND** {系统状态不应发生变化或其他保护机制}

## Constraints

- [{规则编号}](../../harness/invariants/{file}.md) — {规则简述}
- [{规则编号}](../../harness/architecture/{file}.md) — {架构约束或边界要求}

## Notes

- {待人工确认项；仅保留真正未收敛的问题}
