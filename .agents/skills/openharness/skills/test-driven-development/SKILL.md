---
name: test-driven-development
description: 用 specs 和 verification case 驱动 RED-GREEN-REFACTOR 循环来实现行为。适用于 OpenHarness 或 Claude Code 中可以先写测试的功能开发或缺陷修复。
---

# 测试驱动开发

在以下场景使用这个 skill：

- a feature can be expressed as testable behavior
- a bug has a reproducible failing scenario
- you want tighter control over regressions during implementation

## 目标

让实现从失败测试出发，而不是从猜测性的代码出发。

## 工作流

1. 选择一个最小行为切片。
   来源可以是：
   - `spec.md`
   - `verification.md`
   - a reproduced bug scenario

2. 先写失败测试。
   范围要小，意图要明确。

3. 运行测试并确认它确实失败。
   如果一写出来就通过，说明这个测试没有证明行为缺口。

4. 写最小实现，让它通过。

5. 重新运行测试，确认通过。

6. 在保持绿色状态的前提下做必要重构。

7. 跑相关回归检查。

## OpenHarness 对应关系

- `/oh:apply --tdd` 应显式使用这个 skill
- 对 bugfix，应该从已复现的失败场景开始
- 一次处理一个任务切片，不要把整个 feature 变成一个巨大的 RED 步骤

## 证据格式

每个 RED/GREEN 循环至少记录：

- failing test name
- failing output summary
- passing output summary

## 约束

- 选择 TDD 时，在失败测试出现前不要先写实现代码
- 不要把多个无关行为塞进同一个测试循环
- 做重构时必须保持测试持续为绿
