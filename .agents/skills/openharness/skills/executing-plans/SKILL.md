---
name: executing-plans
description: 在已有明确范围后，按计划逐步执行，并配合进度跟踪、验证和受控调整。适用于工作已经定 scope，接下来主要需要在 Claude Code 中做纪律化执行时。
---

# 执行计划

当方向或计划已经确定后，使用这个 skill。

适合触发的场景：

- a written plan exists
- the user says to proceed
- work needs checkpointed execution instead of more design discussion

不要用它来代替 planning。如果下一步还不清楚，应先回到 planning。

## 目标

把计划变成经过验证的进度推进，同时不丢状态、不偏 scope。

## 工作流

1. 重申当前目标。
   确认现在执行的是哪一个 plan step。

2. 一次执行一个有内聚性的步骤。
   不要把无关改动混在同一轮里。

3. 每个关键步骤后都做验证。
   先做最小但足够有说服力的检查，再根据风险扩大。

4. 更新进度状态。
   记录改了什么、验证了什么、还剩什么。

5. 显式处理偏差。
   如果某一步失败，只能选择以下路径之一：
   - debug
   - revise the plan
   - surface a blocker

6. 用真实状态收尾。
   使用：
   - `DONE`
   - `CONCERNS`
   - `BLOCKED`
   - `NEEDS_CONTEXT`

## Claude Code 对应关系

- 如果 `update_plan` 可用，就在执行过程中持续维护
- 优先在本地推进下一步关键路径，再把侧车工作委派出去
- 用验证证据代替“应该已经完成了”的判断
- 如果计划发生实质变化，要更新计划，不要悄悄即兴发挥

## 约束

- 检查失败后不要盲目继续
- 不要让实现偏离已约定目标
- 没有验证就不能宣称完成
- 如果执行重新变成设计讨论，就切回 planning 或 brainstorming
