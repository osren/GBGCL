---
name: writing-plans
description: 把已经达成一致的方向转成可执行的计划，包含步骤顺序、依赖、验证点和明确停止条件。适用于用户明确要 plan，或较大工作需要在 Claude Code 中做结构化拆解时。
---

# 编写计划

当任务已经大到“边做边想”容易出错时，使用这个 skill。

适合触发的场景：

- the user explicitly asks for a plan
- the work spans multiple files or phases
- validation and sequencing matter
- there are blockers, dependencies, or handoff points

## 目标

产出一份简洁、可执行、且便于跟踪的计划。

## 工作流

1. 定义目标结果。
   最终状态必须是可观察、可判断的。

2. 识别约束和依赖。
   包括：
   - required inputs
   - external blockers
   - risky assumptions

3. 把工作拆成具体步骤。
   每一步都应该有可见结果。

4. 增加验证点。
   对每个主要步骤，明确成功如何验证。

5. 标记可能的分支点。
   指出如果某个检查失败，计划在哪里会改变。

6. 保持计划精简。
   一般 3-7 步已经足够。

## Claude Code 对应关系

- 如果 `update_plan` 可用，优先把计划写进去
- 同时最多只保留一个 `in_progress`
- 步骤应对应真实工作单元，不要写成抽象阶段口号
- 当新事实改变了顺序时，要更新计划，不要默默偏航

## 输出格式

- Outcome:
- Constraints:
- Step 1:
- Step 2:
- Step 3:
- Validation:
- Risks:

## 约束

- 任务很简单时，不要强行写计划
- 不要用显而易见的废话把计划写长
- 不要隐藏高风险假设
- 如果写完计划后“下一步做什么”仍不清楚，那这个计划就是失败的
