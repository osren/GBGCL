---
name: brainstorming
description: 在实现前生成并比较具体可落地的方案选项。适用于问题仍有歧义、存在多种可行路径，或用户希望在 Claude Code 中看到 tradeoff、选项空间与结构化方案探索时。
---

# 方案头脑风暴

在正式承诺某一条实现路径前使用这个 skill。

适合触发的场景：

- requirements are still fuzzy
- more than one design is plausible
- the user asks for options, tradeoffs, or recommendations
- a risky change needs alternatives before execution

如果执行已经明确开始，就不要再用它，除非你确实需要重新打开方案空间。

## 目标

先扩展选项空间，再收敛到一个站得住脚的推荐路径。

## 工作流

1. 定义这次要做的决策。
   写清楚：
   - the problem to solve
   - hard constraints
   - what would make an option unacceptable

2. 生成 2-4 个真实可实施的选项。
   每个选项都必须能落地，不要只是口号。

3. 从实务维度比较方案。
   优先比较：
   - complexity
   - delivery speed
   - risk of regression
   - migration cost
   - operability
   - fit with existing architecture

4. 暴露未知项。
   明确指出哪些事情必须在执行前确认。

5. 给出一个推荐方案。
   解释为什么它是“当前最适合”的，而不是抽象上最优。

## Claude Code 对应关系

- 在写计划或写代码前使用
- 输出要简洁，并且面向决策
- 如果代码库上下文重要，先读相关文件，再提方案
- 如果用户希望继续推进，最后要收敛到一个能直接进入 plan 或 execution 的推荐路径

## 输出格式

- Problem:
- Constraints:
- Option A:
- Option B:
- Option C: (optional)
- Recommendation:
- Open questions:

## 约束

- 如果明显只有一条路，不要硬凑很多选项
- 不要默认推荐重写
- 不要用模糊语言掩盖关键 tradeoff
- 一旦方向清楚，就停止 brainstorm，进入下一步
