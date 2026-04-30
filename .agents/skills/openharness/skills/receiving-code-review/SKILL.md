---
name: receiving-code-review
description: 用有纪律的分流、接受决策、定向修复和重新验证来处理评审反馈。适用于 OpenHarness review、verify 或人工 review 返回 finding / 问题的场景。
---

# 处理代码评审反馈

当以下来源返回 finding 后，使用这个 skill：

- `/oh:review`
- `/oh:verify`
- human review
- another agent's review pass

## 目标

把 finding 转成明确决策和经过验证的后续动作，而不是模糊地说一句“已处理”。

## 工作流

1. 标准化 finding。
   对每一项记录：
   - source
   - severity
   - file/area
   - claimed issue

2. 给每项 finding 分类。
   - accept
   - reject with reason
   - defer with reason
   - needs clarification

3. 把已接受的 finding 映射到工件。
   判断以下哪些需要更新：
   - code
   - `design.md`
   - `spec.md`
   - `verification.md`
   - `tasks.md`

4. 一次修一批有内聚性的内容。
   保持每轮修复范围清晰，不要把无关修改混进来。

5. 重新验证已接受的修复。
   先跑最小但足够有说服力的回归检查，必要时再扩大。

6. 带证据回复。
   对每个已接受的 finding 说明：
   - what changed
   - where it changed
   - what verification was rerun

## OpenHarness 对应关系

- 设计评审反馈：可能需要在审批前更新 `design.md`
- apply 后综合 review 的 finding：可能需要在归档前同时更新代码和验证
- verify 的 finding：通常既要修代码，也要修正 spec / design 的一致性

## 约束

- 没有决策记录时，不要悄悄吸收 review 反馈
- 没有具体修改或明确拒绝时，不要说“已解决”
- 如果 finding 暴露的是 spec / design 问题，就更新文档工件，不要盲目补代码
