---
name: requesting-code-review
description: 在发起评审前，用自检清单、证据包和清晰上下文整理代码或设计变更。适用于调用 OpenHarness review 或请其他 agent / 人工评审前。
---

# 发起代码评审

在请求评审前使用这个 skill。

目的不是替代码辩护，而是减少 reviewer 的理解歧义。

## 目标

向 reviewer 提供一份边界清楚、范围明确、证据充分的评审包。

## 工作流

1. 说明这次 review 的目标。
   - Design approval
   - code final review
   - focused follow-up after fixes

2. 总结范围。
   至少包括：
   - feature name
   - files changed
   - intended behavior
   - known risks

3. 先自检。
   检查：
   - matches `spec.md`
   - matches `design.md`
   - tasks/verification are up to date
   - obvious harness violations are addressed

4. 附上证据。
   优先附：
   - compile/test output
   - verification results
   - focused diff summary

5. 明确指出 reviewer 应重点关注的地方。
   例如：
   - boundary handling
   - migration safety
   - performance assumptions
   - greenfield architecture baseline

6. 明确列出仍未解决的问题。
   不要把不确定性藏在模糊表述里。

## OpenHarness 对应关系

- 在 `/oh:review` 的 Phase 1 / Design 审批前：整理 `proposal.md / spec.md / design.md`
- 在 `/oh:review` 的 apply 后综合 review 前：整理 spec / design / code / test 证据
- 在交给人工 review 前：给出范围清晰的摘要，而不是只说一句“帮我看看”

## 约束

- 不要在 `tasks/progress` 还半更新状态时就发起评审
- 不要为了显得干净而省略已知风险
- 没有证据时，不要说“已经完全验证”
