---
name: systematic-debugging
description: 用“复现-缩圈-假设-验证”的流程调试预期与实际不一致的问题。适用于 apply、review、verify 中发现失败、回归或异常行为的场景。
---

# 系统化调试

只有在真实不一致已经出现后，才使用这个 skill：

- expected behavior != actual behavior
- test fails
- compile/runtime error appears
- review/verify finds a regression or inconsistency

不要把这个 skill 用于模糊的设计讨论。那属于 planning，不属于 debugging。

## 目标

基于证据找到根因，再用回归检查确认修复真实有效。

## 工作流

1. 明确不一致是什么。
   写清楚：
   - expected behavior
   - actual behavior
   - where it was observed

2. 稳定复现。
   优先用一个确定性的命令、请求或测试来复现。
   如果复现不稳定，要明确记录。

3. 缩小范围。
   检查：
   - input data
   - call path
   - changed files
   - environment/config assumptions

4. 提出假设。
   假设必须具体且可证伪。
   好的例子：
   - "null value enters mapper X and breaks branch Y"
   不好的例子：
   - "something is wrong with the service"

5. 执行有区分度的检查。
   选择能排除某个假设的下一条命令或检查。

6. 确认根因。
   区分清楚：
   - symptom
   - trigger
   - root cause

7. 做最小且合理的修复。
   除非根因要求，否则不要顺手做大范围重构。

8. 重新执行复现和回归检查。
   不重新验证，修复就不算完成。

## OpenHarness 对应关系

- `/oh:apply`：实现过程被阻塞，或任务出现回归时使用
- `/oh:review`：文档、代码、功能验证中发现异常或不一致时使用
- `/oh:verify`：独立验收中发现契约与结果不一致时使用

## 输出格式

用五行摘要输出调试结果：

- Expected:
- Actual:
- Reproduction:
- Root cause:
- Verified fix:

## 约束

- 在重新跑过失败场景前，不能说“已经修复”
- 在没找到根因前，不要做大范围改写
- 没有证据时，不要把环境问题和代码问题混为一谈
- 如果无法复现，要明确写出来
