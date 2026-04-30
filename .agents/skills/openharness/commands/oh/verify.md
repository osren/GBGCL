---
description: "独立验收命令。apply 完成后，以独立视角验证实现是否满足 spec / design / verification 契约。不看实现过程，只看最终结果。"
---

## User Input

```text
$ARGUMENTS
```

## 核心原则

**Generator 和 Verifier 必须分离。**
- Generator（apply）有"完成任务"的动机，倾向于说"做完了"
- Verifier 有"找问题"的动机，专门验证最终结果
- 你是独立验收方，你的工作是**找出所有问题**，不是确认一切正常

## Feature 上下文解析

按以下统一规则定位目标 feature：

- 用户显式传入 `[feature]` → 直接使用该 feature
- 未传入 `[feature]` 且 `docs/specs/active/` 下只有一个活跃 feature → 直接使用该 feature
- 未传入 `[feature]` 且存在多个活跃 feature → 列出候选并让用户选择
- 未传入 `[feature]` 且无活跃 feature → 终止并提示用户先创建或指定 feature

## 内置 Skills 编排协议

执行 `/oh:verify` 时，以下 skills 属于固定编排：

| Skill | 触发时机 | 必须产出的记录 |
|------|---------|--------------|
| `verification-before-completion` | 默认启用；每次准备给出“通过/不通过”结论时 | `verify-report.md` 中的实际命令、关键输出、结论依据 |
| `systematic-debugging` | 发现失败、回归、spec/result 不一致、影响面不一致时 | 五行调试摘要：Expected / Actual / Reproduction / Root cause / Verified fix |
| `receiving-code-review` | 本次 verify 是对上一轮 finding 修复后的复验时 | finding 对账表：逐条处理结果、涉及工件、重点复验区域 |

## 执行流程

### Step 1: 加载上下文（只读契约和结果，不读实现过程叙事）

读取以下文件，这是**独立验收标准**：

- `docs/specs/active/{feature}/spec.md` — 行为契约与验收标准
- `docs/specs/active/{feature}/design.md` — 技术方案与影响面预期
- `docs/specs/active/{feature}/tasks.md` — 检查任务完成状态与 `exec-done` checkpoint
- `docs/specs/active/{feature}/verification.md` — 轻量交付/证据门禁与测试要求
- `docs/knowledge/index.md` — knowledge 总入口，先建立完整知识地图
- 再按验收需要进入相关知识索引或稳定入口（如 `project-context.md`、`architecture.md`、`domain-model.md`、`api/index.md`、`database/index.md`、`prd/index.md`、`exp/index.md`、`service-boundary.md`）

**前置检查**：如果 `tasks.md` 中没有 `exec-done` checkpoint，终止执行并提示：
```
⛔ 未找到 exec-done checkpoint，/oh:verify 需在 /oh:apply 完成后执行。
```

**不要读取 apply 过程中的对话或日志**——独立验收只看契约和最终结果。

若本次 verify 是“对上一轮 finding 修复后的复验”，先用 `receiving-code-review` skill 检查：
- finding 是否有明确接受/拒绝/延期决策
- 修复是否映射到代码和文档
- 是否指定了需要重点复验的区域

### 行为纪律（验收前必读）

**三条铁律**：
1. **证据先于声明** — 声称"完成/通过/修复"前，必须展示实际命令输出或代码片段作为证据
2. **暂停先于猜测** — 遇到不确定/不明确/有歧义的情况，停下来与用户确认，不猜测不编造
3. **契约先于发挥** — 严格按 design/spec/verification 中的契约执行，不自行添加/删减

**暂停升级协议** — 遇到不确定时，上报以下状态之一：
- **DONE** — 完成，无异常
- **CONCERNS** — 完成，但有疑虑（列出疑虑，让用户判断）
- **BLOCKED** — 无法继续（说明原因，等待指示）
- **NEEDS_CONTEXT** — 缺少信息（列出需要的信息，等待补充）

### Step 2: 加载约束标准

加载以下约束入口作为评判依据：

- `docs/harness/index.md`
- 依次检查各子目录索引，并读取全部叶子规则文件：
  - `docs/harness/invariants/index.md` 及全部叶子规则
  - `docs/harness/architecture/index.md` 及全部叶子规则
  - `docs/harness/infrastructure/index.md` 及全部叶子规则
  - `docs/harness/linters/index.md` 及全部叶子规则
- `docs/specs/active/{feature}/verification.md`

### Step 3: 独立验收维度

按以下维度进行独立验收：

1. **Specs 覆盖率**
   - 逐条对照 `spec.md` 中的场景与 requirements
   - 判断实际实现和测试是否覆盖契约
   - 输出 `COVERED / PARTIAL / MISSING / WRONG`

2. **编译和测试**
   - 运行实际编译/测试命令
   - 检查结果是否与 `verification.md`、代码现状一致
   - 若项目配置了覆盖率报告，检查是否满足门禁要求

3. **代码质量与约束合规**
   - 检查命名、异常处理、日志、分层规范、依赖方向
   - 检查 invariants 与架构/基础设施规则是否被违反

4. **影响面验证**
   - 从 `design.md` 中读取预期影响范围
   - 对照实际改动判断是否存在遗漏或超范围修改

5. **集成测试要求**
   - 若 `design.md` 要求集成测试，则检查是否存在且通过

若出现失败、场景无法映射、影响面不一致或证据冲突，使用 `systematic-debugging` 记录根因，不得直接给笼统结论。

若本次 verify 是对上一轮 finding 的复验，`verify-report.md` 必须包含 finding 对账结果；若 `verification.md` 中的证据与本次实际测试输出不一致，需单列为独立问题，而不是并入一般测试失败。

### Step 4: 生成独立验收报告

生成 `docs/specs/active/{feature}/verify-report.md`：

```markdown
# 独立验收报告：{feature-name}

> 验收时间：{YYYY-MM-DD HH:MM}
> 验收结果：**通过 / 不通过**

## finding 对账（如有）
| finding | 决策 | 涉及工件 | 重点复验区域 | 结果 |
|---------|------|---------|-------------|------|
| F-001 | accept / reject / defer | code / spec / verification / tasks | {模块/文件} | {验证结果} |

## 总览
| 维度 | 结果 | 说明 |
|------|------|------|
| Specs 覆盖率 | ✅ / ❌ | N/M 条场景满足 |
| 编译和测试 | ✅ / ❌ | 全量测试 X passed, Y failed |
| 代码质量 | ✅ / ❌ | N BLOCKER, M MAJOR, K MINOR |
| 影响面验证 | ✅ / ❌ | 实际改动与 design.md 一致 |
| 集成测试 | ✅ / ❌ / ⏭ 跳过 | 核心链路覆盖 |

## 关键问题
- [BLOCKER] {文件:行号} {问题描述}
- [MAJOR] {文件:行号} {问题描述}

## 验证证据
- 编译命令：`{命令}`
- 测试命令：`{命令}`
- 关键输出：{摘要}
```

### Step 5: 输出结果并给出行动建议

#### 如果验收通过：
```
独立验收通过 ✅

- Specs 覆盖率：100% (N/N)
- 全量测试：全部通过 (X tests)
- 代码质量：无 BLOCKER/MAJOR 问题
- 影响面：与 design.md 一致

验收报告：docs/specs/active/{feature}/verify-report.md
可以执行 /oh:archive 归档此需求。
```

#### 如果验收不通过：
```
独立验收不通过 ❌

需要修复的问题：
1. [Specs] N 条场景缺少对应测试
2. [测试] N 个测试用例失败
3. [代码] N 个 BLOCKER 问题

验收报告：docs/specs/active/{feature}/verify-report.md

修复后重新运行 /oh:verify {feature} 验证。
```

若输出 finding 给实现者，要求其后续使用 `receiving-code-review` skill 处理反馈闭环。

## 通过标准

| 维度 | 通过条件 |
|------|---------|
| Specs 覆盖率 | 所有关键场景状态为 COVERED，无 MISSING 或 WRONG |
| 编译和测试 | 编译通过 + 全量测试无失败 |
| 代码质量 | 无 BLOCKER 问题 |
| 影响面验证 | 无遗漏（超范围标注但不阻塞） |
| 集成测试 | 如果 design.md 要求了，则必须存在且通过 |

**MAJOR 问题不阻塞通过，但会在报告中列出，建议修复。**

## Guardrails

- 必须独立判断，**不要因为 apply 说"已完成"就认为没问题**
- 逐条检查关键场景，不跳过任何一条
- 代码质量审查基于实际文件内容，不基于实现报告
- 如果 spec.md 本身有问题（模糊、矛盾），在报告中标注，但不影响评估其他维度
- 验收报告是事实记录，不美化、不淡化问题
- 每次验收都重新运行测试，不信任之前的测试结果
