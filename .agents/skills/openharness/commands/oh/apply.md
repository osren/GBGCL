---
description: "执行需求实现。前置检查 design.md 的 approval: approved 状态和执行文档齐备。按方案逐步实现，每完成一个阶段更新 tasks.md 并触发 cr-checklist.md 检查。全部完成后自动触发综合 /oh:review。"
---

## User Input

```text
$ARGUMENTS
```

## Feature 上下文解析

按以下统一规则定位目标 feature：

- 用户显式传入 `[feature]` → 直接使用该 feature
- 未传入 `[feature]` 且 `docs/specs/active/` 下只有一个活跃 feature → 直接使用该 feature
- 未传入 `[feature]` 且存在多个活跃 feature → 列出候选并让用户选择
- 未传入 `[feature]` 且无活跃 feature → 终止并提示用户先创建或指定 feature

## 内置 Skills 编排协议

执行 `/oh:apply` 时，以下 skills 属于固定编排，不是“可选参考”：

| Skill | 触发时机 | 必须产出的记录 |
|------|---------|--------------|
| `verification-before-completion` | 默认启用；每次准备把 task、阶段或 `exec-done` 标记为完成时 | `tasks.md` 中的证据块：声明、验证命令、关键输出、变更文件列表 |
| `test-driven-development` | 用户传入 `--tdd` 时 | 每个任务至少一组 RED/GREEN 证据；如有重构，再补 REFACTOR 后回归结果 |
| `systematic-debugging` | 编译失败、测试失败、行为与 `spec.md`/`verification.md` 不一致时 | 五行调试摘要：Expected / Actual / Reproduction / Root cause / Verified fix |
| `receiving-code-review` | 当前任务目标是处理 review/verify/human finding 时 | finding 对账表：接受/拒绝/延期决策、涉及工件、回归验证 |

## 条件 Skills 编排

执行 `/oh:apply` 时，以下 skills 按场景条件触发，不是默认每次都进入：

| Skill | 触发条件 | 使用效果 |
|------|---------|---------|
| `executing-plans` | `tasks.md` 已形成明确计划，且本次实现跨多个阶段、模块或检查点时 | 按步骤推进执行，持续校准当前目标、验证结果与 `tasks.md` 进度状态 |
| `parallel-dispatch` | 存在可安全并行的独立子任务、侧车验证或互不重叠的文件改动时 | 先拆出 delegation plan，再把研究、验证或局部实现分发给并行 agent，同时保留主线在本地推进 |

## 执行流程

### Phase 0: 前置门禁检查（design approval + tasks.md + verification.md，必须通过才能继续）

1. 读取 `docs/specs/active/{feature}/design.md`
2. 检查 frontmatter 中的 `mode` 字段：
   - `mode: hub` → **终止执行**，输出：
     ```
     ⛔ Hub 方案不支持 /oh:apply。
     请将 service-briefs/*.md 分发到各服务仓库，并在对应服务仓库执行普通 /oh:propose / /oh:apply。
     ```
3. 检查 frontmatter 中的 `approval` 字段：
   - `approval: approved` → 继续检查
   - `approval: pending` 或缺失 → **终止执行**，输出：
     ```
     ⛔ Design 待审批，无法开始执行。
     请先执行 /oh:review 并获得人工批准（approval: approved）。
     ```
4. 检查以下执行材料是否存在：
   - `docs/specs/active/{feature}/tasks.md`
   - `docs/specs/active/{feature}/verification.md`
   - 若任一缺失 → **终止执行**，输出：
     ```
     ⛔ 缺少执行必需材料，无法开始执行。
     请先执行 /oh:propose <feature> --continue 生成 tasks.md 与 verification.md，
     再重新执行 /oh:apply。
     ```

### 行为纪律（执行前必读）

**三条铁律**：
1. **证据先于声明** — 声称"完成/通过/修复"前，必须展示实际命令输出或代码片段作为证据
2. **暂停先于猜测** — 遇到不确定/不明确/有歧义的情况，停下来与用户确认，不猜测不编造
3. **契约先于发挥** — 严格按 design/spec/verification 中的契约执行，不自行添加/删减

**暂停升级协议** — 遇到不确定时，上报以下状态之一：
- **DONE** — 完成，无异常
- **CONCERNS** — 完成，但有疑虑（列出疑虑，让用户判断）
- **BLOCKED** — 无法继续（说明原因，等待指示）
- **NEEDS_CONTEXT** — 缺少信息（列出需要的信息，等待补充）

#### apply 专属红旗

| 出现这个想法时 | 正确做法 |
|--------------|---------|
| "这个功能 design 没提到但显然需要" | NEEDS_CONTEXT，与用户确认 |
| "这段代码可以顺便优化一下" | 不做，只实现 design 内容 |
| "测试先跳过，后面 review 会看" | 不跳过，每个阶段后立即验证 |
| "这个模块和 design 略有不同但更好" | 按 design 实现，改进建议记录到 CONCERNS |
| "差不多完成了，剩下的细节不重要" | 检查 tasks.md 每一项，遗漏就是未完成 |

### Phase 1: 读取执行材料

读取以下文档（**严格按照这些内容执行，不自行发挥**）：

- `docs/specs/active/{feature}/design.md`
- `docs/specs/active/{feature}/tasks.md`
- `docs/specs/active/{feature}/spec.md`
- `docs/specs/active/{feature}/verification.md`
- `docs/knowledge/index.md`
- 再按实现需要进入相关知识索引或稳定入口
- `docs/harness/index.md`
- 依次检查各子目录索引，并读取每个子目录下的全部叶子规则文件：
  - `docs/harness/invariants/index.md` 及全部叶子规则
  - `docs/harness/architecture/index.md` 及全部叶子规则
  - `docs/harness/infrastructure/index.md` 及全部叶子规则
  - `docs/harness/linters/index.md` 及全部叶子规则

### Phase 2: 按任务逐步实现

按 `tasks.md` 中的任务顺序执行：

1. 若任务来自上一轮 review/verify/human finding，先进入 `receiving-code-review`
2. 标记任务为进行中
3. 若启用 `--tdd`，先执行 RED/GREEN/REFACTOR
4. 按 design.md 方案实现
5. 如发现 design.md 有遗漏或不明确，上报 **NEEDS_CONTEXT**，不得自行决策
6. 若实现过程中出现失败、回归或行为不一致，立即切换 `systematic-debugging`
7. 准备将任务标记完成前，必须进入 `verification-before-completion`
8. 检查本任务对应的验证证据是否已记录到 `tasks.md` 或 `verification.md`
9. 若本任务来自 review / verify / human finding，检查 finding 对账记录是否已补齐
10. 完成任务后更新 `tasks.md` checkpoint 与证据

每完成一个功能模块（如数据层、业务层、接口层），读取并执行 `docs/hooks/cr-checklist.md` 中的检查项；该检查既包含代码规范，也包含 tasks checkpoint、verification 证据、finding 对账等闭环项。

### Phase 3: 全部完成后的自动收尾

所有任务完成后，进入 `exec-done` 之前必须先做一致性检查：

1. `tasks.md` 中所有任务状态均已完成
2. `verification.md` 已记录本轮执行证据，且关键测试/约束检查结果可回溯
3. 若本轮是在处理 review / verify / human finding，finding 对账结果已补齐
4. 不存在未解释的 `CONCERNS / BLOCKED / NEEDS_CONTEXT`

上述检查满足后，**自动依次执行**（无需用户手动触发）：

1. 更新 `tasks.md` — 标记所有任务完成
2. 更新 `tasks.md` — 记录 `exec-done` checkpoint
3. 自动执行 `/oh:review`（apply 后综合 review）

### Phase 4: 输出执行报告

```
## /oh:apply 完成 — {feature-name}

### 实现摘要
- 完成任务：{N} 个
- 新增文件：{列表}
- 修改文件：{列表}

### review 结果
- ✅ 通过 → 可执行 /oh:verify 进行独立验收，或直接 /oh:archive 归档
- ❌ 不通过 → 查看评审报告，按 finding 修复后重新执行 /oh:review
  或执行 /oh:rollback 回退到指定 checkpoint
```

### TDD 模式（可选）

若用户指定 `--tdd`（如 `/oh:apply feature-name --tdd`），每个任务按以下顺序执行：

1. **RED** — 根据 verification.md 中对应的行为切片，先编写失败的测试
2. 运行测试，确认测试**确实失败**（展示输出作为证据）
3. **GREEN** — 编写最少量代码使测试通过
4. 运行测试，确认测试**通过**（展示输出作为证据）
5. **REFACTOR** — 重构代码（保持测试通过）
6. 更新 tasks.md，并在需要时补充 verification.md 证据

## 约束

- **不修改 harness/ 和 knowledge/** — 这些文件在 exec 阶段只读
- **每个阶段后更新 tasks.md** — 便于 rollback 时定位 checkpoint
- **apply 自动收尾只触发 `/oh:review`** — 独立验收由 `/oh:verify` 单独负责
