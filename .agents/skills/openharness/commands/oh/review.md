---
description: "综合 review 命令。统一承担四类场景：Phase 1 文档 review、Phase 2 文档 review、服务仓库 Design 审批 / Hub 方案审批、apply 后综合 review。"
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

执行 `/oh:review` 时，以下 skills 属于固定编排：

| Skill | 触发时机 | 必须产出的记录 |
|------|---------|--------------|
| `requesting-code-review` | 默认启用；正式启动评审 Agent 前 | 送审包摘要：review 目标、scope、证据、已知风险、未决问题 |
| `receiving-code-review` | 上一轮 review/verify/human finding 修复后重新送审时 | finding 对账表：接受/拒绝/延期决策、对应改动、重新验证结果 |
| `verification-before-completion` | 评审过程中遇到“已修复/已实现/已验证”声明时 | 将无证据声明降级为“未验证”，或要求补充新鲜证据后再判断 |

## 条件 Skills 编排

执行 `/oh:review` 时，以下 skill 按场景条件触发：

| Skill | 触发条件 | 使用效果 |
|------|---------|---------|
| `parallel-dispatch` | review 范围较大，除产品/技术/质量三视角外，还需要独立的证据收集、专项检查或额外只读分析时 | 先拆出互不重叠的并行 review 子任务，明确每个子任务的关注点、输入材料和回收方式 |

## review 场景判断

根据 feature 当前状态自动判断 review 场景：

| 场景 | 判断条件 | review scope | 参与 Agent |
|------|---------|-------------|------------|
| **Phase 1 文档 review** | 已生成 `proposal.md / spec.md / design.md`，且尚无 `tasks.md / verification.md` | proposal/spec/design | 产品专家 + 技术专家 |
| **Phase 2 文档 review** | 已生成 `tasks.md / verification.md`，且无 `exec-done` checkpoint | tasks/verification 及其依赖文档 | 产品专家 + 技术专家 |
| **Design 审批** | `design.md` 中 `approval: pending`，且 review 目标是审批 | proposal/spec/design | 产品专家 + 技术专家 |
| **Hub 方案审批** | `design.md` frontmatter 中 `mode: hub`，且 `approval: pending` | proposal/design/service-briefs | 产品专家 + 技术专家 |
| **apply 后综合 review** | `tasks.md` 有 `exec-done` checkpoint | 文档 + 代码 + 功能验证 + 门禁检查 | 产品专家 + 技术专家 + 质量专家 |

若无法自动判断，询问用户当前是哪种 review 场景。

## 执行流程

### Step 1: 读取 review 材料

按场景读取不同材料：

**Phase 1 文档 review / Design 审批**：
- `docs/specs/active/{feature}/proposal.md`
- `docs/specs/active/{feature}/spec.md`
- `docs/specs/active/{feature}/design.md`
- `docs/knowledge/index.md`
- 再按 review 需要进入相关知识索引或稳定入口
- `docs/knowledge/project-context.md`
- `docs/knowledge/architecture.md`
- `docs/harness/index.md`
- 依次检查各子目录索引，并读取全部叶子规则文件：
  - `docs/harness/invariants/index.md` 及全部叶子规则
  - `docs/harness/architecture/index.md` 及全部叶子规则
  - `docs/harness/infrastructure/index.md` 及全部叶子规则
  - `docs/harness/linters/index.md` 及全部叶子规则

**Phase 2 文档 review**：
- `docs/specs/active/{feature}/proposal.md`
- `docs/specs/active/{feature}/spec.md`
- `docs/specs/active/{feature}/design.md`
- `docs/specs/active/{feature}/tasks.md`
- `docs/specs/active/{feature}/verification.md`
- `docs/knowledge/index.md`
- `docs/harness/index.md`
- 依次检查各子目录索引，并读取全部叶子规则文件

**Hub 方案审批**：
- `docs/specs/active/{feature}/proposal.md`
- `docs/specs/active/{feature}/design.md`
- `docs/specs/active/{feature}/service-briefs/`
- `docs/knowledge/index.md`
- 再按 Hub 评审需要进入相关知识入口与叶子文档

**apply 后综合 review**：
- `docs/specs/active/{feature}/proposal.md`
- `docs/specs/active/{feature}/spec.md`
- `docs/specs/active/{feature}/design.md`
- `docs/specs/active/{feature}/tasks.md`
- `docs/specs/active/{feature}/verification.md`
- `docs/knowledge/index.md`
- `docs/harness/index.md`
- 依次检查各子目录索引，并读取全部叶子规则文件
- feature 相关代码变更文件
- 必要时执行功能测试 / 边界测试 / 异常测试 / invariants 与架构约束检查

## 行为纪律（review 前必读）

**三条铁律**：
1. **证据先于声明** — 声称"完成/通过/修复"前，必须展示实际命令输出或代码片段作为证据
2. **暂停先于猜测** — 遇到不确定/不明确/有歧义的情况，停下来与用户确认，不猜测不编造
3. **契约先于发挥** — 严格按 design/spec/verification 中的契约执行，不自行添加/删减

#### review 独立性原则

> **"不信任报告，只信任代码和当前工件。"**

- 文档 review 只 review 当前阶段已生成工件，不得把未生成工件当成失败理由
- apply 后综合 review 不信任 apply 阶段的任何完成声明，必须独立验证
- 发现的每个问题必须附带具体文件路径和行号
- 对所有“已完成/已修复/已通过”的声明，按 `verification-before-completion` 的标准要求证据

### Step 2: 启动多 Agent 并行 review

#### 产品专家 Agent

关注：
- proposal/spec/design 是否准确表达需求与验收标准
- tasks/verification 是否覆盖关键执行准备与验收路径
- apply 后功能与边界场景是否满足契约

#### 技术专家 Agent

关注：
- design 方案是否合理可行
- tasks 与 verification 是否与 design 一致
- 代码是否符合架构约束、业务不变量、技术规范
- apply 后实现是否与 design.md 方案一致

#### 质量专家 Agent（仅 apply 后综合 review）

关注：
- 代码规范、测试覆盖、异常处理、安全问题
- `verification.md` 中的轻量交付门禁与证据门禁是否满足
- 必要时独立执行关键功能验证

### Step 3: 汇总 review 报告

```markdown
## review 报告 — {feature-name}
> review 时间：{YYYY-MM-DD}
> review 场景：Phase 1 文档 review / Phase 2 文档 review / Design 审批 / Hub 方案审批 / apply 后综合 review

### 送审包摘要
- review 目标：{目标}
- scope：{文档/代码/测试/模块范围}
- 证据：{命令与输出摘要}
- 风险与未决问题：{列表}

### finding 对账（复审时）
| finding | 决策 | 涉及工件 | 重新验证 |
|---------|------|---------|---------|
| F-001 | accept / reject / defer | code / spec / verification / tasks | {命令或证据} |

### 产品专家 review
- 需求理解准确性：✅/⚠️/❌
- 验收标准完整性：✅/⚠️/❌
- 遗漏场景：{如有，列出}

### 技术专家 review
- 方案合理性：✅/⚠️/❌
- 架构合规性：✅/⚠️/❌
- Invariants 合规：{规则编号与问题}

### 质量专家 review（apply 后综合 review）
- 代码规范：✅/⚠️/❌
- 功能验证：✅/⚠️/❌
- 安全检查：✅/⚠️/❌

### 改进建议汇总
| 序号 | 来源 | 问题 | 建议操作 | 是否采纳 |
|------|------|------|---------|---------|
| 1 | 技术专家 | {问题} | {建议} | [ ] |

### 结论
- ✅ 通过 → {下一步说明}
- ❌ 不通过 → 请按改进建议调整后重新执行 /oh:review
```

### Step 4: 处理结果

**Phase 1 文档 review**：
- 输出文档问题与改进建议
- 若需要正式放行进入下一阶段，则继续执行 Design 审批逻辑

**Phase 2 文档 review**：
- 输出执行准备问题与改进建议
- 通过后可进入 `/oh:apply`

**Design 审批场景**：
- **通过**：在 `design.md` frontmatter 中设置 `approval: approved`，并明确提示用户先执行 `/oh:propose <feature> --continue`，再执行 `/oh:apply`
- **不通过**：保持 `approval: pending`，等待用户调整后重新评审

**Hub 方案审批场景**：
- **通过**：在 `design.md` frontmatter 中设置 `approval: approved`
- **不通过**：保持 `approval: pending`
- **通过后提示**：将 `service-briefs/*.md` 分发到各服务仓库，继续执行普通 `/oh:propose`

**apply 后综合 review**：
- **通过**：仅当 `verification.md` 中的轻量交付门禁与证据门禁已满足时，提示可以直接 `/oh:archive`，或执行 `/oh:verify` 做独立验收
- **不通过**：先向用户展示 finding，询问是否采纳；采纳后再更新相关文档并修复代码，之后重新执行 `/oh:review`

## 约束

- **并行 review** — 多 Agent 同时执行，提高效率
- **改进建议需人工确认** — 不自动执行改进，等待用户确认后再调整
- **阶段性 review 不读未来工件** — 只能基于当前阶段已存在的工件判断
