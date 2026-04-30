---
description: "服务仓库 propose Phase 1 总控 agent。生成 proposal、spec、design。"
---

## 任务：执行服务仓库 propose Phase 1

### Step 1: 准备工作

1. 解析 `<name>` 和 `<prd-path>`，任一缺失则提示用户补充
2. 检查 `docs/specs/active/{name}/` 是否已存在：
   - 已存在且含 `design.md` → 提示用户使用 `--continue` 进入 Phase 2
   - 已存在但无 `design.md` → 询问是否覆盖重新生成
3. 创建目录 `docs/specs/active/{name}/`

### 行为纪律（生成前必读）

**三条铁律**：

1. **证据先于声明** — 声称"完成/通过/修复"前，必须展示实际命令输出或代码片段作为证据
2. **暂停先于猜测** — 遇到不确定/不明确/有歧义的情况，停下来与用户确认，不猜测不编造
3. **契约先于发挥** — 严格按 proposal/spec/design 的边界和已加载约束执行，不自行添加/删减

**暂停升级协议** — 遇到不确定时，上报以下状态之一：

- **DONE** — 完成，无异常
- **CONCERNS** — 完成，但有疑虑（列出疑虑，让用户判断）
- **BLOCKED** — 无法继续（说明原因，等待指示）
- **NEEDS_CONTEXT** — 缺少信息（列出需要的信息，等待补充）

> 上报 BLOCKED 永远好过猜测后出错。错误的实现比没有实现更糟糕。

#### propose 专属红旗

| 出现这个想法时 | 正确做法 |
|--------------|---------|
| "原始需求没提到但这个场景肯定需要" | 标注 `<!-- TODO: 待确认 -->`，不自行补充 |
| "这个技术方案比 knowledge/ 中的更好" | 按现有架构设计，改进建议放到 design 附录 |
| "行为契约还不够具体，我来脑补" | 列出待澄清项，让用户确认 |
| "原始需求没说清楚，我推断是这个意思" | 标注待确认，不推断 |

### Step 2: 生成 proposal.md

使用 Agent tool 调用 `~/.claude/skills/openharness/agents/prd-parser.md`（安装后重写为 `{install-dir}/commands/oh/agents/prd-parser.md`）。

传递以下上下文：

- 产品文档路径：`{prd-path}`
- feature 目录：`docs/specs/active/{name}/`
- 目标工件：`proposal.md`

agent 产出应收敛为 `docs/specs/active/{name}/proposal.md`（含结构化需求背景、用户角色、关键旅程、范围、风险、待澄清项、信息来源追溯）。

生成 `proposal.md` 时，遵循以下分析纪律：

- 先读 `docs/knowledge/index.md`，再进入 `service-boundary.md` 等稳定入口建立知识地图
- 结合 `domain-mapping.md`（若存在）识别产品别名与业务域映射，避免把产品词直接当系统术语
- 先做“职责边界扫描”，明确目标领域内谁负责什么，再总结需求影响范围
- 识别涉及的系统/代码库时，只收敛目标领域内的直接改动范围，不把领域外系统写成实现主体
- 从上游触发方/用户角色视角梳理关键旅程，说明触发条件、系统响应、业务价值
- 若发现系统归属、核心流程或上下游流向存在关键歧义，标注待确认；不要把推测写成既定范围

> 若产品文档包含图片，提取的图片保存到 `docs/specs/active/{name}/media/`

### Step 3: 读取知识库与约束体系

并行读取以下内容，**必须在调用 agent 前完成**：

- `docs/knowledge/index.md` — knowledge 总入口，先建立完整知识地图
- 再按方案生成需要进入相关知识索引或稳定入口（如 `project-context.md`、`architecture.md`、`domain-model.md`、`api/index.md`、`database/index.md`、`prd/index.md`、`exp/index.md`、`service-boundary.md`）
- `docs/knowledge/project-context.md` — 项目上下文（必须读取，获取 `project-mode`）
- `docs/knowledge/architecture.md` — 技术架构
- `docs/knowledge/domain-model.md` — 领域模型
- `docs/knowledge/domain-mapping.md` — 产品别名到业务域映射（若存在）
- `docs/knowledge/service-boundary.md` — 服务职责边界稳定入口
- `docs/knowledge/api/index.md` — 接口清单（若存在）
- `docs/knowledge/database/index.md` — 数据库概览（若存在）
- 叶子知识文档按需读取，不得绕过 `docs/knowledge/index.md` 直接凭经验判断系统现状
- `docs/harness/index.md` — harness 总入口，建立完整约束地图
- 依次检查所有子目录索引，并读取每个子目录下的全部叶子规则文件：
  - `docs/harness/invariants/index.md` 及全部叶子规则
  - `docs/harness/architecture/index.md` 及全部叶子规则
  - `docs/harness/infrastructure/index.md` 及全部叶子规则
  - `docs/harness/linters/index.md` 及全部叶子规则

若出现以下任一情况，先条件触发 `brainstorming`，再继续 Step 4-6：

- 原始需求对实现路径有多种合理解释
- `knowledge/` 显示存在多个可落地方案
- 需求影响范围大，直接生成单一路径的 `design.md` 风险较高

触发后至少沉淀以下内容：

- 问题定义
- 2-4 个可执行方案
- 每个方案的复杂度 / 风险 / 兼容性 / 演进性对比
- 推荐方案与仍待确认的问题

这些内容可写入 `design.md` 的附录、问题记录或 TODO 区，而不是单独新建文档。

### Step 4: 生成 spec.md

基于 `proposal.md` 与已加载的 knowledge/harness 约束，生成 `docs/specs/active/{name}/spec.md`。

要求：

- `spec.md` 只依赖 `proposal.md` 收敛行为契约
- 使用接近 OpenSpec 的表达方式：`Purpose`、`Scope`、`Requirements`、`Scenario`、`Constraints`、`Notes`
- `Requirements` 使用 `The system SHALL ...`
- `Scenario` 使用 `GIVEN / WHEN / THEN / AND`
- 不写实现方案、任务拆解或测试执行记录
- 若需求仍存在关键冲突或未收敛项，优先标注 `<!-- TODO: 待确认 -->` 或上报 `NEEDS_CONTEXT`

生成 `spec.md` 时，遵循以下契约分析纪律：

1. 先从 `proposal.md` 中拆解每个关键场景的：
   - 核心业务对象
   - 关键数据动作（读 / 写 / 计算 / 通知）
   - 上游触发方与下游响应结果
2. 对每个场景优先做“复用 / 修改 / 新增能力”判断：
   - 优先复用已有能力
   - 仅在非破坏性扩展可行时写“修改”类要求
   - 只有在职责冲突、破坏性过强或确无现有入口时，才写需要新增的系统能力
3. 做字段/语义级对齐，不要停留在名词表面：
   - 警惕产品文档别名与系统现有术语不一致
   - 区分“字段缺失”和“命名不同但语义相同”
4. 每条 requirement 应尽量体现：
   - 输入/触发条件
   - 系统必须执行的行为
   - 关键约束或业务规则
   - 关键异常或边界条件
5. 若无法从 proposal + knowledge 中确认字段、规则或归属：
   - 在对应 requirement/scenario/notes 标注 `<!-- TODO: 待确认 -->`
   - 不把猜测写成 SHALL

### Step 5: 生成 design.md

基于以下输入，生成 `docs/specs/active/{name}/design.md`：

- `docs/knowledge/project-context.md`
- `docs/specs/active/{name}/proposal.md`
- `docs/specs/active/{name}/spec.md`
- `docs/knowledge/architecture.md`
- `docs/knowledge/domain-model.md`
- `docs/harness/index.md` 与各子目录 `index.md` — 完整 harness 约束体系入口
- 按命中的约束继续读取相关叶子规则文件

**文档格式**：严格参考 `~/.claude/skills/openharness/templates/design.md.tpl` 模板（安装后重写为 `{install-dir}/commands/oh/templates/design.md.tpl`）。

生成时：

- frontmatter 必须包含 `project-mode: greenfield|brownfield`，值来自 `docs/knowledge/project-context.md`
- frontmatter 必须包含 `approval: pending`
- 可选章节（架构图/流程图/数据迁移）— 基于需求判断是否生成内容，无需则保留标题注明“不涉及”
- 若 `project-mode: greenfield`：
  - 优先写“首版基线架构选择、模块拆分、扩展边界、运维落地方式”
  - `影响评估` 重点写“后续演进兼容性、版本演进策略、数据初始化/冷启动策略”
  - 所有“对已有系统/老数据/原有逻辑”的检查项统一填 `不涉及（greenfield）`，不得伪造存量影响
- 若 `project-mode: brownfield`：
  - 优先写“兼容性、迁移成本、影响面、灰度/回滚方案”

### Step 6: 输出报告并自动进入 review

```text
## /oh:propose Phase 1 完成 — {name}

### 已生成
- [x] docs/specs/active/{name}/proposal.md
- [x] docs/specs/active/{name}/spec.md
- [x] docs/specs/active/{name}/design.md（approval: pending）

### 自动执行
- [x] 自动触发 /oh:review
- review scope: proposal.md / spec.md / design.md
- 不读取也不要求 tasks.md / verification.md / code

### 下一步
1. 查看 review 报告并按需调整 proposal/spec/design
2. review 通过后执行 /oh:propose {name} --continue 生成执行文档
```
