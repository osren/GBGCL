# Open Harness V2：AI Coding 工作范式

> **版本**: v2.0 | **日期**: 2026-04-01
> **定位**: 面向企业级系统的AI辅助开发范式

---

## 一、范式概述

### 1.1 核心理念与环境架构

**"人类驾驭，Agent执行"** — 以约束为边界，以知识库为基础，以规格驱动开发，构建人机协作的软件开发范式。

**Agent 工作环境**：

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent环境架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. KNOWLEDGE 知识库 （Agent的资料库）                         │
│     └── 业务、流程、架构、领域模型、api、db、exp...              │
│                                                             │
│  2. HARNESS 约束（Agent的约束）                             │
│     └── 业务不变量、技术规范、基础设施规范、linters            │
│                                                             │
│  3. SPECS 工作区域（Agent需求管理和开发工作区）                   │
│                                                             │
│  4. HOOKS 审查区 （Agent审查）                                 │
│                                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 需求开发流程

针对每个需求，推荐按以下链路推进：

`/oh:init` → `/oh:propose <name> <prd-path>` → 自动 `/oh:review`（Phase 1 文档）→ `/oh:propose <name> --continue` → 自动 `/oh:review`（Phase 2 文档）→ `/oh:apply` → 自动 `/oh:review`（apply 后综合）→ 可选 `/oh:verify` → `/oh:archive`

其中，`/oh:review` 负责开发侧综合检查：阶段文档 review、审批、apply 后代码 review / 功能验证 / 门禁检查；`/oh:verify` 负责独立验收，从实现者之外的视角再次核对 spec / design / verification / code 是否真实闭环。若综合 review 已通过且本次无需独立验收，也可以直接进入 `/oh:archive`。

#### 阶段拆解

| 阶段 | 命令 | 主要产物 / 动作 | 作用 |
|------|------|----------------|------|
| 项目初始化 | `/oh:init` | 生成 `docs/harness/`、`docs/knowledge/`、`docs/hooks/`、`CLAUDE.md` | 建立约束、知识、轻量 runtime guard 与协作入口 |
| Phase 1 | `/oh:propose <name> <prd-path>` | 生成 `proposal.md`、`spec.md`、`design.md` | 先收敛需求背景、行为契约和实现方案 |
| Phase 1 文档 review / Design 审批 | `/oh:review` | 自动或手动审核 `proposal.md / spec.md / design.md`，审批通过后将 `design.md` 设为 `approval: approved` | 防止需求与方案未收敛就进入执行准备 |
| Phase 2 | `/oh:propose <name> --continue` | 生成 `tasks.md`、`verification.md`，并自动触发 `/oh:review` | 基于已审批的 design + spec 补齐执行与验证文档 |
| 执行落地 | `/oh:apply` | 按 `tasks.md` 实现代码，更新 checkpoint，并自动执行 `/oh:review`（apply 后综合 review） | 将设计转换为实现，并在完成后自动触发代码 review、功能验证与门禁检查 |
| 独立验收（可选/推荐） | `/oh:verify` | 生成 `verify-report.md` | 从独立视角复验 spec 覆盖、测试结果、设计一致性与影响面 |
| 归档沉淀 | `/oh:archive` | 归档 feature，沉淀 proposal 到 `docs/knowledge/prd/`，并给出知识更新建议 | 结束 active 状态并形成可复用知识资产 |

#### 产物边界

| 文档 | 何时生成 | 职责边界 |
|------|---------|---------|
| `proposal.md` | Phase 1 | 说明为什么做、给谁做、业务范围与关键旅程 |
| `spec.md` | Phase 1 | 说明系统必须做什么，沉淀正式行为契约 |
| `design.md` | Phase 1 | 说明准备怎么做，并承担审批主文档角色 |
| `tasks.md` | Phase 2 | 说明如何按步骤执行，并记录 execution status / checkpoints |
| `verification.md` | Phase 2 | 说明如何验证交付、如何记录证据与门禁状态 |

#### 关键控制点

| 控制点 | 说明 |
|--------|------|
| `design.md` 审批门禁 | `approval: approved` 之前不能进入 `/oh:propose --continue` 或 `/oh:apply` |
| Phase 2 生成门禁 | `tasks.md` / `verification.md` 只在 design 审批通过后生成 |
| propose 阶段自动 review | Phase 1 完成后仅 review `proposal/spec/design`；Phase 2 完成后仅 review `tasks/verification` 及其依赖文档 |
| apply 自动收尾 | `/oh:apply` 完成后自动执行 `/oh:review`（apply 后综合 review），无需手动补跑 |
| verify 推荐位点 | `/oh:verify` 不是强制门，但推荐在 archive 前补一次独立验收 |
| archive 归档门禁 | 归档前必须确保 `verification.md` 中的轻量交付门禁与证据门禁满足 |

#### 失败与回退

- 若 Design 审批不通过：先按评审意见调整 `proposal.md / spec.md / design.md`，再重新执行 `/oh:review`
- 若 apply 后综合 review / verify 不通过：先修复问题，再重新执行对应命令；必要时可用 `/oh:rollback` 回退到 `tasks.md` 中记录的 checkpoint

### 1.3 设计原则

1. **知识库驱动** — 架构、领域模型、API、数据库、Proposal 统一管理，需求完成后自动沉淀
2. **约束前置** — 先定义不可违反的规则，在规则边界内执行
3. **审批控制** — `design.md` 必须审批后才能执行，防止理解偏差导致返工
4. **工作区分区** — active（进行中）/ completed（已完成），状态清晰
5. **Hook分层治理** — 通过 runtime hooks、workflow gates、review checklists 三层协同，避免把所有治理都压成底层自动拦截
6. **多Agent评审** — 产品/技术/质量三专家独立评审，全面覆盖
7. **可回退机制** — 支持 rollback，出错时可回退到任意 checkpoint
8. **灵活调整** — 每个阶段生成文档后，人类都可用自然语言随时调整当前阶段内容
9. **Proposal 持久化** — 归档时自动将 Proposal 沉淀到知识库，形成需求资产库

### 1.4 与 Hub 设计稿的关系

`openHarnessV2.md` 是 **Open Harness 的主范式文档**，描述的是默认、基础、面向单服务仓库的工作方式。

`openHarnessHubDesign.md` 是在此基础上的 **Hub 模式扩展设计稿**，用于补齐以下多服务场景能力：

- 集中式文档仓库
- 跨服务知识聚合
- `system-map.md` / `core-flows/*.md` 这类系统视图层
- `/oh:init --hub` 与 `/oh:propose --hub`
- `service-briefs/*.md` 分发到各服务仓库继续执行

两者关系定义如下：

- `openHarnessV2.md` 负责说明主工作流：`/oh:init` → `/oh:propose` → 自动 `/oh:review`（Phase 1 / Phase 2 / apply 后综合）→ 可选 `/oh:verify` → `/oh:archive`
- `openHarnessHubDesign.md` 负责说明多服务扩展：在**不推翻主工作流**的前提下，为集中式文档库补一层“系统视图 + 跨服务方案母稿”
- Hub 模式不是另一套平行范式，而是 Open Harness V2 在多服务协作场景下的补充设计

阅读建议：

- 关注单服务研发流程时，以 `openHarnessV2.md` 为主
- 关注集中式文档仓库、多服务全景图、跨服务方案母稿时，补充阅读 `openHarnessHubDesign.md`

---

## 二、目录结构

```
project-root/
├── CLAUDE.md                        # [必需] Agent总入口（导航地图）
└── docs/                            # [必需] 所有项目文档
    ├── harness/                     # 约束层（Agent必须遵守）
    │   ├── index.md                 # [索引+约束] harness 入口：约束概览 + 子目录索引
    │   ├── invariants/              # 业务不变量（违反→业务错误/资金风险）
    │   │   ├── index.md             # [索引+约束] 不变量入口
    │   │   ├── amt.md               # [约束] 金额规则
    │   │   ├── txn.md               # [约束] 交易规则
    │   │   └── biz.md               # [约束] 业务规则
    │   ├── architecture/            # 架构约束
    │   │   ├── index.md             # [索引+约束] 架构入口
    │   │   └── {module}.md          # [约束] 模块架构
    │   ├── infrastructure/          # 基础设施实现规范
    │   │   ├── index.md             # [索引+约束] 基础设施入口
    │   │   ├── api.md               # [约束] 接口规范
    │   │   ├── database.md          # [约束] 数据库规范
    │   │   ├── cache.md             # [约束] 缓存规范
    │   │   ├── mq.md                # [约束] 消息队列规范
    │   │   ├── security.md          # [约束] 安全规范
    │   │   ├── apollo.md            # [约束] 配置中心规范
    │   │   ├── discovery.md         # [约束] 服务发现规范
    │   │   ├── xxl-job.md           # [约束] 任务调度规范
    │   │   ├── logging.md           # [约束] 日志规范
    │   │   └── build.md             # [约束] 构建规范
    │   └── linters/                 # 自动化检查工具
    │       ├── index.md             # [索引+工具] linters 入口
    │       ├── invariant-check.sh   # [工具] 不变量检查脚本
    │       ├── code-style-check.sh  # [工具] 代码风格检查
    │       └── security-check.sh    # [工具] 安全漏洞检查
    ├── knowledge/                   # 知识库（业务和技术现状 + 经验沉淀）
    │   ├── index.md                 # [索引+知识] 知识库入口：知识概览 + 更新记录
    │   ├── architecture.md          # [知识] 系统架构：分层、模块、依赖
    │   ├── domain-model.md          # [知识] 领域模型：实体、关系、业务流程
    │   ├── prd/                     # Proposal 知识库
    │   │   ├── index.md             # [索引+知识] Proposal 入口：需求索引
    │   │   └── {feature}-proposal.md # [知识] 各功能 Proposal 文档
    │   ├── api/                     # API知识
    │   │   ├── index.md             # [索引+知识] API入口：接口索引
    │   │   └── {service}-api.md     # [知识] 各服务接口文档
    │   ├── database/                # 数据库知识
    │   │   ├── index.md             # [索引+知识] 数据库入口：表索引
    │   │   ├── er.md                # [知识] ER图：实体关系图
    │   │   └── {table}.md           # [知识] 各表结构文档
    │   └── exp/                     # 经验总结
    │       ├── index.md             # [索引+经验] 经验入口：经验索引 + 搜索指南
    │       └── {YYYYMMDD}-{title}.md # [经验] 经验文档
    ├── specs/                       # 工作区域
    │   ├── active/                  # 进行中
    │   │   └── {feature-name}/
    │   │       ├── proposal.md      # [规格] 需求背景：背景、范围、旅程、风险
    │   │       ├── spec.md          # [规格] 行为契约：What
    │   │       ├── design.md        # [规格] 技术方案：How（需审批后执行）
    │   │       ├── tasks.md         # [规格] 执行清单：任务拆解、状态、checkpoint
    │   │       └── verification.md  # [规格] 验证与门禁：测试、证据、交付门禁
    │   └── completed/               # 已完成
    │       └── {feature-name}/
    │           └── ...（同上，归档状态）
    └── hooks/                       # Hook机制
        ├── index.md                 # [索引+配置] Hook入口：Hook配置 + 触发规则
        ├── post-action.md           # [Hook] 动作后检查：每次AI动作后自动执行
        └── cr-checklist.md          # [Checklist] CR检查清单：代码审查要点 + 闭环核对
```

**设计规则**：凡是承担目录总览 + 子文档索引职责的入口文件，统一命名为 `index.md`（如 `harness/index.md`、`knowledge/index.md`），用于承载整体概览与当前目录下其他文件的导航。

### 目录设计理由

| 层级 | 类型 | 职责 | Agent消费方式 |
|------|------|------|--------------|
| CLAUDE.md | 导航 | 入口地图 | 每次任务首先读取 |
| **harness/** | | **约束层（Agent必须遵守）** | |
| docs/harness/index.md | [索引+约束] | 约束入口：约束概览 + 子目录索引 | 所有任务先读取，建立完整约束地图 |
| docs/harness/invariants/ | [约束目录] | 业务不变量：金额/交易/业务规则 | 读 `index.md`，再读取全部叶子规则 |
| docs/harness/architecture/ | [约束目录] | 架构约束：系统架构 + 模块边界 | 读 `index.md`，再读取全部叶子规则 |
| docs/harness/infrastructure/ | [约束目录] | 技术规范：接口/数据库/缓存/MQ等 | 读 `index.md`，再读取全部叶子规则 |
| docs/harness/linters/ | [工具目录] | 自动化检查：不变量/风格/安全检查 | 读 `index.md`，再读取全部叶子规则 |
| docs/specs/active/{feature}/verification.md | [规格] | 轻量交付/证据门禁：测试、约束检查、执行证据 | 由 `/oh:apply`、`/oh:review`、`/oh:verify`、`/oh:archive` 读写与判定 |
| **knowledge/** | | **知识库（业务和技术现状）** | |
| docs/knowledge/index.md | [索引+知识] | 知识库入口：知识概览 + 更新记录 | 所有任务先读取，建立完整知识地图 |
| docs/knowledge/architecture.md | [知识] | 系统架构：分层、模块、依赖 | 架构相关任务按需读取 |
| docs/knowledge/domain-model.md | [知识] | 领域模型：实体、关系、业务流程 | 领域理解相关任务按需读取 |
| docs/knowledge/prd/ | [知识目录] | Proposal 知识库：各功能需求文档 | 先读 `docs/knowledge/prd/index.md`，再按需读叶子文档 |
| docs/knowledge/prd/index.md | [索引+知识] | Proposal 入口：需求索引 | 查找历史需求文档 |
| docs/knowledge/prd/{feature}-proposal.md | [知识] | 功能 Proposal 文档：背景、范围、旅程、风险 | 理解历史需求 |
| docs/knowledge/api/ | [知识目录] | API文档：各服务接口契约 | 先读 `docs/knowledge/api/index.md`，再按需读叶子文档 |
| docs/knowledge/database/ | [知识目录] | 数据库文档：表结构、ER图 | 先读 `docs/knowledge/database/index.md`，再按需读叶子文档 |
| docs/knowledge/exp/ | [经验目录] | 经验总结：可复用经验和踩坑记录 | 先读 `docs/knowledge/exp/index.md`，再按需读叶子文档 |
| docs/knowledge/exp/index.md | [索引+经验] | 经验入口：经验索引 + 搜索指南 | 搜索相关经验 |
| docs/knowledge/exp/{date}-{title}.md | [经验] | 经验文档：背景、问题、方案、启示 | 参考类似经验 |
| **specs/** | | **工作区域（需求开发）** | |
| docs/specs/active/ | [规格] | 进行中：正在开发的需求 | 当前工作重点 |
| docs/specs/completed/ | [规格] | 已完成：归档的需求 | 历史参考 |
| docs/specs/active/{feature}/proposal.md | [规格] | 需求背景：背景、范围、关键旅程、风险 | 理解需求背景 |
| docs/specs/active/{feature}/spec.md | [规格] | 行为契约：系统必须做什么 | 需求细化 |
| docs/specs/active/{feature}/design.md | [规格] | 技术方案：How（需审批后执行） | 方案设计与审批主稿 |
| docs/specs/active/{feature}/tasks.md | [规格] | 执行清单：任务拆解、状态、checkpoint | 任务执行 |
| docs/specs/active/{feature}/verification.md | [规格] | 验证与门禁：测试、证据、交付门禁 | 测试验证 + 合规检查 |
| **hooks/** | | **审查机制（轻量运行时守门 + 阶段门禁 + 检查清单）** | |
| docs/hooks/index.md | [索引+配置] | 审查机制入口：runtime hooks、workflow gates、review checklists 总览 | 理解治理结构 |
| docs/hooks/post-action.md | [Hook] | 写后轻量检查：invariant 与 specs/evidence 一致性提醒 | 自动触发 |
| docs/hooks/cr-checklist.md | [Checklist] | CR检查清单：代码审查要点 + checkpoint / 证据 / finding 闭环核对 | apply/review 时参考 |

**设计规则**：目录入口文件统一使用 `index.md`，职责为：
1. **整体概览** — 当前目录的核心内容摘要
2. **文件索引** — 指向目录下其他文件的导航，支持渐进式披露
3. **运行期协议入口** — Agent 先读根 `index.md`，再依次检查各子目录 `index.md`，并读取每个子目录下的全部叶子规则文件；约束适用范围始终覆盖完整 harness 体系
4. **知识库按需加载入口** — Agent 先读 `docs/knowledge/index.md` 建立知识地图，再进入相关索引或稳定入口（如 `service-boundary.md`），最后按需读取叶子知识文档

**例外说明**：
- `docs/knowledge/service-boundary.md` 保持稳定语义化文件名，不迁移到 `service-boundary/index.md`
- `docs/specs/**` 继续沿用工作流驱动文件名（如 `proposal.md`、`spec.md`、`design.md`），不适用 `index.md` 规则

**关键区分**：

| 类型 | 定位 | 回答问题 | 更新时机 |
|------|------|----------|---------|
| **harness** | 约束 | "必须遵守什么" | 架构调整/规范变更时 |
| **knowledge** | 知识库 | "系统现状是什么 + 有哪些经验" | 需求完成后自动更新 |
| **specs/active** | 进行中 | "正在做什么" | 需求开发过程 |
| **specs/completed** | 已完成 | "做过什么" | 需求归档后 |
| **hooks** | 审查 | "如何自动检查" | AI每次动作后触发 |

---

### 2.1 项目模式

Open Harness V2 显式区分两类项目模式：

| project-mode | 适用场景 | 设计/评审基线 |
|--------------|---------|--------------|
| `greenfield` | 0→1 新项目，尚无稳定业务代码 | 评审首版基线架构合理性、可扩展性、部署与运维落地性 |
| `brownfield` | 存量系统改造、演进、重构 | 评审兼容性、影响面、迁移成本、回滚与灰度策略 |

项目模式由 `/oh:init` 自动识别并写入 `docs/knowledge/project-context.md`，`/oh:propose` 需将其写入 `design.md` frontmatter，`/oh:review` 按此模式切换评审标准。

对于 `greenfield`：
- 允许在兼容性相关章节填写 `不涉及（greenfield）`
- 不得伪造“已有系统影响/老数据兼容性”分析
- 重点补足基线架构、初始化、演进路径、运维可行性

---

### 2.2 Claude Code Skill Layer

在 Claude Code 生态中，Open Harness V2 保留 `/oh:*` 作为显式工作流入口，同时提供可复用的内置 skills：

| Skill | 作用 | 常见触发阶段 | 典型产物 |
|------|------|-------------|---------|
| `verification-before-completion` | 在宣称完成/修复/通过前强制补齐新鲜验证证据 | apply / review / verify / archive | 命令输出证据块、结论依据 |
| `systematic-debugging` | 对失败、回归、行为不一致执行复现-缩圈-根因-回归流程 | apply / review / verify | 五行调试摘要 |
| `requesting-code-review` | 在送审前整理 review scope、风险、证据、自检项 | review 前 | 送审包摘要 |
| `receiving-code-review` | 对 review / verify finding 做接收、分流、修复、复验闭环 | review / verify 后 | finding 对账表 |
| `test-driven-development` | 用 RED/GREEN/REFACTOR 驱动实现 | apply --tdd / bugfix | RED/GREEN 证据 |
| `brainstorming` | 在执行前扩展并比较可行方案，帮助快速收敛设计方向 | propose 前 / 复杂改造前 | 选项对比与推荐 |
| `writing-plans` | 把已选方向拆成可执行步骤、验证点和依赖 | brainstorm 后 / 动手前 | 可执行计划 |
| `executing-plans` | 按计划逐步推进实现，并持续校准进度与验证 | apply 前 / apply 中 | 进度状态与步骤执行 |
| `parallel-dispatch` | 将独立子任务分发给并行 agent，同时保持清晰 ownership | 复杂任务执行中 | 并行任务拆分与 ownership |

设计原则：
- `/oh:*` 负责工作流编排
- 内置 skills 负责横切方法能力
- agents 负责一次性生成/扫描任务

其中：
- `verification-before-completion`、`systematic-debugging`、`requesting-code-review`、`receiving-code-review`、`test-driven-development` 是执行期方法能力
- `brainstorming`、`writing-plans`、`executing-plans`、`parallel-dispatch` 是更通用的 Claude Code 工作方式能力，可在 Open Harness 之外复用

命令与内置 skills 的固定编排：

| 命令 | 默认/条件加载的 skills | 命令内必须留下的产物 |
|------|----------------------|--------------------|
| `/oh:apply` | `verification-before-completion`；`--tdd` 时加载 `test-driven-development`；失败时切换 `systematic-debugging`；处理 finding 时先用 `receiving-code-review` | `tasks.md` / `verification.md` 证据块、TDD RED/GREEN 记录、finding 对账、调试摘要 |
| `/oh:review` | `requesting-code-review`；重新送审时先用 `receiving-code-review`；所有完成声明受 `verification-before-completion` 约束 | 送审包摘要、finding 对账、带证据的评审报告 |
| `/oh:verify` | `verification-before-completion`；失败或不一致时切换 `systematic-debugging`；复验 finding 时先用 `receiving-code-review` | `verify-report.md` 证据、finding 对账、调试摘要 |

条件触发的工作方式 skills：

| 命令 | 条件触发的 skills | 触发效果 |
|------|------------------|---------|
| `/oh:propose` | `brainstorming` | 当原始需求含糊或存在多种可行设计时，先比较方案，再收敛到推荐路径 |
| `/oh:propose --continue` | `writing-plans` | 当 design 已定但执行顺序不清时，先形成计划骨架，再映射到 `tasks.md / verification.md` |
| `/oh:apply` | `executing-plans` | 当任务跨阶段、跨模块、跨检查点时，按计划推进并持续更新进度 |
| `/oh:apply` / `/oh:review` | `parallel-dispatch` | 当存在安全可并行的读写分离子任务或专项检查时，先定义 delegation plan 再并行处理 |

### 2.3 使用者速查表

下面这张表按“command -> 依赖的 skill / agent -> 作用 -> 产出”整理，供使用者直接查阅。

| Command | 阶段 | 依赖 Skill | 依赖 Agent | 主要作用 | 用户最终会看到的产出 |
|------|------|-----------|-----------|---------|------------------|
| `/oh:init` | 项目初始化 | 无强绑定 skill | `scan-tech`、`scan-db`、`scan-api`、`scan-service`、`scan-boundary` | 扫描项目、初始化 knowledge / harness / hooks | `docs/harness/`、`docs/knowledge/`、`docs/hooks/`、项目 `CLAUDE.md`、`.claude/settings.json` |
| `/oh:propose <name> <prd-path>` | Phase 1 需求与设计生成 | 条件触发 `brainstorming` | `prd-parser` | 解析原始需求，生成 proposal / spec / design；方案不清时先做多方案收敛；完成后自动触发 Phase 1 文档 review | `proposal.md`、`spec.md`、`design.md`、Phase 1 review 报告 |
| `/oh:review` | Phase 1/2 文档 review、审批或 apply 后综合 review | `requesting-code-review`、`receiving-code-review`、`verification-before-completion`；条件触发 `parallel-dispatch` | 产品专家 agent、技术专家 agent、质量专家 agent（apply 后综合 review 时） | 先整理送审包；复审时对账 finding；必要时做并行专项 review；根据当前阶段已生成工件输出统一评审结论 | 评审报告、改进建议汇总、复审对账；审批通过时更新 `design.md` 的 `approval` |
| `/oh:propose <name> --continue` | Phase 2 执行文档生成 | 条件触发 `writing-plans` | `propose-service-execution-prep` | 将已批准 design 转成执行计划结构，再生成 tasks / verification；完成后自动触发 Phase 2 文档 review | `tasks.md`、`verification.md`、Phase 2 review 报告 |
| `/oh:apply <name>` | 执行实现 | `verification-before-completion`、`systematic-debugging`、`receiving-code-review`；`--tdd` 时用 `test-driven-development`；条件触发 `executing-plans`、`parallel-dispatch` | 无固定专项 agent | 按 design 和 tasks 实现；完成前必须留证据；失败先调试；修 finding 先对账；复杂任务按计划推进；完成后自动触发 apply 后综合 review | 代码实现、更新后的 `tasks.md`、`verification.md`、证据块、debug note、delegation note、综合 review 报告 |
| `/oh:verify <name>` | 独立验收 | `verification-before-completion`、`systematic-debugging`、复验 finding 时 `receiving-code-review` | 无固定专项 agent | 从独立视角验证 spec / design / verification / code 契约；不信任 apply 或 review 的完成叙事 | `verify-report.md`、验收结论、finding 对账、调试摘要 |
| `/oh:archive <name>` | 归档收尾 | 当前无强绑定 skill | `archive-feature` | 归档 active 需求，沉淀 proposal 和知识更新建议 | `specs/completed/{feature}/`、`knowledge/prd/` 中的归档 Proposal、知识库更新建议 |

使用原则：
- 你实际调用的是 `/oh:*` 命令，skill 不是替代 command 的另一套主入口。
- skill 是 command 内部依赖的工作方式层：有些是默认硬编排，有些只在特定条件下触发。
- agent 是专项生成/扫描能力，主要集中在 `init` 和 `propose` 阶段。

---

## 三、工作流与分层职责

这一节只说明 **V2 为什么这样组织**，具体命令参数、入口判断、产物约束以 `commands/oh/*.md` 为准。

### 3.1 核心工作流

Open Harness V2 的主链路是：

```text
/oh:init
→ /oh:propose <name> <prd-path>
→ 自动 /oh:review（Phase 1 文档）
→ /oh:propose <name> --continue
→ 自动 /oh:review（Phase 2 文档）
→ /oh:apply
→ 自动 /oh:review（apply 后综合）
→ 可选 /oh:verify
→ /oh:archive
```

其中：
- `init` 负责建立约束层、知识层、工作区与审查机制
- `propose` 负责两段式文档生成：Phase 1 生成 `proposal.md / spec.md / design.md`，Phase 2 生成 `tasks.md / verification.md`，且每个阶段结束后自动触发阶段性 `/oh:review`
- `review` 是统一的开发侧综合检查入口：承担阶段文档 review、审批与 apply 后综合 review
- `apply` 负责实现与执行证据闭环，完成后自动触发 `/oh:review`
- `verify` 是 apply / review 之后的独立验收点，用于从实现者之外的视角复验契约满足情况
- `archive` 负责从 active 收尾到 completed，并沉淀知识

### 3.2 为什么是“两段式 propose”

V2 没有把设计与执行文档一次性全部生成，而是拆成：

1. **Phase 1**：先收敛 `proposal.md`、`spec.md`、`design.md`
2. **审批通过后**：再生成执行文档并进入实现

这样设计的目的：
- 防止需求理解未稳定时过早生成 tasks/verification，造成返工
- 让 `design.md` 成为真正的控制点，而不是形式化产物
- 保留 “fluid not rigid” 的调整空间：人在审批前可以随时自然语言修文档

### 3.3 为什么区分 command / skill / agent

V2 把能力分成三层：

| 层 | 作用 | 面向谁 |
|----|------|-------|
| command (`/oh:*`) | 工作流入口与编排 | 用户 |
| skill | 横切方法能力，如 review/debug/verification | Claude Code 执行过程 |
| agent prompt | 一次性专项扫描/生成任务 | 命令内部调用 |

设计意图：
- 用户只需要记住 `/oh:*`
- skill 不替代命令，而是补强执行方法
- agent 不暴露为主入口，只承接被编排的专项任务

### 3.4 为什么要分 harness / knowledge / specs / hooks

这四层分别回答不同问题：

| 层 | 回答的问题 |
|----|-----------|
| `harness/` | 必须遵守什么 |
| `knowledge/` | 当前系统现状是什么 |
| `specs/` | 这次需求准备做什么 |
| `hooks/` | 如何在执行过程中自动检查 |

设计目标不是“文档很多”，而是把：
- 约束
- 现状
- 当前需求
- 审查机制

明确拆开，减少混写后的漂移和歧义。

### 3.5 Hub 模式在 V2 中的位置

Hub 不是另一套平行范式，而是 V2 在多服务场景下增加的一层系统视图能力。

主关系如下：
- 单服务研发，直接走主链路
- 多服务协同，先在 Hub 仓库聚合事实、生成系统视图和方案母稿
- 再把 `service-briefs/*.md` 分发到各服务仓库，继续普通 propose/apply 流程

更具体的 Hub 设计请参考 `reference/openHarnessHubDesign.md`。

### 3.6 运行期语义以命令文件为准

为了避免文档漂移，以下内容不在本参考文档中重复展开：
- 参数解析
- 模式分支
- 前置检查
- 产物清单
- 详细交互步骤

查阅方式：
- `/oh:init` → `commands/oh/init.md`
- `/oh:propose` → `commands/oh/propose.md`
- `/oh:review` → `commands/oh/review.md`
- `/oh:apply` → `commands/oh/apply.md`
- `/oh:archive` → `commands/oh/archive.md`

---

## 四、命令系统与安装布局

命令前缀：`/oh`（Open Harness 缩写）

### 4.1 命令系统变化

相比 V1，V2 重构为完整的 AI Coding 工作范式，并将 review / verify 重整为更清晰的两层模型：

### 目录结构变化

1. **新增 knowledge/** — 知识库目录，管理架构、领域模型、API、数据库、经验总结等系统知识
2. **knowledge 下包含 exp/** — 经验总结作为知识库的一部分，统一管理
3. **新增 hooks/** — Hook机制目录，分层触发审查
4. **重构 specs/** — 区分 active（进行中）和 completed（已完成）
5. **收敛需求文档** — 统一为 proposal.md、spec.md、design.md、tasks.md、verification.md
6. **database/api/lessons 合并** — 合并到 knowledge/ 目录下
7. **design.md 成为统一技术方案名** — 统一以 design.md 为审批与设计主文档
8. **tasks.md 保留并吸收 checkpoint** — 任务清单同时承担执行状态与 checkpoint 台账
9. **verification.md 统一门禁与证据** — 承载测试用例、交付门禁与验证证据
10. **knowledge/prd/ Proposal 知识库** — 归档时自动将 proposal 沉淀到知识库，便于回溯

### 工作流变化

1. **init 交互式初始化** — 支持导入外部知识库（`--knowledge`）
2. **两段式生成文档** — `/oh:propose <name> <prd-path>` 先生成设计文档，审批通过后 `/oh:propose <name> --continue` 再生成执行文档
3. **Design 审批前置** — Phase 2 和 apply 都依赖 design.md 的审批状态
4. **阶段生成后自动 review** — Phase 1 / Phase 2 结束后立即 review 当前阶段已生成工件，不把未来工件当阻断条件
5. **apply 后自动综合 review** — apply 完成后立即执行代码 review、功能验证与门禁检查
6. **独立验收独立成命令** — `/oh:verify` 专门承担独立验收，不与开发侧 review 混用
7. **归档自动沉淀 Proposal** — 归档时自动将 proposal.md 复制到 knowledge/prd/，更新索引
8. **归档建议式更新知识库** — 输出架构/API/数据库等更新建议，人工确认后执行
9. **Hook 分层审查** — 轻量/中量/重量分层触发，避免过度审查
10. **多Agent评审** — 产品/技术/质量三专家独立评审
11. **可回退机制** — 支持 rollback 到任意 checkpoint

### 命令系统变化

| V1 命令 | V2 命令 | 变化说明 |
|---------|---------|---------|
| `/openharness feature` | `/oh:propose` | 改为两段式文档生成（设计文档 → 审批 → 执行文档） |
| `/openharness plan` | — | 移除，合并到 `/oh:propose` |
| `/openharness tasks` | — | 移除命令，但 tasks.md 文档由 `/oh:propose` 一键生成 |
| `/openharness execute` | `/oh:apply` | 增加审批状态检查 |
| `/openharness verify` | `/oh:verify` | 独立命令 |
| — | `/oh:review` | 新增多Agent评审命令 |
| — | `/oh:rollback` | 新增回退命令 |
| `/openharness` | `/oh:` | 命令前缀统一缩短 |

### 命令源码与安装布局

V2 的安装分为两层：command 层和 skill 层，会一起安装到 `.claude/` 或 `~/.claude/`。

命令层统一为 namespaced 布局：

- 仓库源码：`openharness/commands/oh/*.md`
- 安装位置：`.claude/commands/oh/*.md` 或 `~/.claude/commands/oh/*.md`
- 用户调用：`/oh:<command>`

对应关系示例：

| 源码文件 | 安装后文件 | 用户命令 |
|---------|-----------|---------|
| `openharness/commands/oh/init.md` | `.claude/commands/oh/init.md` | `/oh:init` |
| `openharness/commands/oh/propose.md` | `.claude/commands/oh/propose.md` | `/oh:propose` |
| `openharness/commands/oh/apply.md` | `.claude/commands/oh/apply.md` | `/oh:apply` |

skill 层的安装布局：

- 仓库源码：`openharness/skills/<skill-name>/`
- 安装位置：`.claude/skills/<skill-name>/` 或 `~/.claude/skills/<skill-name>/`
- 作用：为 `/oh:*` 命令提供方法能力层，也可在 Claude Code 中单独复用

当前会随 OpenHarness 一起安装的 skills：

- `verification-before-completion`
- `systematic-debugging`
- `requesting-code-review`
- `receiving-code-review`
- `test-driven-development`
- `brainstorming`
- `writing-plans`
- `executing-plans`
- `parallel-dispatch`

命令文档中引用的 `~/.claude/skills/openharness/agents/...`、`templates/...` 路径在安装时会被重写到同一命名空间根下的 `oh/agents/...`、`oh/templates/...`，避免运行时出现 `oh-*` 与 `/oh:` 混用。

### 核心理念升级

| 维度 | V1 | V2 |
|------|----|----|
| 生成模式 | 分步生成（feature → plan → tasks 命令） | 两段式生成（设计文档 → 审批 → 执行文档）+ 灵活调整 |
| 文档调整 | 按命令逐步更新 | 自然语言随时调整 |
| 约束 | 约束驱动的规格开发 | 约束前置 + 审批控制 |
| 知识库 | 手动管理 | 建议式自动更新 |
| 审查 | 每次动作后触发 | 分层触发策略 |
| 评审 | 单一 review | 多Agent独立评审 |
| 回退 | 无机制 | 支持 rollback |

### 借鉴 OpenSpec 的设计

V2 命令系统借鉴了 [OpenSpec](https://github.com/Fission-AI/OpenSpec) 的设计理念：

| OpenSpec 理念 | Open Harness V2 实现 |
|---------------|---------------------|
| fluid not rigid | 阶段内文档可随时用自然语言调整 |
| iterative not waterfall | 允许反复 review 和修改，但阶段切换有审批门禁 |
| Actions, not phases | 命令是动作（可随时做），不是阶段（必须按顺序） |
| `/opsx:propose` 一键生成 | `/oh:propose` 两段式生成文档 |
