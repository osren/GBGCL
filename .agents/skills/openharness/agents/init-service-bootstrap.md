---
description: "服务仓库初始化总控 agent。运行于单服务代码仓库，扫描代码生成 knowledge/，生成 harness/、hooks/、CLAUDE.md，并产出 service-meta.yaml。"
---

## 任务：执行服务仓库初始化

你负责在**单个服务仓库根目录**完成 Open Harness 初始化。

## 输入

- 当前仓库根目录
- 原始参数：`$ARGUMENTS`

## 执行流程

### Phase 0: 检查已有内容

检查以下目录/文件是否已存在：

- `CLAUDE.md` — **已存在则在末尾追加 OpenHarness 导航区块**（不得覆盖已有内容），不存在则基于模板生成完整文件
- `docs/harness/`
- `docs/knowledge/`
- `docs/knowledge/prd/`
- `docs/specs/`
- `docs/specs/active/`
- `docs/specs/completed/`
- `docs/hooks/`
- `.claude/settings.json`（检查是否已含 `hooks.PostToolUse` 配置）

若全部已存在，询问用户是否需要更新某些部分，否则仅补充缺失内容。

### Phase 1: 收集项目信息

**优先通过代码扫描自动识别**，无法确定时**必须交互式询问用户**，不得捏造：

1. **技术栈**：
   - 扫描 `pom.xml` / `build.gradle` / `package.json` / `go.mod` 等构建文件
   - 识别：语言版本、Web框架、ORM、RPC框架、数据库、缓存、消息队列

2. **架构风格**：
   - 扫描根目录结构判断：单体/多模块/微服务
   - 扫描包结构判断分层：三层（Controller/Service/Repository）/ DDD / CQRS
   - 无法确定时询问用户

3. **业务领域**：
   - 扫描 README.md、包名推断
   - 无法确定时询问用户（出行/保险/信贷/支付/电商/物流/其他）

4. **项目模式**：
   - 扫描源码目录判断：已有业务代码 → `brownfield`，无业务代码 → `greenfield`
   - 无法确定时询问用户

5. **外部知识库**（若有 `--knowledge` 参数）：
   - 读取指定路径/URL的知识文档
   - 导入到对应 `docs/knowledge/` 子目录

### Phase 2: 生成 knowledge/

#### 有 `--knowledge` 参数

- 读取指定路径/URL的文档，导入到 `docs/knowledge/`
- 生成 `docs/knowledge/index.md` 作为入口索引

#### 无 `--knowledge` 参数，通过扫描 Agent 生成

使用 Agent tool 调用以下专项扫描 agent（读取各自的 `.md` 文件执行）：

| Agent | 输出 | 说明 |
|-------|------|------|
| `~/.claude/skills/openharness/agents/scan-tech.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-tech.md`） | `docs/knowledge/architecture.md` | 技术栈、框架、中间件、架构模式、工程规范 |
| `~/.claude/skills/openharness/agents/scan-db.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-db.md`） | `docs/knowledge/database/` | 表结构、字段、索引、ER 关系图 |
| `~/.claude/skills/openharness/agents/scan-api.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-api.md`） | `docs/knowledge/api/` | HTTP REST / Thrift RPC / gRPC 接口清单 |
| `~/.claude/skills/openharness/agents/scan-service.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-service.md`） | `docs/knowledge/domain-model.md` | 服务定位、模块详情、数据归属、事件流、状态机 |
| `~/.claude/skills/openharness/agents/scan-boundary.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-boundary.md`） | `docs/knowledge/service-boundary.md` | 极简一页：核心定位、核心职责、显性边界 |
| `~/.claude/skills/openharness/agents/scan-service-meta.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-service-meta.md`） | `docs/knowledge/service-meta.yaml` | 服务元数据：服务名、业务域、系统角色、上下游、owned data、事件、核心能力 |

**执行顺序**：

1. `scan-tech` / `scan-db` / `scan-api` — 并行执行（互不依赖）
2. `scan-service` — 依赖上述三个 agent 的输出，待其完成后执行
3. `scan-boundary` — 依赖 `scan-service` 的输出
4. `scan-service-meta` — 依赖 `scan-service` / `scan-boundary` 的输出，最后执行

**初始化占位文件**（供后续 `/oh:propose` 使用，初始内容为空模板）：

- `docs/knowledge/project-context.md` — 项目上下文（含 `project-mode: greenfield|brownfield`、技术栈、架构风格、业务域）
- `docs/knowledge/domain-mapping.md` — 产品别名→业务域映射表（**手动维护**，生成空模板，用户填写）
- `docs/knowledge/service-meta.yaml` — 服务元数据（服务名、业务域、系统角色、上下游、owned data、事件、核心能力）
- `docs/knowledge/service-boundary.md` — 服务职责边界文档（由 `scan-boundary` 必定生成，不是 knowledge 主入口）
- `docs/knowledge/service-boundary/` — 服务职责边界增强目录（仅当 `scan-boundary` 能稳定拆分为 `service-boundary/{domain}/` 结构时生成；缺失不影响 `/oh:propose`）

`domain-mapping.md` 空模板格式：

```markdown
# 领域名称映射表

> 手动维护：将产品文档中使用的别名映射到 service-boundary/ 下的业务域目录名称

| 业务域目录名称 | 产品文档别名 | 说明 |
|--------------|------------|------|
| {domain-dir} | {alias1}, {alias2} | |
```

`service-meta.yaml` 生成方式：

- 调用 `~/.claude/skills/openharness/agents/scan-service-meta.md`（安装后重写为 `{install-dir}/commands/oh/agents/scan-service-meta.md`）
- 输出到 `docs/knowledge/service-meta.yaml`
- 无法确认的字段保留 `TODO` 或空数组，不推断
- `owners`、`repo` 默认保留给用户手动补充

**需求归档目录**：创建 `docs/knowledge/prd/` 与 `docs/knowledge/prd/index.md`（空索引，供后续 `/oh:archive` 沉淀 Proposal）

**经验目录**：创建 `docs/knowledge/exp/index.md`（空索引）

**汇总**：生成 `docs/knowledge/index.md`（知识库总入口 + 各文档索引）

> **无法确认的内容**：标注 `<!-- TODO: 待确认 -->`，不推断

### Phase 3: 生成 harness/

#### 绿地项目检测

在生成 harness/ 前，检测是否为绿地（0→1）项目：

**绿地判断条件**：扫描源码目录（`src/main/java`、`src/`、`lib/` 等），若**不存在任何业务源文件**则进入绿地模式。

**绿地模式**（无代码可扫描）：

1. 跳过 Phase 2 的扫描 Agent，改为交互式收集：
   - 询问技术栈选型（语言/框架/数据库/缓存/MQ）
   - 询问业务领域（信贷/支付/电商/物流/其他）
   - 询问架构风格（单体/多模块/微服务）
2. 将项目模式显式标记为 `greenfield`
3. 基于下方模板直接生成 harness/ 规约骨架
4. 无法确认的内容统一标注 `<!-- TODO: 待确认 -->`，不推断

基于收集的技术栈和业务领域，生成约束文件（**内容填充实际情况，非空模板**）：

- 生成的 `docs/harness/` 必须同时满足**结构完整性**与**运行期协议完整性**：所有 `/oh:*` 命令都应能先从 `docs/harness/index.md` 建立完整约束地图，再依次检查各子目录 `index.md`，并读取每个子目录下的全部叶子规则文件；任何子目录都不得缺失其 `index.md` 入口。

**约束总入口**：

- `docs/harness/index.md` — 约束概览 + 子目录索引；所有命令先从这里建立完整约束地图，再进入各子目录读取全部叶子规则

**业务不变量**（目录：`docs/harness/invariants/`；运行期入口：`docs/harness/invariants/index.md`）：

- `index.md` — 不变量入口 + 规则总览
- `biz.md` — 业务规则（基于业务领域填充）
- `amt.md` — 金额规则（仅信贷/支付/电商项目生成）
- `txn.md` — 交易规则（仅信贷/支付项目生成）

**架构约束**（`docs/harness/architecture/`）：

- `index.md` — 基于扫描到的架构模式，写入实际约束

**基础设施规范**（目录：`docs/harness/infrastructure/`；运行期入口：`docs/harness/infrastructure/index.md`）：

- `index.md` — 规范入口（列出各文件链接）
- 按技术栈按需生成以下文件，**读取对应模板作为内容基础，结合扫描结果或交互收集的信息填充**：

| 文件 | 模板 | 生成条件 |
|------|------|---------|
| `api.md` | `~/.claude/skills/openharness/templates/harness/api.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/harness/api.md.tpl`） | 有 HTTP 接口时生成 |
| `database.md` | `~/.claude/skills/openharness/templates/harness/database.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/harness/database.md.tpl`） | 有数据库时生成 |
| `cache.md` | `~/.claude/skills/openharness/templates/harness/cache.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/harness/cache.md.tpl`） | 使用 Redis/Fusion 时生成 |
| `mq.md` | `~/.claude/skills/openharness/templates/harness/mq.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/harness/mq.md.tpl`） | 使用 DDMQ/Carrera 时生成 |
| `security.md` | `~/.claude/skills/openharness/templates/harness/security.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/harness/security.md.tpl`） | 所有项目生成 |
| `logging.md` | `~/.claude/skills/openharness/templates/harness/logging.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/harness/logging.md.tpl`） | 所有项目生成 |

**生成原则**：

- 已有代码的项目：从扫描结果中提取实际使用模式，覆盖模板中的示例值（如连接池参数、Key 前缀、Topic 枚举等）
- 绿地项目：直接使用模板内容，`<!-- TODO: 待确认 -->` 标注部分由用户后续补充
- 模板中的 `<!-- TODO: 待确认 -->` 注释：有明确信息则替换，无法确认则保留

**交付/证据门禁**：

- 不再生成独立 `docs/harness/quality-gates/` 目录
- 运行期门禁统一由 `/oh:apply`、`/oh:review`、`/oh:verify`、`/oh:archive` 在 `docs/specs/active/{feature}/verification.md` 中执行和记录
- `verification.md` 必须承载轻量交付门禁与证据门禁结果

**Linters**（`docs/harness/linters/`）：

- `index.md` — linter 入口
- `invariant-check.sh` — 基于技术栈生成基础检查脚本
- `spec-sync-check.sh` — 轻量一致性提醒脚本：提示代码改动与 `tasks.md` / `verification.md` / finding 对账之间可能存在脱节；默认 warning-only，不做重阻断

### Phase 3.5: 写入项目上下文

生成 `docs/knowledge/project-context.md`，至少包含：

```markdown
# 项目上下文

- **project-mode**: greenfield / brownfield
- **业务域**: {domain}
- **架构风格**: {architecture style}
- **技术栈**: {language} / {framework} / {database}
- **判定依据**:
  - greenfield: 当前仓库尚无稳定业务代码，技术方案按首版基线设计评审
  - brownfield: 当前仓库已有业务代码或既有系统约束，技术方案按兼容演进设计评审
```

要求：

- `greenfield` 项目必须明确写明“评审以基线架构合理性、可扩展性、运维可行性为主，不以历史兼容性为主”
- `brownfield` 项目必须明确写明“评审以现有系统兼容性、影响范围、迁移成本为主”

### Phase 3.6: 补充服务元数据

由 `scan-service-meta` 生成 `docs/knowledge/service-meta.yaml`，至少包含：

```yaml
service: {service-name}
display_name: {service-display-name}
domain: {domain}
role: {gateway|orchestrator|domain|platform}
repo: <!-- TODO: repo url -->
owners:
  - <!-- TODO: owner team -->
upstreams: []
downstreams: []
owned_data: []
published_events: []
consumed_events: []
capabilities: []
status_entities: []
```

要求：

- `service` 默认使用仓库目录名或已识别的服务名
- `display_name` 优先取 README / 项目名中的业务展示名，无法确认则与 `service` 相同
- `role` 根据代码特征判断：
  - 多下游编排明显 → `orchestrator`
  - 以对外接入和参数适配为主 → `gateway`
  - 以领域主写数据和核心业务规则为主 → `domain`
  - 以通用支撑能力为主 → `platform`
- `owned_data` 来自 scan-db + scan-service 可确认的主写表/实体
- `published_events` / `consumed_events` 来自 MQ 扫描结果
- `capabilities` 来自 `service-boundary.md` 和 `domain-model.md` 提炼
- 无法确认的字段保留 `<!-- TODO: 待确认 -->` 或空数组，不推断

### Phase 4: 配置 hooks/

**Step 4.1** 生成 Hook 配置文件：

- `docs/hooks/index.md` — Hook 机制总览（区分 runtime hooks / workflow gates / review checklists）
- `docs/hooks/post-action.md` — 写后的轻量自动检查（invariant 与 specs/evidence 一致性提醒）
- `docs/hooks/cr-checklist.md` — CR 检查清单（代码规范、invariants、架构约束、tasks checkpoint、verification 证据、finding 对账）

**Step 4.2** 将 hooks 写入项目级 `.claude/settings.json`，使其立即生效：

1. 读取 `.claude/settings.json`（不存在则以 `{}` 为基础创建）
2. 合并写入以下 hooks 配置（**不覆盖已有其他字段**）：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash docs/harness/linters/invariant-check.sh 2>&1 | head -50"
          },
          {
            "type": "command",
            "command": "bash docs/harness/linters/spec-sync-check.sh 2>&1 | head -50"
          }
        ]
      }
    ]
  }
}
```

3. 若 `.claude/settings.json` 中已存在 `hooks.PostToolUse`，则**追加**本条目而非替换整个数组
4. 写入完成后输出确认：`✅ hooks 已写入 .claude/settings.json，PostToolUse 触发器已激活`

### Phase 5: 生成或追加 CLAUDE.md

**CLAUDE.md 不存在时**：基于模板 `CLAUDE.md.tpl` 生成完整文件。

**CLAUDE.md 已存在时**：在文件末尾追加 OpenHarness 导航区块，**不得覆盖已有内容**。追加内容必须包含：

```markdown
---

## OpenHarness 导航

### 目录导航

{标准目录导航地图}

### 关键约束速览

> 完整约束见 docs/harness/index.md

{最重要的 3-5 条 invariants}

### 常用命令

{/oh:propose、/oh:apply 等命令速查}

### Agent 工作规则（OpenHarness）

1. **每次任务开始前必须读取本文件**，了解项目概况
2. **harness 约束全量加载**：先读 `docs/harness/index.md`，再依次读取各子目录 `index.md` 及其全部叶子规则文件。harness 下的所有文件不得跳过，不得只按局部约束工作
3. **knowledge 知识库按需加载**：先读 `docs/knowledge/index.md`，再按任务需要进入相关索引或稳定入口（如 `api/index.md`、`database/index.md`、`prd/index.md`、`exp/index.md`、`service-boundary.md`），最后按需读取叶子文档。不得跳过知识入口直接凭经验推断系统现状
4. **不确定的信息通过交互问答确认**，不捏造
5. **新需求开发必须先生成 proposal + spec + design 并审批 design**，再进入执行阶段
```

追加前检查文件末尾是否已包含 `## OpenHarness 导航` 或 `## Agent 工作规则（OpenHarness）` 区块：
- 若已存在 → **不重复追加**，仅确认内容是否需要更新
- 若不存在 → 追加上述完整区块

**无论新增还是追加**，CLAUDE.md 最终必须包含以下核心内容：
- 项目概述（技术栈、架构风格、业务领域）
- 目录导航地图（指向 harness/、knowledge/、specs/、hooks/）
- 关键约束速览（最重要的 3-5 条 invariants）
- 常用命令速查（/oh:propose、/oh:apply 等）
- Agent 工作规则（含 harness 和 knowledge 的渐进加载机制）

### Phase 6: 输出初始化报告

```text
## Open Harness V2 初始化完成

### 项目信息
- 项目模式: {project-mode}
- 语言: {language}
- 框架: {framework}
- 数据库: {database}
- 架构: {architecture style}
- 业务领域: {domain}

### 生成的文件
- [x] CLAUDE.md
- [x] docs/harness/index.md
- [x] docs/harness/invariants/index.md + 相关叶子规则 — {N} 条规则
- [x] docs/harness/architecture/index.md
- [x] docs/harness/infrastructure/index.md + 相关叶子规则 — {N} 个规范文件
- [x] docs/specs/active/{feature}/verification.md — 轻量交付/证据门禁承载点
- [x] docs/harness/linters/index.md
- [x] docs/harness/linters/invariant-check.sh
- [x] docs/knowledge/index.md
- [x] docs/knowledge/prd/
- [x] docs/knowledge/prd/index.md
- [x] docs/knowledge/project-context.md
- [x] docs/knowledge/architecture.md
- [x] docs/knowledge/domain-model.md
- [x] docs/knowledge/domain-mapping.md（空模板，需手动填写）
- [x] docs/knowledge/service-meta.yaml
- [x] docs/knowledge/service-boundary/ — 服务职责边界
- [x] docs/knowledge/api/ — {N} 个接口文档
- [x] docs/knowledge/database/ — {N} 张表
- [x] docs/knowledge/exp/index.md
- [x] docs/specs/active/
- [x] docs/specs/completed/
- [x] docs/hooks/index.md
- [x] docs/hooks/post-action.md
- [x] docs/hooks/cr-checklist.md

### 下一步
1. 审核 docs/harness/ 下的约束文件，补充项目特有规则
2. 审核 docs/knowledge/ 知识库，确认准确性
3. 执行 /oh:propose <feature-name> 开始第一个需求开发
```

## 注意事项

- **不删除已有内容** — 已存在的文件保持不动，只补充缺失文件
- **不捏造信息** — 无法通过扫描或用户确认的内容标注 `<!-- TODO: 待确认 -->`
- **约束内容应基于实际代码** — harness/ 中的约束应反映项目现有规范，而非理想化原则
- **空项目处理** — 若无代码，通过交互收集信息后基于模板生成，并显式标记 `project-mode: greenfield`
