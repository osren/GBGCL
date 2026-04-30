---
description: "Hub 模式初始化总控 agent。运行于集中式文档仓库，校验 services/ 事实层，生成 flow-seeds、core-flows、system-map、knowledge 索引和 Hub 版 CLAUDE.md。"
---

## 任务：执行 Hub 模式初始化

你负责在**集中式文档仓库根目录**完成 Open Harness Hub 初始化。

## 输入

- 当前仓库根目录
- 原始参数：`$ARGUMENTS`

## 目标

生成或刷新以下产物：

- `docs/knowledge/system/flow-seeds.yaml`
- `docs/knowledge/system/core-flows/*.md`
- `docs/knowledge/system/system-map.md`
- `docs/knowledge/index.md`
- `CLAUDE.md`

明确非目标：

- 不生成 `docs/harness/`
- 不生成 `docs/hooks/`
- 不生成 `.claude/settings.json` hooks
- 不生成服务仓库执行用 `docs/specs/` 骨架

## 执行流程

### Phase H0: 检查已有内容

检查以下目录/文件是否已存在，**已存在则保留，缺失则生成**：

- `CLAUDE.md` — **已存在则在末尾追加 OpenHarness Hub 导航区块**（不得覆盖已有内容），不存在则基于模板生成完整文件
- `docs/knowledge/`
- `docs/knowledge/services/`
- `docs/knowledge/system/`
- `docs/knowledge/system/system-map.md`
- `docs/knowledge/system/flow-seeds.yaml`
- `docs/knowledge/system/core-flows/`

若 `docs/knowledge/services/` 不存在，直接终止并提示：

```text
⛔ Hub 模式要求当前仓库已存在 docs/knowledge/services/，其中需放置各服务同步过来的事实文档。
请先准备 services/{service}/service-boundary.md、domain-model.md、service-meta.yaml 等内容后再执行 /oh:init --hub。
```

### Phase H1: 校验服务事实层

扫描 `docs/knowledge/services/*/`，确认每个服务目录至少具备：

- `service-boundary.md`
- `domain-model.md`
- `service-meta.yaml`

可选内容：

- `api/index.md`
- `database/index.md`

若某服务缺少必需文件：

- 在 `system-map.md` 对应位置标注 `<!-- TODO: 待确认 -->`
- 不因单个服务缺失而中止整个初始化

### Phase H2: 初始化系统级种子文件

若 `docs/knowledge/system/flow-seeds.yaml` 不存在：

- 读取 `~/.claude/skills/openharness/templates/flow-seeds.yaml.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/flow-seeds.yaml.tpl`）
- 生成 `docs/knowledge/system/flow-seeds.yaml` 作为示例模板

若已存在：

- 保留已有内容，不覆盖

### Phase H3: 生成核心流程文档

读取 `docs/knowledge/system/flow-seeds.yaml`：

- 若 `flows:` 为空，则仅保留空模板，不生成 `core-flows/*.md`
- 若存在 flow 定义，则对每条 flow 调用 Agent tool：
  - 调用 `~/.claude/skills/openharness/agents/generate-core-flow.md`（安装后重写为 `{install-dir}/commands/oh/agents/generate-core-flow.md`）
  - 传递上下文：
    - flow-id：`{id}`
    - flow 名称：`{name}`
    - trigger：`{trigger}`
    - services：`{services}`
    - 输出路径：`docs/knowledge/system/core-flows/{id}.md`

### Phase H4: 生成系统全景图

调用 Agent tool：

- 调用 `~/.claude/skills/openharness/agents/generate-system-map.md`（安装后重写为 `{install-dir}/commands/oh/agents/generate-system-map.md`）
- 输出：`docs/knowledge/system/system-map.md`

输入来源：

- `docs/knowledge/services/*/service-meta.yaml`
- `docs/knowledge/services/*/service-boundary.md`
- `docs/knowledge/services/*/domain-model.md`
- `docs/knowledge/system/core-flows/*.md`

### Phase H5: 生成知识库入口

生成或刷新 `docs/knowledge/index.md`，至少包含：

- `services/` 目录索引
- `system/system-map.md` 链接
- `system/core-flows/` 索引
- 最近更新时间

### Phase H6: 生成或追加 Hub 版 CLAUDE.md

**CLAUDE.md 不存在时**：读取 `~/.claude/skills/openharness/templates/CLAUDE-hub.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/CLAUDE-hub.md.tpl`），基于模板生成完整文件。

**CLAUDE.md 已存在时**：在文件末尾追加 OpenHarness Hub 导航区块，**不得覆盖已有内容**。追加前检查是否已包含 `## OpenHarness 导航` 区块：
- 若已存在 → **不重复追加**，仅确认内容是否需要更新
- 若不存在 → 追加完整区块

**无论新增还是追加**，CLAUDE.md 最终必须包含：

- Hub 仓库定位
- 目录导航（`knowledge/services/` + `knowledge/system/` + `specs/active/`）
- 常用命令（`/oh:init --hub`、`/oh:propose --hub`）
- 明确声明：Hub 仓库不执行 `/oh:apply`
- Agent 工作规则（含 knowledge 的渐进加载机制）

### Phase H7: 输出初始化报告

```text
## Open Harness Hub 初始化完成

### 已生成 / 刷新
- [x] docs/knowledge/system/flow-seeds.yaml
- [x] docs/knowledge/system/core-flows/
- [x] docs/knowledge/system/system-map.md
- [x] docs/knowledge/index.md
- [x] CLAUDE.md

### 未生成（Hub 模式非目标）
- [ ] docs/harness/
- [ ] docs/hooks/
- [ ] .claude/settings.json hooks
- [ ] 服务仓库执行用 docs/specs/ 骨架

### 下一步
1. 审核 docs/knowledge/system/system-map.md
2. 补充或修正 docs/knowledge/system/flow-seeds.yaml
3. 执行 /oh:propose --hub <feature-name> <prd-path> 生成跨服务方案母稿
```

## Hub 模式约束

- **不生成 harness/** — Hub 仓库不是服务执行仓库
- **不生成 hooks/** — Hub 仓库不承担代码落地和自动检查
- **不生成 `.claude/settings.json` hooks** — 避免把服务侧检查器安装到文档库
- **不推断缺失服务事实** — 无法确认时标注 `<!-- TODO: 待确认 -->`
