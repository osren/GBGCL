---
description: "Hub 模式 propose 总控 agent。运行于集中式文档仓库，生成跨服务 proposal、design 和 service-briefs。"
---

## 任务：生成跨服务方案母稿

> 适用目录：集中式文档仓库根目录
>
> 目标：生成系统级 `design.md` 与面向各服务仓库的 `service-briefs/*.md`
>
> 非目标：不生成 `tasks.md`、`verification.md`，不面向 `oh:apply`

## 输入

- feature 名称
- 产品文档路径
- 当前仓库根目录

## 执行流程

### Step H1: 准备工作

1. 解析 `<name>` 和 `<prd-path>`，任一缺失则提示用户补充
2. 检查 `docs/knowledge/system/system-map.md` 是否存在：
   - 不存在 → 提示先执行 `/oh:init --hub`
3. 检查 `docs/specs/active/{name}/` 是否已存在：
   - 已存在且含 `design.md` → 提示用户手动确认是否覆盖
4. 创建目录：
   - `docs/specs/active/{name}/`
   - `docs/specs/active/{name}/service-briefs/`

### Step H2: 解析产品文档

使用 Agent tool 调用 `~/.claude/skills/openharness/agents/prd-parser.md`（安装后重写为 `{install-dir}/commands/oh/agents/prd-parser.md`）。

传递以下上下文：

- 产品文档路径：`{prd-path}`
- feature 目录：`docs/specs/active/{name}/`

agent 输出：

- `docs/specs/active/{name}/proposal.md`

### Step H3: 读取 Hub 知识库

并行读取以下内容，**必须在调用 Agent 前完成**：

- `docs/knowledge/index.md` — knowledge 总入口，先建立完整知识地图
- 再按 Hub 方案生成需要进入相关知识入口与叶子文档
- `docs/knowledge/system/system-map.md`
- `docs/knowledge/system/core-flows/*.md`
- `docs/knowledge/services/*/service-boundary.md`
- `docs/knowledge/services/*/domain-model.md`
- `docs/knowledge/services/*/service-meta.yaml`

若 `system-map.md` 和 `core-flows/` 显示存在多种合理实现路径，先条件触发 `brainstorming`，再继续后续步骤。

### Step H4: 生成系统级 design.md

基于以下输入，生成 `docs/specs/active/{name}/design.md`：

- `docs/specs/active/{name}/proposal.md`
- `docs/knowledge/index.md`
- `docs/knowledge/system/system-map.md`
- `docs/knowledge/system/core-flows/*.md`
- `docs/knowledge/services/*/service-boundary.md`
- `docs/knowledge/services/*/domain-model.md`
- `docs/knowledge/services/*/service-meta.yaml`

**文档格式**：严格参考 `~/.claude/skills/openharness/templates/hub-design.md.tpl`（安装后重写为 `{install-dir}/commands/oh/templates/hub-design.md.tpl`）。

生成要求：

- frontmatter 至少包含：
  - `feature: {name}`
  - `mode: hub`
  - `approval: pending`
- 必须明确：
  - 影响服务清单
  - 各服务职责分工
  - 跨服务调用链
  - 关键接口 / 关键事件
  - 数据边界与主写归属
  - 风险、迁移、灰度、回滚策略
- 不写仓库内部类、方法、任务拆解细节
- 无法确认的内容标注 `<!-- TODO: 待确认 -->`

### Step H5: 生成服务实施摘要

遍历系统级 `design.md` 中识别出的受影响服务，对每个服务调用 Agent tool：

- 调用 `~/.claude/skills/openharness/agents/generate-service-brief.md`（安装后重写为 `{install-dir}/commands/oh/agents/generate-service-brief.md`）
- 输出：`docs/specs/active/{name}/service-briefs/{service}.md`

每份 brief 必须包含：

- 本服务目标
- 本服务需承担的变更
- 上下游协同
- 关键契约
- 数据影响
- 风险与待确认项

### Step H6: 输出报告

```text
## /oh:propose --hub 完成 — {name}

### 已生成
- [x] docs/specs/active/{name}/proposal.md
- [x] docs/specs/active/{name}/design.md（approval: pending）
- [x] docs/specs/active/{name}/service-briefs/

### 下一步
1. 审核 Hub 级 proposal.md / design.md，确认跨服务职责分工
2. 将 service-briefs/{service}.md 分发到各服务仓库
3. 在各服务仓库执行 /oh:propose <name> <service-brief-path>
```

## Hub 模式约束

- **不支持 `--continue`** — Hub 仓库不生成执行文档
- **不支持直接 `oh:apply`** — 代码实现必须回到各服务仓库
- **service-briefs 是服务仓库输入** — 用于服务仓库继续生成 proposal/spec/design，不是最终可执行方案
