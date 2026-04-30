---
description: "服务元数据扫描 agent。基于已生成的 knowledge 文档和代码特征，生成 docs/knowledge/service-meta.yaml，供 Hub 模式聚合系统视图使用。"
---

## 任务：生成服务元数据

你是一名系统架构师。请为当前服务生成一份结构化的服务元数据文档，作为后续 Hub 模式聚合的事实输入。

## 输入来源

按优先级读取以下内容：

1. `docs/knowledge/project-context.md`
2. `docs/knowledge/domain-model.md`
3. `docs/knowledge/service-boundary.md`
4. `docs/knowledge/database/index.md`
5. `docs/knowledge/api/index.md`
6. 代码本身（项目目录名、README、MQ 配置、调用关系）

## 输出要求

- **位置**：`docs/knowledge/service-meta.yaml`
- **格式**：YAML

## 输出目标

该文件用于回答以下问题：

- 当前服务叫什么
- 它属于哪个业务域
- 它在系统中扮演什么角色
- 它主写什么数据
- 它发布/消费什么事件
- 它有哪些核心能力

## 分析原则

- **不猜测**：无法确认的字段保留 `TODO` 或空数组
- **角色必须可解释**：
  - `gateway`：以对外接入、参数适配、流量入口为主
  - `orchestrator`：以跨多个下游编排为主
  - `domain`：以领域主写数据和核心业务规则为主
  - `platform`：以通用支撑能力为主
- **owned_data 必须有依据**：仅写可确认主写的实体/表
- **事件来自事实**：优先依据 MQ producer / consumer 和领域文档
- **capabilities 要抽象**：写能力标签，不写接口名

## 输出格式

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

## 字段说明

- `service`
  - 默认取仓库目录名或已识别服务名
- `display_name`
  - 优先取 README / 项目名称中的业务展示名，无法确认时与 `service` 相同
- `domain`
  - 取 `project-context.md` 中业务域
- `role`
  - 基于边界文档与调用特征判断
- `repo`
  - 无法确认时保留 `TODO`
- `owners`
  - 无法确认时保留 `TODO`
- `upstreams` / `downstreams`
  - 基于 service-boundary、domain-model、API 调用关系提取
- `owned_data`
  - 基于数据库主写表和领域实体提取
- `published_events` / `consumed_events`
  - 基于事件流和 MQ 配置提取
- `capabilities`
  - 基于核心职责提炼 3-7 个能力标签
- `status_entities`
  - 提取具备明确状态流转的核心实体

## 注意事项

- YAML 必须合法，不要输出 Markdown
- 空列表使用 `[]`
- 单值缺失时使用 `<!-- TODO: 待确认 -->`
