---
description: "项目 Open Harness V2 健康度报告。展示活跃需求状态、知识库完整度、harness 约束覆盖情况、hooks 配置状态。"
---

## User Input

```text
$ARGUMENTS
```

## 执行流程

### Step 1: 扫描项目状态

并行扫描以下内容：

1. `docs/specs/active/` — 活跃 feature 列表
2. `docs/specs/completed/` — 已完成 feature 数量
3. `docs/harness/` — 约束文件完整性
4. `docs/knowledge/` — 知识库文件完整性
5. `docs/hooks/` — hooks 配置完整性

### Step 2: 输出状态报告

```
## Open Harness V2 状态报告
> {YYYY-MM-DD HH:MM}

---

### 活跃需求 ({N} 个)

| Feature | 阶段 | 审批状态 | 最后更新 |
|---------|------|---------|---------|
| {name} | proposal/spec/design / 执行中 / 验证中 | pending/approved | {date} |

### 已完成需求
- 共归档 {N} 个需求
- 最近归档：{feature-name} ({date})

---

### Harness 约束层

| 目录/文件 | 状态 |
|----------|------|
| harness/index.md | ✅/❌ 缺失 |
| harness/invariants/index.md | ✅/❌ |
| harness/invariants/biz.md | ✅/❌ |
| harness/architecture/index.md | ✅/❌ |
| harness/infrastructure/index.md | ✅/❌ |
| harness/linters/invariant-check.sh | ✅/❌ |
| specs/active/*/verification.md 门禁区 | ✅/❌ |

### Knowledge 知识库

| 文件 | 状态 | 最后更新 |
|------|------|---------|
| knowledge/index.md | ✅/❌ | {date} |
| knowledge/architecture.md | ✅/❌ | {date} |
| knowledge/domain-model.md | ✅/❌ | {date} |
| knowledge/api/ | ✅ {N}个文件/❌ 缺失 | {date} |
| knowledge/database/ | ✅ {N}个文件/❌ 缺失 | {date} |
| knowledge/prd/ | ✅ {N}个Proposal/❌ 缺失 | {date} |
| knowledge/exp/ | ✅ {N}个经验/❌ 缺失 | {date} |

### Hooks 配置

| 文件 | 状态 |
|------|------|
| hooks/index.md | ✅/❌ |
| hooks/post-action.md | ✅/❌ |
| hooks/cr-checklist.md | ✅/❌ |

> `docs/hooks/` 表示轻量 runtime guard + workflow gates + review checklists 的机制说明，不代表所有审查都由自动 hook 完成。

---

### 门禁健康度

| Feature | verification.md | 门禁状态 | 说明 |
|---------|-----------------|---------|------|
| {name} | ✅/❌ | 已满足 / 不满足 / 待验证 / 不适用 | {阻断项或缺失项} |

### 待处理事项

{列出需要关注的问题，例如：}
- ⚠️ {feature-name} 等待 Design 审批，执行 /oh:review
- ⚠️ {feature-name} 的 verification.md 仍有门禁项处于“待验证”
- ⚠️ knowledge/architecture.md 可能已过期（{N}天未更新，期间有 {M} 个 feature 归档）
- ❌ harness/linters/invariant-check.sh 缺失，执行 /oh:init 补充

### 健康度评分
- Harness 完整度：{N}/{M} ✅
- Knowledge 完整度：{N}/{M} ✅
- 整体评分：{N}%
```

## 注意事项

- 只报告实际文件状态，不推断或捏造
- 健康度评分基于关键文件的完整性，缺失的文件标注 ❌
