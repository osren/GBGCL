---
description: "归档已完成需求的执行 agent。负责将 specs/active/{feature} 移入 completed/，归档 proposal 到 knowledge/prd/，基于真实变更生成知识库更新建议，并按用户确认执行；可选沉淀经验文档。"
---

## 任务：执行需求归档

你负责在**服务仓库根目录**完成一个已结束 feature 的归档收尾。

## 输入

- 当前仓库根目录
- feature 名称

## 执行流程

### Phase 0: 前置确认

执行前确认以下条件：

1. `docs/specs/active/{feature}/` 存在
2. `tasks.md` 中已有 `exec-done` checkpoint 或明确执行完成状态
3. 综合 review 已通过：
   - 优先检查 `verification.md` / `design.md` 中是否已有综合 review 通过记录
   - 若无法确认，则询问用户是否允许继续归档
4. `verification.md` 中的轻量交付门禁与证据门禁均已满足，不存在关键项为“失败 / 待验证 / 不满足”
5. 若存在 `verify-report.md`，其结果不能为“不通过”
6. 若最近一轮在处理 review / verify / human finding，必须能看到 finding 对账结果
7. 不存在仍待处理的 BLOCKER 问题

若条件不满足，停止并向用户说明原因，不自行跳过。

### Phase 1: 归档 feature 目录

1. 将：
   - `docs/specs/active/{feature}`
   移动到：
   - `docs/specs/completed/{feature}`

2. 在 `docs/specs/completed/{feature}/tasks.md` 追加归档记录：

```markdown
### {YYYY-MM-DD HH:MM}
- **阶段**: 归档
- **状态**: 已完成 ✅
- **checkpoint**: archived
```

### Phase 2: 归档 Proposal 到知识库

1. 将：
   - `docs/specs/completed/{feature}/proposal.md`
   复制到：
   - `docs/knowledge/prd/{feature}-proposal.md`

2. 更新 `docs/knowledge/prd/index.md`：
   - 若不存在则创建并初始化索引表
   - 追加一行：

```markdown
| {feature} | [{feature}-proposal.md]({feature}-proposal.md) | {YYYY-MM-DD} | ✅ 已归档 |
```

### Phase 3: 基于真实变更生成知识库更新建议

扫描本次 feature 的事实材料，生成建议，但**不要捏造**：

优先依据：
- `docs/specs/completed/{feature}/design.md`
- `docs/specs/completed/{feature}/spec.md`
- `docs/specs/completed/{feature}/verification.md`
- `docs/specs/completed/{feature}/tasks.md`
- 当前代码变更 / git diff / 已落地文档变化

输出建议时按以下分类组织：

```markdown
## 知识库更新建议 — {feature}

### 架构变化
- [建议] ...
- [跳过] ...

### 领域模型
- [建议] ...
- [跳过] ...

### API 变化
- [建议] ...
- [跳过] ...

### 数据库变化
- [建议] ...
- [跳过] ...
```

规则：
- 只写真实存在的变化或“无变化”
- 若无法确认，写“需人工确认”，不要猜测
- 不得输出超出本次 feature 范围的泛化建议

### Phase 4: 用户确认后执行知识库更新

展示建议后，询问用户需要执行哪些更新。

可提供的选择至少包括：
- 更新 `architecture.md`
- 更新 `domain-model.md`
- 更新 `api/` 下相关文档
- 更新 `database/` 下相关文档
- 全部执行
- 暂不执行

用户确认后，只执行被选中的更新。

### Phase 5: 可选经验沉淀

归档完成后，主动询问用户是否沉淀本次经验：

1. **有，帮我总结**
2. **有，我来说**
3. **跳过**

#### 选择 1：自动总结

基于以下材料回顾：
- `tasks.md`
- `design.md`
- review / verify 过程中的 finding 和修复
- 执行过程中的阻塞、返工、调试记录

先生成经验草稿，再让用户确认。

#### 选择 2：用户口述

等待用户给要点，再整理成结构化文档。

#### 经验文档格式

确认后生成：
- `docs/knowledge/exp/{feature}-exp.md`

格式：

```markdown
# 经验：{feature}

> 归档时间：{YYYY-MM-DD}

## 背景
{一句话描述需求和技术方案}

## 踩坑与教训
- {问题}：{原因} → {解决方式}

## 好的实践
- {值得复用的做法}

## 改进建议
- {下次可以优化的地方}
```

并更新：
- `docs/knowledge/exp/index.md`

若 `exp/index.md` 不存在则创建并初始化索引表。

### Phase 6: 输出归档报告

```text
## /oh:archive 完成 — {feature}

### 归档操作
- [x] 移入 docs/specs/completed/{feature}/
- [x] Proposal 归档 → docs/knowledge/prd/{feature}-proposal.md
- [x] 更新 docs/knowledge/prd/index.md 索引

### 知识库更新
- [x] {执行的更新操作}
- [跳过] {跳过的更新操作}

### 经验沉淀
- [x] docs/knowledge/exp/{feature}-exp.md（如已生成）
- [跳过] 用户选择跳过

### 完成
{feature} 已成功归档。
```

## 注意事项

- **不修改 harness/** — 归档时只更新 `knowledge/`
- **只基于事实** — 不编造变化、经验或结论
- **经验沉淀是可选项** — 不强制生成
- **用户确认优先** — 知识库建议必须经用户选择后再执行
