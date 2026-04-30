---
description: "回退到指定 checkpoint。读取 tasks.md 中的历史记录，列出可回退点，执行 Git 回退 + 文件清理 + tasks.md 更新。支持 --to <checkpoint> 直接指定回退目标。"
---

## User Input

```text
$ARGUMENTS
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `[feature-name]` | 可选，不指定时自动定位最新活跃 feature | `early-repayment` |
| `--to <checkpoint>` | 可选，直接指定回退目标 | `--to exec-start` |

## Feature 上下文解析

按以下统一规则定位目标 feature：

- 用户显式传入 `[feature]` → 直接使用该 feature
- 未传入 `[feature]` 且 `docs/specs/active/` 下只有一个活跃 feature → 直接使用该 feature
- 未传入 `[feature]` 且存在多个活跃 feature → 列出候选并让用户选择
- 未传入 `[feature]` 且无活跃 feature → 终止并提示用户先创建或指定 feature

## 预定义 Checkpoint

| Checkpoint | 说明 | 回退操作 |
|------------|------|---------|
| `docs-generated` | 回退到文档生成后、人工调整前 | 重新生成所有文档 |
| `review-approved` | 回退到审批通过前 | 重置 `approval: pending`，清除审批记录 |
| `exec-start` | 回退到执行开始前 | Git 回退代码变更，清理代码文件 |
| `exec-done` | 回退到执行完成前（取消 apply 后综合 review / verify 结果） | 清除 review 和 verify 结果 |
| `start` | 回退到 feature 创建前 | 删除整个 feature 目录 |

## 执行流程

### Step 1: 读取 tasks.md

读取 `docs/specs/active/{feature}/tasks.md`，列出所有已记录的 checkpoint。

### Step 2: 确认回退目标

**有 `--to <checkpoint>` 参数时**：
- 验证 checkpoint 是否存在于 tasks.md 中
- 若不存在，提示"未找到该 checkpoint 记录，可用 checkpoint 为: {列表}"

**无 `--to` 参数时**：
- 列出所有可回退点，让用户选择：
  ```
  请选择回退目标：
  1. docs-generated  (2026-04-01 10:00) — 文档生成
  2. review-approved (2026-04-01 11:00) — 审批通过
  3. exec-start      (2026-04-01 11:30) — 开始执行
  ```

### Step 3: 确认回退影响

在执行前，明确告知用户回退影响：

```
⚠️ 即将回退到 {checkpoint}

回退后将：
- {Git 变更说明：如"回退代码变更 N 个文件"}
- {文档变更说明：如"清除 exec 阶段生成的所有内容"}
- {保留说明：如"保留 design.md 审批状态"}

确认执行？(yes/no)
```

等待用户确认后再执行。

### Step 4: 执行回退

根据回退目标执行：

**`exec-start` 或更早**（有代码变更）：
```bash
# 读取 tasks.md 中对应 checkpoint 记录的 git-head
git revert HEAD..{commit-hash}
# 或
git reset --hard {commit-hash}
```
若对应 checkpoint 的 `git-head` 为 `N/A`，说明当时无可回退的 git 基线：
- 不执行 `git revert` / `git reset`
- 改为仅清理该 checkpoint 之后生成的文件，并提示用户人工处理未提交代码
> 询问用户选择 revert（保留历史）还是 reset（直接回退）

**仅文档变更**（无代码需要回退）：
- 清除对应阶段生成的文档内容
- 重置 design.md 中的 approval 状态（如需）

**`start`（删除整个 feature）**：
```bash
# 确认后删除
rm -rf docs/specs/active/{feature}/
```

### Step 5: 更新 tasks.md

在 tasks.md 中记录回退操作：

```markdown
### {YYYY-MM-DD HH:MM}
- **操作**: rollback
- **回退目标**: {checkpoint}
- **原因**: {用户说明或"用户手动回退"}
- **checkpoint**: rollback-{checkpoint}
```

### Step 6: 输出回退报告

```
## /oh:rollback 完成 — {feature-name}

已回退到: {checkpoint} ({timestamp})

### 已清理
- {列出清理的文件/变更}

### 当前状态
- design.md 审批状态: {pending/approved}
- 代码变更: {已回退/无变更}

### 下一步
{根据回退目标给出建议，如：}
- 可以重新调整 design.md 后执行 /oh:review
- 可以重新执行 /oh:apply
```

## 注意事项

- **只回退明确记录的 checkpoint** — tasks.md 中没有记录的 checkpoint 不能回退
- **必须用户确认** — 执行前明确展示影响范围，等待用户确认
- **Git 回退优先使用 revert** — 除非用户明确要求 reset，否则使用 git revert 保留历史
- **不回退 harness/ 和 knowledge/** — 这些文件不应受 feature 回退影响
