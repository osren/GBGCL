# {项目名称} — Agent 工作入口

> Open Harness V2 | {YYYY-MM-DD}

## 项目概述

| 项目 | 内容 |
|------|------|
| 语言 | {language} |
| 框架 | {framework} |
| 数据库 | {database} |
| 缓存 | {cache} |
| 消息队列 | {mq} |
| 架构风格 | {architecture} |
| 业务领域 | {domain} |

## 目录导航

```
docs/
├── harness/          # 约束层（必须遵守）
│   ├── index.md          ← 约束总入口
│   ├── invariants/       ← 业务不变量
│   ├── architecture/     ← 架构约束
│   ├── infrastructure/   ← 技术规范
│   └── linters/          ← 自动化检查
├── knowledge/        # 知识库（系统现状）
│   ├── index.md          ← 知识库入口
│   ├── architecture.md   ← 系统架构
│   ├── domain-model.md   ← 领域模型
│   ├── api/              ← 接口文档
│   ├── database/         ← 数据库文档
│   ├── prd/              ← 产品需求归档
│   └── exp/              ← 经验总结
├── specs/            # 工作区域
│   ├── active/           ← 进行中的需求
│   └── completed/        ← 已完成的需求
└── hooks/            # 审查机制
    ├── index.md          ← Hook 配置入口
    ├── post-action.md    ← 动作后检查
    └── cr-checklist.md   ← CR 检查清单
```

## 关键约束速览

> 完整约束见 [docs/harness/index.md](docs/harness/index.md)

{列出最重要的 3-5 条约束，例如：}
- **AMT-001** 金额计算必须使用 BigDecimal，禁止 double/float
- **SEC-001** 接口必须验证用户权限，禁止越权访问
- **API-001** 对外接口必须实现幂等性
- **BIZ-001** 核心业务状态流转必须满足业务不变量约束

## 常用命令

```bash
/oh:propose <feature> <prd-path>   # Phase 1：生成 proposal/spec/design
/oh:propose <feature> --continue   # Phase 2：生成 tasks/verification
/oh:review [feature]               # 阶段文档 review / Design 审批 / apply 后综合 review
/oh:apply [feature]                # 执行实现（需先审批通过，完成后自动 review）
/oh:verify [feature]               # 独立验收
/oh:archive [feature]              # 归档需求
/oh:status                         # 项目健康度报告
/oh:rollback [feature]             # 回退到指定 checkpoint
```

## Agent 工作规则

1. **每次任务开始前必须读取本文件**，了解项目概况
2. **harness 约束全量加载**：先读 `docs/harness/index.md`，再依次读取各子目录 `index.md` 及其全部叶子规则文件。harness 下的所有文件不得跳过，不得只按局部约束工作
3. **knowledge 知识库按需加载**：先读 `docs/knowledge/index.md`，再按任务需要进入相关索引或稳定入口（如 `api/index.md`、`database/index.md`、`prd/index.md`、`exp/index.md`、`service-boundary.md`），最后按需读取叶子文档。不得跳过知识入口直接凭经验推断系统现状
4. **不确定的信息通过交互问答确认**，不捏造
5. **新需求开发必须先生成 proposal + spec + design 并审批 design**，再进入执行阶段
