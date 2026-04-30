# Open Harness V1：SDD + Harness Engineering 融合开发范式

> **版本**: v1.0 | **日期**: 2026-03-26
> **定位**: 面向企业级系统（以信贷交易系统为例）的AI辅助开发范式

---

## 一、范式概述

### 1.1 核心理念

**"人类驾驭，Agent执行"** — 融合Spec-Driven Development（规格驱动开发）与Harness Engineering（驾驭工程），构建一套让AI Agent高效、安全、合规地执行软件开发的工程范式。

| 传统开发 | Open Harness |
|---------|-------------|
| 人写代码 | 人写Spec，Agent写代码 |
| 文档是附属品 | Spec是唯一truth source |
| 事后review | 约束前置、机械化执行 |
| 经验驱动质量 | 不变量驱动质量 |

### 1.2 设计原则

1. **给地图，不给百科全书** — Agent入口文件控制在100行内，通过指针跳转到详细文档
2. **约束前置，自由执行** — 先定义不可违反的规则，在规则边界内给Agent最大自由度
3. **渐进式披露** — Agent按任务阶段加载最小必要上下文，不一次灌入全部信息
4. **规则可机器执行** — 每条规则最终要能被linter/CI自动验证，而非仅靠文档约定
5. **文档即代码** — Spec和规则文件纳入版本管理，变更可追溯

---

## 二、目录结构

```
project-root/
├── CLAUDE.md                        # [必需] Agent总入口（导航地图）
├── .harness/                        # [必需] 工程约束层
│   ├── constitution.md              # 项目宪法：技术栈、架构原则、安全红线
│   ├── domain-model.md              # 领域模型：核心业务实体和关系
│   ├── invariants.md                # 不变量规则：编号化、可机器执行
│   └── quality-gates.md             # 质量门禁：CI检查项定义
├── specs/                           # [必需] 需求规格层
│   ├── _template.md                 # Spec模板
│   └── {feature-name}/
│       ├── spec.md                  # What + Why（需求规格）
│       ├── plan.md                  # How（技术方案）
│       ├── tasks.md                 # 可执行任务清单
│       └── verification.md          # 功能测试验证记录
├── docs/                            # [必需] 知识文档层
│   ├── architecture.md              # 系统架构概览
│   ├── api/                         # 接口文档
│   │   └── {service-name}-api.md    # 按服务划分的接口文档
│   └── lessons/                     # 经验沉淀
│       └── {YYYYMMDD}-{title}.md    # 技术经验与踩坑记录
└── .linters/                        # [推荐] 自动化规则执行
    └── invariant-check.sh           # 不变量规则自动检查脚本
```

**总计：9-10个核心文档 + 按需的spec、接口文档和经验沉淀文件**

### 目录设计理由

| 层级 | 职责 | 变更频率 | Agent消费方式 |
|------|------|----------|--------------|
| CLAUDE.md | 导航入口 | 极低频 | 每次任务首先读取 |
| .harness/ | 全局约束 | 低频 | 作为system prompt锚点 |
| specs/ | 功能规格 | 每feature创建 | 按当前任务按需读取 |
| docs/ | 架构知识、接口文档、经验 | 中频 | 架构级任务时读取 |
| .linters/ | 规则执行 | 低频 | CI自动执行 |

---

## 三、各文档详细说明与模板

### 3.1 CLAUDE.md — Agent导航入口

**用途**：AI Agent进入项目时的第一个读取文件。提供项目全貌的"地图"，指向详细文档。

**要求**：
- 控制在100行以内
- 只放指针和速览，不放详细内容
- 包含最关键的3-5条硬约束速览

```markdown
# {项目名称} — Agent指引

## 项目概览
{一句话描述项目定位和技术栈}

## 导航地图
| 需要了解 | 位置 |
|----------|------|
| 项目原则和红线 | .harness/constitution.md |
| 领域模型 | .harness/domain-model.md |
| 不变量规则 | .harness/invariants.md |
| 质量门禁 | .harness/quality-gates.md |
| 系统架构 | docs/architecture.md |
| 接口文档 | docs/api/ |
| 经验沉淀 | docs/lessons/ |
| 当前feature开发 | specs/ |
| Spec模板 | specs/_template.md |

## 关键约束（速览）
1. {约束1 — 引用invariants编号}
2. {约束2}
3. {约束3}

## 工作流
1. 读取本文件了解项目全貌
2. 读取 .harness/ 了解约束规则
3. 读取 specs/{feature}/ 了解当前任务
4. 按 tasks.md 逐任务执行
5. 完成后填写 verification.md 记录测试验证结果
6. 确保所有 invariants 规则通过
```

---

### 3.2 .harness/constitution.md — 项目宪法

**用途**：定义不可妥协的工程原则和技术栈。相当于Agent的"行为边界"。

**要求**：
- 只放"绝对不可违反"的原则
- 避免写成百科全书
- 与invariants.md互补：constitution管方向，invariants管具体规则

```markdown
# 项目宪法

## 技术栈
- 语言: {如 Java 17}
- 框架: {如 Spring Boot 3.x}
- 数据库: {如 MySQL 8.0}
- 缓存: {如 Redis 7}
- 消息队列: {如 RocketMQ 5.x}
- 构建工具: {如 Gradle 8.x}

## 架构原则
1. {原则1：如"微服务架构，单一职责"}
2. {原则2：如"API优先设计，接口先于实现"}
3. {原则3：如"数据库变更必须通过migration脚本"}

## 安全红线
1. {红线1：如"密钥和凭据绝不硬编码，使用配置中心"}
2. {红线2：如"所有外部输入必须校验（Parse, Don't Validate）"}
3. {红线3：如"日志禁止输出完整敏感信息"}

## 编码规范
- 代码风格: {如 Google Java Style}
- 命名约定: {如 "Service层方法用动词开头"}
- 异常处理: {如 "业务异常继承BizException，系统异常继承SysException"}
```

---

### 3.3 .harness/domain-model.md — 领域模型

**用途**：描述核心业务实体及其关系，确保Agent正确理解业务语义。

**要求**：
- 使用简洁的实体关系描述
- 包含核心术语的消歧义定义
- 随业务演进及时更新

```markdown
# 领域模型

## 核心术语表
| 术语 | 英文 | 定义 |
|------|------|------|
| {术语1} | {English} | {精确定义} |
| {术语2} | {English} | {精确定义} |

## 核心实体
### {实体名称}
- 描述: {一句话}
- 关键属性: {列出核心字段}
- 关联关系: {与其他实体的关系}

## 实体关系图（文字描述）
{实体A} 1--N {实体B}: {关系描述}
{实体B} N--1 {实体C}: {关系描述}

## 核心业务流程
1. {流程名}: {实体A} → {操作} → {实体B}
2. ...
```

---

### 3.4 .harness/invariants.md — 不变量规则（核心亮点）

**用途**：编号化管理的强制性业务和技术规则。每条规则必须是可机器验证的。Agent生成代码时必须遵守。

**要求**：
- 每条规则有唯一编号（类别缩写-序号）
- 必须包含"检查方式"（如何机器验证）
- 必须包含"修复指引"（Agent违规时如何修复）

**编号体系**：
- `AMT-xxx`: 金额相关规则
- `SEC-xxx`: 安全相关规则
- `API-xxx`: 接口相关规则
- `DAT-xxx`: 数据相关规则
- `AUD-xxx`: 审计相关规则
- `PERF-xxx`: 性能相关规则
- `TXN-xxx`: 交易相关规则

```markdown
# 不变量规则

## AMT-001: 金额类型约束
- **规则**: 所有金额字段类型必须为BigDecimal
- **检查方式**: 静态扫描float/double类型的金额命名字段
- **修复指引**: 替换为BigDecimal，添加RoundingMode.HALF_UP
- **CI脚本**: .linters/invariant-check.sh #AMT-001

## SEC-001: 敏感数据加密
- **规则**: 身份证号、银行卡号、手机号存储时加密，展示时脱敏
- **检查方式**: 扫描Entity中敏感字段是否有@EncryptField注解
- **修复指引**: 添加@EncryptField(存储)和@MaskField(展示)

## API-001: 接口幂等性
- **规则**: 所有写操作API必须支持幂等
- **检查方式**: 检查POST/PUT方法是否有@Idempotent注解
- **修复指引**: 添加@Idempotent注解，实现requestId去重

## TXN-001: 交易流水号唯一
- **规则**: 每笔交易必须生成全局唯一流水号
- **检查方式**: 检查交易入口方法是否调用TradeNoGenerator
- **修复指引**: 使用TradeNoGenerator.generate()生成流水号
```

---

### 3.5 .harness/quality-gates.md — 质量门禁

**用途**：定义CI/CD中的质量检查项，所有PR必须通过才能合并。

```markdown
# 质量门禁

## 编译门禁
- [ ] 编译零错误、零警告

## 测试门禁
- [ ] 单元测试全部通过
- [ ] 新增代码测试覆盖率 >= 80%
- [ ] 关键路径（资金流转）覆盖率 >= 95%

## 代码质量门禁
- [ ] SonarQube扫描零新增Critical/Blocker
- [ ] 无hardcoded密钥/凭据
- [ ] 不变量规则检查全部通过（.linters/invariant-check.sh）

## 安全门禁
- [ ] 依赖漏洞扫描通过（无High/Critical）
- [ ] SQL注入检查通过
- [ ] XSS检查通过

## 文档门禁
- [ ] 新增API有对应的接口文档（docs/api/）
- [ ] spec.md状态已更新
- [ ] verification.md已填写测试验证结果
```

---

### 3.6 specs/_template.md — Feature Spec模板

**用途**：创建新feature时的标准模板。每个feature目录下包含4个文件：spec.md、plan.md、tasks.md、verification.md。

```markdown
# Feature: {feature-name}

> **状态**: draft | spec-review | approved | in-progress | testing | done
> **创建日期**: {YYYY-MM-DD}
> **负责人**: {name}

---

## spec.md — 需求规格（What + Why）

### 需求概述
{一段话描述这个feature解决什么问题，为谁解决}

### 用户故事
- 作为{角色}，我需要{功能}，以便{价值}
- ...

### 验收标准
1. {可验证的标准1}
2. {可验证的标准2}
3. ...

### 非功能需求
- 性能: {如"接口响应 < 200ms"}
- 安全: {如"需要登录鉴权"}
- 合规: {如"涉及征信查询需记录审计日志"}

### 约束和依赖
- 必须遵守的invariants: {如 AMT-001, SEC-001, AUD-001}
- 外部依赖: {如"依赖征信查询接口"}

---

## plan.md — 技术方案（How）

### 方案概述
{简述技术实现思路}

### 影响范围
- 新增: {新增的模块/文件}
- 修改: {修改的模块/文件}
- 依赖: {新引入的依赖}

### 数据模型变更
{表结构变更，如有}

### 接口设计
{新增/修改的API端点}

### 关键设计决策
| 决策 | 方案 | 理由 |
|------|------|------|
| {决策1} | {选择的方案} | {为什么} |

---

## tasks.md — 任务清单

### 执行任务
- [ ] Task 1: {描述}
  - 涉及文件: {file paths}
  - 关联invariants: {如 AMT-001}
- [ ] Task 2: {描述}
- [ ] Task 3: 编写单元测试
- [ ] Task 4: 编写集成测试
- [ ] Task 5: 更新接口文档（docs/api/）

### 验证清单
- [ ] 所有验收标准通过
- [ ] invariants规则检查通过
- [ ] quality-gates全部绿灯

---

## verification.md — 功能测试验证

### 测试环境
- 环境: {dev / staging / pre-prod}
- 数据准备: {测试数据说明}

### 功能测试用例
| 编号 | 测试场景 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|---------|------|---------|---------|------|
| TC-001 | {场景描述} | {输入数据} | {预期} | {实际} | PASS/FAIL |

### 边界条件测试
| 编号 | 边界场景 | 预期结果 | 实际结果 | 状态 |
|------|---------|---------|---------|------|
| BC-001 | {边界描述} | {预期} | {实际} | PASS/FAIL |

### 异常场景测试
| 编号 | 异常场景 | 预期处理 | 实际处理 | 状态 |
|------|---------|---------|---------|------|
| EX-001 | {异常描述} | {预期} | {实际} | PASS/FAIL |

### 验证结论
- 验证日期: {YYYY-MM-DD}
- 验证人: {name}
- 结论: {通过 / 有遗留问题}
- 遗留问题: {如有}
```

---

### 3.7 docs/architecture.md — 系统架构

**用途**：系统级架构概览，帮助Agent理解模块边界和依赖关系。

```markdown
# 系统架构

## 架构风格
{如：微服务架构 / 模块化单体}

## 服务拓扑
{列出核心服务及职责}

| 服务 | 职责 | 技术栈 |
|------|------|--------|
| {服务1} | {职责} | {技术} |

## 分层规则
{如：Controller → Service → Repository，禁止跨层调用}

## 模块依赖规则
{明确允许和禁止的依赖方向}

## 基础设施
- 数据库: {类型、分库分表策略}
- 缓存: {策略}
- 消息队列: {Topic命名规则}
- 监控: {工具链}
```

---

### 3.8 docs/api/{service-name}-api.md — 接口文档

**用途**：按服务记录API接口契约，作为前后端协作和Agent生成代码的参考依据。

**要求**：
- 按服务拆分文件，避免单文件过大
- 每个接口包含：路径、方法、入参、出参、错误码
- 与spec中的接口设计保持一致

```markdown
# {服务名称} 接口文档

> **版本**: v1.0
> **更新日期**: {YYYY-MM-DD}
> **Base URL**: /api/v1/{service-prefix}

## 接口列表

### {接口名称}
- **路径**: {POST|GET|PUT|DELETE} /api/v1/xxx
- **描述**: {一句话描述}
- **鉴权**: {是否需要登录，权限要求}

**请求参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| {field} | {type} | {Y/N} | {描述} |

**响应结果**:
```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

**错误码**:
| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| {code} | {描述} | {建议} |
```

---

### 3.9 docs/lessons/{YYYYMMDD}-{title}.md — 经验沉淀

**用途**：记录开发过程中的技术经验、踩坑记录、最佳实践，形成团队知识库。后续AI Agent和新成员可直接参考，避免重复踩坑。

**要求**：
- 一事一记，标题清晰
- 包含背景、问题、解决方案、启示
- 沉淀可复用的经验，而非流水账

```markdown
# {经验标题}

> **日期**: {YYYY-MM-DD}
> **作者**: {name}
> **标签**: {如 性能优化 | 并发问题 | 数据一致性 | 第三方对接}

## 背景
{什么场景下遇到了这个问题}

## 问题描述
{具体的技术问题或挑战}

## 解决方案
{采用了什么方案，关键代码或配置}

## 根因分析
{问题的根本原因是什么}

## 经验总结
{可复用的经验和教训，对未来开发的指导意义}
```

---

## 四、工作流

### 4.1 标准开发流

```
                    ┌─────────────┐
                    │  需求输入    │
                    └──────┬──────┘
                           ▼
                  ┌─────────────────┐
                  │  创建feature目录  │
                  │  填写spec.md     │
                  └──────┬──────────┘
                         ▼
                  ┌─────────────────┐
              ┌───│  人工审核spec     │
              │   └──────┬──────────┘
              │          ▼ approved
              │   ┌─────────────────┐
              │   │ AI生成plan.md    │
              │   └──────┬──────────┘
              │          ▼
              │   ┌─────────────────┐
              │   │  人工审核plan     │
              │   └──────┬──────────┘
              │          ▼ approved
              │   ┌─────────────────┐
              │   │ AI生成tasks.md   │
              │   └──────┬──────────┘
              │          ▼
              │   ┌─────────────────┐
    reject    │   │ AI逐task执行     │◄───┐
    ◄─────────┤   │ (写代码+测试)     │    │ fix
              │   └──────┬──────────┘    │
              │          ▼               │
              │   ┌─────────────────┐    │
              │   │ CI验证           │    │
              │   │ - 编译测试        │────┘
              │   │ - invariants检查  │  fail
              │   │ - quality-gates   │
              │   └──────┬──────────┘
              │          ▼ pass
              │   ┌──────────────────┐
              │   │ 功能测试验证       │
              │   │ 填写verification │
              │   └──────┬───────────┘
              │          ▼
              │   ┌─────────────────┐
              └──►│  人工最终审核     │
                  │  合并代码         │
                  └──────┬──────────┘
                         ▼
                  ┌─────────────────┐
                  │ 经验沉淀(如有)    │
                  │ 更新接口文档      │
                  └─────────────────┘
```

### 4.2 关键流程要点

| 环节 | 人的工作 | Agent的工作 |
|------|---------|------------|
| 需求 | 编写spec.md | — |
| 方案 | 审核plan | 生成plan.md |
| 执行 | — | 按tasks逐项实现 |
| 验证 | 最终审批 | 运行CI、修复问题、填写verification.md |
| 沉淀 | 审核经验 | 生成接口文档、提炼经验沉淀 |
| 维护 | 更新invariants | 执行invariant检查 |

---

## 五、信贷交易系统完整模板示例

以下是一个完整的信贷交易系统项目模板：

### CLAUDE.md

```markdown
# 信贷交易系统 — Agent指引

## 项目概览
信贷交易核心系统，负责贷款全生命周期的交易处理。Java 17 / Spring Boot 3 / MyBatis Plus / MySQL 8 / Redis / RocketMQ / Gradle 8。

## 导航地图
| 需要了解 | 位置 |
|----------|------|
| 项目原则和红线 | .harness/constitution.md |
| 领域模型和术语 | .harness/domain-model.md |
| 不变量规则 | .harness/invariants.md |
| 质量门禁 | .harness/quality-gates.md |
| 系统架构 | docs/architecture.md |
| 接口文档 | docs/api/ |
| 经验沉淀 | docs/lessons/ |
| 当前开发 | specs/ |

## 关键约束（速览）
1. 金额必须使用BigDecimal（AMT-001）
2. 所有交易接口幂等（API-001）
3. 每笔交易必须有唯一流水号（TXN-001）
4. 敏感数据加密存储脱敏展示（SEC-001）
5. 资金变动必须记录交易流水和审计日志（AUD-001, TXN-002）

## 工作流
1. 读取本文件 → 读取.harness/ → 读取specs/{当前feature}/
2. 按tasks.md逐项执行
3. 完成后填写verification.md
4. 确保invariants全部通过
```

### .harness/constitution.md

```markdown
# 信贷交易系统 — 项目宪法

## 技术栈
- 语言: Java 17
- 框架: Spring Boot 3.2, MyBatis Plus 3.5
- 数据库: MySQL 8.0（读写分离，分库分表）
- 缓存: Redis 7 (Cluster模式)
- 消息队列: RocketMQ 5.x
- 构建: Gradle 8.5 + JDK 17
- 容器: Docker + Kubernetes
- 注册中心: Nacos 2.x
- 分布式ID: Leaf / Snowflake

## 架构原则
1. 微服务架构，按交易域拆分（用户服务、产品服务、交易服务、账务服务、清结算服务、放款服务、还款服务、风控服务）
2. API First：先定义接口契约，再实现
3. 数据库变更必须通过Flyway migration脚本
4. 服务间通信：同步用OpenFeign，异步用RocketMQ事务消息
5. 配置外置，使用Nacos配置中心
6. 资金类操作必须保证最终一致性（TCC / MQ + 本地事务表）

## 安全红线
1. 密钥、凭据、数据库密码绝不硬编码，全部走配置中心+加密
2. 所有外部输入（用户输入、外部API返回）必须校验后使用
3. 日志中绝不输出完整身份证号、银行卡号、CVV
4. SQL拼接绝不使用字符串拼接，必须用参数化查询
5. 内部API必须鉴权，使用JWT + 服务间Token
6. 资金操作必须双重校验（交易密码 + 短信验证码）

## 编码规范
- 代码风格: 遵循阿里巴巴Java开发手册
- Service层方法：动词开头（create/update/delete/query/calculate）
- 异常体系: BizException（业务异常,含错误码） / SysException（系统异常）
- 返回值: 统一使用Result<T>包装
- 交易类接口: 必须记录完整的入参出参日志（脱敏后）

## Gradle构建规范
- 统一依赖版本管理: 使用 libs.versions.toml
- 多模块结构: {service}-api（接口定义）、{service}-service（实现）、{service}-starter（启动）
- 通用插件: spotless(代码格式化)、jacoco(覆盖率)、sonarqube(质量扫描)
```

### .harness/domain-model.md

```markdown
# 信贷交易系统 — 领域模型

## 核心术语表
| 术语 | 英文 | 定义 |
|------|------|------|
| 进件 | Application | 用户提交的贷款申请，包含个人信息、贷款金额、期限等 |
| 授信 | Credit Line | 风控审批通过后给予用户的信用额度 |
| 支用 | Drawdown | 用户在授信额度内发起的实际借款交易 |
| 借据 | Loan | 支用成功后生成的贷款合同记录 |
| 期次 | Installment | 还款计划中的单期还款记录 |
| 交易流水 | Transaction | 每笔资金变动的流水记录，具有唯一流水号 |
| 逾期 | Overdue | 超过应还日期仍未还款的状态 |
| 代偿 | Subrogation | 担保方代为偿还逾期贷款 |
| 结清 | Settlement | 贷款本息全部偿还完毕 |
| 台账 | Ledger | 记录每笔贷款资金进出的会计账本 |

## 核心实体

### 用户(User)
- 关键属性: userId, realName, idCard(加密), phone(加密), creditScore
- 关系: 一个用户可以有多个进件

### 贷款产品(LoanProduct)
- 关键属性: productId, productName, minAmount, maxAmount, annualRate, termRange, repayMethod
- 关系: 一个产品对应一套费率和风控规则

### 进件(Application)
- 关键属性: appId, userId, productId, applyAmount, applyTerm, status
- 状态流转: 提交 → 风控审核 → 人工审批 → 签约 → 放款（或拒绝）
- 关系: 一个进件对应一次风控评估

### 授信额度(CreditLine)
- 关键属性: creditId, userId, totalAmount, availableAmount, frozenAmount, expireDate
- 关系: 一个授信额度可以多次支用

### 借据(Loan)
- 关键属性: loanId, creditId, tradeNo, principal, term, rate, status, loanDate, settleDate
- 关系: 一个借据对应多个期次，对应多条交易流水

### 还款计划(RepaymentPlan)
- 关键属性: planId, loanId, installmentNo, dueDate, principal, interest, penalty, status
- 关系: 一个借据有N期还款计划

### 交易流水(Transaction)
- 关键属性: tradeNo, loanId, type, amount, balanceBefore, balanceAfter, status, tradeTime
- 关系: 每笔资金变动生成一条流水

### 台账(Ledger)
- 关键属性: ledgerId, loanId, tradeNo, debitAmount, creditAmount, balance, bookDate
- 关系: 每条交易流水对应一条或多条台账记录

## 核心业务流程
1. 进件流程: User → Application → RiskAssessment → Approval → Contract
2. 放款流程: Contract → Drawdown → Transaction(放款) → Loan → RepaymentPlan生成 → Ledger记账
3. 还款流程: 到期扣款/主动还款 → Transaction(还款) → 更新RepaymentPlan → 更新Loan状态 → Ledger记账
4. 逾期流程: 还款日T+1检查 → 标记逾期 → 计算罚息 → 催收 → 代偿(如有)
5. 提前还款: 试算 → 申请 → Transaction(提前还款) → 重算还款计划 → Ledger记账
```

### .harness/invariants.md

```markdown
# 信贷交易系统 — 不变量规则

## 金额规则

### AMT-001: 金额类型
- **规则**: 所有金额字段类型必须为BigDecimal
- **检查**: grep扫描float/double类型的金额命名字段(amount/money/fee/interest/principal/balance)
- **修复**: 替换为BigDecimal，运算使用add/subtract/multiply/divide

### AMT-002: 利率精度
- **规则**: 利率计算精度到小数点后8位，金额精度到分（小数点后2位）
- **检查**: 利率相关BigDecimal的scale >= 8，金额scale = 2
- **修复**: setScale(8, RoundingMode.HALF_UP) / setScale(2, RoundingMode.HALF_UP)

### AMT-003: 还款计划平衡
- **规则**: 各期本金之和必须等于借款本金，误差不超过1分钱
- **检查**: 单元测试断言 sum(installment.principal) == loan.principal (允许±0.01)
- **修复**: 尾差调整法 — 最后一期吸收尾差

## 交易规则

### TXN-001: 交易流水号唯一
- **规则**: 每笔交易必须生成全局唯一流水号，格式: {业务类型}{日期}{序号}
- **检查**: 交易入口方法必须调用TradeNoGenerator.generate()
- **修复**: 使用分布式ID生成器生成流水号

### TXN-002: 交易流水完整性
- **规则**: 每笔资金变动必须同时写入Transaction和Ledger
- **检查**: Transaction写入方法必须包含Ledger写入调用
- **修复**: 在同一事务中完成Transaction和Ledger的写入

### TXN-003: 资金借贷平衡
- **规则**: 台账中每笔交易的借方金额之和必须等于贷方金额之和
- **检查**: 单元测试断言 sum(debit) == sum(credit) for each tradeNo
- **修复**: 检查记账分录逻辑

### TXN-004: 额度冻结
- **规则**: 支用前必须先冻结额度，支用失败必须解冻
- **检查**: Drawdown流程必须包含freezeCredit/unfreezeCredit调用
- **修复**: 在支用流程中加入额度冻结/解冻步骤

## 安全规则

### SEC-001: 敏感数据存储
- **规则**: 身份证号、银行卡号、手机号存储时AES加密
- **检查**: Entity中idCard/bankCard/phone字段必须有@EncryptField注解
- **修复**: 添加@EncryptField注解

### SEC-002: 敏感数据展示
- **规则**: 身份证号显示前3后4，银行卡号显示后4位，手机号显示前3后4
- **检查**: VO/DTO中敏感字段必须有@MaskField注解
- **修复**: 添加@MaskField(type=IDCARD/BANKCARD/PHONE)

### SEC-003: 日志安全
- **规则**: 日志中不得输出完整敏感信息
- **检查**: log语句中引用敏感字段时必须使用LogMaskUtil.mask()
- **修复**: 包装为LogMaskUtil.mask(value)

## 接口规则

### API-001: 幂等性
- **规则**: 所有写操作(POST/PUT/DELETE)必须幂等
- **检查**: 写操作方法必须有@Idempotent注解
- **修复**: 添加@Idempotent，使用requestId+Redis去重

### API-002: 统一响应
- **规则**: 所有API返回Result<T>格式
- **检查**: Controller方法返回类型必须为Result<?>
- **修复**: 统一包装为Result.success(data)/Result.fail(code, msg)

### API-003: 入参校验
- **规则**: 所有入参DTO使用JSR-303注解校验
- **检查**: Controller方法参数必须有@Valid/@Validated注解
- **修复**: 添加校验注解，DTO字段添加@NotNull/@Size等

## 审计规则

### AUD-001: 状态变更审计
- **规则**: 所有业务实体状态变更必须记录审计日志
- **检查**: 涉及status字段update的Service方法必须有@AuditLog注解
- **修复**: 添加@AuditLog(action="状态变更描述")

### AUD-002: 审计日志不可变
- **规则**: 审计日志表只允许INSERT，不允许UPDATE/DELETE
- **检查**: AuditLog相关Mapper不得有update/delete方法
- **修复**: 删除违规方法

## 性能规则

### PERF-001: 分页查询
- **规则**: 所有列表查询必须分页，单页上限100条
- **检查**: 列表查询方法必须接受PageParam，且pageSize <= 100
- **修复**: 添加分页参数，Service层强制校验上限

### PERF-002: 禁止全表扫描
- **规则**: WHERE条件必须命中索引
- **检查**: SQL扫描+EXPLAIN分析（集成测试阶段）
- **修复**: 添加合适的索引或优化查询条件
```

### docs/api/trade-service-api.md（接口文档示例）

```markdown
# 交易服务 接口文档

> **版本**: v1.0
> **更新日期**: 2026-03-26
> **Base URL**: /api/v1/trade

## 1. 支用申请

- **路径**: POST /api/v1/trade/drawdown
- **描述**: 用户在授信额度内发起借款
- **鉴权**: 需要登录 + 交易密码 + 短信验证码

**请求参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| creditId | Long | Y | 授信额度ID |
| amount | BigDecimal | Y | 借款金额（元） |
| term | Integer | Y | 借款期数 |
| purpose | String | N | 借款用途 |
| requestId | String | Y | 幂等请求ID |

**响应结果**:
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "tradeNo": "DRW20260326000001",
    "loanId": 100001,
    "amount": "50000.00",
    "term": 12,
    "rate": "0.00045000",
    "status": "PROCESSING"
  }
}
```

**错误码**:
| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| TRADE_001 | 额度不足 | 提示用户可用额度 |
| TRADE_002 | 额度已过期 | 引导用户重新申请授信 |
| TRADE_003 | 交易密码错误 | 提示剩余重试次数 |
| TRADE_004 | 重复请求 | 返回原交易结果 |

## 2. 还款

- **路径**: POST /api/v1/trade/repay
- **描述**: 用户主动还款（当期/提前还款）
- **鉴权**: 需要登录

**请求参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| loanId | Long | Y | 借据ID |
| repayType | String | Y | 还款类型: CURRENT(当期)/EARLY_FULL(全额提前)/EARLY_PARTIAL(部分提前) |
| amount | BigDecimal | N | 部分提前还款金额（repayType=EARLY_PARTIAL时必填） |
| requestId | String | Y | 幂等请求ID |

**响应结果**:
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "tradeNo": "RPY20260326000001",
    "repayPrincipal": "4500.00",
    "repayInterest": "225.00",
    "totalRepay": "4725.00",
    "status": "SUCCESS"
  }
}
```

**错误码**:
| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| REPAY_001 | 借据不存在或已结清 | 检查loanId |
| REPAY_002 | 还款金额不合法 | 校验金额范围 |
| REPAY_003 | 扣款失败 | 提示用户检查余额 |

## 3. 提前还款试算

- **路径**: POST /api/v1/trade/early-repayment/trial
- **描述**: 计算提前还款应还金额明细
- **鉴权**: 需要登录

**请求参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| loanId | Long | Y | 借据ID |
| repayType | String | Y | EARLY_FULL / EARLY_PARTIAL |
| amount | BigDecimal | N | 部分提前还款本金（EARLY_PARTIAL时必填） |

**响应结果**:
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "repayPrincipal": "45000.00",
    "accruedInterest": "180.00",
    "savedInterest": "2700.00",
    "totalRepay": "45180.00",
    "effectiveDate": "2026-03-27"
  }
}
```
```

### docs/lessons/ 示例 — 经验沉淀

**docs/lessons/20260320-repayment-precision.md**
```markdown
# 还款计划生成的精度陷阱

> **日期**: 2026-03-20
> **作者**: 李四
> **标签**: 数据精度 | 金额计算 | 还款计划

## 背景
在等额本息还款计划生成中，12期还款计划各期本金之和与借款本金出现了0.03元的差异，触发了AMT-003不变量校验失败。

## 问题描述
等额本息公式计算每期还款额后，拆分本金和利息时，由于每期都做了四舍五入到分的操作，12次舍入累积导致总本金与原始借款本金不一致。

## 解决方案
采用"尾差调整法"：
1. 前N-1期正常计算并四舍五入
2. 最后一期本金 = 总本金 - 前N-1期本金之和
3. 最后一期利息 = 最后一期月供 - 最后一期本金

关键代码片段：
```java
// 最后一期尾差调整
if (i == term) {
    installment.setPrincipal(remainPrincipal);
} else {
    installment.setPrincipal(monthPrincipal.setScale(2, RoundingMode.HALF_UP));
    remainPrincipal = remainPrincipal.subtract(installment.getPrincipal());
}
```

## 根因分析
BigDecimal四舍五入在多次迭代计算中会产生累积误差，信贷场景要求"分分对齐"，因此必须使用尾差调整而非简单的四舍五入。

## 经验总结
1. 涉及多期拆分的金额计算，最后一期必须使用"倒算法"（用总额减去前面已算的总和）
2. 此模式适用于所有"总额拆分为N份且要求精确对齐"的场景（如分润、分佣）
3. 已将此规则固化为不变量 AMT-003
```

### specs/ 示例 — 实时交易对账功能

**specs/realtime-reconciliation/spec.md**
```markdown
# Feature: 实时交易对账

> **状态**: approved
> **创建日期**: 2026-03-26
> **负责人**: 王五

## 需求概述
实现信贷交易系统与资金方（银行/信托）的实时交易对账能力，确保每笔放款、还款交易的资金状态与资金方一致，及时发现差异并告警。

## 用户故事
- 作为运营人员，我需要实时看到交易对账结果，以便在资金差异发生时第一时间处理
- 作为财务人员，我需要每日对账汇总报表，以便进行资金核对和财务处理
- 作为系统，我需要自动识别对账差异并分类（金额不一致、单边账、状态不一致），以便触发不同的处理流程

## 验收标准
1. 支持逐笔实时对账（交易完成后5分钟内完成对账）
2. 对账差异自动分类：金额差异、单边账（我方有对方无/对方有我方无）、状态差异
3. 差异发现后自动触发告警（企微/钉钉通知）
4. 对账结果持久化，支持历史查询
5. 提供日终对账汇总报表
6. 支持手动触发重新对账

## 非功能需求
- 性能: 单笔对账 < 500ms，日终批量对账10万笔 < 30分钟
- 可用性: 对账系统故障不影响交易主流程
- 合规: 对账差异超过阈值必须上报（AUD-001）

## 涉及不变量
- AMT-001（金额BigDecimal）
- TXN-001（交易流水号唯一）
- TXN-002（交易流水完整性）
- AUD-001（状态变更审计）
- API-001（接口幂等）
```

**specs/realtime-reconciliation/plan.md**
```markdown
# Feature: 实时交易对账 — 技术方案

## 方案概述
新增对账服务(reconciliation-service)，通过消费交易完成MQ消息触发实时对账，调用资金方对账接口比对交易状态和金额。

## 影响范围
- 新增: reconciliation-service（对账服务，独立Gradle模块）
- 新增: reconciliation-api（对账服务接口定义）
- 修改: trade-service（交易完成后发送MQ消息）
- 新增表: t_recon_record（对账记录）、t_recon_diff（对账差异）

## Gradle模块结构
```
reconciliation/
├── reconciliation-api/          # 接口定义
│   └── build.gradle.kts
├── reconciliation-service/      # 服务实现
│   └── build.gradle.kts
└── reconciliation-starter/      # 启动模块
    └── build.gradle.kts
```

## 接口设计
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/recon/trigger | 手动触发单笔对账 |
| POST | /api/v1/recon/batch | 触发批量对账 |
| GET | /api/v1/recon/diff/list | 查询对账差异列表 |
| GET | /api/v1/recon/report/daily | 获取日终对账报表 |

## 数据模型变更
```sql
CREATE TABLE t_recon_record (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_no VARCHAR(32) NOT NULL COMMENT '交易流水号',
    channel_code VARCHAR(16) NOT NULL COMMENT '资金方编码',
    our_amount DECIMAL(18,2) NOT NULL COMMENT '我方金额',
    channel_amount DECIMAL(18,2) COMMENT '对方金额',
    our_status VARCHAR(16) NOT NULL COMMENT '我方状态',
    channel_status VARCHAR(16) COMMENT '对方状态',
    recon_result VARCHAR(16) NOT NULL COMMENT '对账结果:MATCH/DIFF/PENDING',
    recon_time DATETIME NOT NULL COMMENT '对账时间',
    created_at DATETIME NOT NULL,
    INDEX idx_trade_no (trade_no),
    INDEX idx_recon_time (recon_time)
);

CREATE TABLE t_recon_diff (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    recon_id BIGINT NOT NULL COMMENT '对账记录ID',
    diff_type VARCHAR(16) NOT NULL COMMENT '差异类型:AMOUNT/STATUS/ONE_SIDE',
    description TEXT COMMENT '差异描述',
    handle_status VARCHAR(16) NOT NULL DEFAULT 'PENDING' COMMENT '处理状态',
    handler VARCHAR(32) COMMENT '处理人',
    handle_time DATETIME COMMENT '处理时间',
    created_at DATETIME NOT NULL,
    INDEX idx_recon_id (recon_id),
    INDEX idx_handle_status (handle_status)
);
```

## 关键设计决策
| 决策 | 方案 | 理由 |
|------|------|------|
| 对账触发方式 | MQ消息驱动（实时）+ 定时任务（日终兜底） | 兼顾实时性和可靠性 |
| 对账服务部署 | 独立服务 | 对账故障不影响交易主流程 |
| 差异处理 | 先告警后人工处理 | 资金差异需要人工判断，不宜自动处理 |
| 资金方对接 | 适配器模式 | 不同资金方接口不同，通过Adapter统一抽象 |
```

**specs/realtime-reconciliation/tasks.md**
```markdown
# Feature: 实时交易对账 — 任务清单

## 任务列表
- [ ] Task 1: 创建reconciliation Gradle模块（api/service/starter三层）
  - 文件: reconciliation/build.gradle.kts 及子模块
  - 说明: 在根settings.gradle.kts中include新模块

- [ ] Task 2: 创建t_recon_record和t_recon_diff表migration脚本
  - 文件: reconciliation-service/src/main/resources/db/migration/V1__init_recon_tables.sql
  - 约束: AMT-001(金额字段DECIMAL)

- [ ] Task 3: 创建Entity和Mapper
  - 文件: entity/ReconRecord.java, entity/ReconDiff.java, mapper/对应Mapper
  - 约束: 使用MyBatis Plus BaseMapper

- [ ] Task 4: 实现资金方对账适配器（Adapter模式）
  - 文件: adapter/ChannelReconAdapter.java, adapter/impl/BankAReconAdapter.java
  - 说明: 定义统一的queryChannelTransaction()接口

- [ ] Task 5: 实现核心对账比对逻辑
  - 文件: service/ReconService.java
  - 约束: AMT-001(金额比对使用compareTo), TXN-001(流水号关联)

- [ ] Task 6: 实现MQ消费者（实时对账触发）
  - 文件: mq/TradeCompleteReconConsumer.java
  - Topic: TOPIC_TRADE_COMPLETE

- [ ] Task 7: 实现对账差异告警通知
  - 文件: notify/ReconAlertService.java
  - 说明: 接入企微/钉钉webhook

- [ ] Task 8: 实现对账API接口
  - 文件: controller/ReconController.java
  - 约束: API-001(幂等), API-002(统一响应), API-003(入参校验)

- [ ] Task 9: 实现日终批量对账定时任务
  - 文件: job/DailyReconJob.java
  - 说明: 使用@Scheduled或XXL-Job

- [ ] Task 10: 编写单元测试
  - 文件: test/ 目录下对应测试类
  - 重点: 对账比对逻辑覆盖（金额一致/不一致/单边账/状态差异）

- [ ] Task 11: 编写集成测试
  - 文件: test/integration/ReconIntegrationTest.java
  - 重点: 完整流程（交易完成→MQ→对账→差异发现→告警）

- [ ] Task 12: 编写接口文档
  - 文件: docs/api/reconciliation-service-api.md

## 验证清单
- [ ] 对账结果准确：金额一致标记MATCH，不一致标记DIFF
- [ ] 单边账识别正确
- [ ] 告警通知及时发送
- [ ] 幂等性：同一笔交易重复触发对账不产生重复记录
- [ ] 日终批量对账正常执行
- [ ] 所有invariants检查通过
```

**specs/realtime-reconciliation/verification.md**
```markdown
# Feature: 实时交易对账 — 功能测试验证

### 测试环境
- 环境: staging
- 数据准备: 使用mock资金方适配器模拟各种对账场景

### 功能测试用例
| 编号 | 测试场景 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|---------|------|---------|---------|------|
| TC-001 | 正常交易对账一致 | 放款交易tradeNo=DRW001 | recon_result=MATCH | | |
| TC-002 | 金额差异对账 | 我方50000，对方49999 | recon_result=DIFF, diff_type=AMOUNT | | |
| TC-003 | 单边账-我方有对方无 | 我方有交易，对方查无此笔 | recon_result=DIFF, diff_type=ONE_SIDE | | |
| TC-004 | 状态差异 | 我方SUCCESS，对方PROCESSING | recon_result=DIFF, diff_type=STATUS | | |
| TC-005 | 手动触发重新对账 | POST /recon/trigger | 生成新对账记录，覆盖旧结果 | | |

### 边界条件测试
| 编号 | 边界场景 | 预期结果 | 实际结果 | 状态 |
|------|---------|---------|---------|------|
| BC-001 | 资金方接口超时 | 对账状态标记PENDING，5分钟后重试 | | |
| BC-002 | 同一笔交易并发触发对账 | 仅执行一次，幂等保证 | | |
| BC-003 | 日终批量10万笔 | 30分钟内完成 | | |

### 异常场景测试
| 编号 | 异常场景 | 预期处理 | 实际处理 | 状态 |
|------|---------|---------|---------|------|
| EX-001 | 对账服务宕机 | 交易主流程不受影响，MQ消息堆积后自动消费 | | |
| EX-002 | 告警通知发送失败 | 记录失败日志，下次轮询重试 | | |

### 验证结论
- 验证日期: {待填写}
- 验证人: {待填写}
- 结论: {待填写}
- 遗留问题: {待填写}
```

---

## 六、渐进式落地指南

### Phase 1：快速起步（第1-2周）

**目标**：3个试点项目跑通基本流程

```
仅建立：
├── CLAUDE.md
├── .harness/
│   └── constitution.md
└── specs/{feature}/
    └── spec.md (可先用单文件)
```

### Phase 2：建立规范（第3-4周）

**目标**：补充约束层，接入CI

```
新增：
├── .harness/
│   ├── domain-model.md
│   ├── invariants.md (从5条核心规则起步)
│   └── quality-gates.md
├── docs/
│   ├── architecture.md
│   └── api/ (按需建立接口文档)
└── specs/ (拆分为 spec/plan/tasks/verification 四文件)
```

### Phase 3：规模推广（第2月）

**目标**：全团队推广，CI强制执行

```
新增：
├── .linters/
│   └── invariant-check.sh (自动化检查)
├── docs/
│   └── lessons/ (开始积累经验沉淀)
+ CI集成quality-gates
+ invariants规则持续扩充
```

### Phase 4：持续演进（第3月+）

**目标**：按需引入高级能力

```
可选引入：
- docs/quality/grades.md (质量评级)
- scripts/doc-gardening.sh (文档自动清理)
- 更多自定义linter规则
```

---

## 七、FAQ

**Q: invariants.md会不会越来越长难以维护？**
A: 控制在50条以内。超出后按业务域拆分为 invariants-{domain}.md。

**Q: 小团队（3-5人）需要完整的目录结构吗？**
A: 不需要。Phase 1只需CLAUDE.md + constitution.md + spec.md，其余渐进引入。

**Q: 如何确保Agent真的遵守invariants？**
A: 三层保障：①CLAUDE.md中速览提醒 ②spec中关联invariant编号 ③CI中.linters/自动检查。

**Q: 与现有的Jira/Confluence等工具如何集成？**
A: specs/是代码仓库内的truth source。Jira用于任务跟踪，Confluence可放非技术文档。二者指向spec而非替代spec。

**Q: 这个范式与TDD冲突吗？**
A: 不冲突。tasks.md中的测试任务天然支持"先写测试"。SDD + TDD = 先写Spec再写Test再让Agent写Code。

**Q: verification.md 和单元测试什么关系？**
A: 单元测试验证代码逻辑正确性，verification.md 记录功能级别的端到端测试验证结果，二者互补。verification.md侧重业务场景覆盖和边界验证。

**Q: 经验沉淀和代码注释的区别？**
A: 代码注释解释"这段代码做什么"，经验沉淀记录"我们为什么这样做、踩过什么坑、其他场景能复用什么"。经验沉淀是跨项目、跨时间的知识传承。

**Q: 接口文档放在docs/api/还是用Swagger自动生成？**
A: 两者互补。docs/api/记录接口契约（含错误码、业务语义），Swagger生成运行时API文档。前者是"设计态"，后者是"运行态"。

---

> **Open Harness V1** — 让AI Agent成为可信赖的工程伙伴，而非不受约束的代码生成器。
