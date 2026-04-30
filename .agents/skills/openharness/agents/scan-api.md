---
description: "API 接口深度扫描 agent。分析代码库中所有对外接口（HTTP REST / Thrift RPC / gRPC / Dubbo），生成完整的接口清单到 docs/knowledge/api/ 目录，包含完整出入参字段详情。"
---

## 任务：生成项目 API 接口清单（含完整出入参）

你是一名后端架构师。请扫描整个代码库，生成一份完整的 API 接口清单，**每个接口须展开所有请求参数和响应字段**，而非仅列出类型名。

## 输出要求

- **位置**：`docs/knowledge/api/index.md`（接口总览 + 模块索引），各模块详细接口输出到 `docs/knowledge/api/{module}-api.md`
- **格式**：Markdown，按模块分组，每个接口独立小节

## 扫描策略

### 步骤 0：OpenAPI / Swagger 文件优先检测

在扫描代码前，先检查以下路径是否存在 OpenAPI/Swagger 规范文件：

```
src/main/resources/static/swagger*.json
src/main/resources/static/openapi*.yaml
src/main/resources/static/openapi*.json
src/main/resources/*.yaml
src/main/resources/*.json
docs/openapi.*
docs/swagger.*
**/openapi.yaml
**/openapi.json
**/swagger.yaml
**/swagger.json
```

**若找到 OpenAPI/Swagger 文件（优先级最高）：**

- 解析 `paths` 字段获取所有接口路径、方法、summary/description
- 解析 `components.schemas`（OpenAPI 3.x）或 `definitions`（Swagger 2.x）获取完整类型定义
- 对每个接口，展开 `requestBody.content.application/json.schema` 的所有字段
- 对每个接口，展开 `responses.200.content.application/json.schema` 的所有字段
- 若 `$ref` 引用其他 schema，递归解析直到基础类型
- 标注字段是否 `required`、`nullable`，以及 `description`、`example`

> OpenAPI 文件是最权威的信源，优先使用，代码扫描作为补充验证。

---

### 步骤 1：HTTP REST 接口扫描（无 OpenAPI 文件时）

#### 1.1 接口定位

- 查找 `@RestController` / `@Controller` 类
- 拼接类级别 `@RequestMapping` + 方法级别 `@GetMapping` / `@PostMapping` / `@PutMapping` / `@DeleteMapping` / `@PatchMapping`
- 提取参数位置：
  - `@PathVariable` → 路径参数
  - `@RequestParam` → 查询参数（记录 defaultValue、required）
  - `@RequestBody` → 请求体，记录类型名，**进入步骤 1.2 展开字段**
  - `@RequestHeader` → 请求头参数
- 提取返回类型（方法签名），**进入步骤 1.2 展开字段**
- 扫描认证注解：`@PreAuthorize` / `@LoginRequired` / `@AuthCheck` / `@NeedLogin` → 标记是否需要鉴权
- 优先取 `@ApiOperation`（Swagger 2）/ `@Operation`（OpenAPI 3）的 `value` 作为功能描述，其次取方法 Javadoc，最后从方法名推断

#### 1.2 出入参字段展开（关键步骤）

对每个 `@RequestBody` 类型和返回值类型，执行以下操作：

1. **找到类定义**：搜索对应的 Java/Kotlin/Go 类文件
2. **提取所有字段**：
   - Java：读取所有非 `static`、非 `transient` 字段，优先取 `@JsonProperty`  的实际字段名
   - 若字段有 `@ApiModelProperty`（Swagger 2）/ `@Schema`（OpenAPI 3），提取 `value`/`description` 作为字段说明
   - 若字段有 `@NotNull` / `@NotBlank` / `@NotEmpty` 等 Bean Validation 注解，标记为必填
   - 若字段类型是另一个自定义类（非 Java 基础类型/集合），**递归展开**该类的字段（最多递归 3 层，超出则只记录类型名）
   - 若字段类型是 `List<T>` / `Set<T>`，记录为 `Array<T类型>`，并展开 T 的字段
   - 若字段类型是 `Map<K,V>`，记录 key/value 类型
3. **处理继承**：若类继承父类，合并父类字段
4. **处理泛型包装**：若返回值是 `Result<T>` / `ApiResponse<T>` / `CommonResult<T>` 等统一响应包装，先展开包装类的公共字段，再展开 `data` 字段对应的 T 类型

---

### 步骤 2：Thrift RPC 接口扫描

- 查找实现 `*.Iface` 的类（如 `XxxHandler implements XxxService.Iface`）
- **优先读取 `.thrift` 文件**：从 `service` 定义提取方法名、入参 struct、返回 struct
- 展开 `.thrift` 文件中对应 struct 的所有字段（`1: required string name`）
- 无 `.thrift` 文件时，从 Java 方法签名 + 对应的 Request/Response 类展开字段

### 步骤 3：gRPC 接口扫描

- 查找继承 `*Grpc.*ImplBase` 的类
- **优先读取 `.proto` 文件**：从 `service` 定义提取 rpc 方法；从 `message` 定义展开所有字段（field number、类型、名称）
- 标注 `repeated` 字段为数组类型
- 标注 `oneof` 字段

### 步骤 4：Dubbo 接口扫描

- 查找 `@DubboService` / `@Service`（com.alibaba.dubbo）类
- 找到对应的接口定义类，提取所有方法签名
- 按步骤 1.2 展开出入参字段

### 步骤 5：通用规则

- **按模块分组**：根据包名或类名前缀推断所属业务模块
- **API 版本**：识别路径中的 `/v1/`、`/v2/` 等版本前缀并保留
- **枚举类型处理**：若字段类型为枚举，列出所有枚举值及含义（从枚举类注释或 `@ApiModelProperty` 提取）

---

## 输出格式

### api.md（总览）

```markdown
# API 接口清单

> 生成时间：YYYY-MM-DD
> 信息来源：OpenAPI 规范文件 / 代码扫描（二选一标注）
> 接口总数：N 个（HTTP: X 个，Thrift: Y 个，gRPC: Z 个）

## 模块索引

| 序号 | 模块 | 协议 | 接口数 | 详情文件 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 用户 | HTTP REST | 8 | [user-api.md](./user-api.md) |
| 2 | 订单 | HTTP REST | 12 | [order-api.md](./order-api.md) |
| 3 | 积分 | Thrift RPC | 5 | [credit-api.md](./credit-api.md) |
```

---

### {module}-api.md（模块详情，每接口独立小节）

```markdown
# {模块名} API 文档

> 协议：HTTP REST
> 基础路径：/api/v1/users
> 入口类：UserController

---

## 接口列表

| 序号 | 路径 | 方法 | 功能 | 需要鉴权 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | /api/v1/users/{id} | GET | 查询用户详情 | 是 |
| 2 | /api/v1/users/login | POST | 用户登录 | 否 |

---

## GET /api/v1/users/{id} — 查询用户详情

**入口**：`UserController.getById`
**鉴权**：需要（@LoginRequired）
**描述**：根据用户 ID 查询用户详细信息

### 请求参数

| 位置 | 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| Path | id | Long | 是 | 用户 ID |

### 响应（`Result<UserVO>`）

**包装结构**

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| code | Integer | 状态码，0=成功 |
| message | String | 提示信息 |
| data | UserVO | 用户信息，见下方 |

**data 字段（UserVO）**

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| id | Long | 是 | 用户 ID |
| nickname | String | 是 | 用户昵称 |
| phone | String | 是 | 手机号（脱敏） |
| avatarUrl | String | 否 | 头像 URL |
| status | Integer | 是 | 用户状态：0=禁用，1=正常，2=冻结 |
| createdAt | Long | 是 | 注册时间戳（毫秒） |

---

## POST /api/v1/users/login — 用户登录

**入口**：`UserController.login`
**鉴权**：不需要
**描述**：账号密码登录，返回 JWT Token

### 请求参数

**Path 参数**：无

**Query 参数**：无

**请求体（LoginRequest）**

| 字段名 | 类型 | 必填 | 校验规则 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| username | String | 是 | @NotBlank, 长度 6-32 | 用户名或手机号 |
| password | String | 是 | @NotBlank | MD5 加密后的密码 |
| captchaCode | String | 否 | - | 图形验证码（触发风控时必填） |

### 响应（`Result<TokenVO>`）

**data 字段（TokenVO）**

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| accessToken | String | 是 | JWT Token |
| tokenType | String | 是 | 固定值 "Bearer" |
| expireIn | Long | 是 | 有效期（秒） |
| refreshToken | String | 否 | 刷新 Token |

---

## CreditService.addCredit — 发放积分（Thrift RPC）

**服务名**：CreditService
**实现类**：CreditHandler.addCredit
**描述**：向指定用户账户发放积分

### 请求（AddCreditRequest）

| 字段编号 | 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | userId | i64 | 是 | 用户 ID |
| 2 | amount | i32 | 是 | 积分数量，必须 > 0 |
| 3 | bizType | string | 是 | 业务类型：ORDER/SIGN_IN/ACTIVITY |
| 4 | bizId | string | 是 | 业务单号，幂等 key |

### 响应（CreditResult）

| 字段编号 | 字段名 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | success | bool | 是否成功 |
| 2 | balance | i64 | 发放后余额 |
| 3 | errorCode | string | 失败错误码（失败时填充） |
```

---

## 注意事项

- 只输出对外暴露的接口，内部私有方法和 `private` 方法不计入
- 若字段说明无法确定，标注 `<!-- TODO: 待确认 -->`
- 字段嵌套超过 3 层时，只记录类型名并注明"详见 {ClassName} 定义"
- 若存在 OpenAPI 文件，在文档头部标注 `信息来源：OpenAPI 规范文件（{文件路径}）`
- 若仅从代码扫描，在文档头部标注 `信息来源：代码扫描（准确性依赖注解完整性）`
- 若项目无任何接口定义，输出说明并退出
- **不要只输出表格摘要**，每个接口必须有完整的出入参字段节
