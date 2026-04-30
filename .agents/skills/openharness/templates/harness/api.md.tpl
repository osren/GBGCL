# API 接口设计规范

> 技术栈：Spring Boot + Spring MVC + Hibernate Validator + Swagger
> 更新时间：<!-- TODO: 填写日期 -->

## 统一响应格式

所有接口必须返回 `BizResponse<T>` 封装，**禁止**直接返回裸 POJO 或 Map：

```json
{
  "errorCode": "0",
  "errorMsg": "success",
  "data": {}
}
```

- `errorCode = "0"` 表示成功
- 非 0 时 `data` 可为 null，`errorMsg` 必须有可读描述

## 错误码规范

统一在 `ErrorMessage` 枚举中定义，**禁止**在业务代码中散落魔法字符串：

```java
public enum ErrorMessage {
    SUCCESS("0", "success"),
    SYSTEM_ERROR("SYS001", "系统异常"),
    INVALID_PARAMS("SYS002", "参数校验失败"),
    // 业务错误码格式：{服务前缀}{三位数字}
    // TODO: 补充本项目业务错误码
    ;
}
```

## 入参校验

- 使用 Hibernate Validator 注解：`@NotNull`、`@NotBlank`、`@Size`、`@Pattern` 等
- Controller 方法参数必须加 `@Valid` / `@Validated`
- 校验失败由 `@RestControllerAdvice` 统一处理，**禁止**在 Service 层手动判断格式合法性

```java
@RestControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public BizResponse handleValidException(MethodArgumentNotValidException ex) {
        // 统一返回 INVALID_PARAMS 错误码
    }
}
```

## 幂等性

所有**写操作**（创建/修改/删除）必须保证幂等：

- 客户端传入业务唯一 ID（`requestId` / `bizId`）
- 服务端用 Redis Key（`RedisKeys.idempotenceKey(requestId)`）做幂等检查
- TTL 根据业务幂等窗口设置

> <!-- TODO: 确认本项目幂等实现方案（Redis 分布式锁 / 数据库唯一索引 / 版本号） -->

## 接口文档

每个 Controller 及方法必须加 Swagger 注解：

```java
@Api(tags = "xxx 管理")
@RestController
public class XxxController {

    @ApiOperation("创建 xxx")
    @PostMapping("/api/v1/xxx")
    public BizResponse<Long> create(@Valid @RequestBody XxxCreateRequest req) { ... }
}
```

DTO 字段必须加 `@ApiModelProperty`，**标注是否必填、取值范围**。

## HTTP 客户端规范

所有对外 HTTP 调用必须设置超时，**禁止**使用默认无超时配置：

```java
// connectTimeout ≤ 1000ms，readTimeout ≤ 3000ms
// TODO: 确认本项目各下游服务的超时配置
```

## 版本管理

- URL 中包含版本号：`/api/v1/xxx`
- 已上线接口不能直接修改/删除入参字段，只能**新增可选字段**或发布新版本
- 废弃字段用 `@Deprecated` + `@ApiModelProperty(hidden = true)` 标注，至少保留一个版本

## 分页规范

使用 PageHelper，**禁止**在 SQL 中手写 LIMIT/OFFSET：

```java
PageHelper.startPage(pageNum, pageSize);
List<XxxDo> list = mapper.selectByCondition(condition);
PageInfo<XxxDo> pageInfo = new PageInfo<>(list);
```

- 默认 pageSize ≤ 100，超出拒绝或 cap 到 100
