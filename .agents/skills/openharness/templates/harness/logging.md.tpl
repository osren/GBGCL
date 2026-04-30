# 日志规范

> 技术栈：Log4j2（排除 logback）+ 自定义 FinBamaiLayout
> 更新时间：<!-- TODO: 填写日期 -->

## 日志框架

- 使用 **Log4j2**，**排除** logback 和 log4j-slf4j-impl：

```groovy
// build.gradle
configurations {
    all*.exclude module: 'spring-boot-starter-logging'  // 排除 logback
    all*.exclude module: 'log4j-slf4j-impl'
}
dependencies {
    compile "org.springframework.boot:spring-boot-starter-log4j2"
    compile "org.apache.logging.log4j:log4j-api:2.20.0"
    compile "org.apache.logging.log4j:log4j-core:2.20.0"
    compile "org.apache.logging.log4j:log4j-slf4j-impl:2.20.0"
}
```

- 代码中统一使用 SLF4J 接口（`@Slf4j` + `log.info/warn/error`），禁止直接使用 Log4j2 API

## 日志文件分类

| 文件 | 用途 | 级别 |
|------|------|------|
| `business.log` | 业务流程日志（核心流水） | INFO + |
| `error.log` | 错误日志 | ERROR |
| `public.log` | 对外监控/埋点（PatternLayout 纯消息） | INFO |

- 按**小时**滚动（`TimeBasedTriggeringPolicy interval=1`）
- 日志目录：`/home/xiaoju/data1/logs/{service-name}/`

## 日志格式

使用 `FinBamaiLayout`（公司统一格式，含 traceId、spanId、timestamp）。

业务日志 **kv 格式**（便于日志平台解析）：

```java
// 正确：kv 格式，字段用 || 分隔
log.info("consume||topic={}||key={}||offset={}||traceId={}||content={}",
        topic, key, offset, traceId, content);

// 错误：自由文本，难以解析
log.info("消费消息: " + content);
```

## TraceId 规范

- 所有请求入口（HTTP Controller / MQ Consumer）第一行设置 traceId：

```java
// HTTP 请求（通常由 filter 自动注入）
// MQ 消费
String traceId = message.getProperties().get(LogConstant.KEY_TRACE_ID);
LogUtil.setTraceId(traceId);
LogUtil.genSpanId();
```

- 所有业务日志必须能通过 traceId 串联完整调用链
- **禁止**在异步线程中遗失 traceId（异步时需手动传递）

## 日志级别规范

| 场景 | 级别 |
|------|------|
| 业务流程入口/出口、关键状态变更 | INFO |
| 非预期但可恢复的情况（参数校验失败、下游超时）| WARN |
| 需要人工介入的异常（DB 异常、MQ 发送失败）| ERROR |
| 调试信息（默认关闭，不上生产）| DEBUG |

- 业务包日志级别：**INFO**
- 第三方框架（carrera、mybatis 等）：**WARN**，避免日志噪音

## 异常日志规范

打印异常时必须包含完整 stacktrace：

```java
// 正确
log.error("process_error||topic={}||e=", topic, e);

// 错误（丢失 stacktrace）
log.error("process error: " + e.getMessage());
```

## 禁止事项

- **禁止**在日志中打印未脱敏的手机号、身份证、银行卡等 PII 字段
- **禁止**在日志中打印完整的密码、Token、密钥
- **禁止** `System.out.println`（使用 logger）
- **禁止**在高频循环中打印 INFO 日志（会产生海量日志影响性能）
- **禁止**生产环境开启 DEBUG 级别

## Logger 配置示例

```xml
<Loggers>
    <Logger name="com.xxx.{package}" level="INFO" additivity="false">
        <AppenderRef ref="business" />
        <AppenderRef ref="error" />
    </Logger>
    <!-- 第三方包降噪 -->
    <Logger name="com.xiaojukeji.carrera" level="WARN" additivity="false">
        <AppenderRef ref="business" />
    </Logger>
    <Root level="INFO" includeLocation="true">
        <AppenderRef ref="business" />
        <AppenderRef ref="error" />
    </Root>
</Loggers>
```
