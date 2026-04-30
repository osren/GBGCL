# 消息队列使用规范

> 技术栈：DDMQ（Carrera SDK）
> 更新时间：<!-- TODO: 填写日期 -->

## Topic 管理

所有 Topic 名称统一在 `TopicEnum` 枚举中定义，**禁止**在业务代码中硬编码 Topic 字符串：

```java
public enum TopicEnum {
    INNER_DEAL("strategy_center_inner_deal", "内部流转"),
    // ...

    public static String getTopicName(NodeType nodeType, String bizScene) { ... }
}
```

> <!-- TODO: 补充本项目的 Topic 枚举定义 -->

## 生产者规范

### 标准生产模板

```java
Message message = new Message();
// 普通消息
message.setPartitionId(CarreraConfig.PARTITION_RAND);
// 保序消息（相同 hashId 进同一队列）
message.setHashId(Math.abs(sequenceValue.hashCode()));
message.setPartitionId(CarreraConfig.PARTITION_HASH);

message.setBody(content.getBytes());
message.setTopic(TopicEnum.XXX.getName());
// 透传 traceId
message.putToProperties(LogConstant.KEY_TRACE_ID, LogUtil.getTraceId());
message.setKey(RandomKeyUtils.randomKey(CarreraConfig.RANDOM_KEY_SIZE));

Result result = carreraProducer.sendMessage(message);
log.info("produce_result||topic={}||key={}||result={}", topicName, message.getKey(), result);
```

### 生产规则

- 发送后必须打印 INFO 日志（含 topic、key、result）
- 发送失败（result 非成功）必须打印 ERROR 日志，视业务决定是否重试
- **禁止**在事务中发送消息（事务回滚消息无法撤回）

## 消费者规范

### 标准消费模板

```java
@Component
public class XxxConsumer implements MessageProcessor {

    @Override
    public Result process(Message message, Context context) {
        // 第一行：透传 traceId
        String traceId = message.getProperties().get(LogConstant.KEY_TRACE_ID);
        LogUtil.setTraceId(traceId);

        String content = new String(message.getValue());
        log.info("consume||topic={}||key={}||offset={}||content={}",
                context.getTopic(), message.getKey(), message.getOffset(), content);

        // 空消息直接丢弃
        if (InspectionUtils.isEmpty(content)) {
            return Result.SUCCESS;
        }

        try {
            // 业务处理（必须保证幂等）
            doProcess(content);
            return Result.SUCCESS;
        } catch (Exception e) {
            log.error("consume_error||topic={}||e=", context.getTopic(), e);
            // TODO: 根据业务决定是否重试
            // return Result.FAIL;   // 重试（幂等场景可选）
            return Result.SUCCESS;  // 不重试（丢弃）
        }
    }
}
```

### 消费者配置

```java
@Component
@ConfigurationProperties(prefix = "carrera-consumers.xxx")
@Data
public class XxxConsumerConfig {
    private String group = "<!-- TODO: 填写 group 名 -->";
    private Integer threadNum = 300;      // 消费线程数
    private Integer timeout = 5000;       // 超时 ms
    private Integer maxBatchSize = 8;     // 批量拉取（服务端最大 8）
    private Integer retryInterval = 500;  // 重试间隔 ms
    private Integer submitMaxRetries = 2; // 提交重试次数
    private Integer maxLingerTime = 16;   // 服务端等待消息最长时间 ms
}
```

## 幂等性要求

**所有消费者必须保证幂等**，DDMQ 在网络抖动时可能重复投递：

- 推荐方案：用消息 `key` 作为幂等键，写入前查询是否已处理
- 状态机方案：检查业务实体状态，已终态则跳过
- **禁止**假设消息只会消费一次

## Result 决策原则

| 场景 | 返回值 | 说明 |
|------|--------|------|
| 正常处理完成 | `Result.SUCCESS` | 提交 offset |
| 空消息 / 非法格式 | `Result.SUCCESS` | 丢弃，不重试 |
| 下游临时不可用（网络抖动）| `Result.FAIL` | 触发重试 |
| 业务数据问题（上游数据错）| `Result.SUCCESS` | 打 ERROR 日志，人工介入 |

> <!-- TODO: 确认本项目各 Consumer 的重试策略 -->

## 消息体规范

- 消息体统一使用 JSON 序列化
- 消息体必须包含 `bizId`（业务唯一标识，用于幂等）
- **禁止**在消息体中包含密码、密钥等敏感信息
