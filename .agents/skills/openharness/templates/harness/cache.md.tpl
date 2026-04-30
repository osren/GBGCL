# 缓存使用规范

> 技术栈：Fusion（滴滴内部 Redis）+ Jedis + Spring Data Redis
> 更新时间：<!-- TODO: 填写日期 -->

## 存储分类

项目使用两类缓存实例，通过独立 Bean 隔离：

| Bean 名称 | 用途 | 配置前缀 |
|-----------|------|---------|
| `RedisTemplate` | 分布式锁、幂等键 | `custom.redis` |
| `fusionTemplate`（按需命名） | 业务数据缓存 | `custom.fusion` |

> <!-- TODO: 补充本项目实际的 Bean 名称和用途 -->

**禁止**用同一个 RedisTemplate 混用锁和业务缓存。

## 接入方式

使用 DISF 服务名注册，**禁止**在配置中写死 Redis IP：

```java
DisfRedisStandaloneConfiguration config = new DisfRedisStandaloneConfiguration();
config.setDisfName(disfName);   // 通过 DISF 解析，不写 IP
config.setHostName(host);       // fallback host
DynamicJedisConnectionFactory factory = new DynamicJedisConnectionFactory(config, poolConfig);
```

## 序列化规范

- **Key**：`StringRedisSerializer`（可读字符串）
- **Value**：`GenericJackson2JsonRedisSerializer`（JSON，含类型信息）
- **禁止**使用 JDK 默认序列化（不可读、有版本兼容问题）

## Key 命名规范

所有 Key 必须集中定义在 `RedisKeys` 静态类中，**禁止**在业务代码中散落硬编码字符串。

格式：`{service_prefix}:{domain}:{bizId}`

```java
// 正确示例
private static final String PREFIX = "strategy_center:";

public static String submitApproveKey(Long planId) {
    return PREFIX + "submit_approve:" + planId;
}

// 错误示例（禁止）
redisTemplate.opsForValue().set("strategy_center:submit_approve:" + planId, ...);
```

## TTL 规范

- 所有写入必须设置 TTL，**禁止**永不过期的 Key
- CacheManager 默认 TTL：**60 分钟**
- 分布式锁 TTL：**≤ 30 秒**（防死锁）
- 幂等键 TTL：根据业务幂等窗口设置，通常 **5～30 分钟**

> <!-- TODO: 补充本项目各业务场景的 TTL 约定 -->

## 连接池配置

```yaml
# 参考值，按实际压力调整
maxActive: 100
maxIdle: 40
minIdle: 10
maxWait: 40   # ms，等待连接超时
testOnBorrow: true
```

## 缓存穿透/击穿防护

- 查询结果为 null 时**不缓存**（或缓存空对象并设置短 TTL，需在代码注释中说明原因）
- 热点 Key 击穿：使用分布式锁 + 双重检查（先查缓存，加锁，再查缓存，再查 DB）
- **禁止**将 Fusion 作为主存储，数据库是唯一事实来源

## 禁止事项

- 禁止在 Controller / 业务逻辑层直接操作 RedisTemplate，必须封装为 Service/Util
- 禁止缓存敏感字段（手机号、身份证等 PII 数据）明文存储
- 禁止跨服务共享同一个 Key 前缀（会造成 Key 冲突）
