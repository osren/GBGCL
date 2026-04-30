# 安全规范

> 技术栈：KMS（didi-kms-spring-boot-starter）+ desensitization.yml
> 更新时间：<!-- TODO: 填写日期 -->

## 敏感字段加密

涉及 PII（个人身份信息）和敏感业务数据，必须通过 KMS 加密后存储：

```java
// 使用 KMS SDK 加密/解密，禁止自行实现加密算法
@Resource
private KmsService kmsService;

String encrypted = kmsService.encrypt(plainText, keyId);
String decrypted = kmsService.decrypt(cipherText, keyId);
```

**必须加密的字段类型**：
- 手机号、身份证号、银行卡号等 PII 数据
- 合同金额、授信额度等敏感业务字段（按业务合规要求判断）

> <!-- TODO: 补充本项目需要 KMS 加密的字段清单 -->

**禁止**使用 MD5 / Base64 "加密"敏感数据（不可逆哈希不是加密，Base64 不是加密）。

## 日志脱敏

通过 `desensitization.yml` 配置字段级别的日志脱敏规则：

```yaml
desensitization_fields:
  - field_name: mobile
    alias: [mobile, phone, driverMobile, driver_mobile]
    regex: (\d{3})\d{4}(\d{4})
    replace: $1****$2
  # TODO: 补充本项目其他需脱敏的字段
```

**规则**：
- 手机号：保留前3后4，中间4位脱敏为 `****`
- 身份证：保留前6后4，中间8位脱敏
- 银行卡：保留前4后4，中间脱敏

**禁止**在 INFO/DEBUG 日志中打印未脱敏的手机号、身份证等 PII 字段。

## 配置安全

- **禁止**在代码中硬编码密码、密钥、AccessKey 等敏感配置
- 敏感配置通过环境变量或配置中心（Apollo）注入
- `application.yml` 中的密码字段使用 `${ENV_VAR}` 引用，不写明文
- `.gitignore` 必须包含含有真实密码的配置文件

```yaml
# 正确
custom.redis.password: ${REDIS_PASSWORD}

# 禁止
custom.redis.password: mypassword123
```

## 输入校验

- 所有外部输入（HTTP 请求参数）必须通过 Hibernate Validator 校验
- 字符串类型字段必须限制最大长度（`@Size(max = N)`）
- 枚举类型参数必须校验合法值，非法值返回 `INVALID_PARAMS` 错误码
- **禁止**将用户输入直接拼接进 SQL / Redis Key（防注入）

## 接口鉴权

> <!-- TODO: 确认本项目鉴权方案（JWT Token / 网关鉴权 / IP 白名单 / 签名验证） -->
> <!-- TODO: 明确哪些接口需要鉴权，哪些是内部接口 -->

## 依赖安全

- 定期扫描依赖库 CVE 漏洞（Gradle/Maven 依赖检查插件）
- fastjson 使用 1.2.83+（修复历史 RCE 漏洞）
- **禁止**引入版本未知或长期无维护的第三方库
