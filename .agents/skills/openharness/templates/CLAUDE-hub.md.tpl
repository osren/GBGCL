# {文档库名称} — Hub 工作入口

> Open Harness Hub | {YYYY-MM-DD}

## 仓库定位

本仓库用于集中管理各服务事实文档，并生成系统级视图与跨服务方案母稿。

- **服务事实层**：`docs/knowledge/services/`
- **系统视图层**：`docs/knowledge/system/`
- **跨服务方案区**：`docs/specs/active/`

> 注意：Hub 仓库不直接执行代码实现，不运行 `/oh:apply`。

## 目录导航

```text
docs/
├── knowledge/
│   ├── services/         # 各服务事实文档
│   └── system/           # 系统全景图、核心流程
└── specs/
    ├── active/           # 跨服务方案母稿
    └── completed/
```

## 常用命令

```bash
/oh:init --hub                         # 初始化或刷新系统视图层
/oh:propose --hub <feature> <prd>     # 生成跨服务方案母稿
```

## 工作规则

1. 先读取 `docs/knowledge/index.md`，建立 Hub 知识地图
2. 再按任务需要进入 `docs/knowledge/system/` 或 `docs/knowledge/services/` 下的相关索引与稳定入口
3. 设计核心链路时，优先读取 `docs/knowledge/system/system-map.md` 与 `docs/knowledge/system/core-flows/`
4. 需要服务细节时，再下钻 `docs/knowledge/services/{service}/` 下的事实文档
5. Hub 仓库输出的是系统级方案，代码实现必须回到各服务仓库
