## 1. 环境配置与依赖安装

- [ ] 1.1 安装RAG-Anything核心依赖：`pip install raganything`
- [ ] 1.2 安装扩展依赖：`pip install raganything[all]`
- [ ] 1.3 检查LibreOffice安装（如需处理Office文档）：`brew install --cask libreoffice`
- [ ] 1.4 创建.env.example配置文件
- [ ] 1.5 验证MinerU安装：`python -c "from raganything import RAGAnything; print('OK')"`

## 2. 创建RAG服务封装模块

- [ ] 2.1 创建src/rag_service.py核心模块
- [ ] 2.2 实现RAGService类：初始化、文档处理、查询功能
- [ ] 2.3 配置embedding函数（text-embedding-3-large）
- [ ] 2.4 配置LLM函数（gpt-4o-mini）
- [ ] 2.5 添加错误处理和日志

## 3. 创建CLI查询脚本

- [ ] 3.1 创建scripts/rag_query.py脚本
- [ ] 3.2 支持命令行参数：--file, --folder, --query, --mode
- [ ] 3.3 实现批量处理子命令
- [ ] 3.4 添加帮助信息和usage提示

## 4. 测试与文档

- [ ] 4.1 测试单文档处理：处理一篇PDF文档
- [ ] 4.2 测试查询功能：验证hybrid/local/global模式
- [ ] 4.3 测试批量处理：处理docs/目录
- [ ] 4.4 创建docs/rag_usage.md使用说明文档

## 5. 高级功能（可选）

- [ ] 5.1 支持多模态查询（图片、表格、公式）
- [ ] 5.2 配置VLM增强查询（gpt-4o）
- [ ] 5.3 支持增量文档索引更新