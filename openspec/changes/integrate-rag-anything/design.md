## Context

GBGCL项目是一个基于Granular Ball Graph的图表示学习框架。当前项目缺乏对文档（论文、实验报告）的智能问答能力，需要手动检索和理解大量PDF文档。

RAG-Anything是一个开源的All-in-One RAG框架，支持：
- 多格式文档解析（PDF, DOCX, PPT, 图片等）
- 多模态内容识别（图片、表格、公式）
- 多种查询模式（hybrid, local, global, naive）

## Goals / Non-Goals

**Goals:**
- 集成RAG-Anything实现文档自动解析和问答
- 支持批量处理项目中的文档（docs/, figures/等目录）
- 提供简单易用的CLI接口
- 支持多模态查询（带图片、表格的问题）

**Non-Goals:**
- 不修改现有GBGCL训练代码
- 不自建LLM服务（使用外部API）
- 不支持实时OCR识别摄像头输入

## Decisions

### 1. 使用raganything而非自建RAG
**决定**: 直接集成raganything包
**原因**: raganything已封装好LightRAG+MinerU，提供完整的端到端处理流程，减少开发工作量
**替代方案考虑**:
- 自建LightRAG: 需要手动处理文档解析，功能不完整
- 使用LangChain: 需要额外选型 embedding model、vector store 等

### 2. 使用mineru作为解析器
**决定**: parser="mineru"
**原因**: MinerU 2.0对PDF解析效果好，支持表格、公式、图片识别
**替代方案考虑**:
- Docling: 对Office文档更好，但PDF效果一般
- PaddleOCR: CPU可用，但精度较低

### 3. 模型选择
**决定**: 使用gpt-4o-mini作为LLM，text-embedding-3-large作为embedding
**原因**: 平衡成本与效果
**替代方案考虑**:
- gpt-4o: 效果更好但成本高
- text-embedding-3-small: 成本低但精度略低

## Risks / Trade-offs

- **[风险] API密钥依赖** →  mitigation: 在.env中配置，支持多种OpenAI兼容API（如Azure OpenAI）
- **[风险] GPU资源需求** →  mitigation: MinerU解析需要GPU，CPU模式速度慢
- **[风险] 网络调用延迟** →  mitigation: 批量处理后本地查询无延迟
- **[风险] 文档隐私** →  mitigation: 文档存储在本地rag_storage/目录，仅调用外部API时上传内容

## Migration Plan

1. 安装依赖：`pip install raganything`
2. 配置.env文件
3. 创建rag_service.py封装模块
4. 运行测试：处理一篇PDF文档
5. 创建CLI脚本scripts/rag_query.py
6. 批量处理docs/目录

## Open Questions

- [ ] 是否需要支持离线embedding（本地模型）？
- [ ] 批量处理时的并发数配置？
- [ ] 文档更新时如何增量索引？