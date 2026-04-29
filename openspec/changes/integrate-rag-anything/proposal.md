## Why

当前GBGCL项目缺乏对外部文档（如论文、实验报告、需求文档）的智能问答能力。研究过程中需要处理大量的PDF论文、实验结果和需求文档，RAG-Anything可以提供端到端的多模态文档处理和知识问答能力，解决手动检索和理解文档的效率问题。

## What Changes

- 安装RAG-Anything及其依赖项（raganything, lightrag, mineru）
- 创建RAG服务封装模块 `src/rag_service.py`
- 添加环境配置文件 `.env.example` 支持API密钥配置
- 提供CLI命令脚本 `scripts/rag_query.py` 用于文档处理和查询
- 创建示例用法文档 `docs/rag_usage.md`

## Capabilities

### New Capabilities

- `doc-rag-query`: 支持将PDF/文档入库并进行问答
  - 文档解析：支持PDF、DOCX、PPT等格式
  - 多模态内容：支持图片、表格、公式识别
  - 查询模式：hybrid/local/global/naive
- `batch-doc-processing`: 批量处理文件夹中的文档
- `multimodal-query`: 支持带图片、表格、公式的多模态查询

### Modified Capabilities

- （无）

## Impact

- 新增Python依赖：raganything, lightrag
- 需配置OPENAI_API_KEY环境变量
- 新增存储目录 `./rag_storage` 用于向量索引
- 文档处理需要GPU支持（ MinerU）