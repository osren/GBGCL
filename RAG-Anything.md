## 🚀 RAG-Anything 快速入门

URL: [https://github.com/HKUDS/RAG-Anything#-quick-start](https://https://github.com/HKUDS/RAG-Anything#-quick-start)

*开启您的AI之旅*

[![](https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif)](https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif)

### 安装

URL:[https://github.com/HKUDS/RAG-Anything#installation](https://https://github.com/HKUDS/RAG-Anything#installation)

#### 选项 1：从 PyPI 安装（推荐）

URL:[https://github.com/HKUDS/RAG-Anything#option-1-install-from-pypi-recommended](https://github.com/HKUDS/RAG-Anything#option-1-install-from-pypi-recommended)

```shell
# Basic installation
pip install raganything

# With optional dependencies for extended format support:
pip install 'raganything[all]'              # All optional features
pip install 'raganything[image]'            # Image format conversion (BMP, TIFF, GIF, WebP)
pip install 'raganything[text]'             # Text file processing (TXT, MD)
pip install 'raganything[image,text]'       # Multiple features
```

#### 选项 2：从源代码安装

URL:[https://github.com/HKUDS/RAG-Anything#option-2-install-from-source](https://github.com/HKUDS/RAG-Anything#option-2-install-from-source)

```shell
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project with uv
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything

# Install the package and dependencies in a virtual environment
uv sync

# If you encounter network timeouts (especially for opencv packages):
# UV_HTTP_TIMEOUT=120 uv sync

# Run commands directly with uv (recommended approach)
uv run python examples/raganything_example.py --help

# Install with optional dependencies
uv sync --extra image --extra text  # Specific extras
uv sync --all-extras                 # All optional features
```

#### 可选依赖项

URL:[https://github.com/HKUDS/RAG-Anything#optional-dependencies](https://github.com/HKUDS/RAG-Anything#optional-dependencies)

* **`[image]`**- 支持处理 BMP、TIFF、GIF、WebP 图像格式（需要 Pillow 库）
* **`[text]`**- 支持处理 TXT 和 MD 文件（需要 ReportLab）
* **`[all]`**- 包括所有 Python 可选依赖项

> **⚠️办公文档处理要求：**
>
> * 办公文档（.doc、.docx、.ppt、.pptx、.xls、.xlsx）需要安装**LibreOffice。**
> * 从LibreOffice 官方网站下载[](https://www.libreoffice.org/download/download/)
> * **Windows**：从官方网站下载安装程序
> * **macOS**：`brew install --cask libreoffice`
> * **Ubuntu/Debian**：`sudo apt-get install libreoffice`
> * **CentOS/RHEL**：`sudo yum install libreoffice`

**检查 MinerU 安装情况：**

```shell
# Verify installation
mineru --version

# Check if properly configured
python -c "from raganything import RAGAnything; rag = RAGAnything(); print('✅ MinerU installed properly' if rag.check_parser_installation() else '❌ MinerU installation issue')"
```

模型会在首次使用时自动下载。如需手动下载，请参阅[MinerU 模型源配置](https://github.com/opendatalab/MinerU/blob/master/README.md#22-model-source-configuration)。

### 使用示例

URL:[https://github.com/HKUDS/RAG-Anything#usage-examples](https://github.com/HKUDS/RAG-Anything#usage-examples)

#### 1. 端到端文档处理

URL:[https://github.com/HKUDS/RAG-Anything#1-end-to-end-document-processing](https://github.com/HKUDS/RAG-Anything#1-end-to-end-document-processing)

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # Parser selection: mineru, docling, or paddleocr
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Process a document
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # Query the processed content
    # Pure text query - for basic knowledge base search
    text_result = await rag.aquery(
        "What are the main findings shown in the figures and tables?",
        mode="hybrid"
    )
    print("Text query result:", text_result)

    # Multimodal query with specific multimodal content
    multimodal_result = await rag.aquery_with_multimodal(
    "Explain this formula and its relevance to the document content",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "Document relevance probability"
    }],
    mode="hybrid"
)
    print("Multimodal query result:", multimodal_result)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. 直接多模态内容处理

URL:[https://github.com/HKUDS/RAG-Anything#2-direct-multimodal-content-processing](https://github.com/HKUDS/RAG-Anything#2-direct-multimodal-content-processing)

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor

async def process_multimodal_content():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
    )
    await rag.initialize_storages()

    # Process an image
    image_processor = ImageModalProcessor(
        lightrag=rag,
        modal_caption_func=lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: openai_complete_if_cache(
            "gpt-4o",
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]} if image_data else {"role": "user", "content": prompt}
            ],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ) if image_data else openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )

    image_content = {
        "img_path": "path/to/image.jpg",
        "image_caption": ["Figure 1: Experimental results"],
        "image_footnote": ["Data collected in 2024"]
    }

    description, entity_info = await image_processor.process_multimodal_content(
        modal_content=image_content,
        content_type="image",
        file_path="research_paper.pdf",
        entity_name="Experimental Results Figure"
    )

    # Process a table
    table_processor = TableModalProcessor(
        lightrag=rag,
        modal_caption_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )

    table_content = {
        "table_body": """
        | Method | Accuracy | F1-Score |
        |--------|----------|----------|
        | RAGAnything | 95.2% | 0.94 |
        | Baseline | 87.3% | 0.85 |
        """,
        "table_caption": ["Performance Comparison"],
        "table_footnote": ["Results on test dataset"]
    }

    description, entity_info = await table_processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="research_paper.pdf",
        entity_name="Performance Results Table"
    )

if __name__ == "__main__":
    asyncio.run(process_multimodal_content())
```

#### 3. 批量处理

URL:[https://github.com/HKUDS/RAG-Anything#3-batch-processing](https://github.com/HKUDS/RAG-Anything#3-batch-processing)

```python
# Process multiple documents
await rag.process_folder_complete(
    folder_path="./documents",
    output_dir="./output",
    file_extensions=[".pdf", ".docx", ".pptx"],
    recursive=True,
    max_workers=4
)
```

#### 4. 自定义模态处理器

URL:[https://github.com/HKUDS/RAG-Anything#4-custom-modal-processors](https://github.com/HKUDS/RAG-Anything#4-custom-modal-processors)

```python
from raganything.modalprocessors import GenericModalProcessor

class CustomModalProcessor(GenericModalProcessor):
    async def process_multimodal_content(self, modal_content, content_type, file_path, entity_name):
        # Your custom processing logic
        enhanced_description = await self.analyze_custom_content(modal_content)
        entity_info = self.create_custom_entity(enhanced_description, entity_name)
        return await self._create_entity_and_chunk(enhanced_description, entity_info, file_path)
```

#### 5. 查询选项

URL:[https://github.com/HKUDS/RAG-Anything#5-query-options](https://github.com/HKUDS/RAG-Anything#5-query-options)

RAG-Anything 提供三种查询方法：

**纯文本查询**- 使用 LightRAG 进行直接知识库搜索：

```python
# Different query modes for text queries
text_result_hybrid = await rag.aquery("Your question", mode="hybrid")
text_result_local = await rag.aquery("Your question", mode="local")
text_result_global = await rag.aquery("Your question", mode="global")
text_result_naive = await rag.aquery("Your question", mode="naive")

# Synchronous version
sync_text_result = rag.query("Your question", mode="hybrid")
```

**VLM增强查询**- 使用VLM自动分析检索到的上下文中的图像：

```python
# VLM enhanced query (automatically enabled when vision_model_func is provided)
vlm_result = await rag.aquery(
    "Analyze the charts and figures in the document",
    mode="hybrid"
    # vlm_enhanced=True is automatically set when vision_model_func is available
)

# Manually control VLM enhancement
vlm_enabled = await rag.aquery(
    "What do the images show in this document?",
    mode="hybrid",
    vlm_enhanced=True  # Force enable VLM enhancement
)

vlm_disabled = await rag.aquery(
    "What do the images show in this document?",
    mode="hybrid",
    vlm_enhanced=False  # Force disable VLM enhancement
)

# When documents contain images, VLM can see and analyze them directly
# The system will automatically:
# 1. Retrieve relevant context containing image paths
# 2. Load and encode images as base64
# 3. Send both text context and images to VLM for comprehensive analysis
```

**多模态查询**- 具有特定多模态内容分析功能的增强型查询：

```python
# Query with table data
table_result = await rag.aquery_with_multimodal(
    "Compare these performance metrics with the document content",
    multimodal_content=[{
        "type": "table",
        "table_data": """Method,Accuracy,Speed
                        RAGAnything,95.2%,120ms
                        Traditional,87.3%,180ms""",
        "table_caption": "Performance comparison"
    }],
    mode="hybrid"
)

# Query with equation content
equation_result = await rag.aquery_with_multimodal(
    "Explain this formula and its relevance to the document content",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "Document relevance probability"
    }],
    mode="hybrid"
)
```

#### 6. 加载现有 LightRAG 实例

URL:[https://github.com/HKUDS/RAG-Anything#6-loading-existing-lightrag-instance](https://github.com/HKUDS/RAG-Anything#6-loading-existing-lightrag-instance)

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
import os

async def load_existing_lightrag():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # First, create or load existing LightRAG instance
    lightrag_working_dir = "./existing_lightrag_storage"

    # Check if previous LightRAG instance exists
    if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
        print("✅ Found existing LightRAG instance, loading...")
    else:
        print("❌ No existing LightRAG instance found, will create new one")

    # Create/load LightRAG instance with your configuration
    lightrag_instance = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
    )

    # Initialize storage (this will load existing data if available)
    await lightrag_instance.initialize_storages()
    await initialize_pipeline_status()

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return lightrag_instance.llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Now use existing LightRAG instance to initialize RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,  # Pass existing LightRAG instance
        vision_model_func=vision_model_func,
        # Note: working_dir, llm_model_func, embedding_func, etc. are inherited from lightrag_instance
    )

    # Query existing knowledge base
    result = await rag.aquery(
        "What data has been processed in this LightRAG instance?",
        mode="hybrid"
    )
    print("Query result:", result)

    # Add new multimodal document to existing LightRAG instance
    await rag.process_document_complete(
        file_path="path/to/new/multimodal_document.pdf",
        output_dir="./output"
    )

if __name__ == "__main__":
    asyncio.run(load_existing_lightrag())
```

#### 7. 直接插入内容列表

URL:[https://github.com/HKUDS/RAG-Anything#7-direct-content-list-insertion](https://github.com/HKUDS/RAG-Anything#7-direct-content-list-insertion)

对于已经拥有预解析内容列表（例如，来自外部解析器或先前处理）的情况，您可以直接将其插入 RAGAnything 而无需进行文档解析：

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def insert_content_list_example():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define model functions
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Example: Pre-parsed content list from external source
    content_list = [
        {
            "type": "text",
            "text": "This is the introduction section of our research paper.",
            "page_idx": 0  # Page number where this content appears
        },
        {
            "type": "image",
            "img_path": "/absolute/path/to/figure1.jpg",  # IMPORTANT: Use absolute path
            "image_caption": ["Figure 1: System Architecture"],
            "image_footnote": ["Source: Authors' original design"],
            "page_idx": 1  # Page number where this image appears
        },
        {
            "type": "table",
            "table_body": "| Method | Accuracy | F1-Score |\n|--------|----------|----------|\n| Ours | 95.2% | 0.94 |\n| Baseline | 87.3% | 0.85 |",
            "table_caption": ["Table 1: Performance Comparison"],
            "table_footnote": ["Results on test dataset"],
            "page_idx": 2  # Page number where this table appears
        },
        {
            "type": "equation",
            "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
            "text": "Document relevance probability formula",
            "page_idx": 3  # Page number where this equation appears
        },
        {
            "type": "text",
            "text": "In conclusion, our method demonstrates superior performance across all metrics.",
            "page_idx": 4  # Page number where this content appears
        }
    ]

    # Insert the content list directly
    await rag.insert_content_list(
        content_list=content_list,
        file_path="research_paper.pdf",  # Reference file name for citation
        split_by_character=None,         # Optional text splitting
        split_by_character_only=False,   # Optional text splitting mode
        doc_id=None,                     # Optional custom document ID (will be auto-generated if not provided)
        display_stats=True               # Show content statistics
    )

    # Query the inserted content
    result = await rag.aquery(
        "What are the key findings and performance metrics mentioned in the research?",
        mode="hybrid"
    )
    print("Query result:", result)

    # You can also insert multiple content lists with different document IDs
    another_content_list = [
        {
            "type": "text",
            "text": "This is content from another document.",
            "page_idx": 0  # Page number where this content appears
        },
        {
            "type": "table",
            "table_body": "| Feature | Value |\n|---------|-------|\n| Speed | Fast |\n| Accuracy | High |",
            "table_caption": ["Feature Comparison"],
            "page_idx": 1  # Page number where this table appears
        }
    ]

    await rag.insert_content_list(
        content_list=another_content_list,
        file_path="another_document.pdf",
        doc_id="custom-doc-id-123"  # Custom document ID
    )

if __name__ == "__main__":
    asyncio.run(insert_content_list_example())
```

**内容列表格式：**

应`content_list`遵循标准格式，每个条目都是一个字典，包含以下内容：

* **文本内容**：`{"type": "text", "text": "content text", "page_idx": 0}`
* **图片内容**：`{"type": "image", "img_path": "/absolute/path/to/image.jpg", "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}`
* **表格内容**：`{"type": "table", "table_body": "markdown table", "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}`
* **公式内容**：`{"type": "equation", "latex": "LaTeX formula", "text": "description", "page_idx": 3}`
* **通用内容**：`{"type": "custom_type", "content": "any content", "page_idx": 4}`

**重要提示：**

* **`img_path`**必须是图像文件的绝对路径（例如，`/home/user/images/chart.jpg`或`C:\Users\user\images\chart.jpg`） 。
* **`page_idx`**：表示内容在原始文档中出现的页码（从 0 开始索引）
* **内容排序**：项目按照它们在列表中出现的顺序进行处理。

这种方法在以下情况下尤其有用：

* 您有来自外部解析器（非 MinerU/Docling）的内容
* 您希望处理程序生成的内容
* 您需要将来自多个来源的内容插入到单个知识库中。
* 您有一些缓存的解析结果，希望能够重复使用。

---

## 🛠️ 示例

URL:[https://github.com/HKUDS/RAG-Anything#%EF%B8%8F-examples](https://github.com/HKUDS/RAG-Anything#%EF%B8%8F-examples)

*实际应用演示*

URL:[https://user-images.githubusercontent.com/74038190/212257455-13e3e01e-d6a6-45dc-bb92-3ab87b12dfc1.gif](https://user-images.githubusercontent.com/74038190/212257455-13e3e01e-d6a6-45dc-bb92-3ab87b12dfc1.gif)

该`examples/`目录包含全面的使用示例：

* **`raganything_example.py`**使用 MinerU 进行端到端文档处理
* **`modalprocessors_example.py`**直接多模态内容处理
* **`office_document_test.py`**使用 MinerU 进行 Office 文档解析测试（无需 API 密钥）
* **`image_format_test.py`**使用 MinerU 进行图像格式解析测试（无需 API 密钥）
* **`text_format_test.py`**使用 MinerU 进行文本格式解析测试（无需 API 密钥）

**运行示例：**

```shell
# End-to-end processing with parser selection
python examples/raganything_example.py path/to/document.pdf --api-key YOUR_API_KEY --parser mineru

# Direct modal processing
python examples/modalprocessors_example.py --api-key YOUR_API_KEY

# Office document parsing test (MinerU only)
python examples/office_document_test.py --file path/to/document.docx

# Image format parsing test (MinerU only)
python examples/image_format_test.py --file path/to/image.bmp

# Text format parsing test (MinerU only)
python examples/text_format_test.py --file path/to/document.md

# Check LibreOffice installation
python examples/office_document_test.py --check-libreoffice --file dummy

# Check PIL/Pillow installation
python examples/image_format_test.py --check-pillow --file dummy

# Check ReportLab installation
python examples/text_format_test.py --check-reportlab --file dummy
```

---

## 🔧 配置

URL:[https://github.com/HKUDS/RAG-Anything#-configuration](https://github.com/HKUDS/RAG-Anything#-configuration)

*系统优化参数*

### 环境变量

URL:[https://github.com/HKUDS/RAG-Anything#environment-variables](https://github.com/HKUDS/RAG-Anything#environment-variables)

创建`.env`文件（参考`.env.example`）：

```shell
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_base_url  # Optional
OUTPUT_DIR=./output             # Default output directory for parsed documents
PARSER=mineru                   # Parser selection: mineru, docling, or paddleocr
PARSE_METHOD=auto              # Parse method: auto, ocr, or txt
```

**注意：**为了向后兼容，仍然支持旧版环境变量名称：

* `MINERU_PARSE_METHOD`已弃用，请使用`PARSE_METHOD`

> **注意**：只有在使用 LLM 集成进行完整的 RAG 处理时才需要 API 密钥。解析测试文件（`office_document_test.py`和`image_format_test.py`）仅用于测试解析器功能，不需要 API 密钥。

### 解析器配置

URL:[https://github.com/HKUDS/RAG-Anything#parser-configuration](https://github.com/HKUDS/RAG-Anything#parser-configuration)

RAGAnything 现在支持多种解析器，每种解析器都有其特定的优势：

#### MinerU 解析器

URL:[https://github.com/HKUDS/RAG-Anything#mineru-parser](https://github.com/HKUDS/RAG-Anything#mineru-parser)

* 支持 PDF、图像、Office 文档等多种格式
* 强大的OCR和表格提取功能
* GPU加速支持

#### Docling 解析器

URL:[https://github.com/HKUDS/RAG-Anything#docling-parser](https://github.com/HKUDS/RAG-Anything#docling-parser)

* 针对 Office 文档和 HTML 文件进行了优化
* 更好的文档结构保留
* 原生支持多种 Office 格式

#### PaddleOCR 解析器

URL:[https://github.com/HKUDS/RAG-Anything#paddleocr-parser](https://github.com/HKUDS/RAG-Anything#paddleocr-parser)

* 面向图像和PDF的OCR专用解析器
* `content_list`生成与现有处理方式兼容的文本块
* 支持可选的 Office/TXT/MD 解析，方法是先将其转换为 PDF。

安装 PaddleOCR 解析器扩展程序：

```shell
pip install -e ".[paddleocr]"
# or
uv sync --extra paddleocr
```

> **注意**：PaddleOCR 也需要`paddlepaddle`CPU/GPU 驱动程序（具体驱动程序包因平台而异）。请按照官方指南进行安装：[https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)

### MinerU 配置

URL:[https://github.com/HKUDS/RAG-Anything#mineru-configuration](https://github.com/HKUDS/RAG-Anything#mineru-configuration)

```shell
# MinerU 2.0 uses command-line parameters instead of config files
# Check available options:
mineru --help

# Common configurations:
mineru -p input.pdf -o output_dir -m auto    # Automatic parsing mode
mineru -p input.pdf -o output_dir -m ocr     # OCR-focused parsing
mineru -p input.pdf -o output_dir -b pipeline --device cuda  # GPU acceleration
```

您还可以通过 RAGAnything 参数配置解析：

```python
# Basic parsing configuration with parser selection
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output/",
    parse_method="auto",          # or "ocr", "txt"
    parser="mineru"               # Optional: "mineru", "docling", or "paddleocr"
)

# Advanced parsing configuration with special parameters
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output/",
    parse_method="auto",          # Parsing method: "auto", "ocr", "txt"
    parser="mineru",              # Parser selection: "mineru", "docling", or "paddleocr"

    # MinerU special parameters - all supported kwargs:
    lang="ch",                   # Document language for OCR optimization (e.g., "ch", "en", "ja")
    device="cuda:0",             # Inference device: "cpu", "cuda", "cuda:0", "npu", "mps"
    start_page=0,                # Starting page number (0-based, for PDF)
    end_page=10,                 # Ending page number (0-based, for PDF)
    formula=True,                # Enable formula parsing
    table=True,                  # Enable table parsing
    backend="pipeline",          # Parsing backend: pipeline|hybrid-auto-engine|hybrid-http-client|vlm-auto-engine|vlm-http-client.
    source="huggingface",        # Model source: "huggingface", "modelscope", "local"
    # vlm_url="http://127.0.0.1:3000" # Service address when using backend=vlm-http-client

    # Standard RAGAnything parameters
    display_stats=True,          # Display content statistics
    split_by_character=None,     # Optional character to split text by
    doc_id=None                  # Optional document ID
)
```

> **注意**：MinerU 2.0 不再使用`magic-pdf.json`配置文件。所有设置现在都通过命令行参数或函数参数传递。RAG-Anything 支持多种文档解析器，包括 MinerU、Docling 和 PaddleOCR。

### 处理要求

URL:[https://github.com/HKUDS/RAG-Anything#processing-requirements](https://github.com/HKUDS/RAG-Anything#processing-requirements)

不同的内容类型需要特定的可选依赖项：

* **办公文档**（.doc、.docx、.ppt、.pptx、.xls、.xlsx）：安装[LibreOffice](https://www.libreoffice.org/download/download/)
* **扩展图像格式**（.bmp、.tiff、.gif、.webp）：使用以下命令安装`pip install raganything[image]`
* **文本文件**（.txt、.md）：使用以下命令安装`pip install raganything[text]`
* **PaddleOCR 解析器**（`parser="paddleocr"`）：先安装`pip install raganything[paddleocr]`，然后安装`paddlepaddle`适用于您平台的版本。

> **📋 快速安装**：`pip install raganything[all]`用于启用所有格式支持（仅依赖 Python - LibreOffice 仍需单独安装）

---

## 🧪 支持的内容类型

URL:[https://github.com/HKUDS/RAG-Anything#-supported-content-types](https://github.com/HKUDS/RAG-Anything#-supported-content-types)

### 文档格式

URL:[https://github.com/HKUDS/RAG-Anything#document-formats](https://github.com/HKUDS/RAG-Anything#document-formats)

* **PDF文件**——研究论文、报告、演示文稿
* **办公文档**- DOC、DOCX、PPT、PPTX、XLS、XLSX
* **图片格式**- JPG、PNG、BMP、TIFF、GIF、WebP
* **文本文件**- TXT、MD

### 多模态元素

URL:[https://github.com/HKUDS/RAG-Anything#multimodal-elements](https://github.com/HKUDS/RAG-Anything#multimodal-elements)

* **图片**- 照片、图表、示意图、屏幕截图
* **表格**——数据表、对比图表、统计摘要
* **公式**- LaTeX 格式的数学公式
* **通用内容**- 通过可扩展处理器实现自定义内容类型

*有关特定格式依赖项的安装，请参阅[配置](https://github.com/HKUDS/RAG-Anything#-configuration)部分。*

---

## 📖 引用

URL:[https://github.com/HKUDS/RAG-Anything#-citation](https://github.com/HKUDS/RAG-Anything#-citation)

*学术参考*

📖

如果您发现 RAG-Anything 对您的研究有所帮助，请引用我们的论文：

```
@misc{guo2025raganythingallinoneragframework,
      title={RAG-Anything: All-in-One RAG Framework},
      author={Zirui Guo and Xubin Ren and Lingrui Xu and Jiahao Zhang and Chao Huang},
      year={2025},
      eprint={2510.12323},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.12323},
}
```

---

## 🔗 相关项目

URL:[https://github.com/HKUDS/RAG-Anything#-related-projects](https://github.com/HKUDS/RAG-Anything#-related-projects)

*生态系统与扩展*


| [⚡**LightRAG**~简单快速的 RAG~](https://github.com/HKUDS/LightRAG) | [🎥**VideoRAG**~极致长上下文视频 RAG~](https://github.com/HKUDS/VideoRAG) | [✨**迷你抹布**~极其简单的抹布~](https://github.com/HKUDS/MiniRAG) |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------------: | :----------------------------------------------------------------: |

---

## ⭐ 收藏历史

URL:[https://github.com/HKUDS/RAG-Anything#-star-history](https://github.com/HKUDS/RAG-Anything#-star-history)

*社区增长轨迹*

[![星空历史图表](https://camo.githubusercontent.com/ecd2e6786f34337407f35b748bc0513c51f343495659d885798edbdfddacb75d/68747470733a2f2f6170692e737461722d686973746f72792e636f6d2f7376673f7265706f733d484b5544532f5241472d416e797468696e6726747970653d44617465)](https://star-history.com/#HKUDS/RAG-Anything&Date)

---

## 🤝 贡献

URL:[https://github.com/HKUDS/RAG-Anything#-contribution](https://github.com/HKUDS/RAG-Anything#-contribution)

*加入创新*

我们感谢所有投稿人的宝贵贡献。

[![](https://camo.githubusercontent.com/1680c784f122bc349c4b6258a1bacea1304f3219e8ce71ee9d54388eb5f2238e/68747470733a2f2f636f6e747269622e726f636b732f696d6167653f7265706f3d484b5544532f5241472d416e797468696e67)](https://github.com/HKUDS/RAG-Anything/graphs/contributors)

```
