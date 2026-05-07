"""
处理 sgrl.pdf 文档并建立 RAG 知识库
"""
import asyncio
import os
import ssl
import urllib3
import requests

# 禁用 SSL 验证（解决企业网络证书问题）
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    ssl._create_unverified_https_context = _create_unverified_https_context

# 为 requests 禁用警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# MiniMax API 配置
API_KEY = "sk-cp-KPHW8QBA95P1fngZvqM6mH3fZ20WvraNZfD8WfcUY1PwmS1Jl44WRviyb0UeK0MuAKtxFil4YENoql2hxxc_zKNzGj_brTpaaCaVVbCV5Z0Qxlx_yjCI-5U"
BASE_URL = "https://api.minimax.chat/v1"

async def main():
    # 创建配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 定义 LLM 模型函数
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "MiniMax-M2.5",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs,
        )

    # 定义 vision model function
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        if messages:
            return openai_complete_if_cache(
                "MiniMax-M2.5",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "MiniMax-M2.5",
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
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定义 embedding 函数 (MiniMax text-embedding-01)
    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts,
            model="text-embedding-01",
            api_key=API_KEY,
            base_url=BASE_URL,
        ),
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # 处理文档
    pdf_path = "./sgrl.pdf"
    output_dir = "./output"

    print(f"正在处理文档: {pdf_path}")
    await rag.process_document_complete(
        file_path=pdf_path,
        output_dir=output_dir,
        parse_method="auto"
    )
    print(f"文档处理完成，输出到: {output_dir}")
    print("文档解析成功！RAG 知识库已建立。")

if __name__ == "__main__":
    asyncio.run(main())