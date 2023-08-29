import sys
sys.path.append('/langchain-ChatGLM')
# add configs/model_config to sys.path
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K)
from fastapi import Body, Request
from fastapi.responses import StreamingResponse

from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode


def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=[
                                                        "samples"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K,
                                          description="匹配向量数"),
                        history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                        stream: bool = Body(False, description="流式输出"),
                        local_doc_url: bool = Body(
                            False, description="知识文件返回本地路径(true)或URL(false)"),
                        request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    print("kb:", kb)
    print("list doc:", kb.list_docs())
    docs = kb.search_docs(query, top_k)
    context = "\n".join([doc.page_content for doc in docs])
    print("context:", context)


if __name__ == '__main__':
    knowledge_base_chat(query="如何向ChatGPT提问?", knowledge_base_name="samples",
                        top_k=3, stream=False, local_doc_url=False)
    knowledge_base_chat(query="遇到问题找谁??", knowledge_base_name="qa",
                        top_k=3, stream=False, local_doc_url=False)
    knowledge_base_chat(query="遇到问题可以找谁?", knowledge_base_name="qa",
                        top_k=3, stream=False, local_doc_url=False)
    knowledge_base_chat(query="艳艳有哪些职责?", knowledge_base_name="qa",
                        top_k=3, stream=False, local_doc_url=False)
