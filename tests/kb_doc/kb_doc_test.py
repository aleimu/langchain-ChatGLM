import sys
sys.path.append('/langchain-ChatGLM')
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_doc_api import search_docs

data = search_docs(query="遇到问题可以找谁?", knowledge_base_name="cicd",
                   top_k=5, score_threshold=1)
print(data)
