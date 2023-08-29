from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

# 加载已存在的 Faiss 索引
index_path = '/langchain-ChatGLM/knowledge_base/cicd/vector_store/index.faiss'
index = faiss.read_index(index_path)

# 加载m3e-large模型和对应的tokenizer
model_name = "/m3e-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

query = "遇到问题可以找谁?"
inputs = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).numpy()  # 使用句子的平均向量作为表示


# 打印输出向量
print(vector)

# 执行查询
k = 5  # 返回最接近的 5 个邻居


# faiss.normalize_L2(vector)
distances, indices = index.search(vector, k)

print("\n最接近的邻居索引:")
print(indices)
print("\n对应的距离:")
print(distances)

"""
[[ 1.0278939  -0.49014142 -0.31939915 ...  0.3802984  -0.9962534
   1.2474844 ]]

最接近的邻居索引:
[[0 1 2]]

对应的距离:
[[1.5720782 1.573061  1.5732176]]
"""