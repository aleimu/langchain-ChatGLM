import pickle
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# 加载m3e-large模型和tokenizer
model_name = "/m3e-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 加载Faiss向量库
index_file = '/langchain-ChatGLM/knowledge_base/cicd/vector_store/index.faiss'
index = faiss.read_index(index_file)

# 加载索引文件
index_pkl = '/langchain-ChatGLM/knowledge_base/cicd/vector_store/index.pkl'
with open(index_pkl, "rb") as f:
    index_info = pickle.load(f)

# 转换输入句子为向量
input_sentence = "遇到问题可以找谁?"

# 使用tokenizer对输入进行处理
inputs = tokenizer.encode_plus(
    input_sentence, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 获取模型输出
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# 提取最后一层输出作为句子向量
sentence_vector = outputs.last_hidden_state[:, 0, :].numpy()

# 使用Faiss进行相似问题匹配
k = 5  # 返回最相似的前5个问题
D, I = index.search(sentence_vector, k)  # D是距离，I是相似问题的索引

print("\n最接近的邻居索引:")
print(D)
print("\n对应的距离:")
print(I)

# 打印相似问题
print("相似问题：")
for i in range(k):
    similar_question_index = I[0][i]
    similar_question_info = index_info[similar_question_index]
    print(f"- {similar_question_info}")
