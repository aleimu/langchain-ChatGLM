#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@DOC : test.py 
@Date ：2023/8/11 14:12 
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", device_map="auto",
                                             torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
response = model.chat(tokenizer, messages)
print(response)

for inum, response in enumerate(model.chat(
        tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事", history=[], stream=True
)):
    print(inum, response, type(response))
