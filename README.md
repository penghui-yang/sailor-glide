# Sailor2 Speculative Decoding with Glide Model

This repository contains the implementation of speculative decoding for the Sailor2 project, utilizing the Glide model. The primary goal is to enhance the decoding process by integrating the Glide model into the existing Sailor2 framework.

## How to Use

Suppose you save the weights in file_name.
```python
from transformers import AutoTokenizer
from glide_sailor import Qwen2Glide
import torch

file_name = ""
tokenizer = AutoTokenizer.from_pretrained(file_name)
model = Qwen2Glide.from_pretrained(file_name, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Model bahasa adalah model probabilistik"
### The given Indonesian input translates to 'A language model is a probabilistic model of.'
prompt_id = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="right")

output_ids = model.spec_generate(prompt_id, max_gen_len=64)
output = tokenizer.decode(output_ids[0])
print(output)
```
