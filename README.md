# Sailor2 Speculative Decoding with Glide Model

This repository contains the implementation of speculative decoding for the Sailor2 project, utilizing the Glide model. The primary goal is to enhance the decoding process by integrating the Glide model into the existing Sailor2 framework.

## How to Use

Suppose you save the weights in file_name.
```python
from transformers import AutoTokenizer
from glide_sailor import Qwen2Glide
tokenizer = AutoTokenizer.from_pretrained(file_name)
model = Qwen2Glide.from_pretrained(file_name, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Hello, how are you?"
prompt_id = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="right")
prompt_id = prompt_id['input_ids'].cuda()
output_ids = model.spec_generate(prompt_id, max_gen_len=128)
output = tokenizer.decode(output_ids[0])
```
