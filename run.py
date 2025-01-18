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
