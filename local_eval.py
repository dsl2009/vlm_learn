from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download, AutoTokenizer
import torch
model_name = "Qwen/Qwen3-0.6B"

cache_dir = f'./cache/'
model_cache_dir = f'{cache_dir}{model_name}'

snapshot_download(model_name, cache_dir=cache_dir, revision="master")
# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
print(model)


# prepare the model input
prompt = "中国历史上一共多少朝代"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model_inputs["input_ids"].shape)
print(model_inputs["attention_mask"].shape)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)