import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,CLIPVisionModel, CLIPImageProcessor
import swanlab
from torch import nn
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models



class VLM(nn.Module):
    def __init__(self):
        super(VLM, self).__init__()
        model_name = "Qwen/Qwen3-0.6B"
        cache_dir = f'../cache/'
        model_cache_dir = f'{cache_dir}{model_name}'
        snapshot_download(model_name, cache_dir=cache_dir, revision="master")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, use_fast=False, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(model_cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
        self.vision, self.preprocess= load_from_name("ViT-B-16", device='cuda', download_root=cache_dir)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.LayerNorm(1024)
        )
        self.embed_tokens = self.llm.model.embed_tokens
        print(self.embed_tokens)

    def forward(self, img, input_ids=None, attention_mask=None):
        with torch.no_grad():
            img_emb = self.vision.encode_image(img)
        txt_emb = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([img_emb, txt_emb], dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)





if __name__ == '__main__':
    model = VLM()
    img = torch.randn(1, 3, 224, 224).cuda()
    model(img)