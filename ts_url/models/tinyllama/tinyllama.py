import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig
from ...utils.utils import Projector
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from ...registry import MODELS
from torch import nn
import os
from torch.nn import functional as F
from accelerate import PartialState
import subprocess

from transformers import BitsAndBytesConfig



# export BNB_CUDA_VERSION=122

# CUDA_VISIBLE_DEVICES=0,2,3,5 python experiment.py --model_name mmfa --gpu 0 --dataset_name NATOPS --experiment_type PT
model_path = "/home/username/vllm/TinyLlama_v1.1"
if not os.path.exists(model_path):
    model_path = "TinyLlama/TinyLlama_v1.1"
@MODELS.register("TinyLlama")
class SFA_TinyLlama(Module):
    def __init__(self, device, output_dim=320, rank=4, lora_alpha=16, tokenizer="/home/username/vllm/TinyLlama_v1.1", load_wwm_weights=True, **kwargs):
        super(SFA_TinyLlama, self).__init__()
        model_path = tokenizer
        device_string = PartialState().process_index
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
            r=rank, lora_alpha=lora_alpha, lora_dropout=0.1,target_modules= ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"] # , "attn_v", "attn_k"]
        )
        # peft_config = get_peft_config(config)


        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        

        model = AutoModel.from_pretrained(model_path, device_map="auto", quantization_config=nf4_config)
        config = model.config
        
        if not load_wwm_weights:
            model.apply(lambda module: torch.nn.init.uniform_(module.weight) if hasattr(module, 'weight') else None)
                
        self.bert = get_peft_model(model, peft_config)
        self.hidden_dim = config.hidden_size
        # print(config)
        
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim).to(device)
        self.projector = Projector("4096-8192", output_dim).to(device)

    def forward(self, X, **kwargs):
        o_device = X.device
        # print(o_device)
        
        x = self.bert(X)
        
        x = x.last_hidden_state
        x = torch.mean(x, dim=1).to(torch.float32)
        # print(dtype)
        # exit()
        x = self.linear(x)
        features = self.projector(x)
        return x, features

def __main__():
    model = SFA_Gemma()