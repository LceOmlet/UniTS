import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig
from ...utils.utils import Projector
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from ...registry import MODELS
from torch import nn
from transformers import BitsAndBytesConfig

@MODELS.register("LongFormer")
class SFA_Bert(Module):
    def __init__(self, output_dim=320, rank=4, lora_alpha=16, load_wwm_weights=False, **kwargs):
        super(SFA_Bert, self).__init__()
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
            r=rank, lora_alpha=lora_alpha, lora_dropout=0.1,target_modules= ["query", "value", "key"]
        )
        
        
        # peft_config = get_peft_config(config)
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
        # config.max_position_embeddings = 2048
        
        model = AutoModel.from_pretrained("google/bigbird-roberta-base")
        # if not load_wwm_weights:
        #     model.apply(lambda module: torch.nn.init.uniform_(module.weight) if hasattr(module, 'weight') else None)
            
        
                
        self.bert = get_peft_model(model, peft_config)
        self.hidden_dim = 768
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
        self.projector = Projector("4096-8192", output_dim)

    def forward(self, X, **kwargs):
        x = self.bert(X).pooler_output.to(torch.float32)
        x = self.linear(x)
        features = self.projector(x)
        return x, features
