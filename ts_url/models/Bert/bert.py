import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig
from ...utils.utils import Projector
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

class SFA_Bert(Module):
    def __init__(self, output_dim=320):
        super(SFA_Bert, self).__init__()
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1,target_modules= ["query", "value", "key"]
        )
        # config = {
        #   "peft_type": "PREFIX_TUNING",
        #   "task_type": "TOKEN_CLS",
        #   "inference_mode": False,
        #   "num_virtual_tokens": 20,
        #   "token_dim": 768,
        #   "num_transformer_submodules": 1,
        #   "num_attention_heads": 12,
        #   "num_layers": 12,
        #   "encoder_hidden_size": 768,
        #   "prefix_projection": False,
        #     "postprocess_past_key_value_function": None,
        # }

        # peft_config = get_peft_config(config)
        model = AutoModel.from_pretrained("allenai/longformer-base-4096")
        # model = AutoModel.from_config(cfg)
        
        # def forward(input_id, **kwargs):
        #     model(input_id)
        # model.forward = forward
        # print(list(model.named_modules()))
        self.bert = get_peft_model(model, peft_config)
        self.hidden_dim = 768
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
        self.projector = Projector("4096-8192", output_dim)

    def forward(self, X, **kwargs):
        x = self.bert(X).pooler_output
        x = self.linear(x)
        features = self.projector(x)
        return x, features
