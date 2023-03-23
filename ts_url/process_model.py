try:
    from .models.mvts_transformer.src.models.ts_transformer import TSTransformerEncoder
    from .models.ts2vec.ts2vec import TS2Vec
    from .models.ts_tcc.models.model import base_Model
    from .models.ts_tcc.models.TC import TC
    from .models.default_configs.configues import model_configures
    from .models.UnsupervisedScalableRepresentationLearningTimeSeries.networks.causal_cnn import CausalCNNEncoder
except Exception:
    from models.mvts_transformer.src.models.ts_transformer import TSTransformerEncoder
    from models.ts2vec.ts2vec import TS2Vec
    from models.ts_tcc.models.model import base_Model
    from models.ts_tcc.models.TC import TC
    from models.default_configs.configues import model_configures
    from models.UnsupervisedScalableRepresentationLearningTimeSeries.networks.causal_cnn import CausalCNNEncoder
import json
import logging
from torch import nn
import torch
import numpy as np
import os

class TS_TCC(nn.Module):
    def __init__(self, device, kernel_size, feat_dim, stride, dropout, output_dims,
                                num_classes, timesteps, max_len) -> None:
        super(TS_TCC, self).__init__()
        final_out_channels = output_dims
        input_channels = feat_dim
        features_len = max_len
        self.model = base_Model(features_len,kernel_size, input_channels, stride, dropout, final_out_channels,
                                num_classes, timesteps).to(device)
        self.tenporal_contr_model = TC(device, final_out_channels, timesteps).to(device)
    
    def parameters(self):
        return self.model.parameters()
    
    def parameters_tc(self):
        return self.tenporal_contr_model.parameters()

class T_LOSS(CausalCNNEncoder):
    def __init__(self, feat_dim, channels, depth, reduced_size, output_dims, kernel_size, device, max_len):
        out_channels = output_dims
        super().__init__(feat_dim, channels, depth, reduced_size, out_channels, kernel_size)
    
    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths

        causal_cnn = self.network[0]
        linear = self.network[3]
        length = X.shape[2]
        features = np.nan(
            (X.shape[0], self.out_channels, length), np.nan
        )
        features = torch.tensor(features, dtype=X.dtype, device=X.device)
        count = 0
        batch = X
        with torch.no_grad():
            # First applies the causal CNN
            output_causal_cnn = causal_cnn(batch)
            after_pool = torch.empty(
                output_causal_cnn.size(), dtype=torch.double
            )
            if self.cuda:
                after_pool = after_pool.cuda(self.gpu)
            after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
            # Then for each time step, computes the output of the max
            # pooling layer
            for i in range(1, length):
                after_pool[:, :, i] = torch.max(
                    torch.cat([
                        after_pool[:, :, i - 1: i],
                        output_causal_cnn[:, :, i: i+1]
                    ], dim=2),
                    dim=2
                )[0]
            features[
                count * batch_size: (count + 1) * batch_size, :, :
            ] = torch.transpose(linear(
                torch.transpose(after_pool, 1, 2)
            ), 1, 2)
            count += 1

        self.encoder = self.encoder.train()
        return features

logger = logging.getLogger("__main__")

model_dicts = {
    "mvts_transformer": TSTransformerEncoder, # feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False,
    "ts2vec": TS2Vec,
    "ts_tcc": TS_TCC,
    "t_loss": T_LOSS
}

class FussionModel(nn.Module):
    def __init__(self, model_names, optim_configs, dls_setting, model_configs, ckpt_paths, device, agg_method="max", pred_len=None) -> None:
        super(FussionModel, self).__init__()
        self.model_counts = len(model_names)
        self.model_names = model_names
        self.pred_len = pred_len
        self.outputs_dims = []
        for idx, (model_name, ckpt_path, model_config) in enumerate(zip(model_names, ckpt_paths, model_configs)):
            model_config["feat_dim"] = dls_setting["input_feat_dim"]
            model_config["max_len"] = dls_setting["seq_len"]
            model_config["device"] = device
            self.outputs_dims.append(model_config["output_dims"])

            model_class = model_dicts[model_name]
            setattr(self, "model_" + str(idx), model_class(**model_config))
            getattr(self, "model_" + str(idx)).load_state_dict(torch.load(ckpt_path)["state_dict"])
        self.agg_method = agg_method
        print(dls_setting["input_feat_dim"], pred_len)
        self.output = nn.Linear(sum(self.outputs_dims), int(dls_setting["input_feat_dim"] * pred_len) if pred_len else dls_setting["label_num"])
    
    def encode(self, data ,padding_mask):
        batch_size = data.shape[0]
        encoddings_cat = []
        for idx, (model_name, output_dims) in enumerate(zip(self.model_names, self.outputs_dims)):
            model = getattr(self, "model_" + str(idx))
            if model_name == "mvts_transformer":
                encoddings = model.get_encodding(data, padding_mask)
            elif model_name == "ts_tcc":
                _, encoddings = model.model(data)
            elif model_name == "ts2vec":
                encoddings = model.encode_masked(data, 
                                torch.zeros_like(data).to(dtype=torch.bool)) 
            elif model_name == "t_loss":
                encoddings = model(data.permute(0, 2, 1))
            encoddings = encoddings.reshape(batch_size, output_dims, -1)
            if self.agg_method == "max":
                encoddings = torch.max(encoddings, dim=-1).values
            elif self.agg_method == "avg":
                encoddings = torch.mean(encoddings, dim=-1).values
            elif self.agg_method == "last":
                encoddings = encoddings[..., -1]
            encoddings_cat.append(encoddings)
        encoddings_cat = torch.cat(encoddings_cat, dim=-1)
        return encoddings_cat
    
    def forward(self, data, padding_mask):
        encoddings_cat = self.encode(data, padding_mask)
        out = self.output(encoddings_cat)
        # 
        if self.pred_len: out = out.reshape(data.shape[0], -1, data.shape[2])
        # print(out.shape)
        return out


def get_fusion_model(checkpoints, fusion_methods, dls, dls_setting, device='cpu', pred_len=None):
    model_names = []
    optim_configs = []
    model_configs = []
    ckpt_paths = []
    for idx, ckpt in enumerate(checkpoints):
        model_name = "_".join(ckpt.split("_")[6:])
        model_names.append(model_name)
        dirs = os.listdir(ckpt)
        for dr in dirs:
            dr = os.path.join(ckpt, dr)
            if "optim.json" in dr:
                with open(dr, "r") as f:
                    optim_configs.append(json.load(f))
            elif "model.json" in dr:
                with open(dr, "r") as f:
                    model_configs.append(json.load(f))
            elif ".pth" in dr:
                ckpt_paths.append(dr)
    fusion_model = FussionModel(model_names, optim_configs, dls_setting, model_configs, ckpt_paths, device,"max",pred_len)
    print(model_configs)
    new_configs = []
    for cfg in model_configs:
        if 'device' in cfg:
            if isinstance(cfg['device'], torch.device):
                cfg['device'] = cfg['device'].type
    return fusion_model, model_configs
    

            

def get_model(model_name, hp_config_path, dls, dls_setting, p_path=None, task="self-supervised", device="cpu"):
    model_config = model_configures[model_name]
    with open(model_config, "r") as f:
        model_config_ = json.load(f)
    model_config = dict()
    for key in model_config_:
        if key[0] != '@':
            model_config[key] = model_config_[key]
    model_class = model_dicts[model_name]
    if task == "self-supervised":
        model_config["feat_dim"] = dls_setting["input_feat_dim"]
        model_config["max_len"] = dls_setting["seq_len"]
        model_config["device"] = device
        model = model_class(**model_config)
    logger.info(model_config)
    logger.info(model)
    return model, model_config
        


