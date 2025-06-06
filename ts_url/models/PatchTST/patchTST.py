
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ...utils.layers.pos_encoding import *
from ...utils.layers.basics import *
from ...utils.layers.attention import *
from ...registry import MODELS
            
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 
                             'regression', 'classification', 
                             'prototypical', 'sample_aggr', 
                             "set_info_res"], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        self.backbone = PatchTSTEncoder(num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, self.n_vars, num_patch, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)
        elif head_type == 'prototypical':
            self.head = Prototypical(512)
        elif head_type == "sample_aggr":
            self.head = Prototypical(512, 16, "sample_aggr")
        elif head_type == "set_info_res":
            self.head = Prototypical(1024, 16, "set_info_res")

    @staticmethod
    def split_to_fragments(tensor, fragment_length):
        # Calculate the number of fragments and the length of the last fragment
        num_fragments = (len(tensor) + fragment_length - 1) // fragment_length
        last_fragment_length = len(tensor) % fragment_length

        # Split the tensor into fragments
        fragments = [tensor[i * fragment_length : (i + 1) * fragment_length] for i in range(num_fragments - 1)]
        if last_fragment_length != 0:
            fragments.append(tensor[-last_fragment_length:])
        else:
            fragments.append(tensor[-fragment_length:])
        return fragments

    def forward(self, z, y=None, valid=False):                             
        """
        z: tensor [bs x meta x num_patch x n_vars x patch_len]
        """   
        prototypical = isinstance(z, list)
        if prototypical:
            r_channels, r_samples = [], []
            for zi in z:
                if valid:
                    r_channel, r_sample = [], []
                    zi = self.split_to_fragments(zi, 256)
                    for zii in zi:
                        r_channel_, r_sample_ = self.backbone(zii)
                        r_sample.append(r_sample_)
                        r_channel.append(r_channel_)
                    r_channel = torch.cat(r_channel, dim=0)
                    r_sample = torch.cat(r_sample, dim=0)
                else:
                    r_channel, r_sample = self.backbone(zi)
                r_channels.append(r_channel)
                r_samples.append(r_sample)
            pred = self.head(r_samples, y, valid)
            return pred
            
        else:
            r_channel, r_sample = self.backbone(z)
            z = self.head(r_sample)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y



class Prototypical(nn.Module):
    def __init__(self, embedding_dim, n_head=None, head_type="protypical") -> None:
        super().__init__()
        self.n_head = n_head
        self.head_type = head_type
        if head_type=="sample_aggr":
            self.activation = get_activation_fn("gelu")
            self.batch_aggr = MultiheadAttention(embedding_dim, n_head, embedding_dim // n_head, embedding_dim // n_head)
        if head_type=="set_info_res":
            self.activation = get_activation_fn("gelu")
            self.res_linear = nn.Linear(embedding_dim, embedding_dim) 
            self.batch_aggr_0 = MultiheadAttention(embedding_dim, n_head, embedding_dim // n_head, embedding_dim // n_head)
            self.batch_aggr_1 = MultiheadAttention(embedding_dim, n_head, embedding_dim // n_head, embedding_dim // n_head)
            self.ff_linear = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                self.activation,
                nn.Linear(embedding_dim, embedding_dim)
            )
        
    def forward(self, embeddings, labels, valid=False):
        # Step 1: Calculate the class prototypes (centroids) for each class
        logits = []
        for embedding, label in zip(embeddings, labels):
            if valid:
                label_index = []
                for value in torch.unique(label):
                    indices = torch.nonzero(label == value).squeeze()
                    label_index.append(indices)
                min_index = min([len(index_) for index_ in label_index])
                base_num = min(min_index // 2, 128 // len(label_index))
                embedding_ = embedding
                base_index = [index_[:base_num] for index_ in label_index]
                embedding = torch.cat([embedding_[index_] for index_ in base_index], dim=0)
                label = torch.cat([label[index_] for index_ in base_index], dim=0)
            else:
                embedding_ = embedding

            if self.head_type=="sample_aggr":
                eshape = embedding.shape
                embedding = self.activation(embedding)
                embedding, _ = self.batch_aggr(embedding, embedding, embedding)
                embedding = embedding.reshape(eshape)
            if self.head_type=="set_info_res":
                embedding_aggr = self.activation(embedding)
                embedding, _ = self.batch_aggr_0(embedding
                                                    , embedding, embedding)
                embedding = self.activation(embedding)
                embedding, _ = self.batch_aggr_1(embedding
                                                    , embedding, embedding)
                embedding_aggr = self.activation(embedding.mean(dim=0, keepdim=True))
                embedding_aggr = self.res_linear(embedding_aggr.squeeze(1))
            embedding_ = torch.cat([embedding_aggr.tile((embedding_.shape[0], 1)), embedding_], dim=1)
            embedding_ = self.activation(embedding_)
            embedding_ = self.ff_linear(embedding_)

            if valid:
                embedding = torch.cat([embedding_[index_] for index_ in base_index], dim=0)
            else:
                embedding = embedding_

            unique_labels = torch.unique(label)
            num_classes = len(unique_labels)
            logit = torch.zeros(embedding_.size(0), num_classes, device=embedding.device)
            sample_logits = torch.matmul(embedding, embedding_.transpose(0, 1))

            if valid:
                for i, l in enumerate(unique_labels):
                    class_mask = label == l
                    logit[:, i] = sample_logits[class_mask].mean(0)
                for index_ in base_index:
                    logit[index_] = -1
            else:
                eye = torch.eye(sample_logits.shape[0]).to(device=sample_logits.device)
                sample_logits = sample_logits * (1 - eye)
                for i, l in enumerate(unique_labels):
                    class_mask = label == l
                    norm = torch.ones(logit.shape[0]).to(device=sample_logits
                                                         .device) * class_mask.sum() \
                                                            - class_mask.to(sample_logits.dtype)
                    logit[:, i] = sample_logits[class_mask].sum(0) / norm

            # Step 3: Calculate the log-probabilities using the negative squared distances
            logits.append(logit)
        return logits

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, c_in, patch_num, patch_len, dropout, embed_dim=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, patch_len * c_in * patch_num)
        self.n_vars = c_in
        self.patch_num = patch_num
        self.patch_len = patch_len

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = self.linear( self.dropout(x) )\
            .reshape((-1, self.patch_num, self.n_vars, self.patch_len))      # [bs x nvars x num_patch x patch_len]
        return x

def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    # print(xb.shape)
    proto = isinstance(xb, list)
    if not proto:
        xb = [xb]
    
    new_xb = []
    for x in xb:
        seq_len = x.shape[1]
        num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(num_patch-1)
        s_begin = seq_len - tgt_len
        
        x = x[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
        x = x.unfold(dimension=-2, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        new_xb.append(x)
    
    if not proto:
        xb = new_xb[0]
    else:
        xb = new_xb
    
    return xb, num_patch

def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        # layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

@MODELS.register("PatchTST")
class PatchTSTEncoder(nn.Module):
    def __init__(self, patch_len=12, stride=12, num_classes=None,
                 n_layers=2, d_model=320, n_heads=16, 
                 d_ff=320, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, embed_dim=320,
             **kwargs):

        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.pe =pe
        self.learn_pe = learn_pe

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)      

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)
        
        # Channel Encoder
        self.c_encoder = TSTEncoder(embed_dim, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers + 1, 
                                    store_attn=store_attn)
        
        self.activation = get_activation_fn(act)
        
        e1 = embed_dim
        self.embed_patch_aggr = e1
        self.W_E1 = nn.Linear(d_model, e1)

        if num_classes is not None:
            self.logits = nn.Sequential(
                # nn.Linear(self.embed_patch_aggr, num_classes),
                # get_activation_fn(act),
                nn.Linear(self.embed_patch_aggr, num_classes)
            )
        
        self.projector = Projector("4096-8192", self.embed_patch_aggr)

    def forward(self, x, train_mode="ts_tcc") -> Tensor:          
        """
        x: tensor [bs x timesteps x nvars]
        """
        # print(x.shape)
        x, _ = create_patch(x.permute(0,2,1), self.patch_len, self.stride)
        # [bs x num_patch x nvars x patch_len]
        bs, num_patch, n_vars, patch_len = x.shape
        # Positional encoding
        self.W_pos = positional_encoding(self.pe, self.learn_pe, num_patch, self.d_model).to(x.device)                                                    # z: [bs * nvars x num_patch x d_model]

        # Input encoding
        x = self.W_P(x.to(dtype=torch.float))                                                      # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)                                                    # z: [bs * nvars x num_patch x d_model]
        z = self.activation(z)
        z = self.W_E1(z)
        z = self.activation(z)
        # z = torch.mean(z, dim=1)
        if "tcc" in train_mode:
            z = torch.reshape(z, (-1,n_vars, num_patch, self.embed_patch_aggr))               # z: [bs x nvars x d_model]
            z = z.permute(0, 2, 1, 3)
            z = torch.reshape(z, (-1, n_vars, self.embed_patch_aggr))
        elif "spec" in train_mode:
            z = torch.mean(z, dim=1)
            z = torch.reshape(z, (-1, n_vars, self.embed_patch_aggr))
        # print(z.shape)
        r_channel = self.c_encoder(z)
        r_sample = torch.mean(r_channel, dim=1)                 # z: [bs x nvars x d_model x num_patch]

        if "tcc" in train_mode:
            z = torch.reshape(r_sample, (-1, num_patch, self.embed_patch_aggr))
            z = z # z: [bs x num_patch x d_model]
        elif "spec" in train_mode:
            z = r_sample
            project = self.projector(z)
        # print(z.shape)

        if train_mode == "train_vic":
            return z, project        


        if hasattr(self, "logits"):
            if len(z.shape) ==2:
                z = z.unsqueeze(-1)
            z_pool = F.max_pool1d(z, kernel_size=z.size(2))
            z_pool = z_pool.squeeze(-1)
            # print(z_pool.shape)
            logits = self.logits(z_pool)
            return logits, z
        return z
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src



