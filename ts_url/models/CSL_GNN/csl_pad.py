import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import sys
import torch.nn.functional as F

from ...utils.utils import reduce
from ...utils.utils import Projector
from .gcn import GCN
from .utils import generate_binomial_mask

from ...utils.layers.basics import get_activation_fn
from torch_geometric.utils import to_undirected
from ...registry import MODELS
__all__ = ["LearningShapeletsModelMixDistancesGCN"]

class MinEuclideanDistBlock(nn.Module):
  
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                               dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, masking=False):
       
        
        
        
        # print(x.shape)
        pad_size = (self.shapelets_size -1)// 2
        x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        
        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        x = -x
        # print(x.shape)
        # exit()
        
        #x = torch.cdist(x, self.shapelets, p=2)
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        # x, _ = torch.min(x, 2)
        # x = torch.mean(x, dim=1, keepdim=True)
        
        """
        n_dims = x.shape[1]
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :]
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            out += torch.cdist(x_dim, self.shapelets[i_dim : i_dim + 1, :, :], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        x = out
        x = x.transpose(2, 3)
        """
        
        # hard min compared to soft-min from the paper
        
        return x


  
   

        
class MaxCosineSimilarityBlock(nn.Module):
   
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, masking=False):
     
        """
        n_dims = x.shape[1]
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        shapelets_norm = shapelets_norm.transpose(1, 2).half()
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :].half()
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            x_dim = x_dim / x_dim.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
            out += torch.matmul(x_dim, shapelets_norm[i_dim : i_dim + 1, :, :]).float()
        
        x = out.transpose(2, 3) / n_dims
        """
        
        
        pad_size = (self.shapelets_size -1)// 2
        x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        
       
        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        
        shapelets_norm = (self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))
        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
       
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]
        x = self.relu(x)
        # x, _ = torch.max(x, 2)
        # x = torch.sum(x, dim=1, keepdim=True) / n_dims
        
        
        # ignore negative distances
        return x

   
        

class MaxCrossCorrelationBlock(nn.Module):
   
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.reverse_shapelets = nn.ConvTranspose1d(in_channels=self.shapelets.out_channels,
                                                    out_channels=self.shapelets.in_channels,
                                                    kernel_size=self.shapelets.kernel_size,
                                                    stride=self.shapelets.stride,
                                                    padding=self.shapelets.padding)
        self.reverse_shapelets.weight.data=self.shapelets.weight.data
        if self.shapelets.bias is not None:
            self.reverse_shapelets.bias.data = self.shapelets.bias.data
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = to_cuda
        self.in_channels = in_channels
        if self.to_cuda:
            self.cuda()
    
    def reverse(self, x, masking=False):
        x = x.permute(0, 1, 3, 2)

        
        
        
    def forward(self, x, masking=False):
        if self.in_channels == 1:
            new_x = [] 
            x = x.permute(1,0,2)       
            
            pad_size = (self.shapelets_size -1)// 2
            
            x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))
            for x_ in x:
                x_ = x_[:, None, :]
                x_ = self.shapelets(x_)
                new_x.append(x_[None])
            x = torch.cat(new_x, dim=0)
            x = x.permute(1, 0, 3, 2)
            # print(x.shape)
            # exit()
            # x, _ = torch.max(x, 3, keepdim=True)
            # x = x.mean(dim=0)
        else:
            pad_size = (self.shapelets_size -1)// 2
            dim = x.shape[1]
            x = F.pad(x, (pad_size, pad_size + (self.shapelets_size -1) % 2))
            x = self.shapelets(x)
            x = x.unsqueeze(1)
            # print(dim)
            x = x.tile(1, dim, 1, 1)
            x = x.permute(0, 1, 3, 2)
            # x, _ = torch.max(x, 2, keepdim=True)
            # print(x.shape)
            # exit()
        if masking:
            mask = generate_binomial_mask(x.shape)
            x *= mask
        
        return x

    



class ShapeletsDistBlocks(nn.Module):
   
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True, checkpoint=False):
        super(ShapeletsDistBlocks, self).__init__()
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'mix':
            module_list = []
            for shapelets_size, num_shapelets in self.shapelets_size_and_len.items():
                module_list.append(MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCrossCorrelationBlock(shapelets_size=shapelets_size,
                                                            num_shapelets=num_shapelets - 2 * num_shapelets//3,
                                                            in_channels=in_channels, to_cuda=self.to_cuda))
            self.blocks = nn.ModuleList(module_list)
        
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def forward(self, x, masking=False):
       
        out = []
        for block in self.blocks:
            if self.checkpoint and self.dist_measure != 'cross-correlation':
                out.append(checkpoint(block, x, masking))
            
            else:
                out.append(block(x, masking))
        out = torch.cat(out, dim=-1)

        return out



  

class LearningShapeletsModel(nn.Module):
   
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda, checkpoint=checkpoint)
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):
       
        x = self.shapelets_blocks(x, masking)
        
        x = torch.squeeze(x, 1)
        
        # test torch.cat
        #x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)
        
        x = self.projection(x)
        
        if optimize == 'acc':
            x = self.linear(x)
        
        
        return x

   
  

# @MODELS.register("mmfa")
class LearningShapeletsModelMixDistancesGCN(nn.Module):
   
    def __init__(self, feat_dim=1, num_classes=None, seq_len=224, dist_measure='mix',
                 to_cuda=True, checkpoint=False, output_dim=320, **kwargs):
        super(LearningShapeletsModelMixDistancesGCN, self).__init__()
        len_ts = seq_len
        in_channels = feat_dim

        # len_ts = 224
        num_shapelets = output_dim // 8
        dim_out = num_shapelets * 8
        self.output_dim = output_dim
        shapelets_size_and_len = {int(i): num_shapelets for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}
        self.num_shapelets_length = len(shapelets_size_and_len)
        self.num_shapelets = num_shapelets
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        
        self.shapelets_euclidean = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='euclidean', to_cuda=to_cuda, checkpoint=checkpoint)
        
        
        self.shapelets_cosine = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='cosine', to_cuda=to_cuda, checkpoint=checkpoint)
        
        self.shapelets_cross_correlation = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] - 2 * (item[1] // 3) for item in shapelets_size_and_len.items()},
                                                    dist_measure='cross-correlation', to_cuda=to_cuda, checkpoint=checkpoint)
        
        
        # self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )
        # print(sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn1 = nn.BatchNorm2d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn2 = nn.BatchNorm2d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn3 = nn.BatchNorm2d(num_features=sum(num - 2 * (num // 3) for num in self.shapelets_size_and_len.values()))
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        
        self.act = get_activation_fn('gelu')
        
        self.outpt = nn.Linear(dim_out, output_dim)
        
        self.projector = Projector("4096-8192", output_dim)

        # self.gcn = GCN(dim_out)

        

        # if num_classes is not None:
        #     self.logits = nn.Sequential(
        #         # nn.Linear(self.embed_patch_aggr, num_classes),
        #         # get_activation_fn(act),
        #         nn.Linear(output_dim, num_classes)
        #     )
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, train=False, masking=False, **kwargs):
        
        n_samples = x.shape[0]
        num_lengths = len(self.shapelets_size_and_len)
        
        out = []
        
        x_out = self.shapelets_euclidean(x, masking)
        # print(x_out.shape)
        
        x_out = self.bn1(x_out.permute(0, 3, 2 ,1)).permute(0, 3, 2 ,1)
        out.append(x_out)
        
        x_out = self.shapelets_cosine(x, masking)
        x_out = self.bn2(x_out.permute(0, 3, 2 ,1)).permute(0, 3, 2 ,1)
        out.append(x_out)
        
        x_out = self.shapelets_cross_correlation(x, masking)
        x_out = self.bn3(x_out.permute(0, 3, 2 ,1)).permute(0, 3, 2 ,1)
        out.append(x_out)
        
        out = torch.cat(out, dim=-1)

        bs, c, l, s = out.shape

        out, _ = torch.max(out, dim=2)
        
        N = out.shape[1]
        edge_index = torch.tensor([[i, j] for i in range(N) for j in range(N) if i != j], dtype=torch.long).t().cuda()

        out_ = torch.mean(out, dim=1)
        out = out_


        out = out.reshape(n_samples, -1)

        feature = out
        project = self.projector(feature)
        
        
        multi_scale_shapelet_energy = [out[:, length_i * self.num_shapelets: (length_i + 1) * self.num_shapelets] for length_i in range(self.num_shapelets_length)]
        
        if train:
            return feature, project, multi_scale_shapelet_energy   

        z = feature
        if hasattr(self, "logits"):
            if len(z.shape) ==2:
                z = z.unsqueeze(-1)
            z_pool = F.max_pool1d(z, kernel_size=z.size(2))
            z_pool = z_pool.squeeze(-1)
            # print(z_pool.shape)
            logits = self.logits(z_pool)
            return logits, z
        return z
    
    def encode(self, data, reduce_method="mean", **kwargs):
        repr = self(data, train=False, masking=False)
        repr = reduce(repr, reduce_method)
        return repr

if __name__ == "__main__":

    len_ts = 224
    samples = np.random.rand(10, 3, 1000)
    samples = torch.tensor(samples)
    sample = samples[:10].to(torch.float).cuda()
    sample = torch.nn.functional.interpolate(sample, size=(1000, ))
    print(sample.shape)
    SMMD = LearningShapeletsModelMixDistances(to_cuda=True)
    out = SMMD(sample)
    print(out[0].shape)