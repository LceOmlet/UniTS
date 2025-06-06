import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = Conv(hidden_channels, hidden_channels)
        self.conv2 = Conv(hidden_channels, hidden_channels)
        self.conv3 = Conv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
    def forward(self, x, edge_index, batch):
        x_0 = x
        x = self.conv1(x, edge_index)
        x_1 = x
        x += x_0
        x = x.relu()
        x = self.conv2(x, edge_index)
        x_0 = x
        x += x_1
        x = x.relu()
        x = self.conv3(x, edge_index)
        x += x_0
        x = global_mean_pool(x, batch)    # 使用全局平均池化获得图的嵌入
        # x = self.lin(x)
        return x