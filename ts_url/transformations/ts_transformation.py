from ..registry import TRANSFORMATION
import numpy as np
import torch
import tsaug
import random
from torch import nn
__all__ = ["VoidTransfromation", "Augmentation"]

@TRANSFORMATION.register("default")
class VoidTransfromation:
    def __init__(self, X, **kwargs) -> None:
        pass

    def transform(self, x, index, **kwargs):

        return x.detach()
    
@TRANSFORMATION.register("augmentation")
class Augmentation(VoidTransfromation):
    def __init__(self, **kwargs) -> None:
        pass

    def transform(self, x, index=None, **kwargs):
        augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 16 - 1))',
						'Pool(seed=np.random.randint(2 ** 16 - 1))',
						'Quantize(seed=np.random.randint(2 ** 16 - 1))',
						'TimeWarp(seed=np.random.randint(2 ** 16 - 1))']
        # print(x.shape)
        aug1 = np.random.choice(augmentation_list, 1, replace=False)
        ts_l = x.size(2)
        x_q = x.transpose(1,2).cpu().numpy()
        for aug in aug1:
            x_q = eval('tsaug.' + aug + '.augment(x_q)')
        x_q = torch.from_numpy(x_q).float()
        x_q = x_q.transpose(1,2)
        # print(x_q.shape)
        
        return x_q.float()

@TRANSFORMATION.register("temporal_neiborhood")
class TemporalNeiborhood(VoidTransfromation):
    def __init__(self, X, random_window=(24, 32), **kwargs) -> None:
        self.X = X
        self.random_window = random_window

    def transform(self, x, index, **kwargs):
        rand_idx = random.randint(*self.random_window)
        rand_idx = min(index + rand_idx, len(self.X) - 1)
        return torch.from_numpy(self.X[rand_idx]).float().unsqueeze(0)
        