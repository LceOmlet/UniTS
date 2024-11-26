import torch
from torch.nn import functional as F
from ..registry import TRANSFORMATION
from .ts_transformation import VoidTransfromation
__all__ = ["GADF", "R_Plot"]

@TRANSFORMATION.register("GADF")
class GADF(VoidTransfromation):
    def transform(self, x, index, resize_shape=224, **kwargs):
        # Input datatype data_batch: tensor, size (batch_size, n)
        # Output datatype gadf_batch: tensor, size (batch_size, n, n), GADF for each series in the batch
        data_batch = x
        batch_size, channel, n = data_batch.shape
        device = data_batch.device
        data_batch = data_batch.cpu()
        data_batch = data_batch.reshape(batch_size * channel, n)
        datacos = data_batch.clone()
        datasin = torch.sqrt(1 - torch.clamp(datacos**2, 0, 1))
        gadf_batch = (datasin.unsqueeze(2) * datacos.unsqueeze(1)) - (datacos.unsqueeze(2) * datasin.unsqueeze(1))
        gadf_batch = gadf_batch.reshape(batch_size, channel, n, n)
        # if resize_shape > n:
        #     resize_shape = resize_shape // 2
        gadf_batch = F.interpolate(gadf_batch, size=(resize_shape, resize_shape), 
                    mode='bilinear', align_corners=False)
        
        desired_channels = 8
        if channel > desired_channels:
            gadf_batch = gadf_batch.view((batch_size, channel, resize_shape * resize_shape))
            gadf_batch = F.adaptive_avg_pool2d(gadf_batch, (desired_channels, resize_shape * resize_shape))
            gadf_batch = gadf_batch.view((batch_size, desired_channels, resize_shape, resize_shape))
        # print(gadf_batch)
        # exit()
        
        return gadf_batch.detach().to(device).float()

@TRANSFORMATION.register("RP")
class R_Plot(VoidTransfromation):
    def transform(self, x, index, delay=0, resize_shape=224, **kwargs):
        # Input datatype data_batch: tensor, size (batch_size, n)
        # Input datatype delay: int, delay embedding for RP formation, default value is 1
        # Output datatype rp_batch: tensor, size (batch_size, n - delay, n - delay), unthresholded recurrence plots for each series in the batch
        # print(data_batch.device)
        data_batch = x
        # print(data_batch.shape)
        # raise RuntimeError()
        device = data_batch.device
        batch_size, channel, n = data_batch.shape
        data_batch = data_batch.reshape(batch_size * channel, n)
        data_length = n
        
        transformed = torch.zeros(batch_size * channel, 2, data_length - delay)
        transformed[:, 0, :] = data_batch[:, 0:data_length - delay]
        transformed[:, 1, :] = data_batch[:, delay:data_length]
        
        # Vectorized calculation of recurrence plots
        transformed_expanded = transformed.unsqueeze(3)  # Expand dimensions for broadcasting
        temp = transformed_expanded - transformed_expanded.permute(0, 1, 3, 2)
        temp2 = temp ** 2
        rp_batch = torch.sum(temp2, dim=1)
        rp_batch = rp_batch.reshape(batch_size, channel, n, n)
        # if resize_shape > n:
        #     resize_shape = resize_shape // 2
        rp_batch = F.interpolate(rp_batch, size=(resize_shape, resize_shape), 
                    mode='bilinear', align_corners=False)
        desired_channels = 8
        if channel > desired_channels:
            # print(rp_batch.shape)
            rp_batch = rp_batch.view((batch_size, channel, resize_shape * resize_shape))
            rp_batch = F.adaptive_avg_pool2d(rp_batch, (desired_channels, resize_shape * resize_shape))
            rp_batch = rp_batch.view((batch_size, desired_channels, resize_shape, resize_shape))
        
        # print(rp_batch.shape)
        # exit()
        return rp_batch.detach().contiguous().to(device).float()