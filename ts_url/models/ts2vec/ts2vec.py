import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from .models.encoder import TSEncoder
from .models.losses import hierarchical_contrastive_loss
from .utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
from torch import nn

def reduce(reprs, method):
    if method == "mean":
        reprs = torch.mean(reprs, dim=-1)
    elif method == "max":
        reprs = torch.mean(reprs, dim=-1).data
    elif method == "last":
        reprs = reprs[..., -1]
    return reprs

class TS2Vec(nn.Module):
    '''The TS2Vec model'''
    
    def __init__(
        self,
        feat_dim,
        output_dims=320,
        hidden_dims=64,
        max_len=100,
        depth=10,
        device='cpu',
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=feat_dim, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode_batch(self, x, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0):
        ts_l = x.shape[2]
        if sliding_length is not None:
            reprs = []
            for i in range(0, ts_l, sliding_length):
                l = i - sliding_padding
                r = i + sliding_length + (sliding_padding if not casual else 0)
                x_sliding = torch_pad_nan(
                    x[:, max(l, 0) : min(r, ts_l)],
                    left=-l if l<0 else 0,
                    right=r-ts_l if r>ts_l else 0,
                    dim=1
                )
                out = self._eval_with_pooling(
                    x_sliding,
                    mask,
                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                    encoding_window=encoding_window
                )
                reprs.append(out)
            out = torch.cat(reprs, dim=1)
            if encoding_window == 'full_series':
                out = F.max_pool1d(
                    out.transpose(1, 2).contiguous(),
                    kernel_size = out.size(1),
                ).squeeze(1)
        else:
            out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
            if encoding_window == 'full_series':
                out = out.squeeze(1)
        return out
    
    def encode_masked(self, x, mask):
        out = self._eval_with_pooling(x, mask)
        return out
    
    def encode(self, data, reduce_method="mean", **kwarg):
        embeddings = self.encode_masked(data, 
                                torch.zeros_like(data).to(dtype=torch.bool)) 
        
        return embeddings

    
    # def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
    #     ''' Compute representations using the model.
        
    #     Args:
    #         data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
    #         mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
    #         encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
    #         casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
    #         sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
    #         sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
    #         batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
    #     Returns:
    #         repr: The representations for data.
    #     '''
    #     assert self.net is not None, 'please train or load a net first'
    #     assert data.ndim == 3
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #     n_samples, ts_l, _ = data.shape

    #     org_training = self.net.training
    #     self.net.eval()
        
    #     dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
    #     loader = DataLoader(dataset, batch_size=batch_size)
        
    #     with torch.no_grad():
    #         output = []
    #         for batch in loader:
    #             x = batch[0]
    #             if sliding_length is not None:
    #                 reprs = []
    #                 if n_samples < batch_size:
    #                     calc_buffer = []
    #                     calc_buffer_l = 0
    #                 for i in range(0, ts_l, sliding_length):
    #                     l = i - sliding_padding
    #                     r = i + sliding_length + (sliding_padding if not casual else 0)
    #                     x_sliding = torch_pad_nan(
    #                         x[:, max(l, 0) : min(r, ts_l)],
    #                         left=-l if l<0 else 0,
    #                         right=r-ts_l if r>ts_l else 0,
    #                         dim=1
    #                     )
    #                     if n_samples < batch_size:
    #                         if calc_buffer_l + n_samples > batch_size:
    #                             out = self._eval_with_pooling(
    #                                 torch.cat(calc_buffer, dim=0),
    #                                 mask,
    #                                 slicing=slice(sliding_padding, sliding_padding+sliding_length),
    #                                 encoding_window=encoding_window
    #                             )
    #                             reprs += torch.split(out, n_samples)
    #                             calc_buffer = []
    #                             calc_buffer_l = 0
    #                         calc_buffer.append(x_sliding)
    #                         calc_buffer_l += n_samples
    #                     else:
    #                         out = self._eval_with_pooling(
    #                             x_sliding,
    #                             mask,
    #                             slicing=slice(sliding_padding, sliding_padding+sliding_length),
    #                             encoding_window=encoding_window
    #                         )
    #                         reprs.append(out)

    #                 if n_samples < batch_size:
    #                     if calc_buffer_l > 0:
    #                         out = self._eval_with_pooling(
    #                             torch.cat(calc_buffer, dim=0),
    #                             mask,
    #                             slicing=slice(sliding_padding, sliding_padding+sliding_length),
    #                             encoding_window=encoding_window
    #                         )
    #                         reprs += torch.split(out, n_samples)
    #                         calc_buffer = []
    #                         calc_buffer_l = 0
                    
    #                 out = torch.cat(reprs, dim=1)
    #                 if encoding_window == 'full_series':
    #                     out = F.max_pool1d(
    #                         out.transpose(1, 2).contiguous(),
    #                         kernel_size = out.size(1),
    #                     ).squeeze(1)
    #             else:
    #                 out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
    #                 if encoding_window == 'full_series':
    #                     out = out.squeeze(1)
                        
    #             output.append(out)
                
    #         output = torch.cat(output, dim=0)
            
    #     self.net.train(org_training)
    #     return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
