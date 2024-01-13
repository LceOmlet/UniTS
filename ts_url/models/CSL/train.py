import os
import warnings

import numpy as np
import torch

import tsaug
from torch import nn

from tqdm import tqdm

from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances




class LearningShapeletsCL:
    """
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        The keys are the length of the shapelets and the values the number of shapelets of
        a given length, e.g. {40: 4, 80: 4} learns 4 shapelets of length 40 and 4 shapelets of
        length 80.
    loss_func : torch.nn
        the loss function
    in_channels : int
        the number of input channels of the dataset
    num_classes : int
        the number of output classes.
    dist_measure: `euclidean`, `cross-correlation`, or `cosine`
        the distance measure to use to compute the distances between the shapelets.
      and the time series.
    verbose : bool
        monitors training loss if set to true.
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True, l3=0.0, l4=0.0, T=0.1, alpha=0.0, is_ddp=False, checkpoint=False, seed=None):
        self.is_ddp = is_ddp
        self.checkpoint = checkpoint
        self.seed = seed
        if dist_measure == 'mix':
            self.model = LearningShapeletsModelMixDistances(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
                                            to_cuda=to_cuda, checkpoint=checkpoint)
        else:
            self.model = LearningShapeletsModel(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
                                            to_cuda=to_cuda, checkpoint=checkpoint)
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None
        self.scheduler = None

        
     
        self.l3 = l3
        self.l4 = l4
        self.alpha = alpha
        self.use_regularizer = False
        
         
        
        #self.mask = MaskBlock(p=0.5)
        
        #self.bn = nn.BatchNorm1d(num_features=self.model.num_shapelets)
        #self.relu = nn.ReLU()
        
        #if self.to_cuda:
        #    self.mask.cuda()
        #    self.bn.cuda()
        #    self.relu.cuda()
        
        self.T = T
        
        #self.r = 64
        
        #self.num_clusters = [0.01, 0.02, 0.04]
        

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler


    
    def update(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def update_CL(self, x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k):
    
        augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 32 - 1))',
                             'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 32 - 1))',
                             'Pool(seed=np.random.randint(2 ** 32 - 1))',
                             'Quantize(seed=np.random.randint(2 ** 32 - 1))',
                             'TimeWarp(seed=np.random.randint(2 ** 32 - 1))'
                             ]
        #augmentation_list = ['AddNoise()', 'Pool()', 'Quantize()', 'TimeWarp()']
        
        ts_l = x.size(2)
        
        aug1 = np.random.choice(augmentation_list, 1, replace=False)
             
        x_q = x.transpose(1,2).cpu().numpy()
        for aug in aug1:
            x_q = eval('tsaug.' + aug + '.augment(x_q)')
        x_q = torch.from_numpy(x_q).float()
        x_q = x_q.transpose(1,2)
        
        if self.to_cuda:
            x_q = x_q.cuda()
                
        
        aug2 = np.random.choice(augmentation_list, 1, replace=False)
        while (aug2 == aug1).all():
            aug2 = np.random.choice(augmentation_list, 1, replace=False)
        
        x_k = x.transpose(1,2).cpu().numpy()
        for aug in aug2:
            x_k = eval('tsaug.' + aug + '.augment(x_k)')
        x_k = torch.from_numpy(x_k).float()
        x_k = x_k.transpose(1,2)
        
        if self.to_cuda:
            x_k = x_k.cuda()
        
        
        
        #print(x_q, x_k)
        
        num_shapelet_lengths = len(self.shapelets_size_and_len)
        num_shapelet_per_length = self.num_shapelets // num_shapelet_lengths
        
        
        with torch.autograd.set_detect_anomaly(True):
            q = self.model(x_q, optimize=None, masking=False)
            
           
            
            k = self.model(x_k, optimize=None, masking=False)
          
            
            
            
            
            
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)
            logits = torch.einsum('nc,ck->nk', [q, k.t()])
            logits /= self.T
            labels = torch.arange(q.shape[0], dtype=torch.long)
            
            
            if self.to_cuda:
                labels = labels.cuda()
            
            
            loss = self.loss_func(logits, labels)     
            
            
            q_sum = None
            q_square_sum = None
            
            
            k_sum = None
            k_square_sum = None
            
            loss_sdl = 0
            c_normalising_factor_q = self.alpha * c_normalising_factor_q + 1
            
            c_normalising_factor_k = self.alpha * c_normalising_factor_k + 1
            #print(q.shape)
            for length_i in range(num_shapelet_lengths):
                qi = q[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
                ki = k[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
                
                logits = torch.einsum('nc,ck->nk', [nn.functional.normalize(qi, dim=1), nn.functional.normalize(ki, dim=1).t()])
                logits /= self.T
                #print(logits)
                loss += self.loss_func(logits, labels)
                
                
                if q_sum == None:
                    q_sum = qi
                    q_square_sum = qi * qi
                else:
                    q_sum = q_sum + qi
                    q_square_sum = q_square_sum + qi * qi
                    
                C_mini_q = torch.matmul(qi.t(), qi) / (qi.shape[0] - 1)
                C_accu_t_q = self.alpha * C_accu_q[length_i] + C_mini_q
                C_appx_q = C_accu_t_q / c_normalising_factor_q
                loss_sdl += torch.norm(C_appx_q.flatten()[:-1].view(C_appx_q.shape[0] - 1, C_appx_q.shape[0] + 1)[:, 1:], 1).sum()
                #print(length_i)
                C_accu_q[length_i] = C_accu_t_q.detach()
                
                if k_sum == None:
                    k_sum = ki
                    k_square_sum = ki * ki
                else:
                    k_sum = k_sum + ki
                    k_square_sum = k_square_sum + ki * ki
                    
                C_mini_k = torch.matmul(ki.t(), ki) / (ki.shape[0] - 1)
                C_accu_t_k = self.alpha * C_accu_k[length_i] + C_mini_k
                C_appx_k = C_accu_t_k / c_normalising_factor_k
                loss_sdl += torch.norm(C_appx_k.flatten()[:-1].view(C_appx_k.shape[0] - 1, C_appx_k.shape[0] + 1)[:, 1:], 1).sum()
                #print(length_i)
                C_accu_k[length_i] = C_accu_t_k.detach()
                
                
                
                
                
            
            loss_cca = 0.5 * torch.sum(q_square_sum - q_sum * q_sum / num_shapelet_lengths) + 0.5 * torch.sum(k_square_sum - k_sum * k_sum / num_shapelet_lengths)
            
            
            loss += self.l3 * (loss_cca + self.l4 * loss_sdl) 
            
              
                    
                    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        
        return [loss.item(), 0, loss_cca.item(), loss_sdl.item(), 0], C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k
        
    


    def fine_tune(self, X, Y, epochs=1, batch_size=256, epoch_idx=-1):
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")
        
        
        
        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()
        
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.long).contiguous()
        
        
        
        

        train_ds = torch.utils.data.TensorDataset(X, Y)
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None
        
        
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=True)

        # set model in train mode
        self.model.train()

        losses_ce = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        
        
        
        
        
        
        for epoch in progress_bar:
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)
            
            
            
            for (x, y) in train_dl:
                
                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.cuda()
                    y = y.cuda()
                    #print("Training data", idx, " on cuda ", torch.cuda.current_device())
                loss_ce = self.update(x, y)
                losses_ce.append(loss_ce)
        return losses_ce
        
        
    def train(self, X, epochs=1, batch_size=256, epoch_idx=-1):
        
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")
        
        
        
        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()
        
        
        
        
        
        

        train_ds = torch.utils.data.TensorDataset(X, torch.arange(X.shape[0]))
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None
        
        
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=True)

        # set model in train mode
        self.model.train()

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        current_loss_dist = 0
        current_loss_sim = 0
        
        
        
        self.model.train()
        
        for epoch in progress_bar:
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)
            
            if self.to_cuda:
                c_normalising_factor_q = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_q = [torch.tensor([0], dtype=torch.float).cuda() for _ in range(len(self.shapelets_size_and_len))]
                c_normalising_factor_k = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_k = [torch.tensor([0], dtype=torch.float).cuda() for _ in range(len(self.shapelets_size_and_len))]
            else:
                c_normalising_factor_q = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_q = [torch.tensor([0], dtype=torch.float).cuda() for _ in range(len(self.shapelets_size_and_len))]
                c_normalising_factor_k = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_k = [torch.tensor([0], dtype=torch.float).cuda() for _ in range(len(self.shapelets_size_and_len))]
            
            
            for (x, idx) in train_dl:
                
                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.cuda()
                    #print("Training data", idx, " on cuda ", torch.cuda.current_device())
                
                
                    
        
                if not self.use_regularizer:       
                    current_loss_ce, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k = self.update_CL(x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k)
                        
        
                        
                    losses_ce.append(current_loss_ce)
                else:
                    pass
                 
                
                
            if not self.use_regularizer:
                progress_bar.set_description(f"Loss: {current_loss_ce}")
            else:
                if self.l1 > 0.0 and self.l2 > 0.0:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}, "
                                                 f"Loss sim: {current_loss_sim}")
                else:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}")
            if self.scheduler != None:
                self.scheduler.step()
            
            
                
        return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
        losses_ce, losses_dist)

    def transform(self, X, *, batch_size=512, result_type='tensor', normalize=False):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        shapelet_transform = []
        for (x, ) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
            #shapelet_transform = self.model.transform(X)
                shapelet_transform.append(self.model(x, optimize=None).cpu())
        shapelet_transform = torch.cat(shapelet_transform, 0)
        if normalize:
            shapelet_transform = nn.functional.normalize(shapelet_transform, dim=1)
        if result_type == 'tensor':
            return shapelet_transform
        return shapelet_transform.detach().numpy()
    
    def predict(self, X, *, batch_size=512):
 
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        for (x, ) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
            #shapelet_transform = self.model.transform(X)
                preds.append(self.model(x).cpu())
        preds = torch.cat(preds, 0)
        
        return preds.detach().numpy() 









