import os

import numpy as np
import torch
from torch import nn, optim
import random

from train import LearningShapeletsCL
from utils import z_normalize, eval_accuracy, TSC_multivariate_data_loader, get_weights_via_kmeans

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score

import tsaug

from torch.utils.tensorboard import SummaryWriter

import argparse


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest


import numpy as np
import pandas as pd
import joblib
import os
import torch


import torch.distributed as dist
from torch.multiprocessing import Process 
from torch.nn.parallel import DistributedDataParallel as DDP

DATA_PATH = './AD_data'








parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Dataset name')
parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
parser.add_argument('-T', '--temperature', default=0.1, type=float, help='temperature')
parser.add_argument('-l', '--lmd', default=1e-2, type=float, help='multi-scale alignment weight')
parser.add_argument('-ls', '--lmd-s', default=1.0, type=float, help='SDL weight')
parser.add_argument('-a', '--alpha', default=0.5, type=float, help='covariance matrix decay')
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-g', '--to-cuda', default=True, type=bool)
parser.add_argument('-e', '--eval-per-x-epochs', default=10, type=int)
parser.add_argument('-d', '--dist-measure', default='mix', type=str)
#parser.add_argument('-r', '--rank', default=-1, type=int)
parser.add_argument('-w', '--world-size', default=-1, type=int)
parser.add_argument('-p', '--port', default=15535, type=int)
parser.add_argument('-r', '--resize', default=0, type=int)
parser.add_argument('-c', '--checkpoint', default=False, type=bool)
parser.add_argument('--window-size', type=int)
parser.add_argument('--num-shapelets', type=int, default=40)


def evaluate_ad(dataset, augmentation=False, seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=256, to_cuda=True, eval_per_x_epochs=10, dist_measure='mix', rank=-1, world_size=-1, resize=0, checkpoint=False, window_size=100, num_shapelets=40):
    is_ddp = False
    if rank != -1 and world_size != -1:
        is_ddp = True
    if is_ddp:      
        # initialize the process group
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        seed += 1
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
              
        
        
    if dataset == 'SMAP' or dataset == 'MSL':
        all_labels = pd.read_csv(os.path.join(DATA_PATH, 'SMAP&MLS', 'labeled_anomalies.csv'))



        if dataset == 'SMAP':
            data_labels = all_labels[all_labels['spacecraft'] == 'SMAP']
        if dataset == 'MSL':
            data_labels = all_labels[all_labels['spacecraft'] == 'MSL']
    
        all_chans = data_labels['chan_id'].values
    
    elif dataset == 'SMD' or dataset == 'ASD':
        if dataset == 'SMD':
            all_chans = ['machine-1-1', 'machine-1-6', 'machine-1-7',
                        'machine-2-1', 'machine-2-2', 'machine-2-7', 'machine-2-8',
                        'machine-3-3', 'machine-3-4', 'machine-3-6', 'machine-3-8', 'machine-3-11']
        if dataset == 'ASD':
            all_chans = ['omi-' + str(i) for i in range(1, 13)]    
    
    total_preds = np.array([], dtype=np.int32)
    total_labels = np.array([], dtype=np.int32)
    #window_size = args.window_size
    
    progress_bar = tqdm(range(len(all_chans)))
    
    #print(all_chans.shape)
    for ch_idx in progress_bar:
        channel = all_chans[ch_idx]
        if dataset == 'SMAP' or dataset == 'MSL':
            train = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'train', channel + '.npy'))
            test = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'test', channel + '.npy'))
            
            
            label = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'labels', channel + '.npy'))[window_size - 1:]
            
        elif dataset == 'SMD' or dataset == 'ASD':
            train = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD/processed', channel + '_train.pkl'))
            test = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD/processed', channel + '_test.pkl'))
            
            raw_label = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD/processed', channel + '_test_label.pkl'))[window_size - 1:]

            label = np.ones_like(raw_label, dtype=np.int32)
            label[raw_label == 1] = -1
        
        train = np.nan_to_num(train, 0)
        test = np.nan_to_num(test, 0)

        scaler = MinMaxScaler((-1, 1)).fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
        
        train_data = torch.from_numpy(train).unfold(0, window_size, 1).numpy()
        test_data = torch.from_numpy(test).unfold(0, window_size, 1).numpy()
         
        n_ts, n_channels, len_ts = train_data.shape
        loss_func = nn.CrossEntropyLoss()
        num_classes = len(set(label))
        shapelets_size_and_len = {int(i): num_shapelets for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}
        #dist_measure = "cross-correlation"
        #dist_measure = "mix"
        dist_measure = dist_measure
        #dist_measure = "cosine"
        lr = 1e-2
        wd = 0
        learning_shapelets = LearningShapeletsCL(shapelets_size_and_len=shapelets_size_and_len,
                                                in_channels=n_channels,
                                                num_classes=num_classes,
                                                loss_func=loss_func,
                                                to_cuda=to_cuda,
                                                verbose=1,
                                                dist_measure=dist_measure,
                                                l3=l,
                                                l4=ls,
                                                T=T,
                                                alpha=alpha,
                                                is_ddp=is_ddp,
                                                checkpoint=checkpoint,
                                                seed=seed)
    
    
    
        if is_ddp:
            learning_shapelets.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learning_shapelets.model)
            learning_shapelets.model = DDP(learning_shapelets.model, device_ids=[rank], find_unused_parameters=True)
        #optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
        optimizer = optim.SGD(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=0.0001)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [300, 800])
        learning_shapelets.set_optimizer(optimizer)

        
        learning_shapelets.verbose = 0
        
        #if is_ddp and rank == 0:
        #    dist.barrier()

        def sequence_label(y, window_size):
            end_points = []
            for i in range(1, len(y)):
                if y[i] == 1 and y[i - 1] == -1:
                    end_points.append(i - 1)
            for e in end_points:
                y[e:e+window_size] = -1
            return y
        
        label = sequence_label(label, window_size=window_size)
        
        total_progress = tqdm(range(100))
        
        for epoch in total_progress:
            
            if epoch == 0 or (epoch + 1) % eval_per_x_epochs == 0: 
                if not is_ddp or rank == 0:
                    transformation = learning_shapelets.transform(train_data, result_type='numpy', normalize=True, batch_size=batch_size)
                    transformation_test = learning_shapelets.transform(test_data, result_type='numpy', normalize=True, batch_size=batch_size)
                    clf = IsolationForest(random_state=seed).fit(transformation)
                    preds = clf.predict(transformation_test)
                    #print(np.where(label == -1))
                    #print(set(preds), set(label))
                    #print(f1_score(label, preds, pos_label=-1), recall_score(label, preds, pos_label=-1), precision_score(label, preds, pos_label=-1))
                    f1 = f1_score(label, preds, pos_label=-1)
                    print('------------------------------')
                    print(args.dataset, window_size, chidx)
                    print(f1, epoch)
                    print('------------------------------')
                    
            losses = learning_shapelets.train(train_data, epochs=1, batch_size=batch_size, epoch_idx=epoch)
           
            
            
           
            
            
            
        
        
    
    
   
    
    
    return learning_shapelets,
  
  
def main(rank, world_size):
    args = parser.parse_args()
    
    
    
    results = evaluate_ad(args.dataset,
                            seed=args.seed,
                            T=args.temperature,
                            l=args.lmd,
                            ls=args.lmd-s,
                            alpha=args.alpha,
                            batch_size=args.batch_size,
                            to_cuda=args.to_cuda,
                            eval_per_x_epochs=args.eval_per_x_epochs,
                            dist_measure=args.dist_measure,
                            rank=rank,
                            world_size=world_size,
                            resize=args.resize,
                            checkpoint=args.checkpoint,
                            window_size=args.window_size,
                            num_shapelets=args.num_shapelets)
    if results != None:
        print(results[-1])
    


if __name__ == '__main__':
    args = parser.parse_args()
    
    world_size = args.world_size
    if world_size > 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.port)
        processes = []
        for rank in range(world_size):
            p = Process(target=main, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    else:
        main(-1, -1)
  
  
