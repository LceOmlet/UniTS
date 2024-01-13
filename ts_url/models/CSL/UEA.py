import os

import numpy as np
import torch
from torch import nn, optim
import random

from train import LearningShapeletsCL
from utils import z_normalize, eval_accuracy, TSC_multivariate_data_loader, get_weights_via_kmeans

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
import tsaug

import argparse



import torch.distributed as dist
from torch.multiprocessing import Process 
from torch.nn.parallel import DistributedDataParallel as DDP

UEA_path = './Multivariate_ts'
UEA_datasets = os.listdir(UEA_path)
UEA_datasets.sort()








parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='UEA dataset name')
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
parser.add_argument('--task', default='classification', type=str)



def evaluate_UEA(dataset, seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True, eval_per_x_epochs=10, dist_measure='mix', rank=-1, world_size=-1, resize=0, checkpoint=False, task='classification'):
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
              
        
        
    #if is_ddp and rank != 0:
    #    dist.barrier()    
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(UEA_path, dataset)
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)

    if resize > 0:
        X_train = tsaug.Resize(size=resize, seed=seed).augment(X_train.swapaxes(-1, -2)).swapaxes(-1, -2)
        X_test = tsaug.Resize(size=resize, seed=seed).augment(X_test.swapaxes(-1, -2)).swapaxes(-1, -2)
   

   
    n_ts, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    
    # K = MV = 40, R = 8
    # D_repr = RK
    shapelets_size_and_len = {int(i): 40 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}
    
    
    dist_measure = dist_measure
    lr = 1e-2
    wd = 0
    learning_shapelets = LearningShapeletsCL(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=to_cuda,
                                       verbose=0,
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

        
    
    
    total_progress = tqdm(range(200))
    for epoch in total_progress:
        if epoch == 0 or (epoch + 1) % eval_per_x_epochs == 0:
            if not is_ddp or rank == 0:
                if task == 'clustering':
                    
                    transformation_test = learning_shapelets.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
                    scaler = RobustScaler()
                    transformation_test = scaler.fit_transform(transformation_test)
                    
                    
                    pca = PCA(n_components=10)
                    low_dim_test = pca.fit_transform(transformation_test)
                    preds = KMeans(n_clusters=num_classes, init='random').fit_predict(low_dim_test)
                    ri_test = rand_score(preds, y_test)
                    nmi_test = normalized_mutual_info_score(preds, y_test)
                    if not is_ddp or rank == 0:     
                        
                        print('KMeans: ', ri_test, nmi_test, epoch)
                    
               
                else:
                    
                    
                    
                    transformation = learning_shapelets.transform(X_train, result_type='numpy', normalize=True, batch_size=batch_size)
                    transformation_test = learning_shapelets.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
                    scaler = RobustScaler()
                    transformation = scaler.fit_transform(transformation)
                    transformation_test = scaler.transform(transformation_test)
                        
                        
                        
                        
                    acc_val = -1
                    C_best = None    
                    for C in [10 ** i for i in range(-4, 5)]:
                        clf = SVC(C=C, random_state=42)
                        acc_i = cross_val_score(clf, transformation, y_train, cv=5)
                        if acc_i.mean() > acc_val:
                            C_best = C
                    clf = SVC(C=C_best, random_state=42)
                    clf.fit(transformation, y_train)      
                    train_acc = accuracy_score(clf.predict(transformation), y_train)
                    test_acc = accuracy_score(clf.predict(transformation_test), y_test)

                    if not is_ddp or rank == 0:     
                        #pass
                        print('Classification:', train_acc, test_acc, epoch)
                        
        losses = learning_shapelets.train(X_train, epochs=1, batch_size=batch_size, epoch_idx=epoch)
        #total_progress.set_description(f"loss: {np.mean(losses)}")
        total_progress.set_description(f"loss: {np.mean([loss[0] for loss in losses])},"
                                       f"loss_align: {np.mean([loss[2] for loss in losses])},"
                                       f"loss_sdl: {np.mean([loss[3] for loss in losses])}")
       
    
  

    return learning_shapelets, train_acc, test_acc
  
  
def main(rank, world_size):
    args = parser.parse_args()
    
    
    
    results = evaluate_UEA(args.dataset,
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
                            task=args.task)
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
  
  
