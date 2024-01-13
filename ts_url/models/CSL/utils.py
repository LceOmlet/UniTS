import numpy as np
import torch
from torch import nn, optim

import random

from sklearn.metrics import accuracy_score
from tslearn.clustering import TimeSeriesKMeans

def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments


def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters


class MaskBlock(nn.Module):
    def __init__(self, p=0.1):
        super(MaskBlock, self).__init__()
        
        self.net = nn.Dropout(p=p)
    def forward(self, X):
        return self.net(X)



class LinearBlock(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(LinearBlock, self).__init__()
        
        #self.linear = nn.Sequential(nn.Linear(in_channel, 256), nn.ReLU(), nn.Linear(256, n_classes))
        self.linear = nn.Linear(in_channel, n_classes)
    
    def forward(self, X):
        return self.linear(X)

class LinearClassifier():
    def __init__(self, in_channel, n_classes, batch_size=256, lr=1e-3, wd=1e-4, max_epoch=200):
        super(LinearClassifier, self).__init__()
        
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.lr = lr
        
        self.wd = wd
        self.max_epoch = max_epoch
        
        self.net = LinearBlock(in_channel, n_classes)
        
    
    def train(self, X, y):
        X = torch.from_numpy(X)
        X = X.float()
        
        y = torch.from_numpy(y)
        y = y.long()
        
        self.net.cuda()
        
        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200, min_lr=0.0001)
        
        # build dataloader
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=max(int(min(X.shape[0], self.batch_size)), 4), shuffle=True)
        
        
        
        
        self.net.train()
        
        for epoch in range(self.max_epoch):
            losses = []
            for (x, y) in train_loader:
                x = x.cuda()
                y = y.cuda()
                logits = self.net(x)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step(loss)
                
    
    def predict(self, X):
        X = torch.from_numpy(X)
        X = X.float()
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max(int(min(X.shape[0], self.batch_size)), 4), shuffle=False)
        
        predict_list = np.array([])
        
        self.net.eval()
        
        for (x, ) in loader:
            x = x.cuda()
            y_predict = self.net(x)
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
        
        return predict_list



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def z_normalize(a, eps=1e-7):
    return (a - np.mean(a, axis=-1, keepdims=True)) / (eps + np.std(a, axis=-1, keepdims=True))


#def replace_nan_with_row_mean(a):
#    out = np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(axis=1)[:, np.newaxis], a)
#    return np.float32(out)

def replace_nan_with_near_value(a):
    mask = np.isnan(a)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = a[np.arange(idx.shape[0])[:,None], idx]
    return np.float32(out)

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict


def get_data_and_label_from_ts_file(file_path,label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y




def TSC_multivariate_data_loader(dataset_path, dataset_name):
    
    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path,label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path,label_dict)
    
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

def generate_binomial_mask(size, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=size)).cuda()


def eval_accuracy(model, X, Y, X_test, Y_test, normalize=False, lr=1e-3, wd=1e-4):
    transformation = model.transform(X, result_type='numpy', normalize=normalize)
    clf = LinearClassifier(transformation.shape[1], len(set(Y)), lr=lr, wd=wd)
    clf.train(transformation, Y)
    acc_train = accuracy_score(clf.predict(transformation), Y)
    acc_test = accuracy_score(clf.predict(model.transform(X_test, result_type='numpy', normalize=normalize)), Y_test)
    return acc_train, acc_test