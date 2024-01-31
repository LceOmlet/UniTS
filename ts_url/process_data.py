# from tsai.all import *
import numpy as np
import json
from torch.utils.data import Dataset
import os
import pickle
import torch
import random
import tsaug
from itertools import permutations
from random import shuffle
# import tfsnippet as spt

interfusion = ['omi-6', 'omi-9', 'omi-4', 'omi-7', 'machine-2-2', 'omi-10', 'omi-8', 'omi-11', 'machine-1-7', 
 'machine-2-8', 'omi-2', 'omi-3', 'machine-1-6', 'machine-3-3', 'machine-1-1', 'omi-12', 'machine-3-6', 
 'omi-1', 'machine-2-7', 'machine-2-1', 'omi-5', 'machine-3-4', 'machine-3-8', 'machine-3-11']

# MTSC_datasets = get_UCR_multivariate_list()
# UCR_multivariate_list = get_UCR_multivariate_list()

interfusion = set([dsid.lower() for dsid in interfusion])
# UCR_dsid = set([dsid.lower() for dsid in MTSC_datasets + UCR_multivariate_list])

class Dls:
	def __init__(self, X, splits, y=None) -> None:
		train_splits = splits[0]
		test_splits = splits[1]
		# print(X.shape)
		# exit()
		if y is not None:
			self.train_ds = [(X[idx], y[idx]) for idx in train_splits]
			self.valid_ds = [(X[idx], y[idx]) for idx in test_splits]
		else:
			self.train_ds = [(X[idx], ) for idx in train_splits]
			self.valid_ds = [(X[idx], ) for idx in test_splits]
		

def dataTransform(sample, jitter_scale_ratio, jitter_ratio, max_seg):
	weak_aug = scaling(sample, jitter_scale_ratio)
	strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)
	return weak_aug, strong_aug

def permutation(x, max_segments=5, seg_mode="random"):
	orig_steps = np.arange(x.shape[2])

	num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

	ret = np.zeros_like(x)
	for i, pat in enumerate(x):
		if num_segs[i] > 1:
			if seg_mode == "random":
				split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
				split_points.sort()
				splits = np.split(orig_steps, split_points)
			else:
				splits = np.array_split(orig_steps, num_segs[i])
			shuffle(splits)
			warp = np.concatenate(splits).ravel()
			ret[i] = pat[0,warp]
		else:
			ret[i] = pat
	return torch.from_numpy(ret)

def jitter(x, sigma=0.8):
	# https://arxiv.org/pdf/1706.00527.pdf
	return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
	# https://arxiv.org/pdf/1706.00527.pdf
	factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
	ai = []
	for i in range(x.shape[1]):
		xi = x[:, i, :]
		ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
	return np.concatenate((ai), axis=1)

def normalize(memmap, norm_type):
	"""
	Args:
		memmap: input dataframe
	Returns:
		dmemmapf: normalized dataframe
	"""
	if norm_type == "standardization":
		mean = memmap.mean()
		std = memmap.std()
		return (memmap - mean) / (std + np.finfo(float).eps)

	elif norm_type == "minmax":
		max_val = np.max(memmap)
		min_val = np.min(memmap)
		return (memmap - min_val) / (max_val - min_val + np.finfo(float).eps)

	elif norm_type == "per_sample_std":
		return (memmap - np.mean(memmap, axis=0)) / np.std(memmap, axis=0)

	elif norm_type == "per_sample_minmax":
		min_vals = np.min(memmap, axis=0)
		max_vals = np.max(memmap, axis=0)
		return (memmap - min_vals) / (max_vals- min_vals + np.finfo(float).eps)

	else:
		raise (NameError(f'Normalize method "{norm_type}" not implemented'))

def padding_mask(lengths, max_len=None):
	"""
	Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
	where 1 means keep element at this position (time step)
	"""
	batch_size = lengths.numel()
	max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
	return (torch.arange(0, max_len, device=lengths.device)
			.type_as(lengths)
			.repeat(batch_size, 1)
			.lt(lengths.unsqueeze(1)))

def compensate_masking(X, mask):
	"""
	Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
	If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
	Args:
		X: (batch_size, seq_length, feat_dim) torch tensor
		mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
	Returns:
		(batch_size, seq_length, feat_dim) compensated features
	"""

	# number of unmasked elements of feature vector for each time step
	num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
	# to avoid division by 0, set the minimum to 1
	num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
	return X.shape[-1] * X / num_active


def take_per_row(A, indx, num_elem):
	all_indx = indx[:,None] + np.arange(num_elem)
	return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


def collate_superv(data, max_len=None):
	"""Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
	Args:
		data: len(batch_size) list of tuples (X, y).
			- X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
			- y: torch tensor of shape (num_labels,) : class indices or numerical targets
				(for classification or regression, respectively). num_labels > 1 for multi-task models
		max_len: global fixed sequence length. Used for architectures requiring fixed length input,
			where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
	Returns:
		X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
		targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
		target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
			0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
		padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
	"""

	batch_size = len(data)
	features, labels, IDs = zip(*data)

	# Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
	lengths = [X.shape[0] for X in features]  # original sequence length for each time series
	if max_len is None:
		max_len = max(lengths)
	X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
	for i in range(batch_size):
		end = min(lengths[i], max_len)
		X[i, :end, :] = features[i][:end, :]

	targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

	padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
								 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

	return X, targets, padding_masks, IDs

def time_generator(timestamp):
	mins = 60
	hours = 24
	days = 7
	timestamp %= (mins * hours * days)
	res = np.zeros([mins + hours + days])
	res[int(timestamp / hours / mins)] = 1  # day
	res[days + int((timestamp % (mins * hours)) / mins)] = 1  # hours
	res[days + hours + int(timestamp % mins)] = 1  # min
	return res

def collate_superv_regression(data, max_len=None, pred_len=1):

	batch_size = len(data)
	features, IDs = zip(*data)

	features = torch.cat([feature.unsqueeze(0) for feature in features], dim=0)
	lengths = [X.shape[0] for X in features]  # original sequence length for each time series
	features = features[:,: features.shape[1] - pred_len,:]
	preds = features[:,-pred_len:, :]

	# Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
	if max_len is None:
		max_len = max(lengths)
	X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
	for i in range(batch_size):
		end = min(features.shape[1], max_len)
		X[i, :end, :] = features[i][:end, :]

	padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
								 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

	return X, preds, padding_masks, features, IDs


def split_time_series(data, y, window, stride):
	n, d = data.shape
	segments = []
	y_segments = []
	for i in range(0, n - window + 1, stride):
		segment = data[i:i+window]
		y_segment = y[i: i + window]
		segments.append(segment)
		y_segments.append(y_segment)
	return np.array(segments), np.array(y_segments)

def get_elements_by_ratio(lst, ratio):
	"""
	从一个有序的列表中随机选出比例为 ratio 的元素
	:param lst: 有序的列表
	:param ratio: 比例
	:return: 选出的元素组成的列表，有序
	"""
	if ratio <= 0 or ratio > 1:
		raise ValueError("ratio 必须大于 0 且小于 1")
	n = len(lst)
	k = int(n * ratio)
	step = n // k
	start = random.randint(0, step - 1)
	result = []
	for i in range(k):
		result.append(lst[start + i * step])
	return result

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
	
def get_data_and_label_from_ts_file(file_path,label_dict):
    # print(file_path)
    # exit()
    with open(file_path, encoding='utf-8') as file:
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

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                # print(line)
                # exit()
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict   

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a 

def get_unsupervised_data(dsid, filepath="", train_ratio=1, test_ratio=1, window=100, stride=1):
	# 100% train data
	if dsid.lower() in interfusion:
		(x_train, _), (x_test, y_test) = get_interfusion_data(dsid)
		data_length = x_train.shape[0]
		y_train = np.zeros((data_length,))
		train_np, train_y_np = split_time_series(x_train, y_train, window=window,stride=stride)
		test_np, test_y_np = split_time_series(x_test, y_test, window=window, stride=stride)
		splits = (list(range(len(train_np))), list(range(len(train_np), len(train_np) + len(test_np))))
		X = np.transpose(np.concatenate([train_np, test_np], axis=0), (0, 2, 1))
		y = np.concatenate([train_y_np, test_y_np], axis=0)[:, -1]
		train_split = get_elements_by_ratio(splits[0], train_ratio)
	else:
		if filepath == "":
			filepath = "data/UCR"
			Train_dataset_path = filepath + '/' + dsid + '/' + dsid + '_TRAIN.ts'
			Test_dataset_path = filepath + '/' + dsid + '/' + dsid + '_TEST.ts'
		label_dict = get_label_dict(Train_dataset_path)
		X, y = get_data_and_label_from_ts_file(Train_dataset_path, label_dict)
		X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path, label_dict)
		train_size = X.shape[0]
		test_size = X_test.shape[0]
		splits = [list(range(train_size)), list(range(train_size, train_size + test_size))]
		X = np.concatenate([X, X_test], axis=0)
		X = set_nan_to_zero(X)
		y = np.concatenate([y, y_test], axis=0)
	return X, y, splits

def get_datas(data_configs, **kwargs):
	# labels = task != "pretraining" or task != "imputation"
	X_ = None
	y_ = None
	split_ = None
	curr_len = 0
	# tfms = [None, TSClassification()]
	# batch_tfms = [TSStandardize(by_sample=True)]
	for data_config in data_configs:
		# print(data_config)
		X, y, splits_v2 = get_unsupervised_data(**data_config)
		if isinstance(X_, type(None)):
			X_ = X
			y_ = y
			split_ = splits_v2
		else:
			curr_len = X_.shape[0]
			X_ = np.concatenate([X_, X], axis=0)
			y_ = np.concatenate([y_, y], axis=0)
			splits_v2 = ([i + curr_len for i in splits_v2[0]],
						 [i + curr_len for i in splits_v2[1]])
			split_ = (split_[0] + splits_v2[0], split_[1] + splits_v2[1])
			# print(len(split_[0]) + len(split_[1]))
			# print(max(split_[0] + split_[1]), X_.shape[0])
	X_ = normalize(X_, "per_sample_std")
	# if y_.dtype != np.dtype('<U3') and y_.dtype != np.dtype('<U2'):
	# 	tfms[1] = None
	if y_ is not None:
		dls = Dls(X_, y=y_, splits=split_)
	else:
		dls = Dls(X_, splits=split_)
	y_ = y_.reshape(-1)
	return dls, {
		"input_feat_dim": X_.shape[1], 
		"seq_len": X_.shape[2], 
		"sample_num":X_.shape[0],
		"label_num": len(set(y_))
	}


def get_unsupervised_datas(data_configs):
	X_ = None
	y_ = None
	split_ = None
	curr_len = 0
	# tfms = [None, TSStandardScaler, TSClassification()]
	# batch_tfms = [TSStandardize(by_sample=True)]
	for data_config in data_configs:
		X, y, splits_v2 = get_unsupervised_data(**data_config)
		if isinstance(X_, type(None)):
			X_ = X
			y_ = y
			split_ = splits_v2
		else:
			curr_len = X_.shape[0]
			X_ = np.concatenate([X_, X], axis=0)
			y_ = np.concatenate([y_, y], axis=0)
			splits_v2 = ([i + curr_len for i in splits_v2[0]],
						 [i + curr_len for i in splits_v2[1]])
			split_ = (split_[0] + splits_v2[0], split_[1] + splits_v2[1])
			# print(len(split_[0]) + len(split_[1]))
			# print(max(split_[0] + split_[1]), X_.shape[0])
	X_ = normalize(X_, "standardization")
	udls = Dls(X_, splits=split_)
	return udls, {
		"input_feat_dim": X_.shape[1], 
		"seq_len": X_.shape[2], 
		"sample_num":X_.shape[0]
	}


def geom_noise_mask_single(L, lm, masking_ratio):
	"""
	Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
	proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
	Args:
		L: length of mask and sequence to be masked
		lm: average length of masking subsequences (streaks of 0s)
		masking_ratio: proportion of L to be masked

	Returns:
		(L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
	"""
	keep_mask = np.ones(L, dtype=bool)
	p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
	p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
	p = [p_m, p_u]

	# Start in state 0 with masking_ratio probability
	state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
	for i in range(L):
		keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
		if np.random.rand() < p[state]:
			state = 1 - state

	return keep_mask

def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
	"""
	Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
	Args:
		X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
		masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
			feat_dim that will be masked on average
		lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
		mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
			should be masked concurrently ('concurrent')
		distribution: whether each mask sequence element is sampled independently at random, or whether
			sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
			masked squences of a desired mean length `lm`
		exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

	Returns:
		boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
	"""
	if exclude_feats is not None:
		exclude_feats = set(exclude_feats)

	if distribution == 'geometric':  # stateful (Markov chain)
		if mode == 'separate':  # each variable (feature) is independent
			mask = np.ones(X.shape, dtype=bool)
			for m in range(X.shape[1]):  # feature dimension
				if exclude_feats is None or m not in exclude_feats:
					mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
		else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
			mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
	else:  # each position is independent Bernoulli with p = 1 - masking_ratio
		if mode == 'separate':
			mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
									p=(1 - masking_ratio, masking_ratio))
		else:
			mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
											p=(1 - masking_ratio, masking_ratio)), X.shape[1])

	return mask



class ImputationDataset(Dataset):
	"""Dynamically computes missingness (noise) mask for each sample"""

	def __init__(self, data, mean_mask_length=3, masking_ratio=0.15, 
				 mode='separate', distribution='geometric', label=None, exclude_feats=None, mask_row=True):
		super(ImputationDataset, self).__init__()

		
		if label is not None:
			self.label = label
		else:
			self.label = None

		self.data = torch.tensor(data)  # this is a subclass of the BaseData class in data.py
		self.data = torch.squeeze(self.data)
		self.IDs = list(range(len(data)))  # list of data IDs, but also mapping between integer index and ID

		self.masking_ratio = masking_ratio
		self.mean_mask_length = mean_mask_length
		self.mode = mode
		self.distribution = distribution
		self.exclude_feats = exclude_feats
		self.mask_row = mask_row

	def __getitem__(self, ind):
		"""
		For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
		Args:
			ind: integer index of sample in dataset
		Returns:
			X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
			mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
			ID: ID of sample
		"""

		X = self.data[self.IDs[ind]]  # (seq_length, feat_dim) array
		if self.label is not None:
			label= self.label[self.IDs[ind]]
		else:
			label = None
		X = torch.tensor(X).transpose(0, 1)
		mask = noise_mask(X[:,0].unsqueeze(1).numpy() if self.mask_row else X.numpy(), self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
						  self.exclude_feats)  # (seq_length, feat_dim) boolean array

		return X, torch.from_numpy(mask), label, self.IDs[ind]

	def update(self):
		self.mean_mask_length = min(20, self.mean_mask_length + 1)
		self.masking_ratio = min(1, self.masking_ratio + 0.05)

	def __len__(self):
		return len(self.IDs)
	

class ClassiregressionDataset(Dataset):
	def __init__(self, data, labels):
		super(ClassiregressionDataset, self).__init__()

		self.data = data  # this is a subclass of the BaseData class in data.py
		self.IDs = list(range(len(labels)))  # list of data IDs, but also mapping between integer index and ID
		self.feature = torch.tensor(data)

		self.labels = torch.tensor(labels, dtype=torch.long)

	def __getitem__(self, ind):
		"""
		For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
		Args:
			ind: integer index of sample in dataset
		Returns:
			X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
			y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
			ID: ID of sample
		"""

		X = self.feature[self.IDs[ind]].transpose(0, 1)  # (seq_length, feat_dim) array
		y = self.labels[self.IDs[ind]]  # (num_labels,) array

		return X, y, self.IDs[ind]

	def __len__(self):
		return len(self.IDs)
	
class RegressionDataset(Dataset):
	def __init__(self, data):
		super(RegressionDataset, self).__init__()
		self.data = data 
		self.IDs = list(range(len(data)))
		self.feature = torch.tensor(data)

	def __getitem__(self, ind):
		X = self.feature[self.IDs[ind]].transpose(0, 1) 
		return X, self.IDs[ind]

	def __len__(self):
		return len(self.IDs)


def get_data_dim(dataset):
	if dataset == 'SWaT':
		return 51
	elif dataset == 'WADI':
		return 118
	elif str(dataset).startswith('machine'):
		return 38
	elif str(dataset).startswith('omi'):
		return 19
	else:
		raise ValueError('unknown dataset '+str(dataset))


def get_interfusion_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
			 test_start=0, valid_portion=0.3, prefix="./data/InTerFusion/processed"):
	"""
	get data from pkl files
	return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
	"""
	if max_train_size is None:
		train_end = None
	else:
		train_end = train_start + max_train_size
	if max_test_size is None:
		test_end = None
	else:
		test_end = test_start + max_test_size
	print('load data of:', dataset)
	print("train: ", train_start, train_end)
	print("test: ", test_start, test_end)
	x_dim = get_data_dim(dataset)
	f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
	train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
	f.close()
	try:
		f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
		test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
		f.close()
	except (KeyError, FileNotFoundError):
		test_data = None
	try:
		f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
		test_label = pickle.load(f).reshape((-1))[test_start:test_end]
		f.close()
	except (KeyError, FileNotFoundError):
		test_label = None
	# if do_preprocess:
	#	 # train_data = preprocess(train_data)
	#	 # test_data = preprocess(test_data)
	#	 train_data, test_data = preprocess(train_data, test_data, valid_portion=valid_portion)
	print("train set shape: ", train_data.shape)
	print("test set shape: ", test_data.shape)
	print("test set label shape: ", test_label.shape)
	return (train_data, None), (test_data, test_label)
