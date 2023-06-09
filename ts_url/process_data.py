from tsai.all import *
import numpy as np
import json
from torch.utils.data import Dataset
# import tfsnippet as spt

interfusion = ['omi-6', 'omi-9', 'omi-4', 'omi-7', 'machine-2-2', 'omi-10', 'omi-8', 'omi-11', 'machine-1-7', 
 'machine-2-8', 'omi-2', 'omi-3', 'machine-1-6', 'machine-3-3', 'machine-1-1', 'omi-12', 'machine-3-6', 
 'omi-1', 'machine-2-7', 'machine-2-1', 'omi-5', 'machine-3-4', 'machine-3-8', 'machine-3-11']

MTSC_datasets = get_UCR_multivariate_list()
UCR_multivariate_list = get_UCR_multivariate_list()

interfusion = set([dsid.lower() for dsid in interfusion])
UCR_dsid = set([dsid.lower() for dsid in MTSC_datasets + UCR_multivariate_list])

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
			warp = np.concatenate(np.random.permutation(splits)).ravel()
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

def collate_unsuperv(data, max_len=None, mask_compensation=False):
	"""Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
	Args:
		data: len(batch_size) list of tuples (X, mask).
			- X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
			- mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
		max_len: global fixed sequence length. Used for architectures requiring fixed length input,
			where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
	Returns:
		X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
		targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
		target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
			0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
		padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
	"""

	batch_size = len(data)
	features, masks, IDs = zip(*data)

	# Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
	lengths = [X.shape[0] for X in features]  # original sequence length for each time series
	if max_len is None:
		max_len = max(lengths)
	X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
	target_masks = torch.zeros_like(X,
									dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
	for i in range(batch_size):
		end = min(lengths[i], max_len)
		X[i, :end, :] = features[i][:end, :]
		target_masks[i, :end, :] = masks[i][:end, :]

	targets = X.clone()
	X = X * target_masks  # mask input
	if mask_compensation:
		X = compensate_masking(X, target_masks)

	padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
	target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
	return X, targets, target_masks, padding_masks, IDs

def take_per_row(A, indx, num_elem):
	all_indx = indx[:,None] + np.arange(num_elem)
	return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def collate_ts2vec(data, max_len=None, mask_compensation=None):
	batch_size = len(data)
	x, masks, IDs = zip(*data)
	x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	masks = ~masks
	ts_l = x[0].size(1)
	crop_l = np.random.randint(low=2, high=ts_l+1)
	crop_left = np.random.randint(ts_l - crop_l + 1)
	crop_right = crop_left + crop_l
	crop_eleft = np.random.randint(crop_left + 1)
	crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
	crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
	gt1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
	gt2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
	m1 = take_per_row(masks, crop_offset + crop_eleft, crop_right - crop_eleft)
	m2 = take_per_row(masks, crop_offset + crop_left, crop_eright - crop_left)
	return gt1, gt2, crop_l, m1, m2, x, masks, IDs

def collate_ts_tcc(data, jitter_scale_ratio, jitter_ratio, max_seg):
	batch_size = len(data)
	x, masks, IDs = zip(*data)
	x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	masks = ~masks
	aug1, aug2 = dataTransform(x, jitter_scale_ratio, jitter_ratio, max_seg)
	aug1, aug2 = torch.tensor(aug1), torch.tensor(aug2)
	return x, aug1, aug2, masks, IDs

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

def get_unsupervised_data(dsid, filepath, train_ratio, test_ratio, window=100, stride=1):
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
	elif dsid.lower() in UCR_dsid:
		X, y, splits = get_UCR_data(dsid, split_data=False, force_download=True)
		# print(splits,y)
		train_split, valid_split = get_splits(y[splits[0]], train_size=train_ratio, show_plot=False)
	else:
		print(set(MTSC_datasets + UCR_list))
		print(set(interfusion))
		raise NotImplementedError
	test_split = get_splits(y[splits[1]], valid_size=(1 - test_ratio), show_plot=False)[0]
	test_bias = len(splits[0])
	test_split = [i + test_bias for i in test_split]
	splits = (train_split, test_split)
	# check_data(X, y, splits_v2)
	return X, y, splits

def get_datas(data_configs, task):
	labels = task != "pretraining"
	X_ = None
	y_ = None
	split_ = None
	curr_len = 0
	tfms = [None, TSClassification()]
	batch_tfms = [TSStandardize(by_sample=True)]
	for data_config in data_configs:
		X, y, splits_v2 = get_unsupervised_data(**data_config)
		if isinstance(X_, NoneType):
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
	if y_.dtype != np.dtype('<U3') and y_.dtype != np.dtype('<U2'):
		tfms[1] = None
	if labels:
		dls = get_ts_dls(X_, y_, splits=split_, tfms=tfms, batch_tfms=batch_tfms)
	else:
		dls = get_ts_dls(X_, splits=split_, tfms=tfms, batch_tfms=batch_tfms)
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
	tfms = [None, TSStandardScaler, TSClassification()]
	batch_tfms = [TSStandardize(by_sample=True)]
	for data_config in data_configs:
		X, y, splits_v2 = get_unsupervised_data(**data_config)
		if isinstance(X_, NoneType):
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
	udls = get_ts_dls(X_, splits=split_, tfms=tfms, batch_tfms=batch_tfms)
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
				 mode='separate', distribution='geometric', exclude_feats=None, mask_row=True):
		super(ImputationDataset, self).__init__()

		self.data = data  # this is a subclass of the BaseData class in data.py
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

		X = self.data[self.IDs[ind]][0]  # (seq_length, feat_dim) array
		X = torch.tensor(X, dtype=X.dtype).transpose(0, 1)
		mask = noise_mask(X[:,0].unsqueeze(1).numpy() if self.mask_row else X.numpy(), self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
						  self.exclude_feats)  # (seq_length, feat_dim) boolean array

		return X, torch.from_numpy(mask), self.IDs[ind]

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
		self.feature = data

		self.labels = labels

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
		self.feature = data

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
