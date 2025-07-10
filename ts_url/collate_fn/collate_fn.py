

import numpy as np
import torch
from ..registry import COLLATE_FN
from ..process_data import take_per_row,compensate_masking, dataTransform, padding_mask
import tsaug
from collections import defaultdict


@COLLATE_FN.register("time_vae")
def collate_time_vae(data, **kwargs):
    batch_size = len(data)
    x, masks, label, IDs = zip(*data)
    x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
    return {
		"X": x,
		"label": label,
		"IDs": IDs
	}

@COLLATE_FN.register("ts2vec")
def collate_ts2vec(data, seq_len=None, mask_compensation=None):
	batch_size = len(data)
	x, masks, label, IDs = zip(*data)
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
	x = x.permute(0, 2, 1)
	masks = masks.permute(0, 2, 1)
	gt1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
	gt2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
	m1 = take_per_row(masks, crop_offset + crop_eleft, crop_right - crop_eleft)
	m2 = take_per_row(masks, crop_offset + crop_left, crop_eright - crop_left)
	gt1, gt2, m1, m2 = gt1.permute(0, 2, 1), gt2.permute(0, 2, 1), m1.permute(0, 2, 1), m2.permute(0, 2, 1)
	x = x.permute(0, 2, 1)
	masks = masks.permute(0, 2, 1)
	batch = {
		"gt1": gt1,
		"gt2": gt2,
		"crop_l": crop_l,
		"m1": m1,
		"m2":m2,
		"X": x,
		"mask": masks,
		"label": label,
		"IDs": IDs
	}
	return batch

@COLLATE_FN.register("ts_tcc")
def collate_ts_tcc(data, jitter_scale_ratio, jitter_ratio, max_seg):
	batch_size = len(data)
	x, masks, label, IDs = zip(*data)
	x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	masks = ~masks
	x = x
	aug1, aug2 = dataTransform(x, jitter_scale_ratio, jitter_ratio, max_seg)
	aug1, aug2 = torch.tensor(aug1), torch.tensor(aug2)
	aug1 = aug1
	aug2 = aug2
	x = x
	batch = {
		"X": x,
		"aug1": aug1,
		"aug2": aug2,
		"mask": masks,
		"label": label,
		"IDs": IDs
	}
	return batch

@COLLATE_FN.register("unsupervise")
def collate_unsupervise(data):
	X, masks, label, IDs = zip(*data)
	X = torch.cat([x_.unsqueeze(0) for x_ in X], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	return {
		"X": X,
		"mask": masks,
		"label": label,
		"ID": IDs
	}

@COLLATE_FN.register("tnc")
def collate_tnc(batch):
	"""
    Collate function for TNC.
    Each item in batch: (x_t, x_p, x_n, label)

    """
	x_t, x_p, x_n, label, x = zip(*batch)

	x = torch.stack(x)
	x_t = torch.stack(x_t)  # (B, F, L)
	x_p = torch.stack(x_p)
	x_n = torch.stack(x_n)
	label = torch.stack(label)  # (B, mc)

	return {
		"X": x,
		"x_t": x_t,  # (B, F, L)
		"x_p": x_p,  
		"x_n": x_n,  
		"label": label  # (B, mc)
	}


@COLLATE_FN.register("mvts_transformer")
@COLLATE_FN.register("t_loss")
def collate_mvts_transformer(data, seq_len=None, mask_compensation=False):
	"""Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
	Args:
		data: len(batch_size) list of tuples (X, mask).
			- X: torch tensor of shape (feat_dim, seq_length); variable seq_length.
			- mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
		seq_len: global fixed sequence length. Used for architectures requiring fixed length input,
			where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
	Returns:
		X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
		targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
		target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
			0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
		padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
	"""

	batch_size = len(data)
	features, masks, label, IDs = zip(*data)

	# Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
	lengths = [X.shape[0] for X in features]  # original sequence length for each time series
	if seq_len is None:
		seq_len = max(lengths)
	X = torch.zeros(batch_size, seq_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
	target_masks = torch.zeros_like(X,
									dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
	for i in range(batch_size):
		end = min(lengths[i], seq_len)
		X[i, :end, :] = features[i][:end, :]
		target_masks[i, :end, :] = masks[i][:end, :]

	targets = X.clone()
	if mask_compensation:
		X = compensate_masking(X, target_masks)
	padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), seq_len=seq_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
	target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
	batch = {
		"X": X,
		"target": targets,
		"mask": target_masks,
		"padding_mask": padding_masks,
		"label": label,
		"IDs": IDs
	}
	return batch

@COLLATE_FN.register("csl")
def collate_csl(data):
	np.random.seed(0)
	augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 16 - 1))',
						'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 16 - 1))',
						'Pool(seed=np.random.randint(2 ** 16 - 1))',
						'Quantize(seed=np.random.randint(2 ** 16 - 1))',
						'TimeWarp(seed=np.random.randint(2 ** 16 - 1))'
						]
	aug1 = np.random.choice(augmentation_list, 1, replace=False)
	batch_size = len(data)
	x, masks, label, IDs = zip(*data)
	# print(x[0].shape)
	# print(x[0])
	# raise RuntimeError()
	x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	masks = ~masks
	x = x 
	ts_l = x.size(2)
             
	x_q = x.transpose(1,2).cpu().numpy()
	for aug in aug1:
		x_q = eval('tsaug.' + aug + '.augment(x_q)')
	x_q = torch.from_numpy(x_q).float()
	x_q = x_q.transpose(1,2)
	
	aug2 = np.random.choice(augmentation_list, 1, replace=False)
	while (aug2 == aug1).all():
		aug2 = np.random.choice(augmentation_list, 1, replace=False)
	
	x_k = x.transpose(1,2).cpu().numpy()
	for aug in aug2:
		x_k = eval('tsaug.' + aug + '.augment(x_k)')
	x_k = torch.from_numpy(x_k).float()
	x_k = x_k.transpose(1,2)
	# print(x[0])
	# raise RuntimeError()
	batch = {
		"X": x,
		"x_k": x_k,
		"x_q": x_q,
		"mask": masks,
		"label": label,
		"IDs": IDs
	}
	return batch

@COLLATE_FN.register("mmfa_rec")
@COLLATE_FN.register("mmfa")
def collate_mmfa(data):
	# print(data[0]["X"].shape)
	# print(data[0]["X"])
	# raise RuntimeError()
	features = defaultdict(list)
	rst = defaultdict(list)
	
	keys = list(data[0].keys())
	keys.remove("features")
	ts = list(data[0]["features"].keys())

	for d in data:
		for k in keys:
			rst[k].append(torch.tensor(d[k]).unsqueeze(0))
			# print(k, rst[k][0].shape)
		for t in ts:
			features[t].append(torch.tensor(d["features"][t]).unsqueeze(0))
	# 		print(t, features[t][0].shape)
	# raise RuntimeError()
	for k in keys:
		if k != "label" and k != "ID":
			# print(k)
			rst[k] = torch.cat(rst[k])
	for t in ts:
		# print(t)
		features[t] = torch.cat(features[t])
		# print(features[t].shape)
		

	# print(rst["X"].shape)
	# raise RuntimeError()

	rst["features"] = features
	# print(rst["X"[0]])
	# raise RuntimeError()

	return rst