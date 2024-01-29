

import numpy as np
import torch
from ..registry import COLLATE_FN
from ..process_data import take_per_row,compensate_masking, dataTransform, padding_mask
import tsaug

@COLLATE_FN.register("csl")
def collate_csl(data):
	augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 32 - 1))',
						'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 32 - 1))',
						'Pool(seed=np.random.randint(2 ** 32 - 1))',
						'Quantize(seed=np.random.randint(2 ** 32 - 1))',
						'TimeWarp(seed=np.random.randint(2 ** 32 - 1))'
						]
	aug1 = np.random.choice(augmentation_list, 1, replace=False)
	batch_size = len(data)
	x, masks, label, IDs = zip(*data)
	x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	masks = ~masks
	x = x .permute(0, 2 ,1)
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
	# masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	# masks = ~masks
	# aug1, aug2 = dataTransform(x, jitter_scale_ratio, jitter_ratio, max_seg)
	# aug1, aug2 = torch.tensor(aug1), torch.tensor(aug2)
	return x.permute(0, 2, 1), x_k.permute(0, 2, 1), x_q.permute(0, 2, 1), masks, label, IDs

@COLLATE_FN.register("ts2vec")
def collate_ts2vec(data, max_len=None, mask_compensation=None):
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
	gt1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
	gt2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
	m1 = take_per_row(masks, crop_offset + crop_eleft, crop_right - crop_eleft)
	m2 = take_per_row(masks, crop_offset + crop_left, crop_eright - crop_left)
	return gt1, gt2, crop_l, m1, m2, x, masks, label, IDs

@COLLATE_FN.register("ts_tcc")
def collate_ts_tcc(data, jitter_scale_ratio, jitter_ratio, max_seg):
	batch_size = len(data)
	x, masks, label, IDs = zip(*data)
	x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=0)
	masks = torch.cat([x_.unsqueeze(0) for x_ in masks], dim=0)
	masks = ~masks
	x = x.permute(0, 2, 1)
	aug1, aug2 = dataTransform(x, jitter_scale_ratio, jitter_ratio, max_seg)
	aug1, aug2 = torch.tensor(aug1), torch.tensor(aug2)
	aug1 = aug1.permute(0, 2, 1)
	aug2 = aug2.permute(0, 2, 1)
	x = x.permute(0, 2, 1)
	return x, aug1, aug2, masks, label, IDs

@COLLATE_FN.register("mvts_transformer")
@COLLATE_FN.register("t_loss")
@COLLATE_FN.register("unsupervise")
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
	features, masks, label, IDs = zip(*data)

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
	if mask_compensation:
		X = compensate_masking(X, target_masks)
	padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
	target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
	return X, targets, target_masks, padding_masks, label, IDs