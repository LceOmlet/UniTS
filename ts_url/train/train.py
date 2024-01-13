from ..registry.registry import TRAINERS, TRAIN_STEP, PRETRAIN_STEP, TRAIN_AGG
from collections import OrderedDict
import torch
from ..utils.loss import hierarchical_contrastive_loss
import numpy as np
from sklearn.linear_model import Ridge
import torch.nn.functional as F

def list2array(cvt_list):
    if isinstance(cvt_list, list):
        if len(cvt_list) == 0:
            return np.array([])
        if isinstance(cvt_list[0], torch.Tensor):
            cvt_list = torch.cat(cvt_list, dim=0)
        elif isinstance(cvt_list[0], np.ndarray):
            cvt_list = np.concatenate(cvt_list, axis=0)
        else:
            print(cvt_list[0])
            raise NotImplementedError
    if isinstance(cvt_list, torch.Tensor):
        cvt_list = cvt_list.detach().cpu().numpy()
    return cvt_list

def fit_imputation(reprs, targets, masks, valid_ratio, loss_module):
    reprs = list2array(reprs)
    targets = list2array(targets)
    masks = list2array(masks)
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_split = int(len(reprs) * valid_ratio)
    valid_repr, train_repr = reprs[:valid_split], reprs[valid_split:]
    valid_targets, train_targets = targets[: valid_split], targets[valid_split:]
    valid_masks, train_masks = masks[:valid_split], masks[valid_split :] 
    valid_results = []
    for alpha in alphas:
        target_shape = train_targets.shape[1:]
        lr = Ridge(alpha=alpha).fit(
            train_repr.reshape(train_repr.shape[0], -1), 
            train_targets.reshape(train_repr.shape[0], -1)
        )
        valid_pred = lr.predict(valid_repr.reshape((valid_repr.shape[0], -1)))
        valid_pred = valid_pred.reshape((valid_split, target_shape[0], target_shape[1]))
        score = loss_module(torch.tensor(valid_targets), torch.tensor(valid_pred), torch.tensor(valid_masks)).detach().cpu().numpy()
        score = np.mean(score)
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    lr = Ridge(alpha=best_alpha)
    lr.fit(reprs.reshape((reprs.shape[0], -1)), targets.reshape((reprs.shape[0], -1)))
    return lr

@TRAINERS.register("all_train")
def train_epoch(model, dataloader, task, device, loss_module, 
                optimizer, model_name, print_interval, print_callback, val_loss_module,
                logger, optim_config, t_loss_train=None, temporal_contr_optimizer=None, 
                epoch_num=None):
    if task == "clustering":
        raise NotImplementedError()
    epoch_metrics = OrderedDict()
    model.train()
    epoch_loss = 0  # total loss of epoch
    total_active_elements = 0  # total unmasked elements in epoch
    targets, masks, reprs = [], [], []
    for i, batch in enumerate(dataloader):
        train_step_config = {
            "batch": batch, 
            "model": model, 
            "device": device, 
            "optimizer": optimizer, 
            "targets": targets, 
            "masks": masks, 
            "reprs": reprs,
            "model_name": model_name,
            "t_loss_train": t_loss_train,
            "temporal_contr_optimizer": temporal_contr_optimizer,
            "loss_module": loss_module
        }
        loss = TRAIN_STEP.get(task)(**train_step_config)

        if len(loss.shape):
            loss = loss.reshape(loss.shape[0], -1)
            loss = torch.mean(loss, dim=-1)
        if not loss.shape:
            loss = loss.unsqueeze(0)
        batch_loss = torch.sum(loss)
        mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

        metrics = {"loss": mean_loss.item()}
        if i % print_interval == 0:
            ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            print_callback(i, metrics, prefix='Training ' + ending, total_batches=len(dataloader))

        with torch.no_grad():
            total_active_elements += len(loss)
            epoch_loss += batch_loss.item()  # add total loss of batch


    epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
    epoch_metrics['epoch'] = epoch_num
    epoch_metrics['loss'] = float(epoch_loss)
    
    train_agg_kwargs = {
        "reprs": reprs,
        "targets": targets,
        "masks": masks,
        "val_loss_module":val_loss_module,
        "logger": logger,
        "epoch_metrics": epoch_metrics
    }
    train_agg = TRAIN_AGG.get(model_name)
    if train_agg is not None: train_agg(**train_agg_kwargs)

    return epoch_metrics

@TRAIN_AGG.register("ts2vec")
@TRAIN_AGG.register("ts_tcc")
@TRAIN_AGG.register("t_loss")
def train_agg_ts2vec_ts_tcc_t_loss(reprs, targets, masks, val_loss_module, logger, epoch_metrics, **kwargs):
    ridge = fit_imputation(reprs, targets, masks, 0.1, val_loss_module)
    logger.info("Ridge Training.")
    epoch_metrics["ridge"] = ridge

@TRAIN_AGG.register("mvts_transformer")
def train_agg_mvts_transformer(**kwargs):
    pass

@TRAIN_STEP.register("classification")
def step_classification(batch, model, device, loss_module, optimizer, **kwargs):
    X, targets, padding_masks, IDs = batch
    targets = targets.to(device)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
    predictions = model(X.to(device), padding_masks)

    loss = loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("anomaly_detection")
def step_anomaly_detection(batch, model, device, loss_module, optimizer, **kwargs):
    X, targets, padding_masks, IDs = batch
    padding_masks = padding_masks.to(device)
    predictions = model(X.to(device), padding_masks)
    loss = loss_module(predictions, X)
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("regression")
def step_regression(batch, model, device, loss_module, optimizer, **kwargs):
    X, preds, padding_masks, features, IDs = batch
    predictions = model(X.to(device), padding_masks)
    # print(predictions.shape, preds.shape)
    loss = loss_module(predictions, preds)
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("imputation")
def step_imputation(batch, model, device, loss_module, optimizer, **kwargs):
    X, target, target_masks, padding_masks, IDs = batch
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    target = target.to(device)
    predictions = model(X.to(device), padding_masks)
    loss = loss_module(target, predictions, target_masks)
    batch_loss = torch.mean(loss)
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@TRAIN_STEP.register("pretraining")
def step_pretraining(model_name, **kwargs):
    return PRETRAIN_STEP.get(model_name)(**kwargs)

@PRETRAIN_STEP.register("mvts_transformer")
def step_mvts_transformer(batch, model, device, loss_module, optimizer,**kwargs):
    X, target, target_masks, padding_masks, IDs = batch
    # print(torch.mean(torch.abs(X)))
    target = target.to(device)
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    predictions = model(X.to(device), padding_masks)  # (batch_size, padded_length, feat_dim)
    # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
    target_masks = target_masks * padding_masks.unsqueeze(-1)
    loss = loss_module(predictions, target, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
    optimizer.zero_grad()
    loss = torch.mean(loss)
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    return loss

@PRETRAIN_STEP.register("ts2vec")
def step_ts2vec(batch, model, device, loss_module, optimizer, targets, masks, reprs, **kwargs):
    gt1, gt2, crop_l, m1, m2, x, mask, IDs = batch
    gt1 = gt1.to(device)
    gt2 = gt2.to(device)

    out1 = model._net(gt1, m1)[:, -crop_l:]
    out2 = model._net(gt2, m2)[:, :crop_l]
    loss = loss_module(
        out1,
        out2
    )
    targets.append(x)
    masks.append(mask)
    reprs.append(model._net(x, mask).detach())
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    model.net.update_parameters(model._net)
    return loss

@PRETRAIN_STEP.register("ts_tcc")
def step_ts_tcc(batch, model, device, loss_module, optimizer, targets, temporal_contr_optimizer, reprs, masks, **kwargs):
    x, aug1, aug2, mask, IDs = batch
    data = x.float().to(device)
    aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

    # optimizer
    optimizer.zero_grad()
    temporal_contr_optimizer.zero_grad()
    predictions1, features1 = model.model(aug1)
    predictions2, features2 = model.model(aug2)

    # normalize projection feature vectors
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)

    temp_cont_loss1, temp_cont_lstm_feat1 = model.tenporal_contr_model(features1, features2)
    temp_cont_loss2, temp_cont_lstm_feat2 = model.tenporal_contr_model(features2, features1)

    # normalize projection feature vectors
    zis = temp_cont_lstm_feat1 
    zjs = temp_cont_lstm_feat2 

    # compute loss
    lambda1 = 1
    lambda2 = 0.7
    nt_xent_criterion = loss_module
    loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
    # if l2_reg:
    #     loss = mean_loss + l2_reg * l2_reg_loss(model)
    # else:
    #     loss = mean_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    temporal_contr_optimizer.step()
    target = x.detach().clone(); targets.append(target)
    x[mask] = 0; _, reprs_ = model.model(x); reprs.append(reprs_)
    masks.append(mask)
    return loss

@PRETRAIN_STEP.register("t_loss")
def step_t_loss(batch, model, device, loss_module, optimizer, targets, t_loss_train, reprs, masks, **kwargs):
    X, target, target_masks, padding_masks, IDs = batch
    X = X.to(device)
    loss = loss_module(target.permute(0, 2, 1), model, 
                            t_loss_train, save_memory=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
    optimizer.step()
    target = target.detach().clone(); targets.append(target)
    X[target_masks | (~padding_masks.unsqueeze(-1))] = 0; reprs_ = model(X.permute(0, 2, 1)); reprs.append(reprs_)
    masks.append(target_masks)
    return loss

