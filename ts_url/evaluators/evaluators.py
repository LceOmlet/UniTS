try:
    from ..registry import EVALUATORS, EVALUATE_STEP, PRETRAIN_EVALUATE_INFER, EVALUATE_AGG, PRETRAIN_INFER
except:
    from registry import EVALUATORS
import torch
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from sklearn.metrics import classification_report

from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.metrics import roc_auc_score as auc 
import json

def infer_cluster(encodding, targets):
    label_num = np.max(targets) + 1
    pca = PCA(n_components=10)
    reps = pca.fit_transform(encodding)
    kmeans = KMeans(label_num)
    pred = kmeans.fit_predict(reps)
    NMI_score = normalized_mutual_info_score(targets, pred)
    RI_score = rand_score(targets, pred)
    return {"NMI":NMI_score, "RI": RI_score}, pred #, "acc": self.classifier.score(features, y)}


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

def infer_imputation(reprs, targets, masks, ridge, loss_module):
    reprs = list2array(reprs)
    targets = list2array(targets)
    masks = list2array(masks)
    pred = ridge.predict(reprs.reshape((reprs.shape[0], -1)))
    pred = pred.reshape(targets.shape)
    return loss_module(torch.tensor(targets), torch.tensor(pred), torch.tensor(masks)).detach().cpu().numpy().mean()

@EVALUATE_STEP.register("imputation")
def evaluate_imputation(batch, model, device, val_loss_module, per_batch, **kwargs):
    X, targets, target_masks, padding_masks, IDs = batch
    targets = targets.to(device)
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    predictions = model(X.to(device), padding_masks) 
    loss = val_loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
    # print(loss.shape)
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    # print(loss.shape)
    if not loss.shape:
        loss = loss.unsqueeze(0)
    if not loss.shape[0]:
        loss = torch.tensor([0])
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch  
    per_batch['metrics'].append(loss.cpu().numpy())
    return batch_loss, mean_loss, len(loss)

@EVALUATE_STEP.register("pretraining")
def evaluate_pretraining(batch, model, device, model_name, val_loss_module, per_batch, **kwargs):
    X, targets, target_masks, padding_masks, IDs = batch
    targets = targets.to(device)
    target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_masks = padding_masks.to(device)  # 0s: ignore
    target_masks = target_masks * padding_masks.unsqueeze(-1)
    
    
    pretrain_evaluate_kwargs = {
        "targets": targets,
        "target_masks": target_masks,
        "val_loss_module": val_loss_module,
        "model": model,
        "X": X,
        "padding_masks": padding_masks,
        "device": device,
        "per_batch": per_batch
    }
    batch_loss, mean_loss, active_elements, predictions = PRETRAIN_EVALUATE_INFER.get(model_name)(**pretrain_evaluate_kwargs)
    per_batch['X'].append(X.cpu().numpy())
    per_batch['targets'].append(targets.cpu().numpy())
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(np.array(IDs))
    per_batch['target_masks'].append(target_masks.cpu().numpy())
    return batch_loss, mean_loss, active_elements

@PRETRAIN_EVALUATE_INFER.register("mvts_transformer")
def pretrain_evaluate_mvts_transformer(X, model,  device, targets, padding_masks, target_masks, val_loss_module, per_batch, **kwargs):
    predictions = model(X.to(device), padding_masks=padding_masks) 
    loss = val_loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
    # print(loss.shape)
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    # print(loss.shape)
    if not loss.shape:
        loss = loss.unsqueeze(0)
    if not loss.shape[0]:
        loss = torch.tensor([0])
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch   
    per_batch['metrics'].append(loss.cpu().numpy())
    active_elements = len(loss)
    return batch_loss, mean_loss, active_elements, predictions

@PRETRAIN_EVALUATE_INFER.register("ts2vec")
@PRETRAIN_EVALUATE_INFER.register("ts_tcc")
@PRETRAIN_EVALUATE_INFER.register("t_loss")
def pretrain_evaluate_ts2vec(X, model, device, padding_masks, **kwargs):
    predictions = model.encode(X.to(device), padding_mask=padding_masks) 
    mean_loss = float('inf')
    active_elements = 1
    batch_loss = 0
    return batch_loss, mean_loss, active_elements, predictions

@EVALUATE_STEP.register("classification")
def evaluate_classification(batch, model, device, val_loss_module, per_batch, **kwargs):
    X, targets, padding_masks, IDs = batch
    targets = targets.to(device)
    padding_masks = padding_masks.to(device)
    predictions = model(X.to(device), padding_masks)  
    loss = val_loss_module(predictions, targets)
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = torch.mean(loss).cpu().item()
    active_elements = len(loss)
    per_batch['X'].append(X.cpu().numpy())
    per_batch['targets'].append(targets.cpu().numpy())
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(np.array(IDs))
    return batch_loss, mean_loss, active_elements

@EVALUATE_STEP.register("clustering")
def evaluate_clustering(batch, model, device, per_batch, **kwargs):
    X, targets, padding_masks, IDs = batch
    targets = targets.to(device)
    padding_masks = padding_masks.to(device) 
    predictions = model.encode(X.to(device), padding_masks)
    mean_loss = float('inf')
    active_elements = 1
    batch_loss = 0
    per_batch['X'].append(X.cpu().numpy())
    per_batch['targets'].append(targets.cpu().numpy())
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(np.array(IDs))
    return batch_loss, mean_loss, active_elements 

@EVALUATE_STEP.register("anomaly_detection")
def evaluate_anomaly_detection(batch, model, device, val_loss_module, per_batch, **kwargs):
    X, targets, padding_masks, IDs = batch
    targets = targets.to(device)
    padding_masks = padding_masks.to(device)
    predictions = model(X.to(device), padding_masks) 
    loss = val_loss_module(predictions[:, -1, :], X[:, -1, :])
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = torch.mean(loss).cpu().item()
    per_batch["score"].append(loss.detach().cpu().numpy())
    per_batch['metrics'].append(loss.cpu().numpy())
    active_elements = len(loss)
    per_batch['X'].append(X.cpu().numpy())
    per_batch['targets'].append(targets.cpu().numpy())
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(np.array(IDs))
    return batch_loss, mean_loss, active_elements

@EVALUATE_STEP.register("regression")
def evaluate_regression(batch, model, device, val_loss_module, per_batch, **kwargs):
    X, targets, padding_masks, features, IDs = batch
    targets = targets.to(device)
    padding_masks = padding_masks.to(device) 
    predictions = model(X.to(device), padding_masks) 
    X = features
    loss = val_loss_module(predictions, targets)
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = torch.mean(loss).cpu().item()
    per_batch['metrics'].append(loss.cpu().numpy())
    active_elements = len(loss)
    per_batch['X'].append(X.cpu().numpy())
    per_batch['targets'].append(targets.cpu().numpy())
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(np.array(IDs))
    return batch_loss, mean_loss, active_elements

@EVALUATE_AGG.register("mvts_transformer")
@EVALUATE_AGG.register("imputation")
def eval_agg_imputation(**kwargs):
    pass

@EVALUATE_AGG.register("pretraining")
def eval_agg_pretraining(model_name, **kwargs):
    EVALUATE_AGG.get(model_name)(**kwargs)

@EVALUATE_AGG.register("ts2vec")
@EVALUATE_AGG.register("ts_tcc")
@EVALUATE_AGG.register("t_loss")
def eval_agg_ts2vec_ts_tcc_t_loss(ridge, per_batch, val_loss_module, logger, epoch_metrics, **kwargs):
    if ridge is not None:
        epoch_loss = infer_imputation(per_batch["predictions"], per_batch["targets"], 
                                    per_batch["target_masks"], ridge, val_loss_module)
    else:
        epoch_loss = 0
    logger.info("Infer ridge.")
    epoch_metrics["report"] = epoch_loss
    epoch_metrics['loss'] = float(epoch_loss)

@EVALUATE_AGG.register("classification")
def eval_agg_classfication(per_batch, epoch_metrics, **kwargs):
    report = classification_report(np.argmax(per_batch["predictions"], axis=-1), per_batch["targets"])
    epoch_metrics["report"] = report

@EVALUATE_AGG.register("clustering")
def eval_agg_clustering(per_batch, epoch_metrics, **kwargs):
    report, clustering_rst = infer_cluster(per_batch["predictions"], per_batch["targets"])
    epoch_loss = report["NMI"]
    report = json.dumps(report, indent="  ")
    epoch_metrics["report"] = report
    per_batch["clustering_rst"] = clustering_rst
    epoch_metrics['loss'] = float(epoch_loss)

@EVALUATE_AGG.register("regression")
def eval_agg_regression(epoch_metrics, epoch_loss, **kwargs):
    epoch_metrics["report"] = str(epoch_loss)

@EVALUATE_AGG.register("anomaly_detection")
def eval_agg_anomaly_detection(per_batch, epoch_metrics, **kwargs):
    auc_score = auc(per_batch["targets"], per_batch["score"])
    epoch_metrics['auc'] = auc_score
    epoch_metrics['report'] = auc_score

@EVALUATORS.register("all_eval")
def evaluate(model, valid_dataloader, task, device, val_loss_module, 
             model_name, print_interval, print_callback, logger, ridge=None,
               epoch_num=None, keep_all=True):
    model.eval()
    epoch_loss = 0  # total loss of epoch
    total_active_elements = 0  # total unmasked elements in epoch
    # handle = model.output_layer.register_forward_hook(get_rep)
    epoch_metrics = OrderedDict()
    if keep_all:
        per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': [], 'reps': [], "score": [], "X": []}
    for i, batch in enumerate(valid_dataloader):
        target_masks = None

        evaluate_step_kwargs = {
            "batch": batch,
            "model": model,
            "device": device,
            "model_name": model_name,
            "val_loss_module": val_loss_module,
            "per_batch": per_batch
        }
        batch_loss, mean_loss, active_elements = EVALUATE_STEP.get(task)(**evaluate_step_kwargs)
        epoch_loss += batch_loss
        total_active_elements += active_elements

        metrics = {"loss": mean_loss}
        if i % print_interval == 0:
            ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            print_callback(i, metrics, prefix='Evaluating ' + ending, total_batches=len(valid_dataloader))
    
    # print("**" * 50)
    # print(epoch_loss, total_active_elements)
    # print("*" * 100)
    epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
    
    for key in per_batch:
        # print(key)
        per_batch[key] = list2array(per_batch[key])
    per_batch_idxed = deepcopy(per_batch)

    epoch_metrics['epoch'] = epoch_num
    epoch_metrics['loss'] = float(epoch_loss)

    eval_aggr_kwargs = {
        "per_batch": per_batch,
        "epoch_metrics": epoch_metrics,
        "val_loss_module": val_loss_module,
        "logger": logger,
        "ridge": ridge,
        "model_name": model_name,
        "epoch_loss": epoch_loss
    } 

    EVALUATE_AGG.get(task)(**eval_aggr_kwargs)


    # handle.remove()

    if keep_all:
        return epoch_metrics, per_batch_idxed
    else:
        return epoch_metrics