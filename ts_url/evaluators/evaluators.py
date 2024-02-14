try:
    from ..registry import EVALUATE, EVALUATE_STEP, PRETRAIN_EVALUATE_STEP, MAKE_EVAL_REPORT, EVAL_LOOP_INIT
except:
    from registry import EVALUATE
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
from ..utils.utils import list2array

def infer_cluster(encodding, target):
    label_num = np.max(target) + 1
    pca = PCA(n_components=10)
    reps = pca.fit_transform(encodding)
    kmeans = KMeans(label_num)
    pred = kmeans.fit_predict(reps)
    NMI_score = normalized_mutual_info_score(target, pred)
    RI_score = rand_score(target, pred)
    return {"NMI":NMI_score, "RI": RI_score}, pred #, "acc": self.classifier.score(features, y)}

def infer_imputation(reprs, target, masks, ridge, loss_module):
    reprs = list2array(reprs)
    target = list2array(target)
    masks = list2array(masks)
    pred = ridge.predict(reprs.reshape((reprs.shape[0], -1)))
    pred = pred.reshape(target.shape)
    return loss_module(torch.tensor(target), torch.tensor(pred), torch.tensor(masks)).detach().cpu().numpy().mean()

@EVAL_LOOP_INIT.register("pretraining")
def pretraining_eval_loop_init(model, dataloader, evaluator, device, **kwargs):
    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k]
        evaluator.append_train(model, **batch)

@EVALUATE_STEP.register("imputation")
def evaluate_imputation(batch, model, device, val_loss_module, evaluator, **kwargs):
    X, target, mask, padding_mask, label, ID = tuple(batch.values())
    target = target.to(device)
    mask = mask.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
    padding_mask = padding_mask.to(device)  # 0s: ignore
    prediction = model(X.to(device), padding_mask) 
    loss = val_loss_module(prediction, target, mask)  # (num_active,) individual loss (square error per element) for each active value in 
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
    mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the 
    evaluator.append_valid(model, X=X, prediction=prediction, target=target, mask=mask, metrics=loss)
    return batch_loss, mean_loss, len(loss)

@EVALUATE_STEP.register("pretraining")
def evaluate_pretraining(batch, model, device, model_name, val_loss_module, evaluator, **kwargs):
    X, target, mask, padding_mask, label, ID = tuple(batch.values())
    target = target.to(device)
    mask = mask.to(device)  
    padding_mask = padding_mask.to(device) 
    X = X.to(device)
    pretrain_evaluate_kwargs = {
        "target": target,
        "mask": mask,
        "val_loss_module": val_loss_module,
        "model": model,
        "X": X,
        "padding_mask": padding_mask,
        "device": device,
        "evaluator": evaluator
    }
    batch_loss, mean_loss, active_elements = PRETRAIN_EVALUATE_STEP.get(model_name)(**pretrain_evaluate_kwargs)
    evaluator.append_valid(model, X=X, target=target, ID=ID, mask=mask, label=label)
    return batch_loss, mean_loss, active_elements

@PRETRAIN_EVALUATE_STEP.register("mvts_transformer")
def pretrain_evaluate_mvts_transformer(X, model,  device, target, padding_mask, mask, val_loss_module, evaluator, **kwargs):
    prediction = model(X.to(device), padding_masks=padding_mask) 
    loss = val_loss_module(prediction, target, mask)  # (num_active,) individual loss (square error per element) for each active value in 
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
    mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the 
    # evaluator.append_valid(model, X=X, mask=padding_mask, metrics=loss.cpu().numpy())
    active_elements = len(loss)
    return batch_loss, mean_loss, active_elements

@PRETRAIN_EVALUATE_STEP.register("ts2vec")
@PRETRAIN_EVALUATE_STEP.register("ts_tcc")
@PRETRAIN_EVALUATE_STEP.register("t_loss")
@PRETRAIN_EVALUATE_STEP.register("csl")
def pretrain_evaluate_ts2vec(X, model, device, padding_mask, **kwargs):
    mean_loss = float('inf')
    active_elements = 1
    batch_loss = 0
    return batch_loss, mean_loss, active_elements

@EVALUATE_STEP.register("classification")
def evaluate_classification(batch, model, device, val_loss_module, evaluator, **kwargs):
    X, target, padding_mask, ID = tuple(batch.values())
    target = target.to(device)
    padding_mask = padding_mask.to(device)
    prediction = model(X.to(device), padding_mask)  
    loss = val_loss_module(prediction, target)
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = torch.mean(loss).cpu().item()
    active_elements = len(loss)
    evaluator.append_valid(model, mask=padding_mask, X=X, label=target, prediction=prediction, ID=ID)
    return batch_loss, mean_loss, active_elements

@EVALUATE_STEP.register("clustering")
def evaluate_clustering(batch, model, device, evaluator, **kwargs):
    X, target, padding_mask, ID = tuple(batch.values())
    target = target.to(device)
    padding_mask = padding_mask.to(device) 
    prediction = model.encode(X.to(device), padding_mask)
    mean_loss = float('inf')
    active_elements = 1
    batch_loss = 0
    evaluator.append_valid(model, X=X, mask=padding_mask, label=target, prediction=prediction, ID=ID)
    return batch_loss, mean_loss, active_elements 

@EVALUATE_STEP.register("anomaly_detection")
def evaluate_anomaly_detection(batch, model, device, val_loss_module, evaluator, **kwargs):
    X, target, padding_mask, ID = tuple(batch.values())
    target = target.to(device)
    padding_mask = padding_mask.to(device)
    prediction = model(X.to(device), padding_mask) 
    loss = val_loss_module(prediction[:, -1, :], X[:, -1, :])
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = torch.mean(loss).cpu().item()
    active_elements = len(loss)
    evaluator.append_valid(model, X=X, mask=padding_mask, target=target, prediction=prediction, ID=ID, score=loss, metric=loss)
    return batch_loss, mean_loss, active_elements

@EVALUATE_STEP.register("regression")
def evaluate_regression(batch, model, device, val_loss_module, evaluator, **kwargs):
    X, target, padding_mask, ID = tuple(batch.values())
    target = target.to(device)
    padding_mask = padding_mask.to(device) 
    prediction = model(X.to(device), padding_mask) 
    loss = val_loss_module(prediction, target)
    if len(loss.shape) > 1:
        loss = loss.reshape(loss.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
    batch_loss = torch.sum(loss).cpu().item()
    mean_loss = torch.mean(loss).cpu().item()
    active_elements = len(loss)
    evaluator.append_valid(model, metrics=loss, X=X, mask=padding_mask, target=target, prediction=prediction, ID=ID)
    return batch_loss, mean_loss, active_elements

@MAKE_EVAL_REPORT.register("imputation")
def eval_agg_imputation(**kwargs):
    pass

@MAKE_EVAL_REPORT.register("pretraining")
def eval_agg_ts2vec_ts_tcc_t_loss(evaluator, val_loss_module, logger, epoch_metrics, **kwargs):
    if evaluator is not None:
        infer_kwargs = {
            "val_loss_module": val_loss_module
        }
        epoch_metrics.update(evaluator.infer(**infer_kwargs))
    else:
        epoch_metrics.update({
            "report": 0,
            "loss": 0
        })
    return epoch_metrics

@MAKE_EVAL_REPORT.register("classification")
def eval_agg_classfication(evaluator, epoch_metrics, **kwargs):
    report = classification_report(np.argmax(evaluator.get("prediction"), axis=-1), 
                                evaluator.get("label"))
    epoch_metrics["report"] = report

@MAKE_EVAL_REPORT.register("clustering")
def eval_agg_clustering(evaluator, epoch_metrics, **kwargs):
    report = evaluator.infer()
    epoch_loss = report["NMI"]
    report = json.dumps(report, indent="  ")
    epoch_metrics["report"] = report
    epoch_metrics['loss'] = float(epoch_loss)

@MAKE_EVAL_REPORT.register("regression")
def eval_agg_regression(epoch_metrics, epoch_loss, **kwargs):
    epoch_metrics["report"] = str(epoch_loss)

@MAKE_EVAL_REPORT.register("anomaly_detection")
def eval_agg_anomaly_detection(evaluator, epoch_metrics, **kwargs):
    # print(evaluator.get("target").shape, evaluator.get("score").shape)
    auc_score = auc(evaluator.get("target"), evaluator.get("score"))
    epoch_metrics['auc'] = auc_score
    epoch_metrics['report'] = auc_score

@EVALUATE.register("default")
def evaluate(model, valid_dataloader, task, device, val_loss_module, 
             model_name, print_interval, print_callback, logger, evaluator=None,
               epoch_num=None, keep_all=True):
    model.eval()
    epoch_loss = 0  # total loss of epoch
    total_active_elements = 0  # total unmasked elements in epoch
    # handle = model.output_layer.register_forward_hook(get_rep)
    epoch_metrics = OrderedDict()
    # if keep_all:
    #     per_batch = {'mask': [], 'target': [], 'prediction': [], 'metrics': [], 'ID': [], 'reps': [], "score": [], "X": []}
    for i, batch in enumerate(valid_dataloader):
        mask = None

        evaluate_step_kwargs = {
            "batch": batch,
            "model": model,
            "device": device,
            "model_name": model_name,
            "val_loss_module": val_loss_module,
            "evaluator": evaluator
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
    
    epoch_metrics['epoch'] = epoch_num
    epoch_metrics['loss'] = float(epoch_loss)

    eval_aggr_kwargs = {
        "epoch_metrics": epoch_metrics,
        "val_loss_module": val_loss_module,
        "logger": logger,
        "evaluator": evaluator,
        "model_name": model_name,
        "epoch_loss": epoch_loss
    } 

    make_report = MAKE_EVAL_REPORT.get(task)
 
    if make_report is not None:
        make_report(**eval_aggr_kwargs)

    per_batch = evaluator.per_batch_valid
    for key in per_batch:
        # print(key)
        per_batch[key] = list2array(per_batch[key])
    per_batch_idxed = deepcopy(per_batch)

    # handle.remove()

    if keep_all:
        return epoch_metrics, per_batch_idxed
    else:
        return epoch_metrics