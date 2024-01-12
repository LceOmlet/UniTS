try:
    from ..registry import EVALUATORS
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
        if task == "pretraining" or task == "imputation":
            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(device)
            target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(device)  # 0s: ignore
        elif task in ["classification", "clustering", "anomaly_detection"]:
            X, targets, padding_masks, IDs = batch
            targets = targets.to(device)
            padding_masks = padding_masks.to(device) 
        elif task == "regression":
            X, targets, padding_masks, features, IDs = batch
            targets = targets.to(device)
            padding_masks = padding_masks.to(device) 

        # TODO: for debugging
        # input_ok = utils.check_tensor(X, verbose=False, zero_thresh=1e-8, inf_thresh=1e4)
        # if not input_ok:
        #     print("Input problem!")
        #     ipdb.set_trace()
        #
        # utils.check_model(model, verbose=False, stop_on_error=True)
        # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
        if task == "classification":
            predictions = model(X.to(device), padding_masks) 
        elif task == "pretraining":
            target_masks = target_masks * padding_masks.unsqueeze(-1)
        elif task == "clustering":
            predictions = model.encode(X.to(device), padding_masks)
        if task == "anomaly_detection" or task =="imputation":
            predictions = model(X.to(device), padding_masks) 
        elif task == 'regression':
            predictions = model(X.to(device), padding_masks) 
            X = features
        

        if model_name == "mvts_transformer":
            predictions = model(X.to(device), padding_masks)  # (batch_size, padded_length, feat_dim)
        elif model_name == "ts2vec":
            predictions = model.encode_masked(X.to(device), target_masks.to(device)) # repr for the ts2vec method
        elif model_name == "ts_tcc":
            _, predictions = model.model(X.to(device))
        elif model_name == "t_loss":
            predictions = model(X.to(device).permute(0, 2, 1))

        if task == "pretraining" or task == "imputation":
            if model_name not in ["ts2vec", "ts_tcc", 't_loss']:
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
        elif task in ["classification", "regression"]:
            loss = val_loss_module(predictions, targets)
            if len(loss.shape) > 1:
                loss = loss.reshape(loss.shape[0], -1)
                loss = torch.mean(loss, dim=-1)
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = torch.mean(loss).cpu().item()
            # if batch_loss > 100:
            #     print("*" * 100)
            #     print(epoch_loss, batch_loss, i)
            #     print("*" * 100)
        elif task == "anomaly_detection":
            loss = val_loss_module(predictions[:, -1, :], X[:, -1, :])
            if len(loss.shape) > 1:
                loss = loss.reshape(loss.shape[0], -1)
                loss = torch.mean(loss, dim=-1)
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = torch.mean(loss).cpu().item()
            per_batch["score"].append(loss.detach().cpu().numpy())
            # if mean_loss > 20:
            #     continue
            # print(mean_loss)
        
        # if task not in ["classification", "clustering"]: 
        if isinstance(target_masks, torch.Tensor):
            per_batch['target_masks'].append(target_masks.cpu().numpy())

        if model_name not in ["ts2vec", "ts_tcc", "t_loss"] and task != "clustering":
            per_batch['metrics'].append(loss.cpu().numpy())
            
            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        else:
            mean_loss = float('inf')
            total_active_elements = 1
        per_batch['X'].append(X.cpu().numpy())
        per_batch['targets'].append(targets.cpu().numpy())
        per_batch['predictions'].append(predictions.cpu().numpy())
        per_batch['IDs'].append(np.array(IDs))

        metrics = {"loss": mean_loss}
        if i % print_interval == 0:
            ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            print_callback(i, metrics, prefix='Evaluating ' + ending, total_batches=len(valid_dataloader))
    
    # print("**" * 50)
    # print(epoch_loss, total_active_elements)
    # print("*" * 100)
    epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch


    per_batch_idxed = deepcopy(per_batch)
    for key in per_batch:
        # print(key)
        per_batch[key] = list2array(per_batch[key])
    if task == "pretraining":
        if model_name in ["ts2vec", "ts_tcc", "t_loss"]:
            if ridge is not None:
                epoch_loss = infer_imputation(per_batch["predictions"], per_batch["targets"], 
                                            per_batch["target_masks"], ridge, val_loss_module)
            else:
                epoch_loss = 0
            logger.info("Infer ridge.")
            epoch_metrics["report"] = epoch_loss
    elif task == "classification":
        report = classification_report(np.argmax(per_batch["predictions"], axis=-1), per_batch["targets"])
        epoch_metrics["report"] = report
    elif task == "clustering":
        report, clustering_rst = infer_cluster(per_batch["predictions"], per_batch["targets"])
        epoch_loss = report["NMI"]
        report = json.dumps(report, indent="  ")
        epoch_metrics["report"] = report
        per_batch_idxed["clustering_rst"] = clustering_rst
        # print("*" * 100)
        # print(clustering_rst.shape)
        # print(per_batch.keys())
    elif task == "regression":
        epoch_metrics["report"] = str(epoch_loss)
    if task == "anomaly_detection":
        # print(per_batch["score"].shape, per_batch["targets"].shape)
        # print(np.max(per_batch["targes"]))
        auc_score = auc(per_batch["targets"], per_batch["score"])
        epoch_metrics['auc'] = auc_score
        epoch_metrics['report'] = auc_score
        
    epoch_metrics['epoch'] = epoch_num
    epoch_metrics['loss'] = float(epoch_loss)

    # handle.remove()

    if keep_all:
        return epoch_metrics, per_batch_idxed
    else:
        return epoch_metrics