import logging
import time
import torch
from collections import OrderedDict
import os
import numpy as np
try:
    from .utils import utils
    from .utils.loss import *
    from .process_data import *
    from .utils.optimizers import get_optimizer
    from .process_model import get_model, get_fusion_model
    from .models.ts_tcc.models.loss import NTXentLoss
    from .models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLossVaryingLength
    from .models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLoss
except Exception:
    from utils import utils
    from utils.loss import *
    from process_data import *
    from utils.optimizers import get_optimizer
    from process_model import get_model, get_fusion_model
    from models.ts_tcc.models.loss import NTXentLoss
    from models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLossVaryingLength
    from models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLoss
from torch.utils.data import DataLoader
try:
    from ts_url.utils.sklearn_modules import fit_ridge
except Exception:
    import sys
    sys.path.append("/home/liangchen/Desktop/flatkit")
    from ts_url.utils.sklearn_modules import fit_ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.metrics import roc_auc_score as auc 

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)   
    formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')     
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


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

def get_loss_module(task):
    if task == "pretraining" or task == "transduction" or task == "imputation":
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element
    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample
    if task == "regression" or task == "anomaly_detection":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample
    if task == "ts2vec":
        return hierarchical_contrastive_loss
    if task == "clustering":
        return None
    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))
    
def infer_cluster(encodding, targets):
    label_num = np.max(targets) + 1
    pca = PCA(n_components=10)
    reps = pca.fit_transform(encodding)
    kmeans = KMeans(label_num)
    pred = kmeans.fit_predict(reps)
    NMI_score = normalized_mutual_info_score(targets, pred)
    RI_score = rand_score(targets, pred)
    return {"NMI":NMI_score, "RI": RI_score}, pred #, "acc": self.classifier.score(features, y)}

class Trainer:
    def __init__(self, data_configs, model_name, hp_path, p_path, 
                 device, optim_config, task, logger, fine_tune_config=None, ckpt_paths=None) -> None:
        self.val_times = {"total_time": 0, "count": 0}
        self.best_value = None
        self.best_metrics = None
        self.reprs = None
        self.logger = logger
        self.task = task

        self.loss_module = get_loss_module(task)
        self.val_loss_module = get_loss_module(task)
        if task != "classification":
            if model_name == "t_loss": self.loss_varying = TripletLossVaryingLength(
                optim_config["compared_length"], optim_config["nb_random_samples"]
                , optim_config["negative_penalty"]
            ); self.loss_module = TripletLoss(
                    optim_config["compared_length"], optim_config["nb_random_samples"],
                    optim_config["negative_penalty"]
            )

        self.NEG_METRICS = {'loss'}  # metrics for which "better" is less
        self.udls, self.dls_config = get_datas(data_configs, task=task)

        if model_name == "t_loss":
            self.t_loss_train = torch.cat([x[0].unsqueeze(0) for x in self.udls.train_ds], dim=0)
        if task == "pretraining" or task == "imputation":
            if task == "imputation":
                optim_config["masking_ratio"] = 1.0 - fine_tune_config["i_ratio"]
                optim_config["mask_mode"] = "concurrent"
            # print(task)
            # print(optim_config)
            self.train_ds = ImputationDataset(self.udls.train_ds, mean_mask_length=optim_config['mean_mask_length'],
                        masking_ratio=optim_config['masking_ratio'], mode=optim_config['mask_mode'],
                        distribution=optim_config['mask_distribution'], exclude_feats=optim_config['exclude_feats'], mask_row=False)
            self.valid_ds = ImputationDataset(self.udls.valid_ds, mean_mask_length=optim_config['mean_mask_length'],
                        masking_ratio=optim_config['masking_ratio'], mode=optim_config['mask_mode'],
                        distribution=optim_config['mask_distribution'], exclude_feats=optim_config['exclude_feats'])
            self.valid_dataloader = DataLoader(self.valid_ds, batch_size=1, collate_fn=collate_unsuperv)
            if task == "pretraining":
                self.model, self.model_config = get_model(model_name, hp_path, self.udls, self.dls_config, p_path)
            elif task == "imputation":
                fusion = fine_tune_config["fusion"]
                self.model, self.model_config = get_fusion_model(ckpt_paths,fusion, self.udls, self.dls_config, device, pred_len=self.dls_config["seq_len"])
                self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_unsuperv)

        if model_name in ["mvts_transformer", "t_loss"]:
            self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_unsuperv)
        elif model_name == "ts2vec":
            self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_ts2vec)
        elif model_name == "ts_tcc":
            self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], 
            collate_fn=lambda batch: collate_ts_tcc(batch, optim_config["jitter_scale_ratio"], 
            optim_config["jitter_ratio"], optim_config["max_seg"]))
            
        elif task in ["classification", "clustering"]:
            fusion = fine_tune_config["fusion"]
            data = [self.udls.train_ds[i][0] for i in range(len(self.udls.train_ds))]
            label = [self.udls.train_ds[i][1] for i in range(len(self.udls.train_ds))]
            self.train_ds = ClassiregressionDataset(data, labels=label)
            data = [self.udls.valid_ds[i][0] for i in range(len(self.udls.valid_ds))]
            label = [self.udls.valid_ds[i][1] for i in range(len(self.udls.valid_ds))]
            self.valid_ds = ClassiregressionDataset(data, labels=label)
            self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_superv)
            self.valid_dataloader = DataLoader(self.valid_ds, batch_size=1, collate_fn=collate_superv)
            self.model, self.model_config = get_fusion_model(ckpt_paths,fusion, self.udls, self.dls_config, device)
        elif task == "regression":
            fusion = fine_tune_config["fusion"]
            pred_len = fine_tune_config["pred_len"]
            self.pred_len = pred_len
            data = [self.udls.train_ds[i][0] for i in range(len(self.udls.train_ds))]
            self.train_ds = RegressionDataset(data)
            data = [self.udls.valid_ds[i][0] for i in range(len(self.udls.valid_ds))]
            self.valid_ds = RegressionDataset(data)
            self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], collate_fn=
                                         lambda x: collate_superv_regression(x, pred_len=pred_len))
            self.valid_dataloader = DataLoader(self.valid_ds, batch_size=1, collate_fn=
                                               lambda x: collate_superv_regression(x, pred_len=pred_len))
            self.model, self.model_config = get_fusion_model(ckpt_paths,fusion, self.udls, self.dls_config, device, pred_len=pred_len)
        elif task == "anomaly_detection":
            fusion = fine_tune_config["fusion"]
            data = [self.udls.train_ds[i][0] for i in range(len(self.udls.train_ds))]
            label = [self.udls.train_ds[i][1] for i in range(len(self.udls.train_ds))]
            self.train_ds = ClassiregressionDataset(data, labels=label)
            data = [self.udls.valid_ds[i][0] for i in range(len(self.udls.valid_ds))]
            label = [self.udls.valid_ds[i][1] for i in range(len(self.udls.valid_ds))]
            self.valid_ds = ClassiregressionDataset(data, labels=label)
            self.dataloader = DataLoader(self.train_ds, batch_size=optim_config["batch_size"], collate_fn=collate_superv)
            self.valid_dataloader = DataLoader(self.valid_ds, batch_size=1, collate_fn=collate_superv)
            self.model, self.model_config = get_fusion_model(ckpt_paths,fusion, self.udls, self.dls_config, device, pred_len=self.dls_config["seq_len"])
        self.logger.info("train_ds length: " + str(len(self.train_ds)) + ", valid_ds length: " + str(len(self.valid_ds)))
        self.device = device
        
        optim_class = get_optimizer(optim_config['optimizer'])
        optimizer = optim_class(self.model.parameters(), lr=optim_config['lr'], weight_decay=optim_config["l2_reg"])
        self.optimizer = optimizer
        if model_name == "ts_tcc":
            self.temporal_contr_optimizer = torch.optim.Adam(self.model.parameters_tc(), lr=optim_config["lr"], 
                                                             betas=(optim_config["beta1"], optim_config["beta2"]), weight_decay=3e-4)
        self.optim_config = optim_config

        self.l2_reg = optim_config["l2_reg"]
        self.print_interval = optim_config["print_interval"]
        self.printer = utils.Printer(console=True)
        epoch_metrics = OrderedDict()
        self.log_slash_n_flag = False
        self.model_name = model_name
        self.model = self.model.to(self.device)

    def print_callback(self, i_batch, metrics, prefix='', total_batches=0):

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)
    
    def validate(self, epoch, key_metric, save_dir, batch_predictions_path="best_predictions.npz", file_lock=None):
        # raise NotImplementedError("okk")
        self.logger.info("Evaluating on validation set ...")
        eval_start_time = time.time()
        with torch.no_grad():
            aggr_metrics, per_batch = self.evaluate(epoch, keep_all=True)
        eval_runtime = time.time() - eval_start_time
        self.logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))
        self.val_times["total_time"] += eval_runtime
        self.val_times["count"] += 1
        avg_val_time = self.val_times["total_time"] / self.val_times["count"]
        avg_val_batch_time = avg_val_time / len(self.valid_dataloader)
        avg_val_sample_time = avg_val_time / len(self.valid_dataloader.dataset)
        self.logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
        self.logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
        self.logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))
        print()
        print_str = 'Epoch {} Validation Summary: '.format(epoch)
        for k, v in aggr_metrics.items():
            if k == "report":
                print(v)
                continue
            print_str += '{}: {:8f} | '.format(k, v)
        self.logger.info(print_str)

        print(key_metric)
        if key_metric in self.NEG_METRICS:
            if self.best_value is None:
                self.best_value = 1e7
            condition = (aggr_metrics[key_metric] < self.best_value)
        else:
            if self.best_value is None:
                self.best_value = -1e7
            condition = (aggr_metrics[key_metric] > self.best_value)
        if condition:
            self.best_value = aggr_metrics[key_metric]
            utils.save_model(save_dir, 'model_best.pth', epoch, self.model, optim_config=self.optim_config,
                             model_config=self.model_config, model_name=self.model_name)
            self.best_metrics = aggr_metrics.copy()

            pred_filepath = os.path.join(save_dir, batch_predictions_path)
            per_batch_meta = dict()
            for key in per_batch:
                per_batch_meta[key] = dict()
                per_batch_meta[key]["length"] = len(per_batch[key])
                if len(per_batch[key]):
                    per_batch_meta[key]["shape"] = list(per_batch[key][0].shape)
            per_batch_meta_path = os.path.join(save_dir, batch_predictions_path + ".json")

            # with open(per_batch_meta_path, mode="w") as f:
            #     json.dump(per_batch_meta, f)
                
            if file_lock is not None:
                with file_lock:
                    np.savez(pred_filepath, **per_batch)
                    with open(per_batch_meta_path, "w") as f:
                        json.dump(per_batch_meta, f)
            else:
                np.savez(pred_filepath, **per_batch)
                with open(per_batch_meta_path, mode="w") as f:
                    json.dump(per_batch_meta, f)
            if not self.log_slash_n_flag:
                with open("res_file_paths.txt", "a") as rfp:
                    rfp.write(pred_filepath + "\n")
                self.log_slash_n_flag = True
        return aggr_metrics, self.best_metrics, self.best_value

    def get_rep(self, module, input, output):
        self.reprs = input[0].cpu().numpy()

    def evaluate(self, epoch_num=None, keep_all=True):
        self.model.eval()
        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        # handle = self.model.output_layer.register_forward_hook(self.get_rep)
        epoch_metrics = OrderedDict()
        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': [], 'reps': [], "score": [], "X": []}
        for i, batch in enumerate(self.valid_dataloader):
            target_masks = None
            if self.task == "pretraining" or self.task == "imputation":
                X, targets, target_masks, padding_masks, IDs = batch
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
            elif self.task in ["classification", "clustering", "anomaly_detection"]:
                X, targets, padding_masks, IDs = batch
                targets = targets.to(self.device)
                padding_masks = padding_masks.to(self.device) 
            elif self.task == "regression":
                X, targets, padding_masks, features, IDs = batch
                targets = targets.to(self.device)
                padding_masks = padding_masks.to(self.device) 


            # TODO: for debugging
            # input_ok = utils.check_tensor(X, verbose=False, zero_thresh=1e-8, inf_thresh=1e4)
            # if not input_ok:
            #     print("Input problem!")
            #     ipdb.set_trace()
            #
            # utils.check_model(self.model, verbose=False, stop_on_error=True)
            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            if self.task == "classification":
                predictions = self.model(X.to(self.device), padding_masks) 
            elif self.task == "pretraining":
                target_masks = target_masks * padding_masks.unsqueeze(-1)
            elif self.task == "clustering":
                predictions = self.model.encode(X.to(self.device), padding_masks)
            if self.task == "anomaly_detection" or self.task =="imputation":
                predictions = self.model(X.to(self.device), padding_masks) 
            elif self.task == 'regression':
                predictions = self.model(X.to(self.device), padding_masks) 
                X = features
            

            if self.model_name == "mvts_transformer":
                predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)
            elif self.model_name == "ts2vec":
                predictions = self.model.encode_masked(X.to(self.device), target_masks.to(self.device)) # repr for the ts2vec method
            elif self.model_name == "ts_tcc":
                _, predictions = self.model.model(X.to(self.device))
            elif self.model_name == "t_loss":
                predictions = self.model(X.to(self.device).permute(0, 2, 1))

            if self.task == "pretraining" or self.task == "imputation":
                if self.model_name not in ["ts2vec", "ts_tcc", 't_loss']:
                    loss = self.val_loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
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
            elif self.task in ["classification", "regression"]:
                loss = self.val_loss_module(predictions, targets)
                if len(loss.shape) > 1:
                    loss = loss.reshape(loss.shape[0], -1)
                    loss = torch.mean(loss, dim=-1)
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = torch.mean(loss).cpu().item()
                # if batch_loss > 100:
                #     print("*" * 100)
                #     print(epoch_loss, batch_loss, i)
                #     print("*" * 100)
            elif self.task == "anomaly_detection":
                loss = self.val_loss_module(predictions[:, -1, :], X[:, -1, :])
                if len(loss.shape) > 1:
                    loss = loss.reshape(loss.shape[0], -1)
                    loss = torch.mean(loss, dim=-1)
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = torch.mean(loss).cpu().item()
                per_batch["score"].append(loss.detach().cpu().numpy())
                # if mean_loss > 20:
                #     continue
                # print(mean_loss)
            
            # if self.task not in ["classification", "clustering"]: 
            if isinstance(target_masks, torch.Tensor):
                per_batch['target_masks'].append(target_masks.cpu().numpy())

            if self.model_name not in ["ts2vec", "ts_tcc", "t_loss"] and self.task != "clustering":
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
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending, total_batches=len(self.valid_dataloader))
        
        # print("**" * 50)
        # print(epoch_loss, total_active_elements)
        # print("*" * 100)
        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch


        per_batch_idxed = deepcopy(per_batch)
        for key in per_batch:
            # print(key)
            per_batch[key] = list2array(per_batch[key])
        if self.task == "pretraining":
            if self.model_name in ["ts2vec", "ts_tcc", "t_loss"]:
                if hasattr(self, "ridge"):
                    epoch_loss = infer_imputation(per_batch["predictions"], per_batch["targets"], 
                                                per_batch["target_masks"], self.ridge, self.val_loss_module)
                else:
                    epoch_loss = 0
                self.logger.info("Infer ridge.")
                epoch_metrics["report"] = epoch_loss
        elif self.task == "classification":
            report = classification_report(np.argmax(per_batch["predictions"], axis=-1), per_batch["targets"])
            epoch_metrics["report"] = report
        elif self.task == "clustering":
            report, clustering_rst = infer_cluster(per_batch["predictions"], per_batch["targets"])
            epoch_loss = report["NMI"]
            report = json.dumps(report, indent="  ")
            epoch_metrics["report"] = report
            per_batch_idxed["clustering_rst"] = clustering_rst
            # print("*" * 100)
            # print(clustering_rst.shape)
            # print(per_batch.keys())
        elif self.task == "regression":
            epoch_metrics["report"] = str(epoch_loss)
        if self.task == "anomaly_detection":
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

    def train_epoch(self, epoch_num=None):
        if self.task == "clustering":
            raise NotImplementedError()
        epoch_metrics = OrderedDict()
        self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        targets, masks, reprs = [], [], []
        for i, batch in enumerate(self.dataloader):
            if self.task == "classification":
                X, targets, padding_masks, IDs = batch
                targets = targets.to(self.device)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
                predictions = self.model(X.to(self.device), padding_masks)

                loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
                batch_loss = torch.mean(loss)
                self.optimizer.zero_grad()
                batch_loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
            elif self.task == "anomaly_detection":
                X, targets, padding_masks, IDs = batch
                padding_masks = padding_masks.to(self.device)
                predictions = self.model(X.to(self.device), padding_masks)
                loss = self.loss_module(predictions, X)
                batch_loss = torch.mean(loss)
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()

            elif self.task == "regression":
                X, preds, padding_masks, features, IDs = batch
                predictions = self.model(X.to(self.device), padding_masks)
                # print(predictions.shape, preds.shape)
                loss = self.loss_module(predictions, preds)
                batch_loss = torch.mean(loss)
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
            elif self.task == "imputation":
                X, target, target_masks, padding_masks, IDs = batch
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                target = target.to(self.device)
                predictions = self.model(X.to(self.device), padding_masks)
                loss = self.loss_module(target, predictions, target_masks)
                batch_loss = torch.mean(loss)
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
            elif self.model_name == "mvts_transformer":
                X, target, target_masks, padding_masks, IDs = batch
                # print(torch.mean(torch.abs(X)))
                target = target.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)
                # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
                target_masks = target_masks * padding_masks.unsqueeze(-1)
                loss = self.loss_module(predictions, target, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
                self.optimizer.zero_grad()
                loss = torch.mean(loss)
                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
            elif self.model_name == "ts2vec":
                gt1, gt2, crop_l, m1, m2, x, mask, IDs = batch
                gt1 = gt1.to(self.device)
                gt2 = gt2.to(self.device)

                out1 = self.model._net(gt1, m1)[:, -crop_l:]
                out2 = self.model._net(gt2, m2)[:, :crop_l]
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2
                )
                targets.append(x)
                masks.append(mask)
                reprs.append(self.model._net(x, mask).detach())
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
                self.model.net.update_parameters(self.model._net)
            elif self.model_name == "ts_tcc":
                x, aug1, aug2, mask, IDs = batch
                data = x.float().to(device)
                aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

                # optimizer
                self.optimizer.zero_grad()
                self.temporal_contr_optimizer.zero_grad()
                predictions1, features1 = self.model.model(aug1)
                predictions2, features2 = self.model.model(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_lstm_feat1 = self.model.tenporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.model.tenporal_contr_model(features2, features1)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1 
                zjs = temp_cont_lstm_feat2 

                # compute loss
                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(device, self.optim_config["batch_size"]
                                               , self.optim_config["temperature"],
                                            self.optim_config["use_cosine_similarity"])
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
                # if self.l2_reg:
                #     loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
                # else:
                #     loss = mean_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
                self.temporal_contr_optimizer.step()
                target = x.detach().clone(); targets.append(target)
                x[mask] = 0; _, reprs_ = self.model.model(x); reprs.append(reprs_)
                masks.append(mask)
            elif self.model_name == "t_loss":
                X, target, target_masks, padding_masks, IDs = batch
                X = X.to(self.device)
                loss = self.loss_module(target.permute(0, 2, 1), self.model, 
                                        self.t_loss_train, save_memory=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()
                target = target.detach().clone(); targets.append(target)
                X[target_masks | (~padding_masks.unsqueeze(-1))] = 0; reprs_ = self.model(X.permute(0, 2, 1)); reprs.append(reprs_)
                masks.append(target_masks)
            
            if len(loss.shape):
                loss = loss.reshape(loss.shape[0], -1)
                loss = torch.mean(loss, dim=-1)
            if not loss.shape:
                loss = loss.unsqueeze(0)
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending, total_batches=len(self.dataloader))

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        if self.model_name in ["ts2vec", "ts_tcc", "t_loss"]:
            self.ridge = fit_imputation(reprs, targets, masks, 0.1, self.val_loss_module)
            self.logger.info("Ridge Training.")
        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = float(epoch_loss)
        return epoch_metrics
    
    def eval_forecasting(self, train_data, test_data, pred_lens):
        model = self.model
        padding = 50
        
        t = time.time()
        train_repr = model.encode(
            train_data,
            casual=True,
            sliding_length=1,
            sliding_padding=padding,
            batch_size=256
        )
        test_repr = model.encode(
            test_data,
            casual=True,
            sliding_length=1,
            sliding_padding=padding,
            batch_size=256
        )
        ts2vec_infer_time = time.time() - t
        
        ours_result = {}
        lr_train_time = {}
        lr_infer_time = {}
        out_log = {}
        for pred_len in pred_lens:
            train_features, train_labels = self.generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
            test_features, test_labels = self.generate_pred_samples(test_repr, test_data, pred_len)
            
            t = time.time()
            lr = fit_ridge(train_features, train_labels)
            lr_train_time[pred_len] = time.time() - t
            
            t = time.time()
            test_pred = lr.predict(test_features)
            lr_infer_time[pred_len] = time.time() - t

            ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)
                
            out_log[pred_len] = {
                'norm': test_pred,
                'norm_gt': test_labels,
            }
            ours_result[pred_len] = {
                'norm': self.loss_module(test_pred, test_labels),
            }
            
        eval_res = {
            'ours': ours_result,
            'ts2vec_infer_time': ts2vec_infer_time,
            'lr_train_time': lr_train_time,
            'lr_infer_time': lr_infer_time
        }
        return out_log, eval_res
    
    @staticmethod
    def generate_pred_samples(features, data, pred_len, drop=0):
        n = data.shape[1]
        features = features[:, :-pred_len]
        labels = np.stack([data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
        features = features[:, drop:]
        labels = labels[:, drop:]
        return features.reshape(-1, features.shape[-1]), \
                labels.reshape(-1, labels.shape[2]*labels.shape[3])


if __name__ == '__main__':
    data_configs = [{"filepath":"", "train_ratio":0.6, "test_ratio":0.6, "dsid": "lsst"}]
    ckpt = ['/home/liangchen/Desktop/flatkit/ckpts/03_15_17_52_15_omi-6_mvts_transformer', '/home/liangchen/Desktop/flatkit/ckpts/03_15_21_16_44_omi-6_ts2vec']
    # ckpt = None
    save_name = "03_12_14_26_45_lsst_ts2vec"
    model_name = None
    task_name="imputation"
    fine_tune_config = {"fusion":"concat", "i_ratio":0.15}
    # fusion = None
    hp_path, p_path = "", ""
    print_interval = 10
    loss_module = get_loss_module(task_name)
    with open("/home/liangchen/Desktop/flatkit/ts_url/models/default_configs/ts2vec_optim.json") as optim:
        optim_config = json.load(optim)
    start_time = time.strftime("%b_%d_%H_%M_%S", time.localtime()) 
    # task_summary = "/".join(["LSST"]) + "_" + model_name
    # save_name = start_time + "_" + task_summary
    save_path = os.path.join("ckpt", save_name)

    os.makedirs(save_path, exist_ok=True)
    logger = setup_logger("__main__." + save_name, os.path.join(save_path, "run.log"))
    
    trainer = Trainer(data_configs, model_name, hp_path, p_path, 
                torch.device('cpu'), task=task_name, optim_config=optim_config, fine_tune_config=fine_tune_config, logger=logger, ckpt_paths=ckpt)
    """(data_configs, model_name, hp_path, p_path, 
                'cpu', task="pretraining", optim_config=optim_config)"""
    os.makedirs("./test", exist_ok=True)
    trainer.train_epoch(10)
    trainer.validate(10, "loss", save_path)

"""['mvts_transformer', 'ts2vec', 'ts_tcc', 't_loss']
{'data': [{'name': 'lsst', 'sFile': ''}], 'name': 'ts2vec', 'sFile': '', 'pFile': '', 'task': 'pretraining', 'task_id': 0}
task queued: {'data_file_list': [''], 'data_name': ['lsst'], 'model_name': 'ts2vec', 'model_hp_path': '', 'model_path': '', 'task_id': 0, 'task': 'pretraining'}
set neew item stats: [{'task_id': 0, 'alg': 'ts2vec', 'data': 'lsst', 'progress': 0, 'loss': '∞'}]
{'data_file_list': [''], 'data_name': ['lsst'], 'model_name': 'ts2vec', 'model_hp_path': '', 'model_path': '', 'task_id': 0, 'task': 'pretraining'}
None
[{'filepath': '', 'dsid': 'lsst', 'train_ratio': 0.6, 'test_ratio': 0.6}]
None
03_12_14_26_45_lsst_ts2vec
ckpts/03_12_14_26_45_lsst_ts2vec
2023-03-12 14:26:53,992 | INFO : {'output_dims': 320, 'hidden_dims': 64, 'depth': 10, 'feat_dim': 6, 'max_len': 36, 'device': 'cpu'}"""