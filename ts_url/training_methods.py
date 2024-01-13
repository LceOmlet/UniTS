import logging
import time
import torch
from collections import OrderedDict
import os
import numpy as np
from copy import deepcopy
from .models.ts_tcc.models.loss import NTXentLoss
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
from ts_url.utils.sklearn_modules import fit_ridge

from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.metrics import roc_auc_score as auc 
from .registry import EVALUATORS, TRAINERS, DATALOADERS, LOSSES, TRAIN_LOOP_INIT
from .evaluators import evaluators
from .train import train
from .dataloader import dataloaders
from .losses import losses

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)   
    formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')     
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

    

class Trainer:
    def __init__(self, data_configs, model_name, hp_path, p_path, 
                 device, optim_config, task, logger, fine_tune_config=None, ckpt_paths=None) -> None:
        self.val_times = {"total_time": 0, "count": 0}
        self.best_value = None
        self.best_metrics = None
        self.reprs = None
        self.logger = logger
        self.task = task

        loss_config = {
            "model_name": model_name,
            "optim_config": optim_config,
            "device": device
        }

        self.loss_module = LOSSES.get(task)(**loss_config)
        self.val_loss_module = LOSSES.get(task)(train=False, **loss_config)
        self.NEG_METRICS = {'loss'}  # metrics for which "better" is less
        # print(data_configs)
        # exit()
        self.udls, self.dls_config = get_datas(data_configs, task=task)

        if task != "classification" and model_name == "t_loss": 
            self.t_loss_train = torch.cat([torch.tensor(x[0]).to(device).unsqueeze(0) for x in self.udls.train_ds], dim=0)
        
        loader_kwargs = {
            "dls": self.udls,
            "fine_tune_config": fine_tune_config,
            "optim_config": optim_config,
            "model_name": model_name,
            "logger": self.logger
        }

        self.dataloader, self.valid_dataloader = DATALOADERS.get(task)(**loader_kwargs)

        if task == "pretraining":
            self.model, self.model_config = get_model(model_name, hp_path, self.udls, self.dls_config, p_path)
        else:
            fusion = fine_tune_config["fusion"]
            if task in ["classification", "clustering"]:
                pred_len = None
            elif task == "regression":
                pred_len = fine_tune_config["pred_len"]
            else:
                pred_len = self.dls_config["seq_len"]
            self.model, self.model_config = get_fusion_model(ckpt_paths,fusion, self.udls, self.dls_config, device, pred_len=pred_len)
        
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
        self.logger.info("Evaluating on validation set ...")
        eval_start_time = time.time()
        with torch.no_grad():
            aggr_metrics, per_batch = self.evaluate(epoch_num=epoch, keep_all=True)
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

    def evaluate(self, **kwargs):
        kwargs.update({
            "model": self.model,
            "valid_dataloader": self.valid_dataloader,
            "task": self.task,
            "device": self.device,
            "val_loss_module": self.val_loss_module,
            "model_name": self.model_name,
            "print_interval": self.print_interval,
            "print_callback": self.print_callback,
            "logger": self.logger,
        })
        if hasattr(self, "ridge"):
            kwargs["ridge"] = self.ridge
        
        return EVALUATORS.get("all_eval")(**kwargs)

    def train_epoch(self, **kwargs):
        kwargs.update({
            "model": self.model,
            "dataloader": self.dataloader,
            "task": self.task,
            "device": self.device,
            "loss_module": self.loss_module,
            "optimizer": self.optimizer,
            "model_name": self.model_name,
            "print_interval": self.print_interval,
            "print_callback": self.print_callback,
            "val_loss_module": self.val_loss_module,
            "logger": self.logger,
            "optim_config": self.optim_config
        })

        if hasattr(self, "temporal_contr_optimizer"):
            kwargs["temporal_contr_optimizer"] = self.temporal_contr_optimizer

        if hasattr(self, "t_loss_train"):
            kwargs["t_loss_train"] = self.t_loss_train
        results = TRAINERS.get("all_train")(**kwargs)
        self.ridge =  results.get("ridge")
        return results


if __name__ == '__main__':
    data_configs = [{"filepath":"", "train_ratio":1, "test_ratio":1, "dsid": "lsst"}]
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