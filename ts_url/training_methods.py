import logging
import time
import torch
from collections import OrderedDict
import os
import numpy as np
from copy import deepcopy
from .models.ts_tcc.models.loss import NTXentLoss

from .utils import utils
from .utils.loss import *
from .process_data import *
from .utils.optimizers import get_optimizer
from .process_model import get_model, get_fusion_model
from .models.ts_tcc.models.loss import NTXentLoss
from .models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLossVaryingLength
from .models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLoss

from torch.utils.data import DataLoader
from ts_url.utils.sklearn_modules import fit_ridge

from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.metrics import roc_auc_score as auc 
from .registry import EVALUATE, TRAIN_FN, DATALOADERS, LOSSES, TRAIN_LOOP_INIT, TRAINER_INIT, EVALUATOR, EVAL_LOOP_INIT
from .evaluators import evaluators
from .train import train
from .dataloader import dataloaders
from .losses import losses
from .collate_fn import collate_fn
from .test_modules import test_modules
from . import process_model
from .transformations import wavelet
import datetime
import yaml  # 添加在文件开头的import部分

class MicrosecondFormatter(logging.Formatter):
    converter = datetime.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        # Convert the time stamp to a datetime object
        ct = self.converter(record.created)
        if datefmt:
            # Use datetime's strftime, which supports %f
            s = ct.strftime(datefmt)
        else:
            # Default format includes microseconds
            s = ct.strftime("%Y-%m-%d %H:%M:%S.%f")
        return s
    
def extract_meta(per_batch):
    """
    递归地遍历per_batch，提取元信息并存储在per_batch_meta中。
    
    Args:
        per_batch (dict): 输入的包含不同类型数据的字典。
        
    Returns:
        dict: 包含每个键的元信息的字典。
    """
    per_batch_meta = {}
    
    def extract_meta_recursive(data):
        if isinstance(data, dict):
            # 如果data是字典，递归处理每个键
            meta = {}
            for key, value in data.items():
                meta[key] = extract_meta_recursive(value)
            return meta
        else:
            # 否则，根据给定规则处理
            meta = {}

            # 如果是torch.Tensor或numpy数组类型
            if isinstance(data, torch.Tensor):
                if data.ndim == 0:
                    return data.item()  # 将0维Tensor转换为Python标量
                meta["length"] = len(data)
                if len(data):
                    try:
                        meta["shape"] = list(data.shape)
                    except AttributeError:
                        pass
            elif isinstance(data, np.ndarray):
                if data.ndim == 0:
                    return data.item()  # 将0维ndarray转换为Python标量
                meta["length"] = len(data)
                if len(data):
                    try:
                        meta["shape"] = list(data.shape)
                    except AttributeError:
                        pass
            # 检查numpy的标量类型，将其转换为Python原生类型
            elif isinstance(data, (np.generic, np.float32, np.float64, np.int32, np.int64)):
                return data.item()
            # 检查其他原生类型
            elif not hasattr(data, "__len__") or isinstance(data, (str, int, float, bool)):
                return data  # 返回原生标量或字符串等类型
            else:
                meta["length"] = len(data)
                if len(data):
                    try:
                        meta["shape"] = list(data[0].shape)
                    except AttributeError:
                        pass

            return meta
    
    # 开始处理每个键
    for key in per_batch:
        per_batch_meta[key] = extract_meta_recursive(per_batch[key])
    
    return per_batch_meta

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        # 关闭 handler
        handler.close()
        # 从日志记录器中移除 handler
        logger.removeHandler(handler)
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)   
    formatter = MicrosecondFormatter(
        fmt='(%(filename)s:%(lineno)d) %(asctime)s | %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    handler.setFormatter(formatter)
    # 如果已经有 handlers 注册，则移除它们
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

    

class Trainer:
    def __init__(self, data_configs, model_name, model_config, 
                 device, optim_config, task="pretraining", logger=None, save_path=".", fine_tune_config=None, ckpt_paths=None, **kwargs) -> None:
        if isinstance(device, list):
            devices = device
            device = device[0]
        else:
            devices = device
        self.val_times = {"total_time": 0, "count": 0}
        self.best_value = None
        self.best_metrics = None
        self.reprs = None
        self.save_path = save_path 
        self.evaluator = None
        
        os.makedirs(save_path, exist_ok=True)
        
        self.task = task
        
        if logger is None:
            logger = setup_logger("__main__", os.path.join(save_path, "run.log"))
        self.logger = logger

        # 修改optim_config加载逻辑
        if isinstance(optim_config, str):
            if optim_config.endswith('.yaml') or optim_config.endswith('.yml'):
                with open(optim_config, 'r') as f:
                    self.optim_config = yaml.safe_load(f)
            else:
                with open(optim_config, 'r') as f:
                    self.optim_config = json.load(f)
        else:
            self.optim_config = optim_config

        # 修改model_config加载逻辑 
        if isinstance(model_config, str):
            if model_config.endswith('.yaml') or model_config.endswith('.yml'):
                with open(model_config, 'r') as f:
                    self.model_config = yaml.safe_load(f)
            else:
                with open(model_config, 'r') as f:
                    self.model_config = json.load(f)
        else:
            self.model_config = model_config
        # elif task == "pretraining":
        # self.model_config = model_config
        
        loss_config = dict(model_name=model_name, optim_config=optim_config, device=device)
        loss_config.update(optim_config.get("loss", {}))

        self.loss_module = LOSSES.get(task)(**loss_config)
        self.val_loss_module = LOSSES.get(task)(train=False, **loss_config)
        
        self.evaluator = EVALUATOR.get("default")(optim_config=optim_config,)
        self.POS_METRICS = {'accuracy', 'f1'}  # metrics for which "better" is less
        # print(data_configs)
        # exit()
        data_config = optim_config.get("data_config", {})
        self.udls, self.dls_config = get_datas(data_configs, task=task, **data_config)
        # print(self.dls_config)
        
        loader_kwargs = dict(dls=self.udls, data_configs=data_configs, fine_tune_config=fine_tune_config, 
                             optim_config=optim_config, model_name=model_name, logger=self.logger, device=device)

        self.dataloader, self.valid_dataloader = DATALOADERS.get(task)(**loader_kwargs)

        if task == "pretraining":
            self.model, self.model_config = get_model(model_name, self.dls_config, self.model_config)
        else:
            fusion = fine_tune_config["fusion"]
            if task in ["classification", "clustering"]:
                pred_len = None
            elif task == "regression":
                pred_len = fine_tune_config["pred_len"]
            else:
                pred_len = self.dls_config["seq_len"]
            self.model, self.model_config = get_fusion_model(ckpt_paths, self.dls_config, device, pred_len=pred_len)
        
        initer = TRAINER_INIT.get(task)
        self.init_trainer = {}

        self.device = device
        optim_class = get_optimizer(optim_config['optimizer'])
        self.l2_reg = optim_config.get("l2_reg", 0)
        self.print_interval = optim_config.get("print_interval", 10)
        self.evaluate_interval = optim_config.get("evaluate_interval", 1)
        
        
        optimizer = optim_class(self.model.parameters(), lr=optim_config['lr'], weight_decay=self.l2_reg)
        
        
        self.optimizer = optimizer

        if initer is not None:
            initer_kwargs = dict(model_name=model_name, model=self.model, optim_config=optim_config, 
                                 train_ds=self.udls.train_ds, device=devices, optimizer=optimizer)

            self.init_trainer = initer(**initer_kwargs)

        self.optim_config = optim_config

        self.log_slash_n_flag = False
        self.model_name = model_name
        self.model = self.model.to(self.device)

    def print_callback(self, i_batch, metrics, prefix='', total_batches=0):
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\n\t{}".format(met_name) + ": {:g}"
            content.append(met_value)
        template += '\n'
        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.logger.info(dyn_string)
    
    def validate(self, epoch_num, key_metric=None, save_dir=None, save_condition="True", batch_predictions_path="best_predictions.npz", file_lock=None):
        self.logger.info("Evaluating on validation set ...")
        eval_start_time = time.time()
        with torch.no_grad():
            aggr_metrics, per_batch = self.evaluate(epoch_num=epoch_num, keep_all=True)
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
        print_str = 'Epoch {} Validation Summary: '.format(epoch_num)
        for k, v in aggr_metrics.items():
            if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                continue
            if k == "report":
                self.logger.info("report: " + str(v))
                continue
            if k == "train_report":
                self.logger.info("report: " + str(v))
                continue
            print_str += '{}: {:8f} | '.format(k, v)
        self.logger.info(print_str)
        
        # eval(f"{key_metric} = aggr_metrics[key_metric]")
        epoch = epoch_num
        best = self.best_value
        if eval(save_condition):
            if key_metric not in self.POS_METRICS:
                if self.best_value is None:
                    self.best_value = 1e7
                condition = (aggr_metrics[key_metric] <= self.best_value)
            else:
                if self.best_value is None:
                    self.best_value = -1e7
                condition = (aggr_metrics[key_metric] >= self.best_value)
        else:
            condition = False
        if condition and save_dir is not None:
            self.best_value = aggr_metrics[key_metric]
            utils.save_model(save_dir, 'model_best.pth', epoch_num, self.model, optim_config=self.optim_config,
                             model_config=self.model_config, model_name=self.model_name)
            self.best_metrics = aggr_metrics.copy()

            pred_filepath = os.path.join(save_dir, batch_predictions_path)
            per_batch_meta = dict()
            per_batch_meta["best_metric"] = (key_metric, self.best_value)
            per_batch_meta = extract_meta(per_batch)
            per_batch_meta_path = os.path.join(save_dir, batch_predictions_path + ".json")

            # with open(per_batch_meta_path, mode="w") as f:
            #     json.dump(per_batch_meta, f)
            self.logger.info("ckpt updated.")
            if file_lock is not None:
                with file_lock:
                    np.savez(pred_filepath, **per_batch)
                    with open(per_batch_meta_path, "w") as f:
                        json.dump(per_batch_meta, f)
            else:
                np.savez(pred_filepath, **per_batch)
                with open(per_batch_meta_path, mode="w") as f:
                    json.dump(per_batch_meta, f)
        return aggr_metrics, self.best_metrics, self.best_value

    def get_rep(self, module, input, output):
        self.reprs = input[0].cpu().numpy()

    def evaluate(self, clear_evaluator=False, **kwargs):
        
        eval_loop_init = EVAL_LOOP_INIT.get(self.task)

        if eval_loop_init is not None:
            eval_loop_init_kwargs = dict(model=self.model, dataloader=self.dataloader, evaluator=self.evaluator, device=self.device)

            eval_loop_init(**eval_loop_init_kwargs)

        train_agg_kwargs = dict(val_loss_module=self.val_loss_module, logger=self.logger)

        self.evaluator.train_module(**train_agg_kwargs)


        kwargs.update(dict(model=self.model, valid_dataloader=self.valid_dataloader, 
                           task=self.task, device=self.device, val_loss_module=self.val_loss_module, 
                           model_name=self.model_name, print_interval=self.print_interval, 
                           print_callback=self.print_callback, logger=self.logger, 
                           evaluator=self.evaluator))
        
        result = EVALUATE.get("default")(**kwargs)
        if clear_evaluator:
            self.evaluator.clear()
        return result
        

    def train_epoch(self, epoch_num, **kwargs):

        self.evaluator.clear()

        kwargs.update(dict(model=self.model, epoch_num=epoch_num, dataloader=self.dataloader, 
                           task=self.task, device=self.device, evaluator=self.evaluator, 
                           loss_module=self.loss_module, optimizer=self.optimizer, 
                           model_name=self.model_name, print_interval=self.print_interval, 
                           print_callback=self.print_callback, val_loss_module=self.val_loss_module, 
                           logger=self.logger, optim_config=self.optim_config, data=self.udls.train_ds))

        kwargs.update(self.init_trainer)

        # Use TNC-specific training function if model is TNC
        if self.model_name == "tnc":
            trainer_fn = TRAIN_FN.get("tnc")
        else:
            trainer_fn = TRAIN_FN.get("default")
            
        if trainer_fn is not None:
            results = trainer_fn(**kwargs)
        else:
            results = None
        
        
        # self.evaluator =  results.get("evaluator")
        return results
    
    def load_model(self, model_path):
        utils.load_model(self.model, optimizer=self.optimizer, model_path=model_path)
    
    def fit(self):
        for ep in range(self.optim_config['epochs']):
            self.train_epoch(epoch_num=ep)
            key_metric = self.optim_config.get('key_metric', 'loss')
            if ep % self.evaluate_interval == 0:
                save_condition = self.optim_config.get("save_condition", "True")
                aggr_metrics, best_metrics, best_value = self.validate(epoch_num=ep, key_metric=key_metric, save_dir=self.save_path, save_condition=save_condition)
        return best_metrics, self.dls_config


    def _test(self):
        self.validate(0, key_metric="loss", save_dir=self.save_path)


if __name__ == '__main__':
    data_configs = [{"filepath":"", "train_ratio":1, "test_ratio":1, "dsid": "lsst"}]
    ckpt = ['/home/username/Desktop/flatkit/ckpts/03_15_17_52_15_omi-6_mvts_transformer', '/home/username/Desktop/flatkit/ckpts/03_15_21_16_44_omi-6_ts2vec']
    # ckpt = None
    save_name = "03_12_14_26_45_lsst_ts2vec"
    model_name = None
    task_name="imputation"
    fine_tune_config = {"fusion":"concat", "i_ratio":0.15}
    # fusion = None
    hp_path, model_config = "", ""
    print_interval = 10
    loss_module = get_loss_module(task_name)
    with open("/home/username/Desktop/flatkit/ts_url/models/default_configs/ts2vec_optim.json") as optim:
        optim_config = json.load(optim)
    start_time = time.strftime("%b_%d_%H_%M_%S", time.localtime()) 
    # task_summary = "/".join(["LSST"]) + "_" + model_name
    # save_name = start_time + "_" + task_summary
    save_path = os.path.join("ckpt", save_name)

    os.makedirs(save_path, exist_ok=True)
    logger = setup_logger("__main__." + save_name, os.path.join(save_path, "run.log"))
    
    trainer = Trainer(data_configs, model_name, hp_path, model_config, 
                torch.device('cpu'), task=task_name, optim_config=optim_config, fine_tune_config=fine_tune_config, logger=logger, ckpt_paths=ckpt)
    """(data_configs, model_name, hp_path, model_config, 
                'cpu', task="pretraining", optim_config=optim_config)"""
    os.makedirs("./test", exist_ok=True)
    trainer.train_epoch(10)
    trainer.validate(10, "loss", save_path)
