#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from ts_url.training_methods import Trainer
import time
import os
# torch.cuda.set_device(4)
import random
import numpy as np
from torch import nn
import json
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
import argparse
torch.autograd.set_detect_anomaly(True)

# 30个数据集
datasets = [
    # "ArticularyWordRecognition",
    # "AtrialFibrillation",
    # "BasicMotions",
    # "CharacterTrajectories",
    # "Cricket",
    # "DuckDuckGeese",
    # "EigenWorms",
    # "Epilepsy",
    # "EthanolConcentration",
    # "ERing",
    # "FaceDetection",
    # "FingerMovements",
    # "HandMovementDirection",
    "Handwriting",
    # "Heartbeat",
    # "InsectWingbeat",
    # "JapaneseVowels",
    # "Libras",
    # "LSST",
    # "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "Phoneme",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary"
]

# datasets = ["BasicMotions"]

# Try mmfa on LSST dataset with SVM as the test module.

    # "HandMovementDirection",
    # "Handwriting",

transformations = {
    "cgau2": {
        "model":{
            "model_name": "ResNet12"
        },
        "resize_shape": 224
    },
    "RP": {
        "model":{
            "model_name": "ResNet12"
        },
        "resize_shape": 224
    },
    "GADF": {
        "model":{
            "model_name": "ResNet12"
        },
        "max_channel": 20
    },
    "fft":{
        "model":{
            "model_name": "ResNet1d"
        }
    },
    "weasel":{
        "model":{
            "model_name": "LongFormer"
        }
    }
}

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="t_loss", help="vae: 训练vae模型，mmfa: 使用vae训练mmfa")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--dataset_name", type=str, default="Libras", help="30个数据集之一")
parser.add_argument("--experiment_type", type=str, default="test")
parser.add_argument("--mmfa_version", type=str, default="vae", help="vae: 用vae训练mmfa, aug: 采用扩增方式, mmfa: 采用多种可逆变换（具体看default_config/mmfa_mmfa_optim.json）")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--cov_loss_w", type=float, default=10, help="协方差")
parser.add_argument("--std_loss_w", type=float, default=10, help="标准差")
parser.add_argument("--repr_loss_w", type=float, default=1, help="对齐")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1000000)
parser.add_argument("--vae_model_path", type=str, default="exp2/Libras_time_vae/09_21_18_20_56/model_best.pth", help="修改vae模型")
parser.add_argument("--test_transformation", type=str)
# parser.add_argument("")
args = parser.parse_args()

experiment = "exp2"

hp_path = "configs/mmfa_optim.json"
p_path = "configs/mmfa.json"
optim_config = "configs/mmfa_optim.json"
task_name = "pretraining"

device = torch.device('cuda')



def get_config(filepath="/home/liangchen/liangchen/aeon/aeon/datasets/data/test/Multivariate_ts", 
               train_ratio=1, test_ratio=1, dsid="HandMovementDirection"):
    filepath += '/' + dsid
    data_configs = [{
        "filepath": filepath,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "dsid": dsid
    }]
    return data_configs

data_configs = get_config()

data_names = [d['dsid'] for d in data_configs]
task_summary = "_".join(data_names) + "_" + args.model_name

if args.mmfa_version is not None and args.model_name == "mmfa":
    task_summary += "_" + args.mmfa_version
    

start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
# experiment = os.path.join(experiment, task_summary, start_time)

# experiment =  experiment + "_" + args.experiment_type

os.makedirs(experiment, exist_ok=True)

# hp_path = "configs/ts_tcc_optim.json"


##################################################################
#### Search and config optimizing parameters

optim_configuratioin_path = args.model_name + "_" + args.mmfa_version if (args.mmfa_version is not None and args.model_name == "mmfa") else args.model_name 

p_path = f"default_config/{args.model_name}.json"
optim_config = f"default_config/{optim_configuratioin_path}_optim.json"
with open(optim_config, mode="r") as f:
    optim_config = json.load(f)


# if args.experiment_type == "FC":
#     print(f"Current sfa wwm weight loading type: {optim_config['transformations']['sfa']['model']['load_wwm_weights']}")
#     optim_config["transformations"]["sfa"]["model"]["load_wwm_weights"] = False
        
mmfa_loss_config = {"loss":{
        "cov_loss_w":args.cov_loss_w,
        "std_loss_w":args.std_loss_w, 
        "repr_loss_w":args.repr_loss_w
    }}
new_optim_config = {
              "lr": args.lr,
              "epochs": args.epochs,
              "batch_size": args.batch_size
            }

optim_config.update(new_optim_config)
if args.model_name == "mmfa":
    optim_config.update(mmfa_loss_config)
    
##################################################################
##################################################################
#### Set ckpt path for vae model

if args.model_name == "mmfa" and args.vae_model_path is not None:
    optim_config["model_path"] = args.vae_model_path


task_name = "pretraining"
model_name = args.model_name

device = int(args.gpu)
if args.dataset_name != "":
    datasets = [args.dataset_name]
    
# if args.dataset_name == "ucr":
#     for i in range(0, 250):
#         dsid = "ad_ucr_" + str(i)
#         data_configs = get_config(dsid=dsid)
#         data_names = [d['dsid'] for d in data_configs]
#         task_summary = "_".join(data_names) + "_" + model_name
#         start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
#         save_path = os.path.join(experiment, task_summary, start_time)
#         os.makedirs(save_path, exist_ok=True)
#         trainer = Trainer(data_configs, model_name, p_path, 
#                         device, optim_config, task_name, save_path=save_path)

#         # trainer.validate(epoch_num=0, key_metric="loss", save_dir=save_path)
#         trainer.fit()
# else:
dsid = args.dataset_name
data_configs = get_config(dsid=dsid)
data_names = [d['dsid'] for d in data_configs]
task_summary = "_".join(data_names) + "_" + model_name
start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime()) 
save_path = os.path.join(experiment, task_summary, start_time)
os.makedirs(save_path, exist_ok=True)
trainer = Trainer(data_configs, model_name, p_path, 
                device, optim_config, task_name, save_path=save_path)

# trainer.validate(epoch_num=0, key_metric="loss", save_dir=save_path)
trainer.fit()


