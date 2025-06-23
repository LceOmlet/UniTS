#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from ts_url.training_methods import Trainer
from ts_url.config_manager import ConfigManager
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
    "ArticularyWordRecognition",
    # "AtrialFibrillation",
    # "BasicMotions",
    "CharacterTrajectories",
    # "Cricket",
    # "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    # "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    # "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    # "Libras",
    # "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "Phoneme",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    # "StandWalkJump",
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_template.yaml",
                      help="Path to config file")
    parser.add_argument("--model_name", type=str, default="ts_tcc",)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="UWaveGestureLibrary",)
    parser.add_argument("--experiment_type", type=str, default="test")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Initialize config manager
    config_manager = ConfigManager(args.config)
    
    # Get model specific configurations
    model_config = config_manager.get_model_config(args.model_name)
    optim_config = config_manager.get_optimizer_config(args.model_name)
    
    # Update configs with command line arguments
    if args.lr is not None:
        optim_config['lr'] = args.lr
    if args.batch_size is not None:
        optim_config['batch_size'] = args.batch_size
    if args.epochs is not None:
        optim_config['epochs'] = args.epochs
        
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Get data configs
    data_configs = get_config(dsid=args.dataset_name)
    
    # Setup experiment path
    task_summary = f"{args.dataset_name}_{args.model_name}"
    start_time = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    save_path = os.path.join("exp2", task_summary, start_time)
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        data_configs=data_configs,
        model_name=args.model_name,
        model_config=model_config,
        device=device,
        optim_config=optim_config,
        task="pretraining",
        save_path=save_path
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()


