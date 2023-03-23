import os
import numpy as np
import torch
import sys
import torch

sys.path.append("../ts2vec")
from datautils import *
os.system("rm -rf " + "data")
datasets = "ArticularyWordRecognition AtrialFibrillation BasicMotions Epilepsy ERing HandMovementDirection Libras NATOPS PEMS-SF PenDigits StandWalkJump UWaveGestureLibrary"
for data in datasets.split(" "):
    train_data, train_labels, test_data, test_labels = load_UEA(data)
    os.makedirs(os.path.join("data", data))
    with open(os.path.join("data", data, "train.pt"), "wb") as train, open(os.path.join("data", data, "test.pt"), "wb") as test:
        train_data = {"samples": torch.tensor(train_data, dtype=torch.float64).permute(0, 2, 1), "labels": torch.tensor(train_labels, dtype=torch.int64)}
        test_data = {"samples": torch.tensor(test_data, dtype=torch.float64).permute(0, 2, 1), "labels": torch.tensor(test_labels, dtype=torch.int64)}
        torch.save(train_data, train)
        torch.save(test_data, test)

for data in datasets.split(" "):
    root = os.path.join("data", data)
    test = os.path.join(root, "test.pt")
    val  = os.path.join(root, "val.pt")
    os.system("ln -sf test.pt " + val)

# print(json.dumps(performance_dict, indent=" "))
        
