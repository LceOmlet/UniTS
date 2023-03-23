import os
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
import sys
sys.path.append("../ts2vec")
from datautils import *
import json

suffix = ".npz"
performance_dict = dict()
with open("res_file_paths_final.txt", "r") as rfpf:
    for line in rfpf:
        line = line.strip()
        if "experiment" in line:
            path = line + suffix
            reps = np.vstack(dict(np.load(path, allow_pickle=True))["reps"])
            reps = np.mean(reps, axis=1)
            dataset = os.path.join("/home/liangchen/Desktop/3liang/ts2vec/datasets/UEA", dataset_name)
            train_data, train_labels, test_data, test_labels = load_UEA(dataset_name)

            label_num = np.max(test_labels) + 1
            pca = PCA(10)
            new_test_repr = pca.fit_transform(reps)
            kmeans = KMeans(label_num)
            pred = kmeans.fit_predict(new_test_repr)
            

            NMI_score = normalized_mutual_info_score(test_labels, pred)
            RI_score = rand_score(test_labels, pred)
            res = {"NMI":NMI_score, "RI": RI_score}
            if dataset_name in performance_dict:
                performance_dict[dataset_name].append(res)
            else:
                performance_dict[dataset_name] = [res]
        else:
            dataset_name = line
print(json.dumps(performance_dict, indent=" "))
        
