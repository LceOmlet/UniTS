## Implementation of "Multivariate Time Series, Unsupervised Representation Learning, Feature Engineering, Eigenfunctions, Classification, Clustering, Anomaly Detection".
Unsupervised representation learning (URL) for time series data has garnered significant interest due to its remarkable adaptability across diverse downstream applications. It is tricky to ensure the utility for downstream tasks by focusing on patterns implied in the temporal domain features since the goal of URL differs from supervised learning methods. This study introduces an innovative approach that focuses on binding time series representations encoded from different modalities features, thereby guiding the neural encoder to recover local and global associations among these multi-modal features. Though, a variety of feature engineering techniques, e.g., spectral features, wavelet transformed features, features in image form and symbolic features, etc., transform time series to multi-modal informative features, the utilization of intricate feature fusion methods dealing with heterogeneous features hampers their scalability. In contrast to common methods that fuse features from multiple modalities, the proposed approach simplifies the neural architecture by retaining a single time series encoder. We further demonstrate and prove mechanisms for the encoder to maintain better inductive bias. By validating the proposed method on a diverse set of time series datasets, our approach outperforms existing state-of-the-art URL methods across diverse downstream tasks. This paper introduces a novel model-agnostic paradigm for time series URL, paving a new research direction.


### How to Install?
#### Linux
1. Install [miniconda](https://docs.anaconda.com/anaconda/install/linux/).
2. Simply run `bash env.sh` and waiting for the environment to be installed.

Just run `python app.py` in the created python environment "UniTS". After few seconds, you will see the GUI.


### Suported methods & datasets.

#### datasets
* [UEA Archive](http://www.timeseriesclassification.com/) can be downloaded and extracted in `data/UCR`, then the data can be selected easily only by setting the `Name` of the dataset (e.g. LSST).
* [Server Machine
Dataset, SMD](https://dl.acm.org/doi/10.1145/3292500.3330672), the preprocessed datasets are uploaded in [this directory](data/InTerFusion).

- Evaluate
```bash
python experiment.py --gpu 0 --model_name mmfa --dataset_name LSST
```


* [UCR Anomaly Detection](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), providing datasets for various anomaly detection tasks. Preprocessed and raw data can be found in [this directory](data/AnomalyDatasets_2021).

- Evaluate
```bash
python experiment.py --gpu 0 --model_name mmfa --dataset_name ucr
```


* [Application Server Dataset, ASD](https://dl.acm.org/doi/10.1145/3447548.3467075), the preprocessed datasets are uploaded in [this directory](data/InTerFusion).


* Customed data
The costomed datasets can be placed in any directions and be constructed as:
```
|-- path/to/dataset
|   |-- $(dsid) (name of the dataset)
|       |-- $(dsid)_TEST.ts
|       |-- $(dsid)_TRAIN.ts
```
The `.ts` files should be constructed as described in [ts file format](https://www.sktime.net/en/stable/api_reference/file_specifications/ts.html).
When doing classification or anomaly detection tasks, identifiers, `@targetLabel` or `@classLabel` should be contained in the file.

#### methods

* [csl](https://arxiv.org/abs/2305.18888), VLDB-24.
* [ts2vec](https://arxiv.org/abs/1907.05321), AAAI-22.
* [ts-tcc](https://www.ijcai.org/proceedings/2021/0324.pdf), IJCAI-21.
* [mvts-transformer](https://arxiv.org/abs/2010.02803), KDD-21.
* [tnc](https://arxiv.org/abs/2106.00750), ICLR-20.
* [t-loss](https://papers.nips.cc/paper_files/paper/2019/file/53c6de78244e9f528eb3e1cda69699bb-Paper.pdf),NeurIPS-19

### How to change optimization parameters of the training tasks?

Defaults hyper-parameters are defined at `ts_url/models/default_configs` for all the tasks (anomaly_detection, classification, clustering, imputation, regression) and all supported unsupervised methods.
Here we provide an example of hyper-parameters selected for an imputation task, no matter which methods or models are used during the pretraining phase, and how the representations are fused, the parameters can be selected independently.
```json
{
    "batch_size": 64,
    "optimizer": "Adam",
    "@optimier/choice": [
        "Adam",
        "RAdam"
    ],
    "lr": 0.001,
    "l2_reg": 0,
    "print_interval": 10,
    "epochs": 10,
    "mask_distribution": "geometric",
    "@mask_distribution/choice": ["geometric", "bernoulli"],
    "mean_mask_length": 3,
    "exclude_feats": null,
    "evaluator":"svm"
}
```
The hyper-parameters can be changed at anytime before a task being invoked.

