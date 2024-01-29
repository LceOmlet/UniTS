## Official Implementation of "UniTS: A Universal Time Series Analysis Framework Powered by Self-supervised Representation Learning".
UniTS is a powerful framework for time series analysis that incorporates self-supervised representation learning to address practical challenges such as partial labeling and domain shift. With UniTS, users can easily perform analysis tasks using user-friendly GUIs and benefit from its convenience. The components of UniTS are designed with sklearn-like APIs, allowing for flexible extensions. This project's GitHub repository provides access to the UniTS framework and related resources.

### What's new?
An effective and comprehensive representation learning method:

:boom: :boom: [A Shapelet-based Framework for Unsupervised Multivariate Time Series Representation Learning](https://www.vldb.org/pvldb/vol17/p386-wang.pdf) :boom: :boom:.

The theoritical foundation of url for ts is being built by considering shaplet as adaptive wavelet.

### How to Install?
#### Linux
1. Install [miniconda](https://docs.anaconda.com/anaconda/install/linux/).
2. Simply run `bash env.sh` and waiting for the environment to be installed.

Just run `python app.py` in the created python environment "UniTS". After few seconds, you will see the GUI as shown in the image below.
![Pre-training Model](./figures/Pre_training.png)

#### Windows
Almost the same as installing in linux system.
1. Install [miniconda](https://docs.anaconda.com/anaconda/install/linux/).
2. run env.sh as .bat file.
Just run `python app.py` in the created python environment "UniTS". 



### Suported methods & datasets.

#### datasets
* [UEA Archive](http://www.timeseriesclassification.com/) can be downloaded and extracted in `data/UCR`, then the data can be selected easily only by setting the `Name` of the dataset (e.g. LSST).
* [Server Machine
Dataset, SMD](https://dl.acm.org/doi/10.1145/3292500.3330672), the preprocessed datasets are uploaded in [this directory](data/InTerFusion).

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

* [csl](https://arxiv.org/abs/2305.18888), VLDB-23.
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
    "exclude_feats": null
}
```
The hyper-parameters can be changed at anytime before a task being invoked.


### Todo List

- [x] Support insightful visualization for all supported downstream tasks.
- [x] Support customed datasets.
- [x] Support fast and convenient installation.
- [x] Support convenient registrations for more URL models.
- [x] Support convenient registrations for more self-supervise signals.
- [ ] Support [alignment](https://arxiv.org/abs/2312.05698) methods to distil knowledge from multiple encoders. 
- [ ] Support more convenient APIs for advanced development.
