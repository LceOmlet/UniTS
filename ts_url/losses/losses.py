from ..registry import LOSSES, PRETRAIN_LOSSES
from ..models.ts_tcc.models.loss import NTXentLoss
from ..utils.loss import *
from ..models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLoss

@LOSSES.register("imputation")
def get_imputation_loss(**kwargs):
    return MaskedMSELoss(reduction='none') 

@LOSSES.register("regression")
@LOSSES.register("anomaly_detection")
def get_anomaly_detection_loss(**kwargs):
    return nn.MSELoss(reduction='none')

@LOSSES.register("classification")
def get_classification_loss(**kwargs):
    return NoFussCrossEntropyLoss(reduction='none') 

@LOSSES.register("clustering")
def get_clustering_loss(**kwargs):
    return None

@PRETRAIN_LOSSES.register("csl")
def get_csl(**kwargs):
    return nn.CrossEntropyLoss()


@LOSSES.register("pretraining")
def get_loss_module(model_name, train=True, **kwargs):
    if not train:
        return LOSSES.get("imputation")(**kwargs)
    return PRETRAIN_LOSSES.get(model_name)(**kwargs)
    
@PRETRAIN_LOSSES.register("ts2vec")
def get_ts2vec_loss(**kwargs):
    return hierarchical_contrastive_loss

@PRETRAIN_LOSSES.register("ts_tcc")
def get_ts_tcc_loss(device, optim_config, **kwargs):
    return NTXentLoss(device, optim_config["batch_size"]
                        , optim_config["temperature"],
                            optim_config["use_cosine_similarity"])
@PRETRAIN_LOSSES.register("t_loss")
def get_t_loss_loss(optim_config, **kwargs):
    t_loss = TripletLoss(
                    optim_config["compared_length"], optim_config["nb_random_samples"],
                    optim_config["negative_penalty"]
                )
    return t_loss

@PRETRAIN_LOSSES.register("mvts_transformer")
def get_mvts_transformer_loss(**kwargs):
    return LOSSES.get("imputation")(**kwargs)