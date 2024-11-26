from ..registry import LOSSES, PRETRAIN_LOSSES, MODELS
from ..models.ts_tcc.models.loss import NTXentLoss
from ..utils.loss import *
from ..models.UnsupervisedScalableRepresentationLearningTimeSeries.losses.triplet_loss import TripletLoss
from functools import partial
from collections import defaultdict
import math
from matplotlib import pyplot as plt

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
@PRETRAIN_LOSSES.register("time_vae")
def get_time_vae_loss(reconstruction_wt, **kwargs):
    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_recons, dim):
            x_r = torch.mean(X, dim=dim)
            x_c_r = torch.mean(X_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            loss = torch.sum(err)
            return loss

        err = torch.pow(X - X_recons, 2)
        reconst_loss = torch.sum(err)
        
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, dim=1)  # by feature axis 

        return reconst_loss
    def loss_function(self, X, X_recons, z_mean, z_log_var, reconstruction_wt):
        reconstruction_loss = _get_reconstruction_loss(X, X_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss
    
    return partial(loss_function, reconstruction_wt=reconstruction_wt)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cov_loss(x):
    batch_size = x.shape[0]
    num_features = x.shape[1]
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
    )
    return cov_loss

def std_loss(x):
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    # print(std_x)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2
    return std_loss

deep_clustering_acc = 0
normalizing_c = 0
feature_weights_ = defaultdict(lambda :1)
def vicloss(base_model, X, models, transform_features, optimizer, feature_weights=None, train=True, rec=False, alpha=0, rec_loss_w = 25, cov_loss_w=25, std_loss_w=25, repr_loss_w=1, **kwargs):
    global deep_clustering_acc, normalizing_c
    if feature_weights is None:
        feature_weights = feature_weights_

    torch.autograd.set_grad_enabled(True)
    optimizer.zero_grad()

    main_repr = None
    total_loss = 0
    losses = dict()
    # for x in X:
    #     x = x.reshape(-1).detach().cpu().numpy()
    #     plt.plot(x)
    #     plt.show()
    # print(X.shape)
    try:
        if rec:
            main_repr, base_features, multi_scale_shapelet_engergy, rec_ts= base_model(X, train=train)
            rec_loss = F.mse_loss(rec_ts, X)
            total_loss += rec_loss * rec_loss_w
            losses["rec_loss"] = rec_loss
            losses["mean_mse_x"] = F.mse_loss(X, torch.zeros_like(X))
            losses["mean_mse_rec"] = F.mse_loss(rec_ts, torch.zeros_like(rec_ts))
        else:
            main_repr, base_features, multi_scale_shapelet_engergy = base_model(X, train=train)
    except:
        print(X.shape)
        raise RuntimeError()
    
    base_features_ = base_features - base_features.mean(0)

    def check_gradients(model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                # print(param.grad)
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradient of {name}")
                    exit()
                if torch.isinf(param.grad).any():
                    print(f"Inf detected in gradient of {name}")
                    exit()
    
    main_cov_loss = cov_loss(base_features_)
    main_std_loss = std_loss(base_features_)
    losses["main_cov"] = main_cov_loss
    losses["main_std"] = main_std_loss
    cov_loss_all = 0
    std_loss_all = 0
    repr_loss = 0
    reprs = []
    for t in models:
        f_w = feature_weights[t]
        loss = 0
        model_ = models[t]
        
        
        data_ = transform_features[t]
        # print(data_.shape)
        
        # embeddings_layer = model_.bert.get_input_embeddings()
        # exit()
        
        if isinstance(models[t], MODELS.get("mmfa")):
            repr_, features, multi_scale_shapelet_engergy = model_(data_, train=train)
        elif isinstance(models[t], MODELS.get("mmfa_rec")):
            repr_, features, multi_scale_shapelet_engergy, rec_ts = model_(data_, train=train)
        else:
            repr_, features = model_(data_, train=train)
        # print(repr_)
        reprs.append(repr_.unsqueeze(1))
        # print(t)
        # print(repr_)
        inv_loss_ = F.mse_loss(main_repr, repr_)
        repr_loss += inv_loss_
        losses[t + "_inv_loss"] = inv_loss_
        features = features - features.mean(dim=0)
        cov_loss_ = cov_loss(features)
        std_loss_ = std_loss(features)
        losses[t + "_cov_loss"] = cov_loss_
        losses[t + "_std_loss"] = std_loss_
        cov_loss_all += cov_loss_
        std_loss_all += std_loss_
        
    reprs = torch.cat(reprs, dim=1)
    
    deep_clustering_cur = cov_loss_w * (cov_loss_all + main_cov_loss) \
        + std_loss_w * (std_loss_all + main_std_loss)
        
    total_loss +=  deep_clustering_cur + \
             repr_loss_w * repr_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=2.0)
    # check_gradients(model_)
    optimizer.step()
    losses["loss"] = total_loss
    torch.autograd.set_grad_enabled(False)
    return losses

@PRETRAIN_LOSSES.register("mmfa_rec")
def get_mmfa_loss(optim_config, **kwargs):
    return partial(vicloss, rec=True, **optim_config["loss"])

@PRETRAIN_LOSSES.register("mmfa")
def get_mmfa_loss(optim_config, **kwargs):
    return partial(vicloss, **optim_config["loss"])