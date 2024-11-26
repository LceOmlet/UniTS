from ..registry import TRANSFORMATION
import numpy as np
import torch
import tsaug
import random
from torch import nn
import os

from .src.vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    get_prior_samples,
)

__all__ = ["GenerativeModel"]

@TRANSFORMATION.register("time_vae")
class GenerativeModel:
    def __init__(self, X, device, d_name, model_name="timeVAE", model_path=None, verbose=1, train_epochs=1000, percentage=0.01, **kwargs) -> None:
        
        # print(percentage)
        n_samples, n_dims, length = X.shape
        # device = X.device
        
        # if percent < 40, and if n_sample > 40 return 40 else n_sample
        n_samples = int(min(max(int(n_samples * percentage), 100), n_samples))
        
        if train_epochs == "auto":
            train_epochs = int(max(2 * 1000 * 4000 * percentage // n_samples, 2))
        
        random_indices = torch.randperm(len(X))[:n_samples]
        X = X[random_indices]
            
        X = X.transpose(0, 2, 1)
        hyperparameters = kwargs[model_name]
        self.vae_model = instantiate_vae_model(
            vae_type=model_name,
            sequence_length=length,
            feature_dim=n_dims,
            **hyperparameters,
        ).to(device)
        
        storage_path = "generative_models"
        
        if model_path is not None:
            storage_dir = model_path
        else:
            os.makedirs(storage_path, exist_ok=True)
            storage_name = "_".join([model_name, d_name, "perc", str(percentage), "epoch", str(train_epochs)]) + ".ckpt"
            storage_dir = os.path.join(storage_path, storage_name)
        
        if os.path.exists(storage_dir):
            ckpt = torch.load(storage_dir)
            state_dict = ckpt.get("state_dict", ckpt)
            self.vae_model.load_state_dict(state_dict)
        else:
            train_vae(vae=self.vae_model, train_data=X, max_epochs=train_epochs, verbose=verbose)
            torch.save(self.vae_model.state_dict(), storage_dir)
        

    def transform(self, x, index, random_shift_ratio=0.1, **kwargs):
        x = x.permute(0, 2, 1)
        x = self.vae_model.sphere_augmentation(x, random_shift_ratio)
        x = x.permute(0, 2, 1)
        
        return x.detach()