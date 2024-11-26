from ...registry import MODELS
import numpy as np
import torch
import tsaug
import random
from torch import nn
import os
from ...transformations.src.vae.timevae import (
    TimeVAE
)

__all__ = ["TimeVAE4UniTS"]

@MODELS.register("time_vae")
class TimeVAE4UniTS(TimeVAE):
    def __init__(self, **kwargs):
        super(TimeVAE4UniTS, self).__init__(**kwargs)