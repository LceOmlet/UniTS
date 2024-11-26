from ...registry import MODELS


try:
    import torch_geometric
except:
    pass
else:
    from .blocks import *
    