from abc import ABC, abstractmethod
from torch import nn

class BaseMIL(nn.Module, ABC):

    def __init__(self, instance_dim: int=1024, embed_instance_dim: int=512, bag_dim: int=512, n_classes: int=2, **kwargs):
        super().__init__()
        self.instance_dim = instance_dim
        self.embed_instance_dim = embed_instance_dim
        self.bag_dim = bag_dim
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x, return_instance_feat=False, return_attn=False, return_bag_feat=False):
        raise NotImplementedError
