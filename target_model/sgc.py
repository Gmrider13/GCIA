"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
from torch_geometric.nn import SGConv
from .base_model import BaseModel
class SGC(BaseModel):
    """ SGC based on pytorch geometric. Simplifying Graph Convolutional Networks.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nclass : int
        size of output dimension
    K: int
        number of propagation in SGC
    cached : bool
        whether to set the cache flag in SGConv
    lr : float
        learning rate for SGC
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in SGC weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
"""

    def __init__(self, nfeat, nclass,nhid, K=3, cached=True, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(SGC, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = SGConv(nfeat,
                nclass, bias=with_bias, K=K, cached=cached)

        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.name = 'SGC'

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of SGC.
        """
        self.conv1.reset_parameters()



