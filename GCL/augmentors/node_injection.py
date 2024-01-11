import copy

import torch
import numpy as np
from GCL.augmentors.augmentor import Graph, Augmentor


class NodeInjection(Augmentor):
    def __init__(self, feature_mean, feature_var):
        super(NodeInjection, self).__init__()
        self.feature_mean = feature_mean
        self.feature_var = feature_var
    def augment(self, g: Graph) -> Graph:
        print("error")

    def __call__(self, x, edge_index, edge_weight, target_index_list):
        #X中的每一种可能性在定义injection时直接传入
        #这里暂时只考虑X为01的问题
        x1 = copy.deepcopy(x)
        edge_index1 = copy.deepcopy(edge_index)
        for target_index in target_index_list:

            fake_x = torch.normal(mean=self.feature_mean, std=self.feature_var).reshape([1,-1])
            x1 = torch.cat([x1,fake_x],dim=0)

            #这里添加的是无向边
            fake_edge = torch.tensor([[target_index,x1.shape[0]-1],[x1.shape[0]-1, target_index]]).to(edge_index.device)
            edge_index1 = torch.cat([edge_index1, fake_edge],dim=1)
        return x1, edge_index1, edge_weight


class MyFeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(MyFeatureMasking, self).__init__()
        self.pf = pf
    def __call__(self, x, edge_index, edge_weight, target_index_list):
        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.pf
        drop_mask = drop_mask.to(device)
        # x = x.clone()
        x[target_index_list,:][:, drop_mask] = 0
        return x, edge_index, edge_weight
    def augment(self, g: Graph) -> Graph:
        print("error")





class MyCompose(Augmentor):
    def __init__(self, augmentors):
        super(MyCompose, self).__init__()
        self.augmentors = augmentors
    def augment(self, g: Graph) -> Graph:
        print("error")
    def __call__(self, x, edge_index, edge_weight, target_index_list):
        for aug in self.augmentors:
            x, edge_index, edge_weight = aug(x,edge_index, edge_weight, target_index_list)
        return x, edge_index, edge_weight
