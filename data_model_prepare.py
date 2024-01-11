import os

import torch
import os.path as osp
import torch.nn.functional as F
from datasets.reddit_12k import Reddit12k
from torch_geometric.datasets import Planetoid
from target_model.gcn import GCN
from target_model.GraphSAGE import SAGE
from target_model.gat import GAT
from target_model.appnp import APPNP
from target_model.sgc import SGC
import copy


class dgl_model(torch.nn.Module):
    def __init__(self, dataset_name,root,norm, model_name):
        super().__init__()
        self.pyg_model=load_model(dataset_name,model_name,root=root,norm=norm)
        self.pyg_model = self.pyg_model.to("cuda")
    def forward(self,g,feature):
        edge_index = torch.stack(g.edges())
        return self.pyg_model(feature.to("cuda"),edge_index.to("cuda"))

class dpr_model(torch.nn.Module):
    def __init__(self, dataset_name,root,norm, model_name):
        super().__init__()
        self.dpr_model=load_model(dataset_name,model_name,root=root,norm=norm)
        self.dpr_model = self.dpr_model.to("cuda")
    def forward(self,feature,adj,dropout = 0):
        edge_index = adj.to_dense().nonzero().T
        return self.dpr_model(feature.to("cuda"),edge_index.to("cuda"))



def get_classifier(data, model_name='SAGE'):
    if model_name == 'GCN':
        model = GCN(nfeat=data.x.shape[1],
                    nhid=64,
                    nclass=data.y.max().item() + 1,
                    dropout=0.5, device='cuda')

    elif model_name == 'SAGE':
        model = SAGE(nfeat=data.x.shape[1],
                     nhid=64,
                     nclass=data.y.max().item() + 1,
                     dropout=0.5, device='cuda')

    elif model_name == 'SGC':
        model = SGC(nfeat=data.x.shape[1],
                     nhid=64,
                     nclass=data.y.max().item() + 1,
                     device='cuda')

    elif model_name == 'GAT':
        model = GAT(nfeat=data.x.shape[1],
                     nhid=64,
                     nclass=data.y.max().item() + 1,
                     dropout=0.5, device='cuda')

    elif model_name == 'APPNP':
        model = APPNP(nfeat=data.x.shape[1],
                     nhid=64,
                     nclass=data.y.max().item() + 1,
                     dropout=0.5, device='cuda')
    else:
        print("error model name:",model_name)
    return model


def load_data(dataset,root="",norm=False):

    if dataset in ["Cora","Citeseer","PubMed"]:
        data = Planetoid(osp.join(root,'datasets'), name=dataset)
    else:
        data = Reddit12k(osp.join(root,'datasets','Reddit12k'))
    data= data[0]
    if norm:
        data.ori_x = copy.copy(data.x)
        data.x = F.normalize(data.x,dim=1)
    return data


def load_model(dataset, model_name,norm=False,root=""):
    import glob
    data = load_data(dataset,root)
    model = get_classifier(data,model_name)
    if norm:
        model.load_state_dict(torch.load(glob.glob(osp.join(root,'saved_models_norm','{}-{}-*.pth'.format(dataset,model_name)))[0]))
    else:
        model.load_state_dict(torch.load(glob.glob(osp.join(root,'saved_models','{}-{}-*.pth'.format(dataset,model_name)))[0]))

    return model
def load_dgl_data(dataset,root = "",norm=True):
    import dgl
    from torch_geometric.utils import to_networkx
    data=load_data(dataset,root,norm)
    graph = to_networkx(data)
    dgl_graph = dgl.from_networkx(graph)
    dgl_graph.ndata["feat"] = data.x
    dgl_graph.ndata["label"] = data.y
    dgl_graph.ndata["train_mask"] = data.train_mask
    dgl_graph.ndata["val_mask"] = data.val_mask
    dgl_graph.ndata["test_mask"] = data.test_mask
    # print(data.edge_index)
    # print(dgl_graph.edges())
    return dgl_graph
def train_classifier(norm=True):

    if not osp.exists(osp.join('saved_models_norm')):
        os.mkdir(osp.join('saved_models_norm'))
    if not osp.exists(osp.join('saved_models')):
        os.mkdir(osp.join('saved_models'))

    for dataset in ["Cora","Citeseer","PubMed","Reddit"]:
    # for dataset in ["Citeseer","Reddit"]:

        data = load_data(dataset,norm=norm)

        for model_name in ["GCN","SAGE","SGC","GAT","APPNP"]:
            model = get_classifier(data, model_name=model_name)
            model = model.to('cuda')
            model.fit(data,500,patience=100, verbose=True) # train with earlystopping
            acc = model.test()
            if norm:
                torch.save(model.state_dict(), osp.join('saved_models_norm','{}-{}-{:.3f}.pth'.format(dataset,model_name,acc)))
            else:
                torch.save(model.state_dict(), osp.join('saved_models','{}-{}-{:.3f}.pth'.format(dataset,model_name,acc)))

if __name__ == "__main__":
    train_classifier(True)