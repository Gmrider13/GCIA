import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn import SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from target_model.base_model import BaseModel
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
class SAGE(BaseModel):

    def __init__(self, nfeat, nhid, nclass, num_layers=2,
                 dropout=0.5, lr=0.01, weight_decay=0, device='cpu', with_bn=False, **kwargs):
        super(SAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            self_SAGEConv(nfeat, nhid))

        self.bns = nn.ModuleList()
        if 'nlayers' in kwargs:
            num_layers = kwargs['nlayers']
        self.bns.append(nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 2):
            self.convs.append(
                self_SAGEConv(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))

        self.convs.append(
            self_SAGEConv(nhid, nclass))

        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.activation = F.relu
        self.with_bn = with_bn
        self.device = device
        self.name = "SAGE"

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()

        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is not None:
                x = conv(x, adj)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_weight is not None:
            x = self.convs[-1](x, adj)
        else:
            x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
        # return x
class self_SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(self_SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        # Deleted the following line to make propagation differentiable
        # adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, feature, hidden, classes):
        super(GraphSAGE_NET, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def fit(self, pyg_data, train_iters=1000,patience=50, verbose=True):
        """early stopping based on the validation loss
        """
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = pyg_data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index = pyg_data.x, pyg_data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(x, edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = self.accuracy(output[val_mask], labels[val_mask])
            # print(acc)

            # if best_loss_val > loss_val:
            #     best_loss_val = loss_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())
            #     patience = early_stopping
            #     best_epoch = i
            # else:
            #     patience -= 1

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = copy.deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break

        if verbose:
             # print('=== early stopping at {0}, loss_val = {1} ==='.format(best_epoch, best_loss_val) )
            print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
        self.load_state_dict(weights)


if __name__ == '__main__':
    import os.path as osp
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    device = torch.device('cuda')
    path = osp.join('..','datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    model = SAGE(nfeat=data.x.shape[1],
                nhid=32,
                nclass=data.y.max().item() + 1,
                dropout=0.5, device='cuda')
    model = model.to('cuda')
    model.fit(data,train_iters=1000,initialize=True, patience=50, verbose=True) # train with earlystopping
    model.test()
