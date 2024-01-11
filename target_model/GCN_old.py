import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

class GCN(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(nfeat, nclass, bias=with_bias))
        else:
            self.layers.append(GCNConv(nfeat, nhid, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(nhid, nhid, bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GCNConv(nhid, nclass, bias=with_bias))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'GCN'

    def forward(self, x, edge_index, edge_weight=None):
        # x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
        return x

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def test(self,data):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = data.test_mask
        labels = data.y
        output = self.forward(data.x, data.edge_index)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = self.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def accuracy(selff, output, labels):
        """Return accuracy of output compared to labels.

        Parameters
        ----------
        output : torch.Tensor
            output from model
        labels : torch.Tensor or numpy.array
            node labels

        Returns
        -------
        float
            accuracy
        """
        if not hasattr(labels, '__len__'):
            labels = [labels]
        if type(labels) is not torch.Tensor:
            labels = torch.LongTensor(labels)
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def _ensure_contiguousness(self, x, edge_idx, edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

    def fit(self, data,train_iters, patience, verbose=False):

        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = data.y
        train_mask, val_mask = data.train_mask, data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index = data.x, data.edge_index
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
            print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val))
        self.load_state_dict(weights)

