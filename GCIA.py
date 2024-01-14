import copy
import argparse
import os

import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from torch_geometric.utils import k_hop_subgraph
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
import numpy as np
from target_model.gcn import GCN
from target_model.GraphSAGE import SAGE
from data_model_prepare import load_data, load_model
from torch_geometric.loader import NeighborLoader

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z

class MLP(torch.nn.Module):

    def __init__(self,num_i,num_h,num_o):
        super(MLP,self).__init__()

        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.LeakyReLU()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.relu2=torch.nn.LeakyReLU()
        self.linear3=torch.nn.Linear(num_h,num_o)
        self.dropout=0.5
        self.initial()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        x = self.relu2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear3(x)
        # return F.log_softmax(x,dim=1)
        return x

    def initial(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

class Encoder(torch.nn.Module):
    def __init__(self,num_features, hidden_dim, proj_dim,device):
        super(Encoder, self).__init__()
        self.encoder = GConv(input_dim=num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
        self.augmentor = [A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.3)]),
                          A.Compose([A.EdgeRemoving(pe=0.4), A.FeatureMasking(pf=0.4)])]
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)
        self.device = device
    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def fit(self, data,epoch):
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(self.device)
        optimizer = Adam(self.parameters(), lr=0.01,weight_decay=0.005)

        with tqdm(total=epoch, desc='(T)') as pbar:
            for epoch in range(1, epoch + 1):
                self.train()
                optimizer.zero_grad()
                z, z1, z2 = self.forward(data.x, data.edge_index.to(self.device), data.edge_attr)
                h1, h2 = [self.project(x) for x in [z1, z2]]
                loss = contrast_model(h1, h2)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

    def fit_minibatch(self,data,target_node_list, epoch):
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(self.device)
        optimizer = Adam(self.parameters(), lr=0.01)
        target_node_mask = torch.zeros([data.num_nodes], dtype=bool)
        target_node_mask[target_node_list] = True
        kwargs100 = {"batch_size": 500, "num_workers": 0, "persistent_workers": False}
        subgraph_sampler100 = NeighborLoader(
            data,
            input_nodes=target_node_mask,
            num_neighbors=[-1, -1],
            **kwargs100
        )
        with tqdm(total=epoch, desc='(T)') as pbar:
            for epoch in range(1, epoch + 1):
                all_loss = []
                for sub_graph in subgraph_sampler100:
                    sub_graph = sub_graph.to("cuda")
                    self.train()
                    optimizer.zero_grad()
                    z, z1, z2 = self.forward(
                        sub_graph.x,
                        sub_graph.edge_index,
                        sub_graph.edge_attr
                    )
                    h1, h2 = [
                        self.project(x)
                        for x in [z1, z2]
                    ]
                    loss = contrast_model(h1, h2)
                    loss.backward()
                    optimizer.step()
                    all_loss.append(loss.item())
                pbar.set_postfix({"loss": np.mean(all_loss)})
                pbar.update()

    def test(self, data):
        self.eval()
        z, _, _ = self(data.x, data.edge_index.to(self.device), data.edge_attr)
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
        result = LREvaluator()(z, data.y, split)
        return result

class GCL_attacker(torch.nn.Module):
    def __init__(self,ori_data,encoder,discrete_feat,victim,device):
        super(GCL_attacker, self).__init__()

        self.encoder = encoder

        self.discrete_feat = discrete_feat

        if discrete_feat:
            self.feature_mean = torch.mean(ori_data.ori_x.sum(dim=1)).item()
            self.feature_std = torch.std(ori_data.ori_x.sum(dim=1)).item()
            self.feature_max = torch.max(ori_data.ori_x.sum(dim=1)).item()
            self.feature_min = torch.min(ori_data.ori_x.sum(dim=1)).item()
        else:
            self.feature_mean = ori_data.ori_x.mean(dim=0)
            self.feature_std = ori_data.ori_x.std(dim=0)
            self.feature_max,_ = ori_data.ori_x.max(dim=0)
            self.feature_min,_ = ori_data.ori_x.min(dim=0)

        self.ori_data = ori_data
        self.device = device
        self.n_node = ori_data.x.shape[0]
        self.n_feat = ori_data.x.shape[1]
        self.n_class = data.y.max().item() + 1
        self.victim = victim
        self.MLP = MLP(num_i=32, num_h=64, num_o=self.n_class)

    def get_pgd_attack(self,epoch):

        fake_feature = []
        total_e = []
        for target_node_index in tqdm(range(len(self.target_node_list))):
            target_node = self.target_node_list[target_node_index]

            if self.discrete_feat:
                fea_index = np.random.choice(self.n_feat, int(0.3*self.feature_mean),
                                             replace=False)
                fake_x = torch.zeros([1, self.n_feat])
                fake_x[0, fea_index] = 1

            else:
                fake_x = torch.normal(mean=self.feature_mean, std=0.1).reshape([1, -1])
                fake_x=torch.clamp(fake_x,self.feature_min,self.feature_max)

            fake_x = fake_x.to(self.device)
            fake_x.requires_grad = True
            fake_x_norm = F.normalize(fake_x)

            if self.discrete_feat:
                one_index = fake_x.view(-1).nonzero().view(-1)
                one_index = one_index.cpu().numpy()
                zero_index = np.setdiff1d(np.arange(self.n_feat),one_index)

            modified_x_index, modified_edge, target_node_subgraph, _ = k_hop_subgraph(target_node,2,self.ori_data.edge_index,relabel_nodes=True)
            ori_modified_x = self.ori_data.x[modified_x_index]
            modified_x_ini = copy.deepcopy(ori_modified_x.detach())
            modified_x_ini.requires_grad=True

            h_ori = self.encoder.encoder(modified_x_ini, modified_edge.to(self.device), None)[target_node_subgraph,:].reshape([1,-1])

            modified_x = torch.cat([modified_x_ini,fake_x_norm],dim=0)
            fake_edge = torch.tensor([[target_node_subgraph,modified_x.shape[0]-1],[modified_x.shape[0]-1, target_node_subgraph]]).to(self.device)
            modified_edge = torch.cat([modified_edge, fake_edge],dim=1)

            for grad_e in range(epoch):

                h_adv = self.encoder.encoder(modified_x, modified_edge.to(self.device), None)[target_node_subgraph, :].reshape([1, -1])
                diff_adv =F.normalize(self.encoder.project(h_ori)) @ F.normalize(self.encoder.project(h_adv)).T
                MLP_pre = F.softmax(self.MLP(h_adv),dim=1)
                MLP_loss = F.nll_loss(MLP_pre, self.ori_data.y[[target_node]])

                loss = -diff_adv+MLP_loss

                if len(self.fake_nodes[target_node])>=2:
                    self_diff = 0
                    for i in range(1,len(self.fake_nodes[target_node])):
                        self_emb = torch.from_numpy(self.fake_nodes[target_node][i]['h_adv']).to(self.device)
                        tmp_diff = F.normalize(self.encoder.project(self_emb)) @ F.normalize(self.encoder.project(h_adv)).T
                        tmp_diff = torch.clamp(tmp_diff, min=0)
                        self_diff = self_diff + tmp_diff
                    loss = loss - self_diff

                grad_all = torch.autograd.grad(loss, fake_x)[0]
                adv_grad = grad_all.detach()[-1, :].reshape([1,-1])
                lr = 200 / np.sqrt(grad_e+1)
                grad = lr * adv_grad

                with torch.no_grad():
                    if self.discrete_feat:
                        zero_value = grad[:,zero_index]
                        choosed_index0=zero_value.argmax()
                        choosed_value0 =zero_value[0,choosed_index0]
                        choosed_zero = zero_index[choosed_index0]
                        if len(one_index)==0:
                            zero_index=np.delete(zero_index,choosed_index0.item())
                            one_index=np.append(one_index,choosed_zero)
                        else:
                            one_value = grad[:,one_index]
                            choosed_index1=one_value.argmin()
                            choosed_value1 =-one_value[0,choosed_index1]
                            choosed_one = one_index[choosed_index1]
                            assert fake_x[-1][choosed_zero] == 0
                            assert fake_x[-1][choosed_one] == 1

                            if choosed_value0 > choosed_value1:
                                zero_index=np.delete(zero_index,choosed_index0.item())
                                one_index=np.append(one_index,choosed_zero)
                            else:
                                one_index=np.delete(one_index,choosed_index1.item())
                                zero_index=np.append(zero_index,choosed_one)

                        fake_x = torch.zeros([1, self.n_feat])
                        fake_x[0, one_index] = 1
                        if len(one_index) >= self.feature_mean:
                            break
                    else:
                        fake_x += grad[0,:]
                        fake_x=torch.clamp(fake_x,self.feature_min,self.feature_max)

                fake_x = fake_x.to(self.device)
                fake_x = fake_x.detach()
                fake_x.requires_grad = True
                fake_x_norm = F.normalize(fake_x)

                modified_x = torch.cat([modified_x_ini,fake_x_norm],dim=0)

            self.fake_nodes[target_node].append({
                "diff": diff_adv.item(),
                "fake_x": fake_x.cpu().detach().numpy(),
                'h_adv': h_adv.cpu().detach().numpy()})

            if self.discrete_feat:
                fake_feature.append(torch.sum(fake_x).item())
            else:
                fake_feature.append(fake_x)
        #check if the fake node
        if self.discrete_feat:
            print(np.mean(fake_feature))
            print(self.feature_mean)
        else:
            fake_xs = torch.cat(fake_feature,dim=0)
            fake_max,_ = fake_xs.max(dim=0)
            fake_min,_ = fake_xs.min(dim=0)
            print(((self.feature_min-fake_min)<=0).all())
            print(((self.feature_max-fake_max)>=0).all())
    def attack_initial(self):
        fake_nodes = {}

        ori_pre = self.victim(self.ori_data.x, self.ori_data.edge_index)
        self.target_pre = {target_node:ori_pre[target_node][self.ori_data.y[target_node]].cpu().detach().numpy() for target_node in target_node_list}
        self.target_pre_all = {target_node:ori_pre[target_node].cpu().detach().numpy() for target_node in target_node_list}

        self.target_node_list = []
        for target_node in self.ori_target_node_list:
            if np.argmax(self.target_pre_all[target_node]) == self.ori_data.y[target_node]:
                self.target_node_list.append(target_node)

        print("initial mis rate:",(1-len(self.target_node_list)/len(self.ori_target_node_list)))

        for target_node in self.target_node_list:

            target_x_index, target_edge, target_node_subgraph, _ = k_hop_subgraph(target_node,2,self.ori_data.edge_index,relabel_nodes=True)
            ori_targey_x = self.ori_data.x[target_x_index]

            h_ori = self.encoder.encoder(ori_targey_x, target_edge.to(self.device), None)[target_node_subgraph,:].reshape([1,-1])

            if self.discrete_feat:
                fake_feature_n = self.feature_mean
                fea_index = np.random.choice(self.n_feat, int(fake_feature_n),
                                             replace=False)
                fake_x = torch.zeros([1, self.n_feat])
                fake_x[0, fea_index] = 1
            else:
                fake_x = torch.normal(mean=self.feature_mean, std=1).reshape([1,-1])
                fake_x=torch.clamp(fake_x,self.feature_min,self.feature_max)

            fake_nodes[target_node]=[{
                "fake_x": fake_x.cpu().numpy(),
                "h_adv":h_ori.cpu().detach().numpy(),
                "reward":self.target_pre_all[target_node]
            }]
        return fake_nodes
    def get_modified_graph(self, fake_x, target_node):
        modified_graph = copy.deepcopy(self.ori_data)
        fake_x_tensor = torch.from_numpy(fake_x).to(self.device).reshape([1,-1])
        modified_graph.x = torch.cat([modified_graph.x, fake_x_tensor], dim=0)
        modified_graph.x.requires_grad=True
        fake_edge = torch.tensor([[target_node, modified_graph.x.shape[0] - 1], [modified_graph.x.shape[0] - 1, target_node]]).to(self.device)
        modified_graph.edge_index = torch.cat([modified_graph.edge_index, fake_edge], dim=1)
        return modified_graph

    def query(self,e):
        query_acc = []
        for target_node in self.target_node_list:
            fake_x = self.fake_nodes[target_node][-1]['fake_x']
            modified_graph = self.get_modified_graph(fake_x,target_node)
            target_pre = self.victim(modified_graph.x, modified_graph.edge_index )[target_node]
            self.fake_nodes[target_node][-1]['reward']=target_pre.cpu().detach().numpy()
            self.fake_nodes[target_node][-1]['reward_targ'] = target_pre.cpu().detach().numpy()[self.ori_data.y[target_node]]
            self.fake_nodes[target_node][-1]['delt_r'] = self.target_pre[target_node] - target_pre.cpu().detach().numpy()[self.ori_data.y[target_node]]

            if torch.argmax(target_pre) == self.ori_data.y[target_node]:
                query_acc.append(target_node)

        self.target_node_list = query_acc
        print("GCL attack mr in epoch :",e,1-len(query_acc)/len(target_node_list))

    def train_MLP(self, epoch):
        patience = 50
        early_stopping = 100
        min_loss = 1000000
        self.MLP.initial()
        self.MLP.train()
        optimizer = Adam(self.MLP.parameters(), lr=0.01)

        with tqdm(total=epoch, desc='MLP training') as pbar:
            for e in range(epoch):
                delt_z = []
                delt_r = []
                delt_mask_index = []
                for target_node in self.fake_nodes.keys():
                    for item in self.fake_nodes[target_node]:
                        delt_z.append(torch.from_numpy(item['h_adv']).reshape(1, -1))
                        delt_r.append(torch.from_numpy(item['reward']).reshape(1, -1))
                        delt_mask_index.append(self.ori_data.y[target_node])

                delt_mask = torch.zeros([len(delt_z), self.n_class], dtype=bool)
                delt_mask[torch.arange(len(delt_z)), delt_mask_index] = True
                delt_mask = delt_mask.to(self.device)

                shuffel = torch.randperm(len(delt_z))

                delt_mask = delt_mask[shuffel, :]

                delt_z = torch.cat(delt_z, dim=0)
                delt_z = delt_z[shuffel, :]
                delt_z = delt_z.to(self.device)
                delt_z.requires_grade = True
                delt_z = F.normalize(delt_z, dim=1)

                delt_r = torch.cat(delt_r, dim=0)
                delt_r = delt_r[shuffel, :]
                delt_r = delt_r.to(self.device)
                delt_r.requires_grade = True

                optimizer.zero_grad()
                out = self.MLP(delt_z)
                out = F.log_softmax(out, dim=1)
                loss = F.mse_loss(out[delt_mask], delt_r[delt_mask])
                loss.backward()
                optimizer.step()
                added_loss = loss.item()
                torch.cuda.empty_cache()
                pbar.set_postfix({'loss': added_loss})
                pbar.update()
                if added_loss < min_loss:
                    min_loss = added_loss
                    weights = copy.deepcopy(self.MLP.state_dict())
                    patience = 50
                    best_epoch = e
                else:
                    patience -= 1

                if e > early_stopping and patience <= 0:
                    break

        print('=== early stopping at {0}, loss = {1} ==='.format(best_epoch, min_loss))
        self.MLP.load_state_dict(weights)
        torch.cuda.empty_cache()
        self.MLP.eval()

    def attack(self, target_node_list,k_hop):
        self.ori_target_node_list = copy.deepcopy(target_node_list)
        self.fake_nodes = self.attack_initial()

        for e in range(10):
            self.train_MLP(1000)
            self.get_pgd_attack(500)
            self.query(e)
            if len(self.target_node_list) == 0:
                break

    def _CWloss(self, output, labels):
        eye = torch.eye(self.ori_data.y.max() + 1).to(self.device)
        onehot_mx = eye[labels]
        onehot = onehot_mx.to(labels.device)
        best_second_class = (output - 1000 * onehot).argmax(1)
        margin = output[np.arange(len(output)), labels] - \
                 output[np.arange(len(output)), best_second_class]
        k = -0.3
        loss = -torch.clamp(margin, min=k).mean()
        return loss

def random_attack(data,target_node_list, victim,discrete_feat):

    atk_acc = []
    ori_miss = []
    if discrete_feat:
        feature_mean = torch.mean(data.x.sum(dim=1))
        feature_std = torch.std(data.x.sum(dim=1))
    else:
        feature_mean = data.x.mean(dim=0)
        feature_std = data.x.std(dim=0)
    for target_node in target_node_list:
        ori_pre = victim(data.x, data.edge_index.to('cuda'))[target_node]
        if np.argmax(ori_pre.cpu().detach().numpy()) != data.y[target_node]:
            ori_miss.append(target_node)

        for e in range(1):
            if discrete_feat:
                fea_index = np.random.choice(data.x.shape[1], int(50),
                                             replace=False)
                fake_x = torch.zeros([1, data.x.shape[1]])
                fake_x[0, fea_index] = 1
                fake_x = fake_x.to('cuda')
            else:
                fake_x = torch.normal(mean=feature_mean, std=feature_std).reshape([1,-1]).to('cuda')
            modified_x = copy.deepcopy(data.x)
            modified_edge = copy.deepcopy(data.edge_index)
            modified_x = torch.cat([modified_x,fake_x],dim=0)
            fake_edge = torch.tensor([[0,modified_x.shape[0]-1],[modified_x.shape[0]-1, 0]]).to('cuda')
            modified_edge = torch.cat([modified_edge, fake_edge],dim=1)

            target_pre = victim(modified_x, modified_edge.to('cuda'))[target_node]
            if np.argmax(target_pre.cpu().detach().numpy()) != data.y[target_node]:
                atk_acc.append(target_node)
                break
    print(len(target_node_list))
    print("ori miss:",len(ori_miss)/len(target_node_list))
    print("attack mr:",len(atk_acc)/len(target_node_list))

def train_classifier(data, model_name='SAGE'):
    if model_name == 'GCN':
        model = GCN(nfeat=data.x.shape[1],
                    nhid=16,
                    nclass=data.y.max().item() + 1,
                    dropout=0.5, device='cuda')
        model = model.to('cuda')
        model.fit(data,200,30, verbose=True) # train with earlystopping
        model.test(data)
    else:
        model = SAGE(nfeat=data.x.shape[1],
                     nhid=32,
                     nclass=data.y.max().item() + 1,
                     dropout=0.5, device='cuda')
        model = model.to('cuda')
        model.fit(data, 1000, 50, verbose=True)  # train with earlystopping
        model.test()
    return model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Reddit', help='dataset to attack')
    parser.add_argument('--model_name', type=str, default='SAGE', help='dataset to attack')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    data_name = args.dataset
    device = torch.device('cuda:0')
    if args.dataset in ["Cora","Citeseer"]:
        discrete_feat = True
    else:
        discrete_feat = False

    data = load_data(args.dataset,norm=True)
    data = data.to(device)
    feat_sum_before = torch.sum(data.x,dim=1)
    no_zero_before = data.x[0,:][data.x[0,:].nonzero()]
    feat_sum_after = torch.sum(data.x,dim=1)
    no_zero_after = data.x[0,:][data.x[0,:].nonzero()]
    test_nodes = torch.tensor([i for i in range(len(data.test_mask))]).to(device)[data.test_mask]
    k_hop = 2

    GCN_classifier = load_model(args.dataset, args.model_name,norm=True)
    GCN_classifier = GCN_classifier.to(device)
    GCN_classifier.eval()

    target_node_list = test_nodes.cpu().detach().tolist()


    GCL_model = Encoder(num_features=data.x.shape[1], hidden_dim=32, proj_dim=32,device=device)
    GCL_model_dir = osp.join('saved_GCL_model', '{}.pth'.format(args.dataset))
    if osp.exists(GCL_model_dir):
        GCL_model.load_state_dict(torch.load(GCL_model_dir))
        GCL_model=GCL_model.to(device)
    else:
        GCL_model=GCL_model.to(device)
        GCL_model.fit_minibatch(data,target_node_list,1000)
        if not osp.exists('saved_GCL_model'):
            os.makedirs('saved_GCL_model')
        torch.save(GCL_model.state_dict(), GCL_model_dir)


    GCL_model.eval()
    attack_model = GCL_attacker(ori_data=data,encoder=GCL_model,discrete_feat=discrete_feat,victim=GCN_classifier,device=device).to(device)
    attack_model.attack(target_node_list,k_hop)
    # np.save(osp.join("GCL_fake_nodes","add_MLP{}-{}.npy".format(args.dataset,args.model_name)),attack_model.fake_nodes)

