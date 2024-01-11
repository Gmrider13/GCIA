import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
import os

class Reddit12k(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['12k_reddit.npy', '12k_reddit.npy']
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return 'data.pt'

    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        with np.load(os.path.join(self.root,"12k_reddit.npz")) as loader:
            loader = dict(loader)
            adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                             loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                attr_matrix = None

            labels = loader.get('labels')
        split = np.load(os.path.join(self.root,"12k_reddit_split.npy"), allow_pickle=True).item()

        train_mask = torch.zeros(adj_matrix.shape[0], dtype=torch.bool)
        train_mask[split['train']]=True
        val_mask = torch.zeros(adj_matrix.shape[0], dtype=torch.bool)
        val_mask[split['val']]=True
        test_mask = torch.zeros(adj_matrix.shape[0], dtype=torch.bool)
        test_mask[split['test']]=True

        reddit_data = Data(x=torch.tensor(attr_matrix.todense()).float(),
                           edge_index=torch.LongTensor(adj_matrix.nonzero()),
                           y=torch.tensor(labels),
                           train_index=split['train'],
                           val_index = split['val'],
                           test_index=split['test'],
                           train_mask=train_mask,
                           val_mask = val_mask,
                           test_mask=test_mask
                           )

        data_list = [reddit_data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    reddit = Reddit12k("datasets/Reddit12k")