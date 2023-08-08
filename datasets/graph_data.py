# From https://github.com/divelab/DIG/blob/main/dig/xgraph/datasets/load_datasets.py

from ast import Num
import os
import yaml
import glob
import json
import random
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from rand5fold import load5foldData
from pathlib import Path


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(folder,prefix,data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)),0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, col = data.edge_index

    edge_slice=np.load(osp.join(folder, prefix + f'_{"edge_slice"}.npy')).tolist()

    edge_slice = torch.cumsum(torch.from_numpy(np.array(edge_slice)), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])
    
    # Edge indices should start at zero for every graph.

    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    slices['edges_num'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    slices['sentence_tokens'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):

    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    batch: np.array = read_file(folder, prefix, 'node_indicator') 
    x: torch.FloatTensor = torch.from_numpy(np.load(osp.join(folder, prefix + f'_{"X"}.npy')))
    # print(x.shape)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.int64).T
    
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)
    edge_attr = torch.ones((edge_index.size(1), 1)).float()
    name = torch.tensor(range(y.size(0)))
    supplement = dict()

    edges_num: torch.tensor = torch.from_numpy(np.load(osp.join(folder, prefix + f'_{"edge_slice"}.npy'))).unsqueeze(1)

    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens
    
    data = Data(name=name, x=x, edge_index=edge_index, y=y.reshape(-1, 1).float(), edges_num=edges_num.reshape(-1, 1).int(), sentence_tokens=list(sentence_tokens.values()))
    data, slices = split(folder, prefix, data, batch)
    return data, slices, supplement


def get_dataset(dataset_dir, dataset_name, task=None, transform=None):
    
    sentigraph_names = ['Graph_Weibo','Graph_MCFake']
    sentigraph_names = [name.lower() for name in sentigraph_names]
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

    if dataset_name.lower() in sentigraph_names:
        return load_SeniGraph(dataset_dir, dataset_name, transform=transform)
    else:
        raise NotImplementedError


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [ 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
              = read_sentigraph_data(self.raw_dir, self.name)
        
        if self.pre_filter is not None:
            
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])



class SentiGraphTransform(object):

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, data):
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        # integrate further transform
        if self.transform is not None:
            return self.transform(data)
        return data

def load_SeniGraph(dataset_dir, dataset_name, transform=None):
    sent_transform = SentiGraphTransform(transform)
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name, transform=sent_transform)
    return dataset


def get5fold(dataname, batch_size):

    fold0_x_test, fold0_x_train,\
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train,  \
    fold3_x_test, fold3_x_train,  \
    fold4_x_test, fold4_x_train = load5foldData(dataname)

    dataloader = dict()
    loader = dict()
    loader['train'] = DataLoader(fold0_x_train, batch_size=batch_size, shuffle=True)
    loader['test'] = DataLoader(fold0_x_test, batch_size=batch_size, shuffle=False)
    dataloader['split1'] = loader

    loader = dict()
    loader['train'] = DataLoader(fold1_x_train, batch_size=batch_size, shuffle=True)
    loader['test'] = DataLoader(fold1_x_test, batch_size=batch_size, shuffle=False)
    dataloader['split2'] = loader

    loader = dict()
    loader['train'] = DataLoader(fold2_x_train, batch_size=batch_size, shuffle=True)
    loader['test'] = DataLoader(fold2_x_test, batch_size=batch_size, shuffle=False)
    dataloader['split3'] = loader

    loader = dict()
    loader['train'] = DataLoader(fold3_x_train, batch_size=batch_size, shuffle=True)
    loader['test'] = DataLoader(fold3_x_test, batch_size=batch_size, shuffle=False)
    dataloader['split4'] = loader

    loader = dict()
    loader['train'] = DataLoader(fold4_x_train, batch_size=batch_size, shuffle=True)
    loader['test'] = DataLoader(fold4_x_test, batch_size=batch_size, shuffle=False)
    dataloader['split5'] = loader
    
    return dataloader, (fold0_x_train, fold0_x_test)
