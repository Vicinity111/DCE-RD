import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader

from datasets import graph_data


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state):
    multi_label = False
    if dataset_name == 'Graph_MCFake':
        dataset = graph_data.get_dataset(dataset_dir=data_dir, dataset_name='Graph_MCFake', task=None)
        dataloader, (train_set, test_set) = graph_data.get5fold(dataset_name, batch_size)
        print('[INFO] Using default splits!')
        test_set = dataset  # used for visualization

    elif dataset_name == 'Graph_Weibo':
        dataset = graph_data.get_dataset(dataset_dir=data_dir, dataset_name='Graph_Weibo', task=None)
        dataloader, (train_set, test_set) = graph_data.get5fold(dataset_name, batch_size)
        print('[INFO] Using default splits!')
        test_set = dataset  # used for visualization


    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    
    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label}
    
    return dataloader, test_set, x_dim, edge_attr_dim, num_class, aux_info

