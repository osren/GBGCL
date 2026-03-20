import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork, PPI
import torch
import random
import torch_geometric
import copy
import time
from tqdm import tqdm
from torch_geometric.utils import train_test_split_edges
import warnings
import argparse
import sys
import os
import numpy as np
import torch
import toponetx as tnx
from torch_geometric.utils import to_networkx
from topomodelx.utils.sparse import from_sparse
from cwn.utils import convert_graph_dataset_with_rings


def convert_graph_to_cell_complex(data, directed=False,max_cycles=10000, max_cycle_size=5):
    """
    将PyTorch Geometric的图数据转换为细胞复形数据

    参数:
    data: PyTorch Geometric的Data对象

    返回:
    cell_complex: 细胞复形对象
    x_0: 节点特征
    x_1: 边特征 (如果原图没有边特征，则生成全1特征)
    adjacency_matrices: 包含各种拓扑结构矩阵的字典
    """
    '''
    拓扑均匀约束（Topological Uniformity Constraint）
    理论本质：通过拉普拉斯矩阵正则项 tr(H T LH) 保持拓扑结构在低维空间的分布均匀性
    代码实现：
        拓扑矩阵提取（data_topo.py）
        拓扑感知的消息传递（model_topo.py）
    '''
    # 1. 将PyG数据转换为NetworkX图
    G = to_networkx(data, to_undirected=(not directed))

    G.add_edges_from(data.edge_index.T.tolist())
    # 2. 寻找环 - 处理有向图的情况
    if directed:
        # 为了找环，创建一个无向版本
        undirected_G = G.to_undirected()
        all_cycles = nx.cycle_basis(undirected_G)
    else:
        all_cycles = nx.cycle_basis(G)

    # 过滤环：限制数量和大小
    filtered_cycles = [c for c in all_cycles if len(c) <= max_cycle_size][:max_cycles]
    print(f"找到 {len(all_cycles)} 个环")
    print(f"使用其中 {len(filtered_cycles)} 个环进行计算 (限制: 最大{max_cycles}个环, 最大大小{max_cycle_size})")

    # 2. 使用toponetx创建细胞复形
    print(G)
    cell_complex = tnx.CellComplex(G)
    cell_complex.add_cells_from(filtered_cycles, rank=2)
    print(cell_complex)
    # 3. 提取邻域结构
    # 0阶邻接矩阵（节点-节点连接）
    adjacency_0 = cell_complex.adjacency_matrix(rank=0)
    adjacency_0 = torch.from_numpy(adjacency_0.todense()).to_sparse()

    # 下拉普拉斯矩阵
    down_laplacian = cell_complex.down_laplacian_matrix(rank=1)
    down_laplacian = from_sparse(down_laplacian)

    # 上拉普拉斯矩阵
    try:
        up_laplacian = cell_complex.up_laplacian_matrix(rank=1)
        up_laplacian = from_sparse(up_laplacian)
    except ValueError:
        # # 如果上拉普拉斯矩阵不存在，创建零矩阵
        # up_laplacian = np.zeros((down_laplacian.shape[0], down_laplacian.shape[0]))
        # up_laplacian = torch.from_numpy(up_laplacian).to_sparse()
        edge_count = down_laplacian.shape[0]
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0)
        up_laplacian = torch.sparse.FloatTensor(indices, values, (edge_count, edge_count))

    # 4. 准备特征数据
    x_0 = data.x  # 节点特征

    # 如果没有边特征，创建全1特征
    num_edges = cell_complex.shape[1]  # 边数量
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        x_1 = data.edge_attr
    else:
        # 为每条边创建一个简单的全1特征
        x_1 = torch.ones(num_edges, 1)

    # 收集所有邻接结构
    adjacency_matrices = {
        'adjacency_0': adjacency_0,
        'down_laplacian': down_laplacian,
        'up_laplacian': up_laplacian
    }

    return cell_complex, x_0, x_1, adjacency_matrices


def get_cell_complex_dataset(data, directed=False,max_cycles=10000, max_cycle_size=5):
    """
    将数据集转换为细胞复形格式

    参数:
    data: PyTorch Geometric的Data对象

    返回:
    cell_complex_data: 包含细胞复形表示的字典
    """
    cell_complex, x_0, x_1, adjacency_matrices = convert_graph_to_cell_complex(data, directed,max_cycles=max_cycles, max_cycle_size=max_cycle_size)

    cell_complex_data = {
        'x_0': x_0,
        'x_1': x_1,
        'adjacency_0': adjacency_matrices['adjacency_0'],
        'down_laplacian': adjacency_matrices['down_laplacian'],
        'up_laplacian': adjacency_matrices['up_laplacian'],
        'y': data.y,
        'train_mask': data.train_mask if hasattr(data, 'train_mask') else None,
        'val_mask': data.val_mask if hasattr(data, 'val_mask') else None,
        'test_mask': data.test_mask if hasattr(data, 'test_mask') else None
    }

    return cell_complex_data


def load_dataset(dataset_name, dataset_dir):
    print('Dataloader: Loading Dataset', dataset_name)
    assert dataset_name in ['Cora', 'CiteSeer', 'PubMed',
                            'dblp', 'Photo','Computers',
                            'CS','Physics',
                            'ogbn-products', 'ogbn-arxiv', 'Wiki','ppi',
                           'Cornell', 'Texas', 'Wisconsin',
                           'chameleon', 'crocodile', 'squirrel']

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures())

    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name,
                               transform=T.NormalizeFeatures()
                              )

    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name,
                         transform=T.NormalizeFeatures())

    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name,
                           transform=T.NormalizeFeatures())

    elif dataset_name in ['Wiki']:
        dataset = WikiCS(dataset_dir,
                                         transform=T.NormalizeFeatures())
    elif dataset_name in ['ppi']:
        train = PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'train')
        val = PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'val')
        test = PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'test')
        dataset = [train, val, test]

    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
            return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
            return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())

    print('Dataloader: Loading success.')
    print(dataset[0])
    return dataset


def get_wiki_cs(root, transform=T.NormalizeFeatures()):
    from torch_geometric.utils import to_undirected

    dataset = WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)