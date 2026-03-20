import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
import torch
import random
import torch_geometric
import copy
import time
from tqdm import tqdm
import warnings
import argparse
import sys
import os
import torch
import toponetx as tnx
from torch_geometric.utils import to_networkx
from topomodelx.utils.sparse import from_sparse
from model_topo import CellComplexOnline, CellComplexTarget, CellComplexLayer


def convert_graph_to_cell_complex(data):
    """将PyTorch Geometric的图数据转换为细胞复形数据"""
    # 1. 将PyG数据转换为NetworkX图
    G = to_networkx(data, to_undirected=True)

    # 2. 创建细胞复形
    cell_complex = tnx.CellComplex(G)

    # 3. 提取邻域结构
    # 0阶邻接矩阵（节点-节点连接）
    adjacency_0 = cell_complex.adjacency_matrix(rank=0)
    adjacency_0 = torch.from_numpy(adjacency_0.todense()).to_sparse()

    # 下拉普拉斯矩阵
    down_laplacian = cell_complex.down_laplacian_matrix(rank=1)
    down_laplacian = from_sparse(down_laplacian)

    # 上拉普拉斯矩阵 - 修改此部分
    try:
        up_laplacian = cell_complex.up_laplacian_matrix(rank=1)
        up_laplacian = from_sparse(up_laplacian)
    except ValueError:
        # 创建稀疏零矩阵，而不是稠密矩阵
        edge_count = down_laplacian.shape[0]
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0)
        up_laplacian = torch.sparse.FloatTensor(indices, values, (edge_count, edge_count))

    # 4. 准备特征数据
    x_0 = data.x  # 节点特征

    # 边特征处理
    num_edges = cell_complex.shape[1]  # 边数量
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        x_1 = data.edge_attr
    else:
        x_1 = torch.ones(num_edges, 1)

    # 收集所有邻接结构
    adjacency_matrices = {
        'adjacency_0': adjacency_0,
        'down_laplacian': down_laplacian,
        'up_laplacian': up_laplacian
    }

    return cell_complex, x_0, x_1, adjacency_matrices


def get_cell_complex_dataset(data):
    """获取细胞复形数据集"""
    cell_complex, x_0, x_1, adjacency_matrices = convert_graph_to_cell_complex(data)

    # 创建包含所有必要数据的字典
    cell_complex_data = {
        'x_0': x_0,
        'x_1': x_1,
        'adjacency_0': adjacency_matrices['adjacency_0'],
        'down_laplacian': adjacency_matrices['down_laplacian'],
        'up_laplacian': adjacency_matrices['up_laplacian']
    }

    return cell_complex_data


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # 转换目标为one-hot向量
    one_hot_encoder = OneHotEncoder(categories='auto')
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # 归一化特征
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # 获取训练/验证/测试掩码
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # 创建自定义CV
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # 网格搜索
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    return accuracies


def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)


def train_online(online, optimizer, cell_complex_data):
    online.train()
    h, h_pred, h_target = online(cell_complex_data)
    loss = online.get_loss(h_pred, h_target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    online.update_target_encoder()
    return loss.item()


def train_target(target, optimizer, cell_complex_data):
    target.train()
    h_target = target(cell_complex_data)
    loss = target.get_loss(h_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def run(args):
    log_dir = args.log_dir
    path = args.data_dir

    with open(log_dir, 'a') as f:
        f.write(str(args))
        f.write('\n\n\n')

    trials = args.trials
    torch_geometric.seed.seed_everything(args.seed)
    device = torch.device('cpu')

    # 创建保存目录
    os.makedirs('pkl/pkl_online', exist_ok=True)
    os.makedirs('pkl/pkl_target', exist_ok=True)

    for trial in range(trials):
        # 设置超参数
        e1_lr = args.e1_lr
        e2_lr = args.e2_lr
        weight_decay = args.weight_decay
        hidden_dim = args.hidden_dim
        num_epochs = args.num_epochs
        momentum = args.momentum
        dataset_name = args.dataset_name

        # 加载数据集
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(args.data_dir)
        data = dataset[0]
        data = data.to(device)

        # 转换为细胞复形数据
        cell_complex_data = get_cell_complex_dataset(data)
        for key in cell_complex_data:
            if isinstance(cell_complex_data[key], torch.Tensor):
                cell_complex_data[key] = cell_complex_data[key].to(device)

        # 创建细胞复形模型
        model = CellComplexOnline(
            in_channels_0=data.x.size(1),
            hidden_dim=hidden_dim,
            momentum=momentum
        ).to(device)

        # 创建细胞复形Target模型
        target_cc_layer = CellComplexLayer(
            in_channels_0=data.x.size(1),
            in_channels_1=1,
            hidden_dim=hidden_dim
        ).to(device)
        target_model = CellComplexTarget(target_cc_layer).to(device)

        # 优化器
        online_optimizer = torch.optim.Adam(model.parameters(), lr=e1_lr, weight_decay=weight_decay)
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=e2_lr, weight_decay=weight_decay)

        # 记录最佳模型
        best_online_loss = float('inf')
        best_target_loss = float('inf')

        tag = f"{dataset_name}_{time.time()}"

        # 训练循环
        with tqdm(total=num_epochs, desc='训练') as pbar:
            for epoch in range(num_epochs):
                # 训练 Online 模型
                online_loss = train_online(model, online_optimizer, cell_complex_data)

                if online_loss < best_online_loss:
                    best_online_loss = online_loss
                    torch.save(model.state_dict(), f'pkl/pkl_online/best_online_{tag}.pkl')

                # 训练 Target 模型
                target_loss = train_target(target_model, target_optimizer, cell_complex_data)

                if target_loss < best_target_loss:
                    best_target_loss = target_loss
                    torch.save(target_model.state_dict(), f'pkl/pkl_target/best_target_{tag}.pkl')

                pbar.set_postfix({'loss': online_loss})
                pbar.update()

        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(f'pkl/pkl_online/best_online_{tag}.pkl'))
        model.eval()

        # 提取嵌入
        node_embeds, edge_embeds = model.embed(cell_complex_data)
        embeds = node_embeds + edge_embeds

        # 评估嵌入
        scores = fit_logistic_regression_preset_splits(
            embeds.detach().cpu().numpy(),
            data.y.cpu().numpy(),
            train_masks,
            val_masks,
            test_masks
        )
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        print(f"Trial {trial+1}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

        with open(log_dir, 'a') as f:
            f.write(f'Trial {trial+1} - Topological Cell Complex Mean: {mean_acc:.6f}, Std: {std_acc:.6f}\n')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('拓扑细胞复形神经网络 - Wiki数据集')
    parser.add_argument('--dataset_name', type=str, default='Wiki', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='../../datasets/Wiki', help='数据目录')
    parser.add_argument('--log_dir', type=str, default='./log/log_wiki_topo.txt', help='日志目录')
    parser.add_argument('--e1_lr', type=float, default=0.0001, help='online模型学习率')
    parser.add_argument('--e2_lr', type=float, default=0.0001, help='target模型学习率')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA动量')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=700, help='训练轮数')
    parser.add_argument('--seed', type=int, default=66666, help='随机种子')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='隐藏层维度')
    parser.add_argument('--trials', type=int, default=5, help='实验重复次数')

    args = parser.parse_args()
    run(args)