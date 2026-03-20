import glob

import numpy as np
import torch_geometric.transforms as T
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from data_topo import load_dataset, get_cell_complex_dataset
from model_topo import CellComplexOnline, CellComplexTarget, CellComplexLayer
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
torch.cuda.empty_cache()  # 清除未使用的缓存
def adj_norm(adj_t):
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t


def fit_logistic_regression(X, y, data_random_seed=1, repeat=3):
    one_hot_encoder = OneHotEncoder(categories='auto')
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    X = normalize(X, norm='l2')

    rng = np.random.RandomState(data_random_seed)
    accuracies = []

    for _ in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)

    return accuracies


# 在文档2中修改train_online和train_target函数
def train_online(online, optimizer, cell_complex_data, accumulation_steps=4):
    online.train()
    total_loss = 0
    for i in range(accumulation_steps):
        h, h_pred, h_target = online(cell_complex_data)
        loss = online.get_loss(h_pred, h_target.detach()) / accumulation_steps
        loss.backward()
        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    return total_loss
# def train_online(online, optimizer, cell_complex_data):
#     online.train()
#     h, h_pred, h_target = online(cell_complex_data)
#     loss = online.get_loss(h_pred, h_target.detach())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     online.update_target_encoder()
#     return loss.item()


def train_target(target, optimizer, cell_complex_data):
    target.train()
    h_target = target(cell_complex_data)
    loss = target.get_loss(h_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 在文档2的run函数开头添加
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"显存使用: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.max_memory_allocated()/1024**2:.2f}MB")
def run(args):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制为80%显存
        print("已设置显存使用上限为80%")
    log_dir = args.log_dir
    path = args.data_dir

    with open(log_dir, 'a') as f:
        f.write(str(args))
        f.write('\n\n\n')

    trials = args.trials
    torch_geometric.seed.seed_everything(args.seed)

    # 设置设备
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(args.gpu_id)
        print(f"使用GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

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
        dataset = load_dataset(dataset_name, path)
        data = dataset[0]
        data = data.to(device)

        # 转换为细胞复形数据
        # 在run函数中
        cell_complex_data = get_cell_complex_dataset(
            data,
            directed=args.directed,
            max_cycles=1000,  # 限制最大环数
            max_cycle_size=4  # 限制环大小
        )
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
                    # 删除旧的模型文件
                    for f in glob.glob(f'pkl/pkl_online/best_online_{tag}_*.pkl'):
                        try:
                            os.remove(f)
                        except:
                            pass
                    torch.save(model.state_dict(), f'pkl/pkl_online/best_online_{tag}_{epoch}.pkl')

                # 训练 Target 模型
                target_loss = train_target(target_model, target_optimizer, cell_complex_data)

                if target_loss < best_target_loss:
                    best_target_loss = target_loss
                    torch.save(target_model.state_dict(), f'pkl/pkl_target/best_target_{tag}.pkl')

                pbar.set_postfix({'loss': online_loss})
                pbar.update()
                # 定期清理显存
                if epoch % 10 == 0:
                    print_gpu_memory()
                    torch.cuda.empty_cache()

        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(f'pkl/pkl_online/best_online_{tag}.pkl'))
        model.eval()

        # 提取嵌入
        node_embeds, edge_embeds = model.embed(cell_complex_data)
        embeds = node_embeds + edge_embeds

        # 评估嵌入
        scores = fit_logistic_regression(embeds.detach().cpu().numpy(), data.y.cpu().numpy())
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        print(f"Trial {trial+1}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

        with open(log_dir, 'a') as f:
            f.write(f'Trial {trial+1} - Topological Cell Complex Mean: {mean_acc:.6f}, Std: {std_acc:.6f}\n')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('拓扑细胞复形神经网络')
    parser.add_argument('--dataset_name', type=str, default='Computers', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='../../datasets', help='数据目录')
    parser.add_argument('--log_dir', type=str, default='./log/log_computers_topo.txt', help='日志目录')
    parser.add_argument('--e1_lr', type=float, default=0.0001, help='online模型学习率')
    parser.add_argument('--e2_lr', type=float, default=0.0001, help='target模型学习率')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA动量')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=700, help='训练轮数')
    parser.add_argument('--seed', type=int, default=66666, help='随机种子')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='隐藏层维度')
    parser.add_argument('--trials', type=int, default=5, help='实验重复次数')
    # 在参数列表末尾添加
    parser.add_argument('--directed', action='store_true', help='是否将图视为有向图处理')
    # 在参数解析器中添加
    parser.add_argument('--max_cycles', type=int, default=5000, help='最大环数量')
    parser.add_argument('--max_cycle_size', type=int, default=4, help='最大环大小')
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU ID')
    args = parser.parse_args()
    run(args)