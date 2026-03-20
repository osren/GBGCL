import glob
import numpy as np
import torch_geometric.transforms as T
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from data_topo import load_dataset, get_cell_complex_dataset
from model_topo import CellComplexOnline, CellComplexTarget, HierarchicalMessagePassing
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

def topological_contrastive_loss(h_pred, h_target, cell_data, alpha=0.3):
    """拓扑对比损失函数，包含TCM机制"""
    # 基础对比损失
    h_pred_norm = F.normalize(h_pred, dim=-1)
    h_target_norm = F.normalize(h_target, dim=-1)
    logits = torch.mm(h_pred_norm, h_target_norm.t()) / 0.1
    labels = torch.arange(h_pred.size(0)).to(h_pred.device)
    contrastive_loss = F.cross_entropy(logits, labels)
    
    # 拓扑一致性损失 (TCM)
    topo_loss = 0
    for dim in range(2, cell_data['dimension'] + 1):
        boundary_key = dim
        if boundary_key in cell_data['boundary_matrices']:
            boundary = cell_data['boundary_matrices'][boundary_key]
            
            # 使用边界操作符计算拓扑特征
            if boundary.is_sparse:
                online_topo = torch.sparse.mm(boundary, h_pred)
                target_topo = torch.sparse.mm(boundary, h_target)
            else:
                online_topo = torch.mm(boundary, h_pred)
                target_topo = torch.mm(boundary, h_target)
            
            # 计算拓扑特征差异
            topo_loss += F.mse_loss(online_topo, target_topo.detach())
    
    return contrastive_loss + alpha * topo_loss

def train_online(online, optimizer, cell_complex_data, accumulation_steps=4):
    online.train()
    total_loss = 0
    
    # 多次累积梯度
    for i in range(accumulation_steps):
        h_online, h_pred, h_target = online(cell_complex_data)
        
        # 使用改进的拓扑对比损失
        loss = topological_contrastive_loss(h_pred, h_target, cell_complex_data) / accumulation_steps
        
        # 反向传播
        loss.backward()
        total_loss += loss.item()
        
        # 累积梯度后更新
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    return total_loss
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


def train_target(target, optimizer, cell_complex_data):
    target.train()
    h_target = target(cell_complex_data)
    loss = target.get_loss(h_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"显存使用: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.max_memory_allocated()/1024**2:.2f}MB")


def load_best_model(model, tag, model_type='online'):
    """
    加载最佳模型，支持带epoch后缀的文件名
    
    Args:
        model: 要加载的模型
        tag: 文件标签
        model_type: 'online' 或 'target'
    
    Returns:
        bool: 是否成功加载
    """
    try:
        # 构建文件模式
        if model_type == 'online':
            pattern = f'pkl/pkl_online/best_online_{tag}*.pkl'
        else:
            pattern = f'pkl/pkl_target/best_target_{tag}*.pkl'
        
        # 查找匹配的文件
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # 选择最新的文件（按修改时间排序）
            best_file = max(matching_files, key=os.path.getmtime)
            model.load_state_dict(torch.load(best_file))
            print(f"成功加载{model_type}模型: {best_file}")
            return True
        else:
            print(f"警告: 找不到匹配的{model_type}模型文件 {pattern}")
            return False
            
    except Exception as e:
        print(f"加载{model_type}模型时出错: {e}")
        return False


def run(args):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        print("已设置显存使用上限为80%")
    
    log_dir = args.log_dir
    path = args.data_dir

    with open(log_dir, 'a') as f:
        f.write(str(args))
        f.write('\n\n\n')

    trials = args.trials
    torch_geometric.seed.seed_everything(args.seed)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("使用CPU")

    # 创建保存目录
    os.makedirs('pkl/pkl_online', exist_ok=True)
    os.makedirs('pkl/pkl_target', exist_ok=True)

    for trial in range(trials):
        print(f"\n开始第 {trial+1}/{trials} 轮实验")
        
        # 加载数据集
        dataset = load_dataset(args.dataset_name, path)
        data = dataset[0].to(device)
        
        # 转换为细胞复形数据 (支持k维)
        cell_complex_data = get_cell_complex_dataset(
            data,
            directed=args.directed,
            max_dim=args.max_dim,
            max_cycles=args.max_cycles,
            max_cycle_size=args.max_cycle_size
        )
        
        # 计算各维度的输入通道数
        dim_channels = []
        for dim in range(cell_complex_data['dimension'] + 1):
            if dim in cell_complex_data['x_features']:
                dim_channels.append(cell_complex_data['x_features'][dim].size(1))
            else:
                dim_channels.append(1)  # 默认维度
        
        # 将数据转移到设备
        for key in ['x_features', 'adjacency_matrices', 'boundary_matrices']:
            if key in cell_complex_data:
                for k, v in cell_complex_data[key].items():
                    if isinstance(v, torch.Tensor):
                        cell_complex_data[key][k] = v.to(device)
        
        # 创建模型
        model = CellComplexOnline(
            dim_channels=dim_channels,
            hidden_dim=args.hidden_dim,
            momentum=args.momentum
        ).to(device)
        
        # 创建Target模型
        target_model = CellComplexTarget(
            model.target_encoder,
            hidden_dim=args.hidden_dim
        ).to(device)

        # 优化器
        online_optimizer = torch.optim.Adam(model.parameters(), lr=args.e1_lr, weight_decay=args.weight_decay)
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=args.e2_lr, weight_decay=args.weight_decay)

        # 训练循环
        best_online_loss = float('inf')
        best_target_loss = float('inf')
        tag = f"{args.dataset_name}_{time.time()}"
        
        with tqdm(total=args.num_epochs, desc='训练') as pbar:
            for epoch in range(args.num_epochs):
                # 训练Online模型
                online_loss = train_online(model, online_optimizer, cell_complex_data)
                
                # 训练Target模型
                target_loss = train_target(target_model, target_optimizer, cell_complex_data)
                
                # 更新EMA
                model.update_target_encoder()
                
                # 保存最佳模型
                if online_loss < best_online_loss:
                    best_online_loss = online_loss
                    torch.save(model.state_dict(), f'pkl/pkl_online/best_online_{tag}_{epoch}.pkl')
                
                if target_loss < best_target_loss:
                    best_target_loss = target_loss
                    torch.save(target_model.state_dict(), f'pkl/pkl_target/best_target_{tag}_{epoch}.pkl')
                
                pbar.set_postfix({'online_loss': online_loss, 'target_loss': target_loss})
                pbar.update()
                
                # 定期清理显存
                if epoch % 10 == 0:
                    print_gpu_memory()
                    torch.cuda.empty_cache()

        # 评估模型
        model_loaded = load_best_model(model, tag, 'online')
        if not model_loaded:
            print("使用当前训练完成的模型进行评估")
        
        model.eval()
        with torch.no_grad():
            embeds = model.embed(cell_complex_data).cpu().numpy()
        
        # 评估嵌入
        scores = fit_logistic_regression(embeds, data.y.cpu().numpy())
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        print(f"Trial {trial+1}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
        with open(log_dir, 'a') as f:
            f.write(f'Trial {trial+1} - Topological Cell Complex Mean: {mean_acc:.6f}, Std: {std_acc:.6f}\n')

    print(f"\n所有 {trials} 轮实验完成！")


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
    parser.add_argument('--directed', action='store_true', help='是否将图视为有向图处理')
    parser.add_argument('--max_cycles', type=int, default=5000, help='最大环数量')
    parser.add_argument('--max_cycle_size', type=int, default=4, help='最大环大小')
    parser.add_argument('--max_dim', type=int, default=3, help='最大细胞维度')
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU ID')
    
    args = parser.parse_args()
    run(args)