import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from topomodelx.nn.cell.can import CAN
import numpy as np

class KCellMessagePassing(nn.Module):
    """k维细胞消息传递层"""
    def __init__(self, in_channels, out_channels, k_dim):
        super().__init__()
        self.k_dim = k_dim
        self.linear = nn.Linear(in_channels, out_channels)
        self.attention = nn.Parameter(torch.Tensor(1, out_channels))
        nn.init.xavier_uniform_(self.attention)
        
    def forward(self, x, adjacency, boundary_down, boundary_up):
        """
        x: [num_cells, in_channels]
        adjacency: 同维度邻接矩阵
        boundary_down: 下边界矩阵
        boundary_up: 上边界矩阵
        """
        # 同维度消息传递
        x_trans = self.linear(x)
        
        # 稀疏矩阵乘法
        if adjacency.is_sparse:
            msg_same_dim = torch.sparse.mm(adjacency, x_trans)
        else:
            msg_same_dim = torch.mm(adjacency, x_trans)
        
        # 跨维度消息传递
        msg_down = torch.zeros_like(x_trans)
        msg_up = torch.zeros_like(x_trans)
        
        if boundary_down is not None:
            if boundary_down.is_sparse:
                lower_features = torch.sparse.mm(boundary_down, x_trans)
                msg_down = torch.sparse.mm(boundary_down.t(), lower_features)
            else:
                lower_features = torch.mm(boundary_down, x_trans)
                msg_down = torch.mm(boundary_down.t(), lower_features)
                
        if boundary_up is not None:
            if boundary_up.is_sparse:
                upper_features = torch.sparse.mm(boundary_up, x_trans)
                msg_up = torch.sparse.mm(boundary_up.t(), upper_features)
            else:
                upper_features = torch.mm(boundary_up, x_trans)
                msg_up = torch.mm(boundary_up.t(), upper_features)
        
        # 注意力机制融合
        combined = msg_same_dim + 0.5 * msg_down + 0.5 * msg_up
        att_weights = torch.sigmoid(torch.matmul(combined, self.attention.t()))
        return att_weights * combined

class HierarchicalMessagePassing(nn.Module):
    """层级消息传递网络"""
    def __init__(self, dim_channels, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.hidden_dim = hidden_dim
        
        # 为每个维度创建消息传递层
        for dim, in_channels in enumerate(dim_channels):
            if in_channels > 0:
                self.layers[f'dim_{dim}'] = KCellMessagePassing(
                    in_channels, hidden_dim, k_dim=dim
                )
    
    def forward(self, cell_data):
        outputs = {}
        max_dim = cell_data['dimension']
        
        # 逐维度处理
        for dim in range(max_dim + 1):
            dim_key = f'dim_{dim}'
            if dim_key in self.layers:
                # 获取当前维度的输入特征
                x = cell_data['x_features'][dim]
                
                # 获取邻接和边界矩阵
                adj = cell_data['adjacency_matrices'].get(dim, None)
                b_down = cell_data['boundary_matrices'].get(dim, None)
                b_up = cell_data['boundary_matrices'].get(dim + 1, None)
                
                # 消息传递
                outputs[dim_key] = self.layers[dim_key](x, adj, b_down, b_up)
        
        # 融合不同维度的表示
        if not outputs:
            return torch.zeros(cell_data['x_features'][0].size(0), self.hidden_dim).to(x.device)
        
        # 使用均值池化融合
        all_features = torch.stack(list(outputs.values()), dim=0)
        return all_features.mean(dim=0)

class CellComplexOnline(nn.Module):
    """基于细胞复形的Online模型，支持k维层级消息传递"""
    def __init__(self, dim_channels, hidden_dim, momentum=0.99):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.momentum = momentum
        
        # 在线编码器
        self.online_encoder = HierarchicalMessagePassing(dim_channels, hidden_dim)
        
        # 目标编码器
        self.target_encoder = HierarchicalMessagePassing(dim_channels, hidden_dim)
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 初始化目标编码器参数
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def update_target_encoder(self):
        """使用EMA更新目标编码器"""
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_o.data

    def forward(self, cell_complex_data):
        # 在线编码
        h_online = self.online_encoder(cell_complex_data)
        
        # 预测
        h_pred = self.predictor(h_online)
        
        # 目标编码(无梯度)
        with torch.no_grad():
            h_target = self.target_encoder(cell_complex_data)
        
        return h_online, h_pred, h_target

    def get_loss(self, h_pred, h_target):
        """计算对比损失"""
        # 确保维度一致
        if h_pred.shape != h_target.shape:
            min_size = min(h_pred.size(0), h_target.size(0))
            h_pred = h_pred[:min_size]
            h_target = h_target[:min_size]
            
        h_pred = F.normalize(h_pred, dim=-1, p=2)
        h_target = F.normalize(h_target, dim=-1, p=2)
        
        # 余弦相似度损失
        loss = - (h_pred * h_target).sum(dim=-1).mean()
        return loss

    def embed(self, data):
        """从细胞复形中提取嵌入"""
        with torch.no_grad():
            embeds = self.online_encoder(data)
        return embeds

class CellComplexTarget(nn.Module):
    """基于细胞复形的Target模型"""
    def __init__(self, target_encoder, hidden_dim=None):
        super().__init__()
        self.target_encoder = target_encoder
        self.hidden_dim = hidden_dim

    def forward(self, cell_complex_data):
        with torch.no_grad():
            h_target = self.target_encoder(cell_complex_data)
        return h_target

    def get_loss(self, z):
        """计算target模型的损失 (TCM机制)"""
        if not z.requires_grad:
            return torch.tensor(0.0, device=z.device)
        
        # 拓扑一致性损失 (Topological Consistency Mechanism)
        z = F.normalize(z, dim=-1, p=2)
        
        # 中心远离策略
        center = z.mean(dim=0, keepdim=True)
        distances = torch.norm(z - center, dim=1)
        
        # 最大化节点到中心的距离
        loss = -distances.mean()
        return loss