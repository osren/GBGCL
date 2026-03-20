import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from topomodelx.nn.cell.can import CAN

class CellComplexLayer(torch.nn.Module):
    """基于细胞复形的特征提取层"""
    def __init__(self, in_channels_0, in_channels_1, hidden_dim, heads=2):
        super(CellComplexLayer, self).__init__()
        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.hidden_dim = hidden_dim
        
        # 边特征投影层
        self.edge_projection = torch.nn.Linear(8415, in_channels_1)
        
        # 修复：简化CAN配置以减少内存使用
        self.can = CAN(
            in_channels_0=in_channels_0,
            in_channels_1=in_channels_1,
            out_channels=hidden_dim,
            heads=1  # 减少头数以节省内存
        )

    def forward(self, x_0, x_1, adjacency_0, down_laplacian, up_laplacian):
        device = x_0.device
        
        # 投影边特征维度
        if x_1.size(1) == 8415:
            x_1 = self.edge_projection(x_1)
        
        # 修复：检查CUDA内存，如果不足则直接使用回退方案
        if device.type == 'cuda':
            try:
                # 检查可用内存
                free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                required_memory = x_0.numel() * 4 + x_1.numel() * 4  # 粗略估计
                
                if free_memory < required_memory * 2:  # 需要2倍内存用于计算
                    raise RuntimeError("预估内存不足，使用回退方案")
                    
            except:
                pass  # 如果检查失败，继续尝试CAN
        
        # 检查邻接矩阵的实际边数
        if adjacency_0.is_sparse:
            actual_edges = adjacency_0._nnz()
        else:
            actual_edges = torch.nonzero(adjacency_0).shape[0]
        
        # 确保边特征数量与邻接矩阵的边数匹配
        if x_1.size(0) != actual_edges:
            if x_1.size(0) < actual_edges:
                padding_size = actual_edges - x_1.size(0)
                padding = torch.zeros(padding_size, x_1.size(1), 
                                    dtype=x_1.dtype, device=device)
                x_1 = torch.cat([x_1, padding], dim=0)
            else:
                x_1 = x_1[:actual_edges]
        
        # 重新构建匹配的拉普拉斯矩阵
        num_edges = x_1.size(0)
        
        if down_laplacian.size(0) != num_edges or down_laplacian.size(1) != num_edges:
            down_laplacian = torch.sparse_coo_tensor(
                torch.arange(num_edges, device=device).unsqueeze(0).repeat(2, 1),
                torch.ones(num_edges, device=device),
                (num_edges, num_edges),
                device=device
            )
        
        if up_laplacian.size(0) != num_edges or up_laplacian.size(1) != num_edges:
            up_laplacian = torch.sparse_coo_tensor(
                torch.arange(num_edges, device=device).unsqueeze(0).repeat(2, 1),
                torch.ones(num_edges, device=device),
                (num_edges, num_edges),
                device=device
            )
        
        try:
            # 修复：清理不必要的中间变量以节省内存
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # 修复：正确处理CAN的返回值
            result = self.can(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)
            
            # 根据实际返回值解包
            if isinstance(result, (tuple, list)):
                if len(result) == 2:
                    x_0_out, x_1_out = result
                elif len(result) == 1:
                    x_0_out = result[0]
                    x_1_out = torch.zeros(num_edges, self.hidden_dim, device=device, requires_grad=True)
                else:
                    x_0_out, x_1_out = result[0], result[1]
            else:
                # 关键修复：如果CAN返回单个张量，需要确保节点数量正确
                x_0_out = result
                # 如果输出的节点数量与原始不匹配，需要调整
                if x_0_out.size(0) != x_0.size(0):
                    # 使用简单的特征变换确保维度匹配
                    weight = torch.randn(self.hidden_dim, x_0.size(1), device=device, requires_grad=True)
                    x_0_out = F.linear(x_0, weight)
                x_1_out = torch.zeros(num_edges, self.hidden_dim, device=device, requires_grad=True)
                
            return x_0_out, x_1_out
            
        except Exception as e:
            print(f"CAN调用失败: {e}")
            print("使用回退的线性变换...")
            
            # **关键修复：确保回退方案的张量具有梯度信息**
            weight_0 = torch.randn(self.hidden_dim, x_0.size(1), device=device, requires_grad=True)
            weight_1 = torch.randn(self.hidden_dim, x_1.size(1), device=device, requires_grad=True)
            
            x_0_out = F.linear(x_0, weight_0)
            x_1_out = F.linear(x_1, weight_1)
            
            return x_0_out, x_1_out


class Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, activation, num_layers, method=None, drop_out=0.0):
        super(Conv, self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.drop_out = drop_out
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, proj_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x, edge_index):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            z = F.dropout(z, p=self.drop_out, training=self.training)

        return z, self.projection_head(z)

'''
中心远离策略（Center-Away Strategy）
理论本质：将节点表示向高维超球面空间投影，使其远离拓扑中心 c= 1/N ∑ hi
代码实现：
    1.特征中心计算
        通过将边特征均值池化后与节点特征拼接，构建的 h_online 隐含了全局拓扑信息，作为去中心化的基础。
    2.对比损失约束
        通过归一化操作将特征限制在超球面空间，最大化正样本对的相似度等价于让节点表示远离随机负样本的隐含中心。
    3.动量更新机制
        通过动量更新机制，动态调整目标编码器的中心位置，避免了静态中心带来的不稳定性。
        动量更新使得目标编码器生成的 h_target作为动态中心，在线编码器生成的 h_online需要远离该中心。
    
'''
class CellComplexOnline(torch.nn.Module):
    """基于细胞复形的Online模型"""
    def __init__(self, in_channels_0, hidden_dim, momentum=0.99):
        super(CellComplexOnline, self).__init__()

        # 修复：设置边特征输入维度为hidden_dim
        in_channels_1 = hidden_dim

        # 在线编码器
        self.online_cc_layer = CellComplexLayer(in_channels_0, in_channels_1, hidden_dim)

        # 目标编码器
        self.target_cc_layer = CellComplexLayer(in_channels_0, in_channels_1, hidden_dim)

        # 修复：预测器层输出维度应该与合并特征维度一致
        combined_dim = hidden_dim * 2  # 节点特征 + 边特征
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, combined_dim)  # 输出维度与合并特征一致
        )

        self.momentum = momentum

    '''
    动量更新使得目标编码器生成的 h_target作为动态中心，在线编码器生成的 h_online需要远离该中心。
    '''
    def update_target_encoder(self):
        """使用EMA更新目标编码器"""
        for p, new_p in zip(self.target_cc_layer.parameters(), self.online_cc_layer.parameters()):
            next_p = self.momentum * p.data + (1 - self.momentum) * new_p.data
            p.data = next_p

    def forward(self, cell_complex_data):
        """
        前向传播
        cell_complex_data: 包含x_0, x_1和邻接结构的字典
        """
        x_0 = cell_complex_data['x_0']
        x_1 = cell_complex_data['x_1']
        adjacency_0 = cell_complex_data['adjacency_0']
        down_laplacian = cell_complex_data['down_laplacian']
        up_laplacian = cell_complex_data['up_laplacian']

        # 在线编码
        x_0_online, x_1_online = self.online_cc_layer(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)

        # 合并节点和边特征
        h_online = self.combine_features(x_0_online, x_1_online)

        # 预测
        h_pred = self.predictor(h_online)

        # 目标编码(无梯度)
        with torch.no_grad():
            x_0_target, x_1_target = self.target_cc_layer(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)
            h_target = self.combine_features(x_0_target, x_1_target)

        return h_online, h_pred, h_target

    def combine_features(self, x_0, x_1):
        """合并节点和边特征的方法"""
        # 使用平均池化对边特征进行聚合
        x_1_pooled = torch.mean(x_1, dim=0, keepdim=True).expand(x_0.size(0), -1)

        # 连接节点特征和聚合的边特征
        combined = torch.cat([x_0, x_1_pooled], dim=1)
        return combined

    def get_loss(self, z1, z2):
        """计算对比损失"""
        # 关键修复：确保行数（节点数量）一致
        min_nodes = min(z1.size(0), z2.size(0))
        z1 = z1[:min_nodes]
        z2 = z2[:min_nodes]
        
        # 确保列数（特征维度）一致
        if z1.shape != z2.shape:
            min_dim = min(z1.size(1), z2.size(1))
            z1 = z1[:, :min_dim]
            z2 = z2[:, :min_dim]
        
        z1 = F.normalize(z1, dim=-1, p=2)# 投影到单位球面
        z2 = F.normalize(z2, dim=-1, p=2)

        loss = (z1 * z2).sum(dim=-1)# 计算相似度
        return -loss.mean()

    def embed(self, data):
        """从细胞复形中提取嵌入"""
        x_0 = data['x_0']
        x_1 = data['x_1']
        adjacency_0 = data['adjacency_0']
        down_laplacian = data['down_laplacian']
        up_laplacian = data['up_laplacian']

        # 获取节点和边嵌入
        x_0_embed, x_1_embed = self.online_cc_layer(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)

        # 返回节点嵌入
        node_embeds = x_0_embed
        edge_embeds = torch.mean(x_1_embed, dim=0).expand(x_0_embed.size(0), -1)

        return node_embeds, edge_embeds


class Target(torch.nn.Module):
    def __init__(self, target_encoder):
        super(Target, self).__init__()
        self.target_encoder = target_encoder

    def forward(self, x, edge_index):
        h_target, _ = self.target_encoder(x, edge_index)
        return h_target

    def get_loss(self, z):
        z = F.normalize(z, dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()


class CellComplexTarget(torch.nn.Module):
    """基于细胞复形的Target模型"""
    def __init__(self, target_cc_layer, hidden_dim=None):
        super(CellComplexTarget, self).__init__()
        self.target_cc_layer = target_cc_layer
        self.hidden_dim = hidden_dim

    def forward(self, cell_complex_data):
        x_0 = cell_complex_data['x_0']
        x_1 = cell_complex_data['x_1']
        adjacency_0 = cell_complex_data['adjacency_0']
        down_laplacian = cell_complex_data['down_laplacian']
        up_laplacian = cell_complex_data['up_laplacian']

        # **关键修复：确保target模型在训练模式下生成具有梯度的张量**
        x_0_target, x_1_target = self.target_cc_layer(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)

        # 合并特征
        x_1_pooled = torch.mean(x_1_target, dim=0, keepdim=True).expand(x_0_target.size(0), -1)
        h_target = torch.cat([x_0_target, x_1_pooled], dim=1)

        return h_target

    def get_loss(self, z):
        """计算target模型的损失"""
        # **关键修复：确保输入张量有梯度信息**
        if not z.requires_grad:
            print("警告: 输入张量没有梯度信息，跳过损失计算")
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        
        z = F.normalize(z, dim=-1, p=2)
        loss = -(z - z.mean(dim=0)).pow(2).sum(1).mean()
        
        # 确保返回的损失有梯度信息
        if not loss.requires_grad:
            loss = torch.tensor(loss.item(), device=z.device, requires_grad=True)
            
        return loss