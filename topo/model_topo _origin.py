import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from topomodelx.nn.cell.can import CAN

class CellComplexLayer(torch.nn.Module):
    """基于细胞复形的特征提取层"""
    def __init__(self, in_channels_0, in_channels_1, hidden_dim, heads=2):
        super(CellComplexLayer, self).__init__()
        self.can = CAN(
            in_channels_0=in_channels_0,
            in_channels_1=in_channels_1,
            out_channels=hidden_dim,
            heads=heads
        )

    def forward(self, x_0, x_1, adjacency_0, down_laplacian, up_laplacian):
        # 对节点特征和边特征使用CAN进行处理
        x_0_out, x_1_out = self.can(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)
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

        # 假设边特征是一维的(如果没有边特征，我们会生成全1特征)
        in_channels_1 = 1

        # 在线编码器
        self.online_cc_layer = CellComplexLayer(in_channels_0, in_channels_1, hidden_dim)

        # 目标编码器
        self.target_cc_layer = CellComplexLayer(in_channels_0, in_channels_1, hidden_dim)

        # 预测器层
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),  # *2是因为我们连接节点和边特征
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
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
        return torch.cat([x_0, x_1_pooled], dim=1)

    def get_loss(self, z1, z2):
        """计算对比损失"""
        z1 = F.normalize(z1, dim=-1, p=2)# 投影到单位球面
        z2 = F.normalize(z2, dim=-1, p=2)

        loss = (z1 * z2).sum(dim=-1)# 计算相似度
        return -loss.mean()

    def embed(self, data):
        """从细胞复形中提取嵌入"""
        # 使用细胞复形数据进行前向传播
        x_0 = data['x_0']
        x_1 = data['x_1']
        adjacency_0 = data['adjacency_0']
        down_laplacian = data['down_laplacian']
        up_laplacian = data['up_laplacian']

        # 获取节点和边嵌入
        x_0_embed, x_1_embed = self.online_cc_layer(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)

        # 返回节点嵌入和边嵌入的汇总
        # 这里可以使用不同的汇总方式，如连接、平均等
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
    def __init__(self, target_cc_layer):
        super(CellComplexTarget, self).__init__()
        self.target_cc_layer = target_cc_layer

    def forward(self, cell_complex_data):
        x_0 = cell_complex_data['x_0']
        x_1 = cell_complex_data['x_1']
        adjacency_0 = cell_complex_data['adjacency_0']
        down_laplacian = cell_complex_data['down_laplacian']
        up_laplacian = cell_complex_data['up_laplacian']

        x_0_target, x_1_target = self.target_cc_layer(x_0, x_1, adjacency_0, down_laplacian, up_laplacian)

        # 合并特征
        x_1_pooled = torch.mean(x_1_target, dim=0, keepdim=True).expand(x_0_target.size(0), -1)
        h_target = torch.cat([x_0_target, x_1_pooled], dim=1)

        return h_target

    def get_loss(self, z):
        z = F.normalize(z, dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()