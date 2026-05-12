import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge,mask_feature

class Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, activation, num_layers, method=None, drop_out=0.0):
        super(Conv, self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.drop_out = drop_out

        # 输入维度：只在第一次调用时根据是否有gb特征动态决定
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

# Option A v2: 特征融合层（可学习拼接）
class GBFusion(torch.nn.Module):
    """可学习的粒球特征融合模块"""
    def __init__(self, hidden_dim, fusion_type='concat'):
        super(GBFusion, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # 拼接需要一个投影层
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif fusion_type == 'gate':
            # 门控机制
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )

    def forward(self, h_gcn, h_gb):
        """
        Args:
            h_gcn: [N, d] GCN 输出
            h_gb: [N, d] 粒球增强特征
        """
        if self.fusion_type == 'concat':
            # 拼接后投影
            h_combined = torch.cat([h_gcn, h_gb], dim=-1)
            h_fused = self.fusion_proj(h_combined)
            return h_gcn + h_fused  # 残差连接
        elif self.fusion_type == 'gate':
            # 门控融合
            h_combined = torch.cat([h_gcn, h_gb], dim=-1)
            gate = self.gate(h_combined)
            return gate * h_gb + (1 - gate) * h_gcn
        else:
            return h_gcn + h_gb  # 简单相加

class Online(torch.nn.Module):
    def __init__(self, online_encoder, target_encoder, hidden_dim, slsp_adj, num_hop, momentum):
        super(Online, self).__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.slsp_adj = slsp_adj
        self.num_hop = num_hop
        self.momentum = momentum
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        # Option A v1: 可学习的融合层
        self.gb_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def update_target_encoder(self):
        for p, new_p in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            next_p = self.momentum * p.data + (1 - self.momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, gb_feature=None):
        """Option A v1: 在 hidden 层做特征融合

        Args:
            x: [N, d] 原始特征（已有拼接的粒球特征的话是 2*d 维度）
            edge_index: [2, E] 边索引
            gb_feature: [N, d] 粒球增强特征
        """
        or_embeds, pr_embeds = self.embed(x, edge_index, self.slsp_adj, self.num_hop)
        h = or_embeds + pr_embeds

        # Option A v1: 在 hidden 层做特征融合
        if gb_feature is not None:
            # 将 GB 特征也通过 encoder 获取 hidden 表示
            h_gb, _ = self.online_encoder(gb_feature, edge_index)
            # 拼接 + 可学习融合
            h_combined = torch.cat([h, h_gb], dim=-1)
            h = self.gb_fusion(h_combined) + h  # 残差

        h_pred = self.predictor(h)
        with torch.no_grad():
            h_target, _ = self.target_encoder(x, edge_index)

        return h, h_pred, h_target

    def get_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        loss = (z1 * z2).sum(dim=-1)
        return -loss.mean()

    def embed(self, seq, edge_index, adj, Globalhop=10):
        h_1, _ = self.online_encoder(seq, edge_index)
        h_2 = h_1.clone()
        for _ in range(Globalhop):
            h_2 = adj @ h_2
        return h_1, h_2
    
    
class Target(torch.nn.Module):
    def __init__(self,target_encoder):
        super(Target,self).__init__()
        self.target_encoder = target_encoder

    def forward(self,x,edge_index):
        h_target,_ = self.target_encoder(x,edge_index)
        return h_target

    def get_loss(self,z):
        z = F.normalize(z,dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()


