import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge, mask_feature, scatter


class BallConv(torch.nn.Module):
    """GCN-style conv 球特征注入消息函数（BTCM）。

    消息：m_ij = [x_src || x_dst || ball_src || ball_dst]
    聚合：scatter_mean → BatchNorm → PReLU
    """

    def __init__(self, in_dim: int, out_dim: int, ball_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2 + ball_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, ball_feat: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m = torch.cat([x[src], x[dst], ball_feat[src], ball_feat[dst]], dim=-1)
        out = self.lin(m)
        out = scatter(out, dst, dim=0, dim_size=x.size(0), reduce='mean')
        return self.act(self.bn(out))


class Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, activation, num_layers, method=None, drop_out=0.0, btcm=False, ball_dim=0):
        super(Conv, self).__init__()
        self.activation = activation
        self.drop_out = drop_out
        self.btcm = btcm

        if btcm:
            assert ball_dim > 0, "ball_dim must be > 0 when btcm=True"
            self.layers = nn.ModuleList()
            self.layers.append(BallConv(input_dim, hidden_dim, ball_dim))
            for _ in range(num_layers - 1):
                self.layers.append(BallConv(hidden_dim, hidden_dim, ball_dim))
        else:
            self.layers = torch.nn.ModuleList()
            self.layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, proj_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x, edge_index, ball_feat=None):
        z = x
        for conv in self.layers:
            if self.btcm:
                assert ball_feat is not None, "BallConv requires ball_feat"
                z = conv(z, edge_index, ball_feat)
            else:
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

    def forward(self, x, edge_index, gb_feature=None, gb_ball_feat=None):
        """Option A v1: 在 hidden 层做特征融合；BTCM: 球特征注入消息函数

        Args:
            x: [N, d] 原始特征
            edge_index: [2, E] 边索引
            gb_feature: [N, hidden_dim] 粒球增强特征（已投影到 hidden 维）
            gb_ball_feat: [N, ball_dim] 球特征张量（BTCM 通路；为 None 时退化为 GCN）
        """
        or_embeds, pr_embeds = self.embed(x, edge_index, self.slsp_adj, self.num_hop, ball_feat=gb_ball_feat)
        h = or_embeds + pr_embeds

        # Option A v1: 在 hidden 层做特征融合（gb_feature 已经是 hidden 维）
        if gb_feature is not None:
            # 拼接 + 可学习融合
            h_combined = torch.cat([h, gb_feature], dim=-1)
            h_fused = self.gb_fusion(h_combined)
            h = h + h_fused  # 残差连接

        h_pred = self.predictor(h)
        with torch.no_grad():
            if self.online_encoder.btcm and gb_ball_feat is not None:
                h_target, _ = self.target_encoder(x, edge_index, ball_feat=gb_ball_feat)
            else:
                h_target, _ = self.target_encoder(x, edge_index)

        return h, h_pred, h_target

    def get_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        loss = (z1 * z2).sum(dim=-1)
        return -loss.mean()

    def embed(self, seq, edge_index, adj, Globalhop=10, ball_feat=None):
        if self.online_encoder.btcm and ball_feat is not None:
            h_1, _ = self.online_encoder(seq, edge_index, ball_feat=ball_feat)
        else:
            h_1, _ = self.online_encoder(seq, edge_index)
        h_2 = h_1.clone()
        for _ in range(Globalhop):
            h_2 = adj @ h_2
        return h_1, h_2


class Target(torch.nn.Module):
    def __init__(self,target_encoder):
        super(Target,self).__init__()
        self.target_encoder = target_encoder

    def forward(self, x, edge_index, ball_feat=None):
        if self.target_encoder.btcm and ball_feat is not None:
            h_target, _ = self.target_encoder(x, edge_index, ball_feat=ball_feat)
        else:
            h_target, _ = self.target_encoder(x, edge_index)
        return h_target

    def get_loss(self,z):
        z = F.normalize(z,dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()


