# gb_utils.py
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix
from granular import Granular

@torch.no_grad()
def build_granules_and_rewrite(node_embed: torch.Tensor,
                               edge_index: torch.Tensor,
                               quity: str = "homo",
                               sim: str = "dot",
                               alpha_write: float = 0.5):
    """
    node_embed: [N, d] 当前节点表示（来自 online encoder 的 h 或 or_embeds）
    edge_index: [2, E] PyG 格式
    alpha_write: 回写强度，0~1。 0=全用球均值，1=保持原样
    return: z_new [N, d],  granule_sizes (list[int])
    """
    device = node_embed.device
    N = node_embed.size(0)

    # 1) PyG -> scipy csr
    adj_csr = to_scipy_sparse_matrix(edge_index, num_nodes=N)

    # 2) 构建粒球器，并注入嵌入（granular 里会用到 self.z_detached）
    gb = Granular(quity=quity, sim=sim)
    gb.z_detached = node_embed.detach().cpu()

    # 3) 生成粒球（得到：每个球的成员节点列表）
    GB_node_list, GB_graph_list, GB_center_list = gb.forward(adj_csr)

    # 4) 计算每个粒球的均值向量，并回写给其成员
    z_new = node_embed.clone()
    for members in GB_node_list:
        idx = torch.tensor(members, dtype=torch.long, device=device)
        if idx.numel() == 0:
            continue
        ball_mean = node_embed[idx].mean(dim=0, keepdim=True)
        z_new[idx] = alpha_write * node_embed[idx] + (1.0 - alpha_write) * ball_mean

    return z_new, [len(m) for m in GB_node_list]
