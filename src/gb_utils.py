from typing import List, Tuple
import math
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from granular import Granular


# =========================================================
# 自适应质量函数选择 (Option A)
# =========================================================
def get_auto_quality(edge_index: torch.Tensor, labels: torch.Tensor = None) -> str:
    """Wrapper for Granular.auto_quality()"""
    return Granular.auto_quality(edge_index, labels)


# =========================================================
# 1) 粒球构建与球心计算
# =========================================================
@torch.no_grad()
def build_granules(node_embed: torch.Tensor,
                   edge_index: torch.Tensor,
                   quity: str = "auto",
                   sim: str = "dot",
                   labels: torch.Tensor = None) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    构建粒球（不回写），返回球成员、球心索引、球图结构。

    Args:
        node_embed: 节点嵌入 [N, d]（可在 GPU 上）
        edge_index: 图边索引 [2, E]（可能在 GPU 上）
        quity: 粒球划分方式 ('homo'/'detach'/'edges'/'auto')
              'auto' 时自动根据图结构选择
        sim: 相似度度量方式（'dot' / 'cos' / 'per'）
        labels: 节点标签 [N]，用于 auto_quality 计算同质率

    Returns:
        GB_node_list: List[List[int]]，每个球的成员节点列表
        GB_center_list: List[int]，球中心索引
        GB_graph_list: List[int]，球级图结构（来自 granular.forward 的返回）
    """
    # Auto mode: 根据图统计自动选择 quity
    if quity == "auto":
        quity = get_auto_quality(edge_index, labels)
        print(f"[Auto] Selected quity: {quity}")

    # to_scipy_sparse_matrix 需要 CPU 张量
    edge_index_cpu = edge_index.detach().cpu()
    adj_csr = to_scipy_sparse_matrix(edge_index_cpu, num_nodes=node_embed.size(0))

    gb = Granular(quity=quity, sim=sim)
    gb.z_detached = node_embed.detach().cpu()
    GB_node_list, GB_graph_list, GB_center_list = gb.forward(adj_csr)
    return GB_node_list, GB_center_list, GB_graph_list


def compute_ball_centers(node_embed: torch.Tensor,
                         GB_node_list: List[List[int]]) -> torch.Tensor:
    """
    GPU 向量化计算球心（成员节点嵌入均值）。

    Args:
        node_embed: [N, d] 节点嵌入（GPU/CPU 均可）
        GB_node_list: 每个球的成员节点索引

    Returns:
        H: [B, d] 球心向量
    """
    device = node_embed.device
    d = node_embed.size(1)
    B = len(GB_node_list)
    if B == 0:
        return torch.empty(0, d, device=device)

    member_idx_list, ball_idx_list = [], []
    for b, members in enumerate(GB_node_list):
        if len(members) == 0:
            continue
        member_idx_list.append(torch.tensor(members, dtype=torch.long, device=device))
        ball_idx_list.append(torch.full((len(members),), b, dtype=torch.long, device=device))

    if len(member_idx_list) == 0:
        return torch.zeros(B, d, device=device)

    member_idx = torch.cat(member_idx_list)
    ball_idx = torch.cat(ball_idx_list)

    H_sum = torch.zeros((B, d), dtype=node_embed.dtype, device=device)
    H_sum = H_sum.index_add_(0, ball_idx, node_embed[member_idx])

    counts = torch.zeros((B,), dtype=node_embed.dtype, device=device)
    counts = counts.index_add_(0, ball_idx, torch.ones_like(ball_idx, dtype=node_embed.dtype))
    counts = counts.clamp(min=1.0)

    return H_sum / counts.unsqueeze(1)


# 向后兼容旧接口名
def _compute_ball_centers(node_embed, GB_node_list):
    return compute_ball_centers(node_embed, GB_node_list)


# =========================================================
# 2) 构建球图并执行扩散
# =========================================================
def _build_ball_graph(GB_node_list: List[List[int]],
                      GB_center_list: List[int],
                      node_embed: torch.Tensor,
                      edge_index: torch.Tensor,
                      w_mode: str = "topo+center",
                      knn: int = 10) -> torch.Tensor:
    """
    构造球-球图邻接矩阵 W~。

    Args:
        GB_node_list: 每个球的成员节点
        GB_center_list: 球心节点索引（占位，不强依赖）
        node_embed: 节点嵌入
        edge_index: 原图边索引（与 node_embed 在同设备或可 .to ）
        w_mode: 权重融合模式 ['topo', 'center', 'topo+center']
        knn: KNN 稀疏化

    Returns:
        W: 球图邻接矩阵 [B, B]（含自环）
    """
    device = node_embed.device
    B = len(GB_node_list)
    if B == 0:
        return torch.empty(0, 0, device=device)

    # 球心相似度（cos）
    H0 = _compute_ball_centers(node_embed, GB_node_list)   # [B, d]
    Hn = torch.nn.functional.normalize(H0, dim=-1)
    sim_center = torch.mm(Hn, Hn.t())                      # [-1, 1]

    # 跨球拓扑边统计
    node2ball = torch.full((node_embed.size(0),), -1, dtype=torch.long, device=device)
    for b, members in enumerate(GB_node_list):
        if members:
            node2ball[torch.tensor(members, dtype=torch.long, device=device)] = b

    ei = edge_index.to(device)
    b_u, b_v = node2ball[ei[0]], node2ball[ei[1]]
    mask = (b_u >= 0) & (b_v >= 0) & (b_u != b_v)
    bu, bv = b_u[mask], b_v[mask]

    topo_w = torch.zeros(B, B, device=device)
    if mask.any():
        topo_w.index_put_((bu, bv), torch.ones_like(bu, dtype=topo_w.dtype), accumulate=True)
        topo_w.index_put_((bv, bu), torch.ones_like(bv, dtype=topo_w.dtype), accumulate=True)

    # KNN 稀疏化（按中心相似度）
    if knn > 0 and B > knn:
        _, topk_idx = torch.topk(sim_center, k=min(knn + 1, B), dim=1)  # 含自邻
        mask_knn = torch.zeros_like(sim_center, dtype=torch.bool)
        for i in range(B):
            mask_knn[i, topk_idx[i]] = True
        sim_center = torch.where(mask_knn, sim_center, torch.zeros_like(sim_center))

    # 权重融合
    if w_mode == "topo":
        W = topo_w
    elif w_mode == "center":
        W = torch.relu(sim_center)
    else:
        W = torch.relu(sim_center) + topo_w

    # 加自环
    W = W + torch.eye(B, device=device)
    return W


def _diffuse_on_ball_graph(H0: torch.Tensor,
                           W_tilde: torch.Tensor,
                           beta: float = 0.2,
                           K: int = 10) -> torch.Tensor:
    """
    球图上的 K 步扩散：H^{t+1} = (1 - β)H^t + β D^{-1}W H^t
    """
    if H0.numel() == 0:
        return H0
    D = W_tilde.sum(dim=1, keepdim=True) + 1e-9
    P = W_tilde / D
    H = H0
    for _ in range(K):
        H = (1 - beta) * H + beta * (P @ H)
    return H


@torch.no_grad()
def granule_diffuse_and_write(node_embed: torch.Tensor,
                              edge_index: torch.Tensor,
                              quity: str = "homo",
                              sim: str = "dot",
                              alpha_write: float = 0.5,
                              beta: float = 0.2,
                              K: int = 10,
                              w_mode: str = "topo+center",
                              knn: int = 10):
    """
    执行粒球扩散并回写节点表示。

    Returns:
        z_new: 节点新表示 [N, d]
        gb_sizes: 每个粒球大小 List[int]
        H_ball: 扩散后的球向量 [B, d]
        GB_node_list: 球成员索引 List[List[int]]
    """
    device = node_embed.device
    GB_node_list, GB_center_list, GB_graph_list = build_granules(node_embed, edge_index, quity, sim)
    H0 = _compute_ball_centers(node_embed, GB_node_list)
    Wt = _build_ball_graph(GB_node_list, GB_center_list, node_embed, edge_index, w_mode, knn)
    HK = _diffuse_on_ball_graph(H0, Wt, beta, K)

    # 回写到节点（残差式融合）
    z_new = node_embed.clone()
    for b, members in enumerate(GB_node_list):
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.long, device=device)
        z_new[idx] = alpha_write * node_embed[idx] + (1 - alpha_write) * HK[b]

    return z_new, [len(m) for m in GB_node_list], HK, GB_node_list


# =========================================================
# 3) 球级散射 + 匈牙利对齐 + 球级 InfoNCE
# =========================================================
def ball_scatter_loss(H_ball: torch.Tensor,
                      angle_thresh_deg: float = 15.0,
                      neighbor_mask: torch.Tensor = None,
                      tau_u: float = 0.1) -> torch.Tensor:
    """
    球级散射（RSM 升维版）+ 角度阈值控制。
    返回值越小越好。
    """
    if H_ball.numel() == 0:
        return torch.tensor(0.0, device=H_ball.device)

    Hb = torch.nn.functional.normalize(H_ball, dim=-1)
    sim = Hb @ Hb.t()
    B = Hb.size(0)

    mask = ~torch.eye(B, device=Hb.device, dtype=torch.bool)
    if neighbor_mask is not None:
        mask &= neighbor_mask

    # 角度阈值刹车
    angles = torch.acos(torch.clamp(sim, -1 + 1e-6, 1 - 1e-6)) * 180.0 / math.pi
    brake = torch.clamp(angles / max(1e-6, angle_thresh_deg), max=1.0)

    sep = (1.0 - sim) * brake
    sep = sep.masked_select(mask).mean() if mask.any() else torch.tensor(0.0, device=Hb.device)

    Hb2 = (Hb ** 2).sum(dim=1, keepdim=True)
    dist2 = Hb2 + Hb2.t() - 2.0 * (Hb @ Hb.t())
    uni = torch.exp(-2.0 * dist2 / max(1e-6, tau_u))
    uni = uni.masked_select(mask).mean() if mask.any() else torch.tensor(0.0, device=Hb.device)

    return sep + uni


def jaccard_between_balls(B1: List[List[int]], B2: List[List[int]]) -> torch.Tensor:
    """计算两组球成员的 Jaccard 相似度矩阵 [B1, B2]"""
    B, C = len(B1), len(B2)
    J = torch.zeros(B, C)
    sets1 = [set(s) for s in B1]
    sets2 = [set(s) for s in B2]
    for i in range(B):
        for j in range(C):
            inter = len(sets1[i] & sets2[j])
            union = len(sets1[i] | sets2[j])
            J[i, j] = 0.0 if union == 0 else inter / union
    return J


def hungarian_matching(sim_mat: torch.Tensor):
    """最大化 sim 的匈牙利匹配（-sim 为代价）→ 返回 List[(i,j)]"""
    from scipy.optimize import linear_sum_assignment
    cost = (-sim_mat).cpu().numpy()
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def ball_infonce(Ha: torch.Tensor, Hb: torch.Tensor,
                 pos_pairs,
                 temp: float = 0.2,
                 weak_pos_knn: int = 3) -> torch.Tensor:
    """
    球级 InfoNCE 对齐（正样=匹配对；分母含全体 Hb）。
    """
    if Ha.numel() == 0 or Hb.numel() == 0 or len(pos_pairs) == 0:
        return torch.tensor(0.0, device=Ha.device)

    Ha = torch.nn.functional.normalize(Ha, dim=-1)
    Hb = torch.nn.functional.normalize(Hb, dim=-1)
    Sa = Ha @ Hb.t()  # [Ba, Bb]

    loss_all = []
    for ia, ib in pos_pairs:
        pos = Sa[ia, ib] / temp
        denom = Sa[ia] / temp
        loss = - (pos - torch.logsumexp(denom, dim=0))
        loss_all.append(loss)

    return torch.stack(loss_all).mean()
