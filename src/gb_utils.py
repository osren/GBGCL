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


def build_ball_tensor(H_ball: torch.Tensor,
                      GB_node_list: List[List[int]],
                      num_nodes: int,
                      device: torch.device) -> torch.Tensor:
    """节点→所属球嵌入的查表张量（BTCM 用）。

    Args:
        H_ball: 球心嵌入 [B, ball_dim]
        GB_node_list: 每个球的成员节点索引 List[List[int]]
        num_nodes: 原图节点数 N（决定输出张量第一维）
        device: 目标设备

    Returns:
        ball_tensor: [N, ball_dim]，每个节点取其所在球的嵌入。
                     若该节点不在任何球中（GB_node_list 不覆盖），对应行=0。
                     若 H_ball 为空 / GB_node_list 为空，返回 None。
    """
    if H_ball is None or H_ball.numel() == 0 or len(GB_node_list) == 0:
        return None

    ball_dim = H_ball.size(1)
    node2ball = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    for bid, nlist in enumerate(GB_node_list):
        if len(nlist) > 0:
            idx = torch.as_tensor(nlist, dtype=torch.long, device=device)
            node2ball[idx] = bid

    valid = node2ball >= 0
    ball_tensor = torch.zeros(num_nodes, ball_dim, dtype=H_ball.dtype, device=device)
    ball_tensor[valid] = H_ball[node2ball[valid]]
    return ball_tensor


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
                              knn: int = 10,
                              use_ensemble: bool = False,
                              ensemble_quities: List[str] = None,
                              ensemble_temp: float = 1.0,
                              select: str = "hard"):
    """
    执行粒球扩散并回写节点表示。

    Args:
        use_ensemble: 是否启用多 quity 投票
        ensemble_quities: 要投票的 quity 列表
        ensemble_temp: 投票温度系数
        select: 'hard' 选择最佳；'soft' 软融合

    Returns:
        z_new: 节点新表示 [N, d]
        gb_sizes: 每个粒球大小 List[int]
        H_ball: 扩散后的球向量 [B, d]
        GB_node_list: 球成员索引 List[List[int]]
        selected_quity: 实际使用的 quity（用于日志）
    """
    device = node_embed.device

    # --- Ensemble 模式 ---
    if use_ensemble and ensemble_quities:
        best_quality, GB_node_list, weights_dict = build_granules_ensemble(
            node_embed, edge_index, ensemble_quities, sim, temp=ensemble_temp
        )
        # 记录选中的 quity 和权重
        for log_q, log_w in weights_dict.items():
            print(f"[Ensemble] quity={log_q}, weight={log_w:.4f}")
        selected_quity = best_quality
        print(f"[Ensemble] Selected: {selected_quity}")
    else:
        # 单 quity 模式
        GB_node_list, GB_center_list, GB_graph_list = build_granules(node_embed, edge_index, quity, sim)
        selected_quity = quity

    # 构建球图并扩散
    GB_center_list = []  # 简化：不使用预定义球心
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

    return z_new, [len(m) for m in GB_node_list], HK, GB_node_list, selected_quity


# =========================================================
# 2) 粒球质量评估（用于投票）
# =========================================================
@torch.no_grad()
def _evaluate_ball_quality(node_embed: torch.Tensor,
                          edge_index: torch.Tensor,
                          GB_node_list: List[List[int]],
                          quity: str) -> float:
    """
    评估粒球质量：基于球间分离度 + 球内紧凑度。

    分离度：不同球心之间的相似度（越低越好）
    紧凑度：球内节点到球心的距离（越低越好）
    score = 分离度的负值 - 紧凑度

    Returns:
        质量得分（越高越好）
    """
    device = node_embed.device
    B = len(GB_node_list)
    if B <= 1:
        return 0.0

    # 计算球心
    H0 = compute_ball_centers(node_embed, GB_node_list)
    if H0.numel() == 0:
        return 0.0

    Hn = torch.nn.functional.normalize(H0, dim=-1)

    # 1) 球间分离度（所有球心对相似度均值，越低越好）
    sim_center = torch.mm(Hn, Hn.t())  # [B, B]
    mask = ~torch.eye(B, dtype=torch.bool, device=device)
    if mask.any():
        separation = sim_center.masked_select(mask).mean().item()
    else:
        separation = 0.0

    # 2) 球内紧凑度（节点到球心的平均距离，越低越好）
    compactness = 0.0
    total_dist = 0.0
    total_nodes = 0
    for b, members in enumerate(GB_node_list):
        if not members or len(members) == 0:
            continue
        idx = torch.tensor(members, dtype=torch.long, device=device)
        member_embed = node_embed[idx]
        center_embed = H0[b:b+1]
        # L2 距离
        dist = torch.norm(member_embed - center_embed, dim=-1).sum()
        total_dist += dist.item()
        total_nodes += len(members)

    if total_nodes > 0:
        compactness = total_dist / total_nodes

    # 综合得分：分离度越低越好，compactness 越低越好
    # 归一化到合理范围
    quality_score = -separation - compactness * 0.1

    return quality_score


# =========================================================
# 2.5) 并行构建多种 quity 的粒球（投票Ensemble）
# =========================================================
def build_granules_ensemble(node_embed: torch.Tensor,
                           edge_index: torch.Tensor,
                           quities: List[str],
                           sim: str = "dot",
                           labels: torch.Tensor = None,
                           temp: float = 1.0) -> Tuple[str, List[List[int]], dict]:
    """
    并行构建多种 quity 的粒球，计算权重，返回最优结构。

    Args:
        node_embed: 节点嵌入 [N, d]
        edge_index: 图边索引 [2, E]
        quities: 要测试的 quity 列表，如 ['homo', 'detach', 'edges']
        sim: 相似度度量方式
        labels: 节点标签 [N]，用于 auto_quality 计算
        temp: 温度系数（用于 softmax 归一化）

    Returns:
        best_quity: 选中的最优 quity
        best_GB_node_list: 最优 quity 对应的球成员列表
        weights_dict: dict[quity] -> weight
    """
    results = {}
    for q in quities:
        # 构建粒球
        GB_nodes, GB_centers, GB_graphs = build_granules(node_embed, edge_index, q, sim, labels)
        # 评估质量
        quality_score = _evaluate_ball_quality(node_embed, edge_index, GB_nodes, q)
        results[q] = {
            'GB_node_list': GB_nodes,
            'GB_center_list': GB_centers,
            'GB_graph_list': GB_graphs,
            'quality_score': quality_score,
            'num_balls': len(GB_nodes)
        }

    # 计算 softmax 权重（基于质量得分）
    scores = torch.tensor([results[q]['quality_score'] for q in quities])
    weights = torch.softmax(scores / temp, dim=0).tolist()
    weights_dict = dict(zip(quities, weights))

    # 选择权重最高的 quity
    best_idx = weights.index(max(weights))
    best_quity = quities[best_idx]
    best_GB_node_list = results[best_quity]['GB_node_list']

    return best_quity, best_GB_node_list, weights_dict


# =========================================================
# 3) 构建球图并执行扩散
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


# =========================================================
# 4) 增量扩散（方案4）：不重建粒球结构，只更新球心并持续扩散
# =========================================================
@torch.no_grad()
def incremental_diffuse_and_write(node_embed: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  GB_node_list: List[List[int]],
                                  alpha_write: float = 0.5,
                                  beta: float = 0.2,
                                  K: int = 10,
                                  w_mode: str = "topo+center",
                                  knn: int = 10):
    """
    增量扩散：不重建粒球结构，只用当前嵌入更新球心并扩散。

    适用于每 epoch 都执行扩散的场景，避免频繁重建粒球的开销。

    Args:
        node_embed: 当前 epoch 的节点嵌入 [N, d]
        edge_index: 图边索引 [2, E]
        GB_node_list: 已有的粒球结构（成员列表），不重新构建
        alpha_write: 回写混合系数
        beta: 扩散系数
        K: 扩散步数
        w_mode: 权重模式
        knn: KNN 稀疏化

    Returns:
        z_new: 节点新表示 [N, d]
        H_ball: 扩散后的球心 [B, d]
    """
    device = node_embed.device
    B = len(GB_node_list)
    if B == 0:
        return node_embed.clone(), torch.empty(0, device=device)

    # 1. 用当前嵌入计算新球心
    H0 = compute_ball_centers(node_embed, GB_node_list)

    # 2. 构建球图（复用结构，更新中心相似度）
    #    拓扑权重从 edge_index 重新统计，中心相似度用当前嵌入
    Hn = torch.nn.functional.normalize(H0, dim=-1)
    sim_center = torch.mm(Hn, Hn.t())  # [-1, 1]

    # 跨球拓扑边统计（基于当前嵌入重新计算 node2ball 映射）
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

    # KNN 稀疏化
    if knn > 0 and B > knn:
        _, topk_idx = torch.topk(sim_center, k=min(knn + 1, B), dim=1)
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

    # 3. 球图扩散
    D = W.sum(dim=1, keepdim=True) + 1e-9
    P = W / D
    H = H0
    for _ in range(K):
        H = (1 - beta) * H + beta * (P @ H)

    # 4. 回写到节点
    z_new = node_embed.clone()
    for b, members in enumerate(GB_node_list):
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.long, device=device)
        z_new[idx] = alpha_write * node_embed[idx] + (1 - alpha_write) * H[b]

    return z_new, H
