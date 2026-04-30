# granular.py  —— cleaned & self-contained
import math
import numpy as np
import torch
import networkx as nx
from scipy.stats import pearsonr

class Granular:
    def __init__(self, quity='homo', sim='dot'):
        self.labels = None
        self.adj_tensor = None
        self.center_cluster = None
        self.z_detached = None
        self.gb_q = []
        self.predsoft = None
        self.methods = dict(quity=quity, sim=sim)
        self.total_edges = 0
        self.init_num = 2  # will be reset per-graph

    # ===== 自适应质量函数选择 (Option A) =====
    @staticmethod
    def auto_quality(edge_index: torch.Tensor, labels: torch.Tensor = None) -> str:
        """
        根据图统计自动选择最优 quity。

        决策规则：
        - high homophily (>0.6) → 'homo': 同质性强，边两端多为同类节点
        - low homophily + high edge density → 'detach': 需要嵌入区分
        - noisy graph (many low-degree nodes) → 'edges': 边密度异常，存在噪声边

        Args:
            edge_index: [2, E] 边索引
            labels: [N] 节点标签（可选，用于计算 homophily）

        Returns:
            'homo' | 'detach' | 'edges'
        """
        num_nodes = edge_index.max().item() + 1
        num_edges = edge_index.size(1)

        # 1. 计算边密度
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        edge_density = num_edges / max(1, max_possible_edges)

        # 2. 计算同质率（如果提供了 labels）
        if labels is not None:
            labels = labels.cpu().numpy() if labels.device.type == 'cuda' else labels.numpy()
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            same_label = (labels[src] == labels[dst]).sum()
            homophily = same_label / max(1, num_edges)
        else:
            # 3. 无标签时，用图结构推断
            # 计算平均度
            degrees = np.bincount(edge_index[0].cpu().numpy(), minlength=num_nodes)
            avg_degree = degrees.mean()
            degree_std = degrees.std()

            # 低同质推断：度分布方差大（异质节点相连时度差异大）
            degree_cv = degree_std / max(1, avg_degree)  # 变异系数

            # 如果度分布偏斜（很多低度节点 = 可能噪声边）
            low_degree_ratio = (degrees < avg_degree * 0.5).sum() / max(1, num_nodes)

            # 决策
            if degree_cv > 0.8:
                # 度分布差异大，可能异质图
                return 'detach'
            elif low_degree_ratio > 0.3:
                # 大量低度节点，可能是噪声图
                return 'edges'
            else:
                # 默认用 detach
                return 'detach'

        # 3. 有标签时的决策
        if homophily > 0.6:
            # 高同质性，用 homo
            return 'homo'
        elif edge_density < 0.01:
            # 稀疏图，用 detach
            return 'detach'
        else:
            # 中等同质性，检查是否有噪声
            if low_degree_ratio > 0.3:
                return 'edges'
            else:
                return 'detach'

    # ===== 基础工具 =====
    def process_graph(self, adj_csr):
        """csr 邻接 -> nx.Graph（不删自环）"""
        return nx.from_scipy_sparse_array(adj_csr)

    def get_sub_adj_z(self, sub_g, cluster):
        node_indices = torch.tensor(cluster, dtype=torch.long)
        sub_adj = torch.tensor(nx.to_scipy_sparse_array(sub_g, format='csr').toarray())
        sub_z = self.z_detached[node_indices]
        return sub_adj, sub_z

    # ===== ���量度量 =====
    def quity(self, adj_s, z_detach):
        num_edges = torch.sum(adj_s) // 2
        x = torch.matmul(torch.t(z_detach).double(), adj_s.double().to(z_detach.device))
        x = torch.matmul(x, z_detach.double())
        x = torch.trace(x)
        degree_s = adj_s.sum(dim=1)
        y = torch.matmul(torch.t(z_detach).double(), degree_s.double().to(z_detach.device))
        y = (y ** 2).sum() / (2 * num_edges + 1e-9)
        return ((x - y) / (2 * num_edges + 1e-9))

    def quity_degree(self, adj_s):
        total_edges = adj_s.sum() // 2
        num_nodes = adj_s.shape[0]
        return total_edges / max(1, num_nodes)

    def quity_homo(self, adj_s, z_detach):
        sim_matrix = torch.mm(z_detach, z_detach.T)
        sim_matrix.fill_diagonal_(0)
        weighted = adj_s.to(sim_matrix.device) * sim_matrix
        total_similarity = weighted.sum()
        num_edges = adj_s.sum() / 2
        avg_homo = total_similarity / (num_edges + 1e-9)
        return avg_homo

    def quity_edges(self, adj_s):
        sub_edges = adj_s.sum() // 2
        degree_sum = adj_s.sum(dim=1).sum().item()
        m_exp = (degree_sum ** 2) / (4 * max(1, self.total_edges))
        return (sub_edges / max(1, self.total_edges)) - (m_exp / max(1, self.total_edges))

    def get_quality(self, adj_s, z_detach=None):
        if adj_s.shape[0] <= 1:
            return torch.tensor(0.0)
        q = self.methods['quity']
        if q == 'detach':
            return self.quity(adj_s, z_detach)
        elif q == 'homo':
            return self.quity_homo(adj_s, z_detach)
        elif q == 'edges':
            return self.quity_edges(adj_s)
        elif q == 'deg':
            return self.quity_degree(adj_s)
        return self.quity_homo(adj_s, z_detach)

    # ===== 相似度 =====
    def get_sim(self, node1, node2):
        emb1 = self.z_detached[node1]; emb2 = self.z_detached[node2]
        s = self.methods['sim']
        if s == 'cos':
            return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        if s == 'per':
            v1, v2 = emb1.cpu().numpy(), emb2.cpu().numpy()
            sim, _ = pearsonr(v1, v2); return torch.tensor(sim, dtype=torch.float32)
        return torch.dot(emb1, emb2)  # dot (default)

    # ===== 初始化 & 分裂 =====
    def init_GB(self, graph: nx.Graph):
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get)
        center_nodes = sorted_nodes[-self.init_num:]
        points = set(sorted_nodes[:-self.init_num])

        clusters = [[c] for c in center_nodes]
        neighbors_list = [set(graph.neighbors(c)) for c in center_nodes]
        located_point = set(); point_to_cluster = dict(); sim_dic = dict()

        while len(points) > 0:
            for i in range(self.init_num):
                for nb in neighbors_list[i]:
                    if nb in points:
                        if nb not in point_to_cluster:
                            point_to_cluster[nb] = i
                            sim_dic[nb] = self.get_sim(nb, center_nodes[i])
                        else:
                            old_sim = sim_dic[nb]
                            new_sim = self.get_sim(nb, center_nodes[i])
                            if old_sim < new_sim:
                                point_to_cluster[nb] = i; sim_dic[nb] = new_sim
            new_neighbors = [set() for _ in range(self.init_num)]
            for p in point_to_cluster:
                idx = point_to_cluster[p]
                clusters[idx].append(p)
                located_point.add(p)
                new_neighbors[idx].update(list(graph.neighbors(p)))
            neighbors_list = new_neighbors
            point_to_cluster.clear()
            points -= located_point; located_point.clear()

        init_subgraphs = [graph.subgraph(c) for c in clusters]
        cluster_Q = []
        for idx, cluster in enumerate(clusters):
            adj_s, z_s = self.get_sub_adj_z(init_subgraphs[idx], cluster)
            cluster_Q.append(self.get_quality(adj_s, z_s))
        return init_subgraphs, clusters, cluster_Q, center_nodes

    def split_bfs(self, graph, split_GB_list, split_graph_list, split_center_list, center_f, quality_f):
        node_num = graph.number_of_nodes()
        if node_num <= 3:
            split_GB_list.append(list(graph.nodes())); split_graph_list.append(graph)
            split_center_list.append(center_f); self.gb_q.append(quality_f); return

        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        center_nodes = [sorted_nodes[1], sorted_nodes[2]]

        points = set(sorted_nodes); points.remove(center_nodes[0]); points.remove(center_nodes[1])
        clusters = [[center_nodes[0]], [center_nodes[1]]]
        neighbors_list = [set(graph.neighbors(center_nodes[0])), set(graph.neighbors(center_nodes[1]))]
        common_neighbors = neighbors_list[0] & neighbors_list[1]

        while len(points) > 0:
            new_neighbors = [set(), set()]
            for nb in common_neighbors:
                if nb in points:
                    sim1 = self.get_sim(nb, center_nodes[0]); sim2 = self.get_sim(nb, center_nodes[1])
                    idx = 0 if sim1 > sim2 else 1
                    clusters[idx].append(nb); new_neighbors[idx].update(graph.neighbors(nb)); points.remove(nb)
            for i in range(2):
                for nb in neighbors_list[i]:
                    if nb in points:
                        clusters[i].append(nb); new_neighbors[i].update(graph.neighbors(nb)); points.remove(nb)
            neighbors_list = new_neighbors
            common_neighbors = neighbors_list[0] & neighbors_list[1]

        if len(clusters[0]) < 2 or len(clusters[1]) < 2:
            split_GB_list.append(list(graph.nodes())); split_graph_list.append(graph)
            split_center_list.append(center_f); self.gb_q.append(quality_f); return

        sub_a = graph.subgraph(clusters[0]); sub_b = graph.subgraph(clusters[1])
        if (not nx.is_connected(sub_a)) or (not nx.is_connected(sub_b)):
            split_GB_list.append(list(graph.nodes())); split_graph_list.append(graph)
            split_center_list.append(center_f); self.gb_q.append(quality_f); return

        adj_a, z_a = self.get_sub_adj_z(sub_a, clusters[0])
        adj_b, z_b = self.get_sub_adj_z(sub_b, clusters[1])
        qa = self.get_quality(adj_a, z_a); qb = self.get_quality(adj_b, z_b)

        if quality_f > (qa + qb) / 2.5:
            split_GB_list.append(list(graph.nodes())); split_graph_list.append(graph)
            split_center_list.append(center_f); self.gb_q.append(quality_f); return
        else:
            self.split_bfs(sub_a, split_GB_list, split_graph_list, split_center_list, center_nodes[0], qa)
            self.split_bfs(sub_b, split_GB_list, split_graph_list, split_center_list, center_nodes[1], qb)

    def get_GB_graph(self, graph):
        n = graph.number_of_nodes()
        self.init_num = max(2, int(math.isqrt(n)))   # 至少 2
        init_subgraphs, clusters, cluster_Q, init_center = self.init_GB(graph)

        GB_list, GB_graph_list, GB_center_list = [], [], []
        for i, subg in enumerate(init_subgraphs):
            split_GB_list, split_graph_list, split_center_list = [], [], []
            self.split_bfs(subg, split_GB_list, split_graph_list, split_center_list, init_center[i], cluster_Q[i])
            GB_list.extend(split_GB_list); GB_graph_list.extend(split_graph_list); GB_center_list.extend(split_center_list)
        return GB_list, GB_graph_list, GB_center_list

    def generate_GB(self, graph):
        GB_node_list, GB_graph_list, GB_center_list = [], [], []
        if nx.is_connected(graph):
            return self.get_GB_graph(graph)
        for comp in nx.connected_components(graph):
            subg = graph.subgraph(comp)
            if subg.number_of_nodes() <= 3:
                deg = dict(subg.degree()); max_node = max(deg, key=deg.get)
                GB_graph_list.append(subg); GB_node_list.append(list(subg.nodes())); GB_center_list.append(max_node)
            else:
                n, g, c = self.get_GB_graph(subg)
                GB_node_list.extend(n); GB_graph_list.extend(g); GB_center_list.extend(c)
        return GB_node_list, GB_graph_list, GB_center_list

    # ===== 主入口 =====
    def forward(self, adj_csr):
        graph = self.process_graph(adj_csr)
        self.total_edges = graph.number_of_edges()
        GB_node_list, GB_graph_list, GB_center_list = self.generate_GB(graph)
        return GB_node_list, GB_graph_list, GB_center_list