import math
from math import isqrt

import numpy as np
import torch
from graph_datasets import load_data  # 引入自定义函数`load_data`，用于加载图数据集
import networkx as nx
from scipy.stats import pearsonr


class Granular():
    def __init__(self,quity='homo',sim='dot'):
        # self.sim = None
        self.labels = None
        # self.preds = None
        self.adj_tensor = None
        self.center_cluster = None
        self.z_detached = None
        self.gb_q = []
        self.predsoft = None
        self.methods = dict()
        self.methods['quity'] = quity
        self.methods['sim'] = sim
        pass
    def process_graph(self,adj):
        """
        将 csr_martrix邻接矩阵转换成nx的Graph
        然后删除图中的自环
        """
        graph = nx.from_scipy_sparse_array(adj)
        # self_loops = list(nx.selfloop_edges(graph))
        # graph.remove_edges_from(self_loops)
        return graph

    # 获取子图的邻接矩阵
    def get_sub_adj_z(self, sub_g, cluster):
        node_indices = torch.tensor(cluster)
        # 提取邻接矩阵
        sub_adj = torch.tensor(nx.to_scipy_sparse_array(sub_g, format='csr').toarray())
        # 提取 cluster 中节点的相似度子矩阵
        # sub_sim = self.sim[node_indices][:, node_indices]
        # 提取 cluster 中节点的嵌入子张量
        sub_z = self.z_detached[node_indices]
        return sub_adj, sub_z
    # def quity(self, adj_s, sim_s):
    #
    #     num_nodes = adj_s.shape[0]  # 获取球中节点数
    #
    #     # 计算 Tr(BXX^T)，直接使用 sim_s 替代 XX^T
    #     x = torch.sum(sim_s * adj_s.to_dense().to(sim_s.device))  # Tr(B * XX^T)
    #
    #     # 计算度矩阵项
    #     degree_s = adj_s.sum(dim=1)  # 每个节点的度数之和
    #     y = torch.matmul(degree_s.T.double(), degree_s.double())
    #     y = y / (2 * num_nodes)
    #
    #     # 计算最终的qu
    #     scaling = num_nodes ** 2 / (num_nodes ** 2)
    #     quality = ((x - y) / (2 * num_nodes)) * scaling
    #     return quality
    def quity(self, adj_s, z_detach):
        # num_nodes = adj_s.shape[0]  # 获取球中节点数
        num_edges = torch.sum(adj_s) // 2  # 计算球中的边数

        # 使用嵌入矩阵计算 Tr(BXX^T)，代替 sim_s * adj_s 的直接计算
        x = torch.matmul(torch.t(z_detach).double(), adj_s.double().to(z_detach.device))
        x = torch.matmul(x, z_detach.double())
        x = torch.trace(x)  # Tr(BXX^T)

        # 计算度矩阵项
        degree_s = adj_s.sum(dim=1)  # 每个节点的度数之和
        y = torch.matmul(torch.t(z_detach).double(), degree_s.double().to(z_detach.device))
        y = (y ** 2).sum() / (2 * num_edges)
        # 计算最终的质量指标 q
        quality = ((x - y) / (2 * num_edges))
        return quality

    def quity_degree(self, adj_s):
        # 计算总边数：邻接矩阵中的非零元素之和的一半（因为无向图的边被计算了两次）
        total_edges = adj_s.sum() // 2

        # 计算节点总数
        num_nodes = adj_s.shape[0]

        # 计算边/点数比率
        edge_to_node_ratio = total_edges / num_nodes

        return edge_to_node_ratio

    def quity_homo(self, adj_s, z_detach):
        # 获取节点相似度矩阵：计算每两个节点之间的相似度
        sim_matrix = torch.mm(z_detach, z_detach.T)
        sim_matrix.fill_diagonal_(0)
        # 将邻接矩阵与相似度矩阵逐元素相乘，以仅保留图中实际存在的边的相似度
        weighted_sim_matrix = adj_s.to(sim_matrix.device) * sim_matrix

        # 计算总的边权重之和 (仅计算图中实际存在的边)
        total_similarity = weighted_sim_matrix.sum()
        # 计算子图的边数
        num_edges = adj_s.sum() / 2
        # 计算平均同质性（每条边的平均相似度）
        avg_homogeneity = total_similarity / num_edges
        if math.isnan(avg_homogeneity):
            print('质量是nan:',11111)
        return avg_homogeneity
    def quity_edges(self, adj_s):
        # 子图的边数
        sub_edges = adj_s.sum() // 2
        # 子图的度数和
        degree_sum = adj_s.sum(dim=1).sum().item()

        # 期望的边数
        m_exp = (degree_sum ** 2) / (4 * self.total_edges)
        # 计算Q_s
        Q_s = (sub_edges / self.total_edges) - (m_exp / self.total_edges)
        return Q_s

    def get_quity(self,adj_s,z_detach=None):
        if adj_s.shape[0] == 1:
            return torch.tensor(0)
        if self.methods['quity'] == 'detach':
            return self.quity(adj_s, z_detach)
        elif self.methods['quity'] == 'homo':
            return self.quity_homo(adj_s, z_detach)
        elif self.methods['quity'] == 'edges':
            return self.quity_edges(adj_s)
        elif self.methods['quity'] == 'deg':
            return self.quity_degree(adj_s)

    def get_sim(self, node1, node2):
        """
        获取两个节点的相似度，dot为点积
        :param node1:
        :param node2:
        :param sim:
        :return:
        """
        emb_node1 = self.z_detached[node1]
        emb_node2 = self.z_detached[node2]
        if self.methods['sim']=='dot':
            return self.dot_sim(emb_node1,emb_node2)
        elif self.methods['sim']=='cos':
            return self.cos_sim(emb_node1,emb_node2)
        elif self.methods['sim'] == 'per':
            return  self.per_sim(emb_node1,emb_node2)
        else:
            return self.dot_sim(emb_node1, emb_node2)
    def dot_sim(self, emb1,emb2):
        """
        计算点击相似度
        """
        return torch.dot(emb1,emb2)
    def cos_sim(self, emb1,emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)

    def per_sim(self, emb1, emb2):
        # 转换为 numpy 数组并计算皮尔逊相似度
        emb1_np = emb1.cpu().numpy()
        emb2_np = emb2.cpu().numpy()
        similarity, _ = pearsonr(emb1_np, emb2_np)  # 返回相似度值和 p 值
        return similarity

    def init_GB(self, graph:nx.Graph):
        # 图中边的总数
        # m = graph.number_of_edges()
        # 度排序，选度数最大的几个点
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get)
        center_nodes = sorted_nodes[-self.init_num:]
        # center_nodes.reverse()
        # 初始化剩余非中心节点
        points = set(sorted_nodes[0:-self.init_num])
        # 初始化簇和邻居列表
        clusters = []
        neighbors_list = []
        clusters_len = []
        for node in center_nodes:
            clusters.append([node])
            neighbors_list.append(list(graph.neighbors(node)))
            clusters_len.append(1)
        # 每轮被分配的节点集合
        located_point = set()
        # 每轮被配分节点的簇映射
        point_to_cluster = dict()
        sim_dic = dict()
        # 进行bfs分配
        while len(points) > 0:
            # 遍历中心点
            for i in range(self.init_num):
                # 遍历每个中心点的邻居
                for neighbor in neighbors_list[i]:
                    if neighbor in points:
                        # 该节点未被分配
                        if point_to_cluster.get(neighbor) is None:
                            # 将该节点分配给 第 i 个簇
                            point_to_cluster[neighbor] = i
                            # 存储两个节点的点积相似度
                            sim_dic[neighbor] = self.get_sim(neighbor, center_nodes[i])

                        # 该节点已被分配，说明该节点是两个簇的共同邻居
                        else:
                            # 获取当前邻居与旧簇的相似度
                            # old_sim = self.sim[center_nodes[point_to_cluster[neighbor]], neighbor]
                            old_sim = sim_dic[neighbor]
                            # 获得当前邻居与即将分配的簇的相似度
                            # new_sim = self.sim[center_nodes[i], neighbor]
                            new_sim = self.get_sim(neighbor, center_nodes[i])
                            # 与新簇更相似
                            # if old_sim.sum()/len(clusters[point_to_cluster[neighbor]]) < new_sim.sum()/len(clusters[i]):
                            #     point_to_cluster[neighbor] = i
                            if old_sim < new_sim:
                                point_to_cluster[neighbor] = i
                                sim_dic[neighbor] = new_sim
            new_neighbors = []
            # 初始化邻居
            for i in range(self.init_num):
                tset = set()
                new_neighbors.append(tset)
            # 本轮结束，正式分配
            for point in point_to_cluster:
                idx = point_to_cluster[point]
                clusters[idx].append(point)
                clusters_len[idx] += 1
                located_point.add(point)
                new_neighbors[idx].update(list(graph.neighbors(point)))
            neighbors_list.clear()
            neighbors_list = new_neighbors
            point_to_cluster.clear()
            points -= located_point
            located_point.clear()

        # for gb in init_GB_list:
        #     self.drawer.draw_GB(gb)
        cluster_Q = []

        init_GB_list = [graph.subgraph(cluster) for cluster in clusters]
        for idx,cluster in enumerate(clusters):
            sub_adj, sub_z = self.get_sub_adj_z(init_GB_list[idx], cluster)
            sub_q = self.get_quity(sub_adj, sub_z)
            cluster_Q.append(sub_q)
        return init_GB_list, clusters, cluster_Q, center_nodes

    # def init_GB_pred(self, graph:nx.Graph):
    #     # 度排序，选度数最大的几个点
    #     # degree_dict = dict(graph.degree())
    #     graph_nodes = np.array(graph.nodes())
    #     # 获取 graph 中节点对应的 preds_soft 值
    #     preds_soft_values = self.predsoft[graph_nodes]
    #     # 获取前 init_num 个最大值对应的索引
    #     top_indices = np.argpartition(preds_soft_values, -self.init_num)[-self.init_num:]
    #     # 通过这些索引得到对应的节点 ID
    #     center_nodes = graph_nodes[top_indices]
    #     center_nodes_set = set(center_nodes)
    #
    #     points = {node_id for node_id in graph_nodes if node_id not in center_nodes_set}
    #     # 初始化簇和邻居列表
    #     clusters = []
    #     neighbors_list = []
    #     for node in center_nodes:
    #         clusters.append([node])
    #         neighbors_list.append(list(graph.neighbors(node)))
    #     # 每轮被分配的节点集合
    #     located_point = set()
    #     # 每轮被配分节点的簇映射
    #     point_to_cluster = dict()
    #     sim_dic = dict()
    #     # 进行bfs分配
    #     while len(points) > 0:
    #         # 遍历中心点
    #         for i in range(self.init_num):
    #             # 遍历每个中心点的邻居
    #             for neighbor in neighbors_list[i]:
    #                 if neighbor in points:
    #                     # 该节点未被分配
    #                     if point_to_cluster.get(neighbor) is None:
    #                         # 将该节点分配给 第 i 个簇
    #                         point_to_cluster[neighbor] = i
    #                         # 存储两个节点的点积相似度
    #                         sim_dic[neighbor] = self.get_sim(neighbor, center_nodes[i])
    #                     # 该节点已被分配，说明该节点是两个簇的共同邻居
    #                     else:
    #                         # 获取当前邻居与旧簇的相似度
    #                         # old_sim = self.sim[center_nodes[point_to_cluster[neighbor]], neighbor]
    #                         old_sim = sim_dic[neighbor]
    #                         # 获得当前邻居与即将分配的簇的相似度
    #                         # new_sim = self.sim[center_nodes[i], neighbor]
    #                         new_sim = self.get_sim(neighbor, center_nodes[i])
    #                         # 与新簇更相似
    #                         # if old_sim.sum()/len(clusters[point_to_cluster[neighbor]]) < new_sim.sum()/len(clusters[i]):
    #                         #     point_to_cluster[neighbor] = i
    #                         if old_sim < new_sim:
    #                             point_to_cluster[neighbor] = i
    #                             sim_dic[neighbor] = new_sim
    #         new_neighbors = []
    #         # 初始化邻居
    #         for i in range(self.init_num):
    #             tset = set()
    #             new_neighbors.append(tset)
    #         # 本轮结束，正式分配
    #         for point in point_to_cluster:
    #             idx = point_to_cluster[point]
    #             clusters[idx].append(point)
    #             located_point.add(point)
    #             new_neighbors[idx].update(list(graph.neighbors(point)))
    #         neighbors_list.clear()
    #         neighbors_list = new_neighbors
    #         point_to_cluster.clear()
    #         points -= located_point
    #         located_point.clear()
    #     init_GB_list = [graph.subgraph(cluster) for cluster in clusters]
    #     # for gb in init_GB_list:
    #     #     self.drawer.draw_GB(gb)
    #     cluster_Q = []
    #
    #     for cluster in clusters:
    #
    #         sub_adj, sub_z = self.get_sub_adj_z(cluster)
    #         sub_q = self.get_quity(sub_adj, sub_z)
    #         cluster_Q.append(sub_q)
    #     return init_GB_list, clusters, cluster_Q, center_nodes

    def split_bfs(self,graph, split_GB_list, split_graph_list,split_center_list, center_f, quality_f):
        node_num = graph.number_of_nodes()
        # 如果是孤立点的时候
        if node_num <= 3:
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            split_center_list.append(center_f)
            self.gb_q.append(quality_f)
            return
        # 给度数排序
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        # 初始化中心点，选择度数第二大和第三大的节点
        center_nodes = [sorted_nodes[1], sorted_nodes[2]]
        # 初始化剩余节点，删除分配的中心点
        points = set(sorted_nodes)
        points.remove(center_nodes[0])
        points.remove(center_nodes[1])
        # 初始化簇和邻居列表
        clusters = [[center_nodes[0]], [center_nodes[1]]]
        neighbors_list = [set(graph.neighbors(center_nodes[0])), set(graph.neighbors(center_nodes[1]))]
        common_neighbors = neighbors_list[0] & neighbors_list[1]
        # bfs分配
        while len(points) > 0:
            new_neighbors = [set() , set()]
            # 先分配公共邻居
            for neighbor in common_neighbors:
                if neighbor in points:
                    points.remove(neighbor)
                    sim_1 = self.get_sim(neighbor,center_nodes[0])
                    sim_2 = self.get_sim(neighbor,center_nodes[1])
                    if sim_1 > sim_2:
                        to_idx = 0
                    else:
                        to_idx = 1
                    clusters[to_idx].append(neighbor)
                    new_neighbors[to_idx].update(graph.neighbors(neighbor))
            # 分配非公共邻居
            for i in range(2):
                for neighbor in neighbors_list[i]:
                    # 该节点未被分配
                    if neighbor in points:
                        points.remove(neighbor)
                        clusters[i].append(neighbor)
                        new_neighbors[i].update(graph.neighbors(neighbor))
            # 更新邻居
            neighbors_list.clear()
            neighbors_list = new_neighbors
            common_neighbors = neighbors_list[0] & neighbors_list[1]

        # 如果分出来了一个孤立点的情况
        if len(clusters[0]) < 2 or len(clusters[1]) < 2:
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            self.gb_q.append(quality_f)
            split_center_list.append(center_f)
            return
        # 如果分出来断边的情况
        # 提取子图
        subgraph_a = graph.subgraph(clusters[0])
        subgraph_b = graph.subgraph(clusters[1])
        if (not nx.is_connected(subgraph_a)) or (not nx.is_connected(subgraph_b)):
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            self.gb_q.append(quality_f)
            split_center_list.append(center_f)
            return
        # 提取邻接矩阵和相似度矩阵
        adj_a,z_a = self.get_sub_adj_z(subgraph_a, clusters[0])
        adj_b,z_b = self.get_sub_adj_z(subgraph_b, clusters[1])
        # 计算质量
        quality_a = self.get_quity(adj_a, z_a)
        quality_b = self.get_quity(adj_b, z_b)

        # 父球质量更好的情况下
        if quality_f > (quality_a + quality_b)/2.5:
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            self.gb_q.append(quality_f)
            split_center_list.append(center_f)
            return
        else:
            self.split_bfs(subgraph_a, split_GB_list, split_graph_list,split_center_list,center_nodes[0], quality_a)
            self.split_bfs(subgraph_b, split_GB_list, split_graph_list,split_center_list,center_nodes[1], quality_b)
    # 三分裂
    # def split_three(self,graph, split_GB_list, split_graph_list, quality_f):
    #     node_num = graph.number_of_nodes()
    #     # 如果是孤立点的时候
    #     if node_num <= 2:
    #         split_GB_list.append(list(graph.nodes()))
    #         split_graph_list.append(graph)
    #         self.gb_q.append(quality_f)
    #         return
    #     # 给度数排序
    #     degree_dict = dict(graph.degree())
    #     sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    #
    #     # 初始化中心点，选择度数第二大、第三大、和第一大的节点
    #     center_nodes = [sorted_nodes[1], sorted_nodes[2], sorted_nodes[0]]
    #     # 初始化剩余节点，删除分配的中心点
    #     points = set(sorted_nodes)
    #     points.remove(center_nodes[0])
    #     points.remove(center_nodes[1])
    #     points.remove(center_nodes[2])
    #     # 初始化簇和邻居列表
    #     clusters = [[center_nodes[0]], [center_nodes[1]], [center_nodes[2]]]
    #     neighbors_list = [set(graph.neighbors(center_nodes[0])), set(graph.neighbors(center_nodes[1])), set(graph.neighbors(center_nodes[2]))]
    #     common_neighbors = neighbors_list[0] & neighbors_list[1] & neighbors_list[2]
    #     # bfs分配
    #     while len(points) > 0:
    #         new_neighbors = [set(), set(), set()]
    #         # 先分配共同邻居
    #         for neighbor in common_neighbors:
    #             if neighbor in points:
    #                 sims = torch.tensor([self.get_sim(center_nodes[0], neighbor),
    #                                  self.get_sim(center_nodes[1], neighbor),
    #                                  self.get_sim(center_nodes[2], neighbor)])
    #                 # 获取最大索引
    #                 idx = torch.argmax(sims)
    #                 # 分配
    #                 clusters[idx].append(neighbor)
    #                 # 更新邻居
    #                 new_neighbors[idx].update(graph.neighbors(neighbor))
    #                 points.remove(neighbor)
    #         for i in range(3):
    #             for j in range(i+1,3):
    #                 t_common_neighbors = neighbors_list[i] & neighbors_list[j]
    #                 for neighbor in t_common_neighbors:
    #                     if neighbor in points:
    #                         sim0 = self.get_sim(center_nodes[i], neighbor)
    #                         sim1 = self.get_sim(center_nodes[j], neighbor)
    #                         if sim0>sim1:
    #                             idx = i
    #                         else:
    #                             idx = j
    #                         # 分配
    #                         clusters[idx].append(neighbor)
    #                         # 更新邻居
    #                         new_neighbors[idx].update(graph.neighbors(neighbor))
    #                         # 删除节点
    #                         points.remove(neighbor)
    #         # 非公共邻居
    #         for i in range(3):
    #             for neighbor in neighbors_list[i]:
    #                 if neighbor in points:
    #                     clusters[i].append(neighbor)
    #                     new_neighbors[i].update(graph.neighbors(neighbor))
    #                     points.remove(neighbor)
    #         # 更新邻居
    #         neighbors_list.clear()
    #         neighbors_list = new_neighbors
    #         common_neighbors = neighbors_list[0] & neighbors_list[1] & neighbors_list[2]
    #
    #
    #     # 提取子图
    #     subgraph_a = graph.subgraph(clusters[0])
    #     subgraph_b = graph.subgraph(clusters[1])
    #     subgraph_c = graph.subgraph(clusters[2])
    #     # 提取邻接矩阵和相似度矩阵
    #     adj_a, z_a = self.get_sub_adj_z(clusters[0])
    #     adj_b, z_b = self.get_sub_adj_z(clusters[1])
    #     adj_c, z_c = self.get_sub_adj_z(clusters[2])
    #     # 计算质量
    #     quality_a = self.get_quity(adj_a, z_a)
    #     quality_b = self.get_quity(adj_b, z_b)
    #     quality_c = self.get_quity(adj_c, z_c)
    #     # 父球质量更好的情况下
    #     if quality_f > (quality_a + quality_b + quality_c)/2 :
    #         split_GB_list.append(list(graph.nodes()))
    #         split_graph_list.append(graph)
    #         self.gb_q.append(quality_f)
    #         return
    #     else:
    #         self.split_three(subgraph_a, split_GB_list, split_graph_list, quality_a)
    #         self.split_three(subgraph_b, split_GB_list, split_graph_list, quality_b)
    #         self.split_three(subgraph_c, split_GB_list, split_graph_list, quality_c)


    @staticmethod
    def get_node_subgraph_edges(graph, node, subgraph):
        edge_count = 0
        for point in subgraph:
            if graph.has_edge(node, point):
                edge_count += 1

        return edge_count
    def get_GB_graph(self,graph):
        init_GB_num = math.isqrt(len(graph))
        self.init_num = int(init_GB_num)
        init_GB_list, clusters, cluster_Q, init_center = self.init_GB(graph)

        # 粒球节点簇列表
        GB_list = []
        # 粒球节点图列表
        GB_graph_list = []
        # 粒球中心节点列表
        GB_center_list = []
        for i, init_GB in enumerate(init_GB_list):
            split_GB_list = []
            split_graph_list = []
            split_center_list = []
            self.split_bfs(init_GB, split_GB_list, split_graph_list,split_center_list, init_center[i], cluster_Q[i])
            GB_list.extend(split_GB_list)
            GB_graph_list.extend(split_graph_list)
            GB_center_list.extend(split_center_list)
        return GB_list, GB_graph_list,GB_center_list
    def generate_GB(self, graph):
        GB_node_list = []
        GB_graph_list = []
        GB_center_list = []
        if nx.is_connected(graph):
            GB_node_list, GB_graph_list,GB_center_list = self.get_GB_graph(graph)
        else:
            connected_components = nx.connected_components(graph)
            for component in connected_components:
                subgraph = graph.subgraph(component)
                if len(subgraph) <= 3:
                    # 度排序，选度数最大的几个点
                    degree_dict = dict(subgraph.degree())
                    max_node = max(degree_dict, key=degree_dict.get)
                    GB_graph_list.append(subgraph)
                    GB_node_list.append(list(subgraph.nodes()))
                    GB_center_list.append(max_node)
                else:
                    node_list, graph_list,center_list = self.get_GB_graph(subgraph)
                    GB_node_list.extend(node_list)
                    GB_graph_list.extend(graph_list)
                    GB_center_list.extend(center_list)
        # 返回 粒球节点列表，粒球图列表
        return GB_node_list, GB_graph_list, GB_center_list
    def forward(self, adj_csr):
        graph = self.process_graph(adj_csr)
        self.total_edges = graph.number_of_nodes()
        GB_node_list, GB_graph_list, GB_center_list = self.generate_GB(graph)

        return GB_node_list, GB_graph_list,GB_center_list
    def forward_batch(self, adj_csr):
        graph = self.process_graph(adj_csr)
        GB_node_list, GB_graph_list = self.generate_GB(graph)
        return GB_node_list, GB_graph_list
if __name__ == '__main__':
    # 数据集来源
    source = {
        "Cora": "dgl",
        "Citeseer": "dgl",
        "ACM": "sdcn",
        "Pubmed": "dgl",
        "BlogCatalog": "cola",
        "Flickr": "cola",
        "Reddit": "dgl",
    }
    # 加载数据集，包括图结构、标签和聚类数
    graph, labels, n_clusters = load_data(
        directory='../data',
        dataset_name='Cora',
        source=source['Cora'],
        verbosity=2,
    )
    print(123134)
