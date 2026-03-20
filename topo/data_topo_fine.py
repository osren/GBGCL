import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork, PPI
import torch
import random
import torch_geometric
import copy
import time
import gc
from tqdm import tqdm
from torch_geometric.utils import train_test_split_edges
import warnings
import argparse
import sys
import os
import numpy as np
import torch
import toponetx as tnx
from torch_geometric.utils import to_networkx, to_undirected, is_undirected
from topomodelx.utils.sparse import from_sparse
from data.utils import convert_graph_dataset_with_rings

def create_simple_complex_fallback(data, original_y, original_train_mask, original_val_mask, original_test_mask):
    """
    回退方案：创建简单的细胞复形结构
    """
    print("使用回退方案创建简单细胞复形...")
    
    # 确保所有数据都在CPU上
    x_0 = data.x.cpu() if data.x.is_cuda else data.x
    edge_index = data.edge_index.cpu() if data.edge_index.is_cuda else data.edge_index
    
    num_edges = edge_index.shape[1]
    x_1 = torch.ones(num_edges, 1)
    
    # 构建基本邻接矩阵
    num_nodes = x_0.shape[0]
    adjacency_0 = torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.shape[1]), 
        (num_nodes, num_nodes)
    )
    
    # 创建空的拉普拉斯矩阵
    down_laplacian = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros(0),
        (num_edges, num_edges)
    )
    
    up_laplacian = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros(0),
        (num_edges, num_edges)
    )
    
    adjacency_matrices = {
        'adjacency_0': adjacency_0,
        'down_laplacian': down_laplacian,
        'up_laplacian': up_laplacian
    }
    
    # 创建虚拟复形对象
    class MockComplex:
        def __init__(self):
            self.dimension = 1
            self.cochains = [MockCochain(x_0), MockCochain(x_1)]
    
    class MockCochain:
        def __init__(self, x):
            self.x = x
    
    mock_complex = MockComplex()
    
    return mock_complex, x_0, x_1, adjacency_matrices

def create_identity_laplacian(size, device='cpu'):
    """
    创建单位对角线拉普拉斯矩阵，避免计算复杂拉普拉斯矩阵
    """
    indices = torch.arange(size, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    values = torch.ones(size)
    return torch.sparse_coo_tensor(indices, values, (size, size))

def compute_simplified_laplacian(boundary_indices, num_rows, num_cols):
    """
    计算简化的拉普拉斯矩阵，避免内存问题
    为大型图使用更保守的策略
    """
    print(f"计算简化拉普拉斯矩阵: {num_rows}x{num_cols}, 关系数: {boundary_indices.shape[1]}")
    
    # 对于大型矩阵，直接返回身份矩阵或空矩阵
    if num_rows > 20000 or num_cols > 20000 or boundary_indices.shape[1] > 100000:
        print("矩阵过大，返回身份拉普拉斯矩阵")
        return create_identity_laplacian(num_rows)
    
    # 进一步限制关系数量
    max_relations = min(boundary_indices.shape[1], 10000)
    if boundary_indices.shape[1] > max_relations:
        print(f"关系数过多，采样到 {max_relations}")
        perm = torch.randperm(boundary_indices.shape[1])[:max_relations]
        boundary_indices = boundary_indices[:, perm]
    
    try:
        # 计算度矩阵而不是完整的拉普拉斯矩阵
        # 对每个节点计算其度数
        row_indices = boundary_indices[0]
        degrees = torch.bincount(row_indices, minlength=num_rows).float()
        
        # 创建对角度矩阵作为拉普拉斯矩阵的近似
        diag_indices = torch.arange(num_rows, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        laplacian = torch.sparse_coo_tensor(
            diag_indices,
            degrees,
            (num_rows, num_rows)
        ).coalesce()
        
        print(f"简化拉普拉斯矩阵计算完成: {laplacian.shape}")
        return laplacian
        
    except Exception as e:
        print(f"简化拉普拉斯矩阵计算失败: {e}")
        return create_identity_laplacian(num_rows)

def safe_sparse_tensor(indices, values, size, max_nnz=1000000):
    """
    安全创建稀疏张量，避免内存问题
    """
    try:
        # 检查非零元素数量
        if values.numel() > max_nnz:
            print(f"警告: 非零元素过多 ({values.numel()}), 进行采样到 {max_nnz}...")
            # 随机采样
            perm = torch.randperm(values.numel())[:max_nnz]
            indices = indices[:, perm]
            values = values[perm]
        
        return torch.sparse_coo_tensor(indices, values, size)
    except Exception as e:
        print(f"创建稀疏张量失败: {e}")
        # 返回空稀疏张量
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0),
            size
        )

def convert_graph_to_cell_complex(data, directed=False, max_dim=3,max_cycles=10000, max_cycle_size=5):
    """
    将PyTorch Geometric的图数据转换为细胞复形数据，确保三角形被正确转换为面
    """
    print("\n======= 开始细胞复形转换 (支持k维) =======")
    print(f"目标维度: {max_dim}, 最大环大小: {max_cycle_size}, 最大环数量: {max_cycles}")
    print(f"原始数据: 节点数={data.num_nodes}, 边数={data.edge_index.shape[1]}")
    
    
    # 内存清理
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 首先确保所有张量数据都在CPU上
    cpu_data = copy.deepcopy(data)
    print("将数据从GPU移至CPU...")
    # 转移所有可能的张量属性到CPU
    for attr in ['x', 'edge_index', 'edge_attr', 'y', 'train_mask', 'val_mask', 'test_mask']:
        if hasattr(cpu_data, attr):
            attr_value = getattr(cpu_data, attr)
            if attr_value is not None and isinstance(attr_value, torch.Tensor) and attr_value.is_cuda:
                setattr(cpu_data, attr, attr_value.cpu())
                
    print(f"已迁移至CPU")
    
    # 针对大图的优化策略
    num_edges = data.edge_index.shape[1]
    num_nodes = data.num_nodes
    
    # 根据图的大小动态调整参数
    if num_nodes > 30000 or num_edges > 400000:
        print("检测到大规模图，应用内存优化策略...")
        # 对于大图，限制环的大小和数量
        adjusted_max_cycle_size = min(max_cycle_size, 3)
        print(f"调整最大环大小为: {adjusted_max_cycle_size}")
        
        # 限制处理的边数，采样策略
        if num_edges > 200000:
            print("图过大，采用智能边采样策略...")
            # 计算目标边数
            target_edges = min(25000, num_edges // 20)  # 限制为原来的1/20或25000条边
            
            # 转换为无向边进行采样
            edge_index_np = cpu_data.edge_index.numpy()
            
            # 去重处理，只保留 i < j 的边
            edges = []
            seen_edges = set()
            for i in range(edge_index_np.shape[1]):
                u, v = edge_index_np[0, i], edge_index_np[1, i]
                if u != v:  # 排除自环
                    edge = tuple(sorted([u, v]))
                    if edge not in seen_edges:
                        edges.append(edge)
                        seen_edges.add(edge)
            
            print(f"原始边数: {num_edges}, 无向边数: {len(edges)}")
            
            # 采样边
            if len(edges) > target_edges // 2:
                sampled_edges = random.sample(edges, target_edges // 2)
            else:
                sampled_edges = edges
            
            # 重新构建边索引（双向）
            new_edges = []
            for u, v in sampled_edges:
                new_edges.extend([[u, v], [v, u]])
            
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
            print(f"目标无向边数: {len(sampled_edges)}, 目标总边数: {new_edge_index.shape[1]}")
            
            # 创建采样后的数据
            sampled_data = copy.deepcopy(cpu_data)
            sampled_data.edge_index = new_edge_index
            print(f"采样完成: {num_edges} -> {new_edge_index.shape[1]} (无向边: {len(sampled_edges)})")
        else:
            sampled_data = cpu_data
        
        max_cycle_size = adjusted_max_cycle_size
    else:
        sampled_data = cpu_data
     
    # 临时保存标签信息并移除
    original_y = sampled_data.y
    original_train_mask = getattr(sampled_data, 'train_mask', None)
    original_val_mask = getattr(sampled_data, 'val_mask', None)
    original_test_mask = getattr(sampled_data, 'test_mask', None)
    
    # 临时移除标签相关属性
    sampled_data.y = None
    if hasattr(sampled_data, 'train_mask'):delattr(sampled_data, 'train_mask')
    if hasattr(sampled_data, 'val_mask'):delattr(sampled_data, 'val_mask')
    if hasattr(sampled_data, 'test_mask'):delattr(sampled_data, 'test_mask')

    # 创建一个只包含当前数据的列表
    dataset_list = [sampled_data]
    print("\n开始构建环复形结构...")
    start_time = time.time()
    
    # 尝试修复graph-tool库的调用问题
    try:
        
        # **关键修复：正确调用 convert_graph_dataset_with_rings 函数**
        result = convert_graph_dataset_with_rings(
            dataset_list,
            max_ring_size=max_cycle_size,
            include_down_adj=True,
            init_method='sum',
            init_edges=True,
            init_rings=True,  # 重要：确保环被初始化为面
            n_jobs=1
        )
        
        # **关键修复：正确解包返回值**
        if isinstance(result, tuple) and len(result) == 3:
            ring_complexes, dimension, num_features = result
            print(f"函数返回: complexes={len(ring_complexes)}, dimension={dimension}, num_features={num_features}")
        elif isinstance(result, list):
            # 如果只返回复形列表
            ring_complexes = result
            print(f"函数返回复形列表: {len(ring_complexes)}")
        else:
            print(f"函数返回异常格式: {type(result)}")
            raise ValueError("convert_graph_dataset_with_rings 返回格式错误")
        
        end_time = time.time()
        print(f"环复形构建完成，耗时: {end_time - start_time:.2f}秒")
        
        # 获取第一个（也是唯一的）复形
        if len(ring_complexes) > 0:
            complex_data = ring_complexes[0]
            print(f"获取复形数据，类型: {type(complex_data)},维度: {complex_data.dimension}")
        else:
            print("环复形列表为空")
            raise ValueError("环复形构建失败：返回空列表")
        
        # 内存清理
        del dataset_list
        gc.collect()
        
    except Exception as e:
        end_time = time.time()
        print(f"环复形构建失败: {str(e)}，耗时: {end_time - start_time:.2f}秒")
        # import traceback
        # traceback.print_exc()
        # 如果环复形构建失败，回退到简单的图结构
        print("回退到简单图结构...")
        return create_simple_complex_fallback(cpu_data, original_y, original_train_mask, original_val_mask, original_test_mask)
    
    print("\n提取复形特征和结构...")
    # 提取各维度特征
    x_features = {}
    for dim in range(complex_data.dimension + 1):
        if dim < len(complex_data.cochains) and hasattr(complex_data.cochains[dim], 'x'):
            x_features[dim] = complex_data.cochains[dim].x
            print(f"维度 {dim} 特征形状: {x_features[dim].shape}")
        else:
            # 创建默认特征
            if dim == 0:
                num_nodes = cpu_data.num_nodes
                x_features[0] = torch.ones(num_nodes, 1)
            else:
                # 计算上一维度的细胞数量
                prev_cells = x_features.get(dim-1, None)
                num_cells = prev_cells.shape[0] if prev_cells is not None else 0
                x_features[dim] = torch.ones(num_cells, 1)
            print(f"维度 {dim} 无特征，创建默认特征: {x_features[dim].shape}")
    
    # 提取边界矩阵
    boundary_matrices = {}
    for dim in range(1, complex_data.dimension + 1):
        if hasattr(complex_data.cochains[dim], 'boundary_index'):
            boundary_matrices[dim] = complex_data.cochains[dim].boundary_index
            print(f"维度 {dim} 边界矩阵形状: {boundary_matrices[dim].shape}")
        else:
            print(f"维度 {dim} 无边界矩阵")
    
    # 提取邻接矩阵
    adjacency_matrices = {}
    for dim in range(complex_data.dimension + 1):
        # 0维使用原始图邻接
        if dim == 0:
            edge_index = complex_data.cochains[0].upper_index if hasattr(complex_data.cochains[0], 'upper_index') else cpu_data.edge_index
            num_nodes = x_features[0].shape[0]
            adjacency_0 = safe_sparse_tensor(
                edge_index, 
                torch.ones(edge_index.shape[1]), 
                (num_nodes, num_nodes)
            )
            adjacency_matrices[0] = adjacency_0
            print(f"维度 0 邻接矩阵: {adjacency_0.shape}")
        else:
            # 更高维度使用上邻接关系
            if dim < len(complex_data.cochains) and hasattr(complex_data.cochains[dim], 'upper_index'):
                upper_index = complex_data.cochains[dim].upper_index
                num_cells = x_features[dim].shape[0]
                adjacency = safe_sparse_tensor(
                    upper_index,
                    torch.ones(upper_index.shape[1]),
                    (num_cells, num_cells)
                )
                adjacency_matrices[dim] = adjacency
                print(f"维度 {dim} 邻接矩阵: {adjacency.shape}")
    
    # 恢复标签
    complex_data.y = original_y
    if original_train_mask is not None: complex_data.train_mask = original_train_mask
    if original_val_mask is not None: complex_data.val_mask = original_val_mask
    if original_test_mask is not None: complex_data.test_mask = original_test_mask
    
    return {
        'complex': complex_data,
        'x_features': x_features,
        'adjacency_matrices': adjacency_matrices,
        'boundary_matrices': boundary_matrices,
        'dimension': min(max_dim, complex_data.dimension)
    }
    # **检查复形数据结构**
    # try:
    #     # 检查复形对象是否有正确的属性
    #     if hasattr(complex_data, 'cochains'):
    #         print(f"复形有 {len(complex_data.cochains)} 个cochain")
            
    #         # 提取节点特征
    #         if len(complex_data.cochains) > 0 and hasattr(complex_data.cochains[0], 'x'):
    #             x_0 = complex_data.cochains[0].x  # 节点特征
    #             print(f"节点特征形状: {x_0.shape}")
    #         else:
    #             print("未找到节点特征，使用原始特征")
    #             x_0 = sampled_data.x
            
    #         # 处理边特征
    #         if len(complex_data.cochains) > 1 and hasattr(complex_data.cochains[1], 'x') and complex_data.cochains[1].x is not None:
    #             x_1 = complex_data.cochains[1].x  # 边特征
    #             print(f"使用复形提供的边特征，形状: {x_1.shape}")
    #         else:
    #             # 如果没有边特征，使用边的数量创建全1特征
    #             num_edges = sampled_data.edge_index.shape[1]
    #             x_1 = torch.ones(num_edges, 1)
    #             print(f"未找到边特征，创建全1特征，形状: {x_1.shape}")
        
    #     else:
    #         print("复形对象缺少cochains属性，使用回退方案")
    #         return create_simple_complex_fallback(cpu_data, original_y, original_train_mask, original_val_mask, original_test_mask)      
    # except Exception as e:
    #     print(f"提取复形特征失败: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     return create_simple_complex_fallback(cpu_data, original_y, original_train_mask, original_val_mask, original_test_mask)
    
    # # **检查面特征**
    # x_2 = None
    # if len(complex_data.cochains) > 2 and hasattr(complex_data.cochains[2], 'x') and complex_data.cochains[2].x is not None:
    #     x_2 = complex_data.cochains[2].x  # 面特征
    #     print(f"找到面特征，形状: {x_2.shape}")
    #     print(f"成功构建 {x_2.shape[0]} 个面!")
    # else:
    #     print("未找到面特征，检查是否有三角形被转换为面...")
        
    #     # 手动检查是否有环被转换为面
    #     if hasattr(complex_data, 'cochains') and len(complex_data.cochains) > 2:
    #         print(f"复形维度: {len(complex_data.cochains)}")
    #         for i, cochain in enumerate(complex_data.cochains):
    #             if hasattr(cochain, 'x') and cochain.x is not None:
    #                 print(f"  第{i}阶单纯形数量: {cochain.x.shape[0]}")
    #             else:
    #                 print(f"  第{i}阶单纯形: 无特征")
    
    # print("\n构建邻接结构...")
    
    # # **关键修复：确保维度一致性**
    # # 获取实际的边数量（从上邻接关系）
    # actual_edge_count = None
    # if (hasattr(complex_data, 'cochains') and len(complex_data.cochains) > 0 and 
    #     hasattr(complex_data.cochains[0], 'upper_index') and complex_data.cochains[0].upper_index is not None):
    #     # 从上邻接关系构建邻接矩阵
    #     edge_index = complex_data.cochains[0].upper_index
    #     actual_edge_count = edge_index.shape[1]
    #     print(f"从复形的上邻接关系构建0阶邻接矩阵，边数: {actual_edge_count}")
    #     num_nodes = x_0.shape[0]
    #     adjacency_0 = safe_sparse_tensor(
    #         edge_index, 
    #         torch.ones(edge_index.shape[1]), 
    #         (num_nodes, num_nodes)
    #     )
    # else:
    #     # 使用原始的edge_index
    #     print("使用原始边索引构建0阶邻接矩阵")
    #     actual_edge_count = sampled_data.edge_index.shape[1]
    #     adjacency_0 = safe_sparse_tensor(
    #         sampled_data.edge_index, 
    #         torch.ones(sampled_data.edge_index.shape[1]), 
    #         (x_0.shape[0], x_0.shape[0])
    #     )
    
    # # **关键修复：调整边特征以匹配邻接矩阵的维度**
    # if x_1.shape[0] != actual_edge_count:
    #     print(f"边特征维度不匹配: {x_1.shape[0]} vs {actual_edge_count}")
    #     if x_1.shape[0] < actual_edge_count:
    #         # 如果边特征较少，需要扩展
    #         print(f"扩展边特征从 {x_1.shape[0]} 到 {actual_edge_count}")
    #         # 复制最后一个特征或用零填充
    #         additional_features_needed = actual_edge_count - x_1.shape[0]
    #         if x_1.shape[0] > 0:
    #             # 重复最后一个特征
    #             last_feature = x_1[-1:].repeat(additional_features_needed, 1)
    #             x_1 = torch.cat([x_1, last_feature], dim=0)
    #         else:
    #             # 创建零特征
    #             x_1 = torch.zeros(actual_edge_count, x_1.shape[1])
    #     else:
    #         # 如果边特征较多，需要截断
    #         print(f"截断边特征从 {x_1.shape[0]} 到 {actual_edge_count}")
    #         x_1 = x_1[:actual_edge_count]
        
    #     print(f"调整后边特征形状: {x_1.shape}")
    
    # # **使用简化的拉普拉斯矩阵计算方法，避免内存问题**
    # print("\n构建简化拉普拉斯矩阵...")
    
    # # 下拉普拉斯矩阵（节点-边关系）
    # if (len(complex_data.cochains) > 1 and 
    #     hasattr(complex_data.cochains[1], 'lower_index') and complex_data.cochains[1].lower_index is not None):
    #     cochain_1 = complex_data.cochains[1]
    #     lower_index = cochain_1.lower_index
    #     print(f"构建下拉普拉斯矩阵，关系数: {lower_index.shape[1]}")
    #     num_edges = x_1.shape[0]
    #     num_nodes = x_0.shape[0]
        
    #     down_laplacian = compute_simplified_laplacian(lower_index, num_nodes, num_edges)
    # else:
    #     num_edges = x_1.shape[0]
    #     print("未找到下邻接关系，创建身份拉普拉斯矩阵")
    #     down_laplacian = create_identity_laplacian(num_edges)
    # # **上拉普拉斯矩阵（边-面关系）**
    # if (len(complex_data.cochains) > 2 and hasattr(complex_data.cochains[2], 'x') and complex_data.cochains[2].x is not None and
    #     hasattr(complex_data.cochains[1], 'upper_index') and complex_data.cochains[1].upper_index is not None):
    #     cochain_1 = complex_data.cochains[1]
    #     cochain_2 = complex_data.cochains[2]
        
    #     upper_index = cochain_1.upper_index
    #     print(f"构建上拉普拉斯矩阵，关系数: {upper_index.shape[1]}")
    #     num_edges = x_1.shape[0]
    #     num_faces = cochain_2.x.shape[0]
    #     print(f"检测到 {num_faces} 个面")
        
    #     if num_faces > 0:
    #         up_laplacian = compute_simplified_laplacian(upper_index, num_edges, num_faces)
    #     else:
    #         print("面数量为0，创建身份拉普拉斯矩阵")
    #         up_laplacian = create_identity_laplacian(num_edges)
    # else:
    #     num_edges = x_1.shape[0]
    #     print("检测到0个面，创建身份拉普拉斯矩阵")
    #     up_laplacian = create_identity_laplacian(num_edges)
    
    # # 收集所有邻接结构
    # adjacency_matrices = {
    #     'adjacency_0': adjacency_0,
    #     'down_laplacian': down_laplacian,
    #     'up_laplacian': up_laplacian
    # }
    
    # print("\n====== 细胞复形构建完成 ======")
    # print(f"复形维度: {complex_data.dimension if hasattr(complex_data, 'dimension') else 'unknown'}")
    # print(f"节点数量: {x_0.shape[0]}")
    # print(f"边数量: {x_1.shape[0]}")
    # print(f"实际邻接矩阵边数: {actual_edge_count}")
    # if x_2 is not None:
    #     print(f"面数量: {x_2.shape[0]}")
    # else:
    #     print("面数量: 0")
    # print("==============================\n")
    
    # # 恢复原始标签信息
    # if original_y is not None:
    #     complex_data.y = original_y
    # if original_train_mask is not None:
    #     complex_data.train_mask = original_train_mask
    # if original_val_mask is not None:
    #     complex_data.val_mask = original_val_mask
    # if original_test_mask is not None:
    #     complex_data.test_mask = original_test_mask
    
    # return complex_data, x_0, x_1, adjacency_matrices

# 其余函数保持不变...
def get_cell_complex_dataset(data, directed=False, max_dim=3,max_cycles=10000, max_cycle_size=5):
    """
    将数据集转换为细胞复形格式

    参数:
    data: PyTorch Geometric的Data对象
    directed: 是否为有向图
    max_cycles: 最大环数量
    max_cycle_size: 最大环大小

    返回:
    cell_complex_data: 包含细胞复形表示的字典
    """
    # cell_complex, x_0, x_1, adjacency_matrices = convert_graph_to_cell_complex(
    #     data, directed, max_cycles=max_cycles, max_cycle_size=max_cycle_size
    # )
    complex_dict = convert_graph_to_cell_complex(
        data, directed, max_dim=max_dim, max_cycles=max_cycles, max_cycle_size=max_cycle_size
    )
    # cell_complex_data = {
    #     'x_0': x_0,
    #     'x_1': x_1,
    #     'adjacency_0': adjacency_matrices['adjacency_0'],
    #     'down_laplacian': adjacency_matrices['down_laplacian'],
    #     'up_laplacian': adjacency_matrices['up_laplacian'],
    #     'y': data.y,
    #     'train_mask': data.train_mask if hasattr(data, 'train_mask') else None,
    #     'val_mask': data.val_mask if hasattr(data, 'val_mask') else None,
    #     'test_mask': data.test_mask if hasattr(data, 'test_mask') else None
    # }
    cell_complex_data = {
        **complex_dict,
        'y': data.y,
        'train_mask': data.train_mask if hasattr(data, 'train_mask') else None,
        'val_mask': data.val_mask if hasattr(data, 'val_mask') else None,
        'test_mask': data.test_mask if hasattr(data, 'test_mask') else None
    }
    return cell_complex_data

# # load_dataset等其他函数保持不变...
# def load_dataset(dataset_name, dataset_dir):
    print(f'Dataloader: 正在加载数据集 {dataset_name}')

    # 标准化数据集名称格式，处理大小写问题
    valid_datasets = ['Cora', 'CiteSeer', 'PubMed',
                      'dblp', 'Photo', 'Computers',
                      'CS', 'Physics',
                      'ogbn-products', 'ogbn-arxiv', 'Wiki', 'ppi',
                      'Cornell', 'Texas', 'Wisconsin',
                      'chameleon', 'crocodile', 'squirrel']

    # 创建名称映射字典（小写 -> 标准格式）
    name_map = {name.lower(): name for name in valid_datasets}

    # 检查和规范化数据集名称
    if dataset_name.lower() in name_map:
        dataset_name = name_map[dataset_name.lower()]
        print(f"数据集名称规范化为: {dataset_name}")
    else:
        raise ValueError(f"未知数据集: {dataset_name}. 支持的数据集: {', '.join(valid_datasets)}")

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            processed_dir = os.path.join(dataset_dir, f"planetoid-{dataset_name.lower()}", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                dataset = Planetoid(dataset_dir, name=dataset_name,
                                    transform=T.NormalizeFeatures(), force_reload=False)
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, f"planetoid-{dataset_name.lower()}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise
    elif dataset_name == 'CS':
        processed_dir = os.path.join(dataset_dir, "coauthor-cs", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = Coauthor(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures(), force_reload=False)
            return dataset
        else:
            print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
            raw_dir = os.path.join(dataset_dir, "coauthor-cs", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            print(f"请将原始数据文件放入: {raw_dir}")
            raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
    elif dataset_name == 'dblp':
        try:
            processed_dir = os.path.join(dataset_dir, f"citation-full-{dataset_name}", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                dataset = CitationFull(dataset_dir, name=dataset_name,
                                       transform=T.NormalizeFeatures(), force_reload=False)
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, f"citation-full-{dataset_name}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise

    elif dataset_name in ['Photo', 'Computers']:
        try:
            # 检查处理后的文件是否存在
            processed_dir = os.path.join(dataset_dir, f"amazon-{dataset_name.lower()}", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                dataset = Amazon(dataset_dir, name=dataset_name,
                                 transform=T.NormalizeFeatures(), force_reload=False)
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                # 如果您已有原始数据，可以手动放置到正确位置
                raw_dir = os.path.join(dataset_dir, f"amazon-{dataset_name.lower()}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise
    elif dataset_name in ['Physics']:
        try:
            processed_dir = os.path.join(dataset_dir, f"coauthor-{dataset_name.lower()}", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                # 强制只用本地数据，不联网
                dataset = Coauthor(dataset_dir, name=dataset_name,
                                transform=T.NormalizeFeatures(), force_reload=False)
                # 检查dataset是否为空
                if len(dataset) == 0:
                    raise FileNotFoundError("本地数据集为空，请检查 processed 目录内容")
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, f"coauthor-{dataset_name.lower()}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise
    # elif dataset_name in ['CS', 'Physics']:
    #     try:
    #         # 检查处理后的文件是否存在
    #         processed_dir = os.path.join(dataset_dir, f"coauthor-{dataset_name.lower()}", "processed")
    #         if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
    #             print(f"使用本地缓存的 {dataset_name} 数据集")
    #             dataset = Coauthor(dataset_dir, name=dataset_name,
    #                                transform=T.NormalizeFeatures(), force_reload=False)
    #         else:
    #             print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
    #             # 如果您已有原始数据，可以手动放置到正确位置
    #             raw_dir = os.path.join(dataset_dir, f"coauthor-{dataset_name.lower()}", "raw")
    #             os.makedirs(raw_dir, exist_ok=True)
    #             print(f"请将原始数据文件放入: {raw_dir}")
    #             raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
    #     except Exception as e:
    #         print(f"加载数据集时出错: {str(e)}")
    #         raise

    elif dataset_name in ['Wiki']:
        try:
            processed_dir = os.path.join(dataset_dir, "wikics", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                dataset = WikiCS(dataset_dir, transform=T.NormalizeFeatures(), force_reload=False)
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, "wikics", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise

    elif dataset_name in ['ppi']:
        try:
            processed_dir = os.path.join(dataset_dir, "ppi", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                train = PPI(root=dataset_dir, transform=T.NormalizeFeatures(), split='train', force_reload=False)
                val = PPI(root=dataset_dir, transform=T.NormalizeFeatures(), split='val', force_reload=False)
                test = PPI(root=dataset_dir, transform=T.NormalizeFeatures(), split='test', force_reload=False)
                dataset = [train, val, test]
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, "ppi", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise

    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        try:
            processed_dir = os.path.join(dataset_dir, f"webkb-{dataset_name.lower()}", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                dataset = WebKB(dataset_dir, dataset_name, transform=T.NormalizeFeatures(), force_reload=False)
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, f"webkb-{dataset_name.lower()}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise

    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
        try:
            processed_dir = os.path.join(dataset_dir, f"wikipedia-network-{dataset_name}", "processed")
            if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
                print(f"使用本地缓存的 {dataset_name} 数据集")
                dataset = WikipediaNetwork(dataset_dir, dataset_name, transform=T.NormalizeFeatures(),
                                           force_reload=False)
            else:
                print(f"无法找到处理好的数据集，请确保已预先下载 {dataset_name} 数据集")
                raw_dir = os.path.join(dataset_dir, f"wikipedia-network-{dataset_name}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                print(f"请将原始数据文件放入: {raw_dir}")
                raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise

    print('Dataloader: 加载成功')
    print(dataset[0])
    return dataset
def load_dataset(dataset_name, dataset_dir):
    print(f'Dataloader: 正在加载数据集 {dataset_name}')

    valid_datasets = ['Cora', 'CiteSeer', 'PubMed',
                      'dblp', 'Photo', 'Computers',
                      'CS', 'Physics',
                      'ogbn-products', 'ogbn-arxiv', 'Wiki', 'ppi',
                      'Cornell', 'Texas', 'Wisconsin',
                      'chameleon', 'crocodile', 'squirrel']

    name_map = {name.lower(): name for name in valid_datasets}

    if dataset_name.lower() in name_map:
        dataset_name = name_map[dataset_name.lower()]
        print(f"数据集名称规范化为: {dataset_name}")
    else:
        raise ValueError(f"未知数据集: {dataset_name}. 支持的数据集: {', '.join(valid_datasets)}")

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        processed_dir = os.path.join(dataset_dir, f"planetoid-{dataset_name.lower()}", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = Planetoid(dataset_dir, name=dataset_name,
                                transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name == 'CS':
        processed_dir = os.path.join(dataset_dir, "coauthor-cs", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = Coauthor(dataset_dir, name=dataset_name,
                               transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name == 'dblp':
        processed_dir = os.path.join(dataset_dir, f"citation-full-{dataset_name}", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = CitationFull(dataset_dir, name=dataset_name,
                                   transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name in ['Photo', 'Computers']:
        processed_dir = os.path.join(dataset_dir, f"amazon-{dataset_name.lower()}", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = Amazon(dataset_dir, name=dataset_name,
                             transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name in ['Physics']:
        processed_dir = os.path.join(dataset_dir, f"coauthor-{dataset_name.lower()}", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = Coauthor(dataset_dir, name=dataset_name,
                               transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name in ['Wiki']:
        processed_dir = os.path.join(dataset_dir, "wikics", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = WikiCS(dataset_dir, transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name in ['ppi']:
        processed_dir = os.path.join(dataset_dir, "ppi", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            train = PPI(root=dataset_dir, transform=T.NormalizeFeatures(), split='train', force_reload=False)
            val = PPI(root=dataset_dir, transform=T.NormalizeFeatures(), split='val', force_reload=False)
            test = PPI(root=dataset_dir, transform=T.NormalizeFeatures(), split='test', force_reload=False)
            dataset = [train, val, test]
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        processed_dir = os.path.join(dataset_dir, f"webkb-{dataset_name.lower()}", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = WebKB(dataset_dir, dataset_name, transform=T.NormalizeFeatures(), force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    elif dataset_name in ['chameleon', 'crocodile', 'squirrel']:
        processed_dir = os.path.join(dataset_dir, f"wikipedia-network-{dataset_name}", "processed")
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            print(f"使用本地缓存的 {dataset_name} 数据集")
            dataset = WikipediaNetwork(dataset_dir, dataset_name, transform=T.NormalizeFeatures(),
                                       force_reload=False)
        else:
            raise FileNotFoundError(f"本地未找到 {dataset_name} 数据集，请将数据放入 {processed_dir}")
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    print('Dataloader: 加载成功')
    print(dataset[0])
    return dataset
def get_wiki_cs(root, transform=T.NormalizeFeatures()):
    from torch_geometric.utils import to_undirected

    dataset = WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)