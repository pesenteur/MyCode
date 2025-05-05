from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
from parse_args import args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_Data_180():
    data_folder = args.data_path
    # 加载三通道 mobility pattern（7, 3, 180, 180）
    mob_pattern = np.load(data_folder+"/mob_patterns_3channel.npy")  # shape: (7, 3, 180, 180)
    pattern_tensor = torch.tensor(mob_pattern, dtype=torch.float)  # 转为 PyTorch 张量
    pattern_tensor = pattern_tensor.to(device)
    # 加载邻接矩阵（区域流动图）
    mob_adj = np.load(data_folder+"/actual_flow.npy")  # shape: (180, 180)
    mob_adj_tensor = torch.tensor(mob_adj, dtype=torch.float)
    mob_adj_tensor = mob_adj_tensor.to(device)
    # 加载路径数据
    road = np.load(data_folder+"/path_p.npy")  # shape: (180, 180)
    road_tensor = torch.tensor(road, dtype=torch.float)
    road_tensor =road_tensor.to(device)
    return pattern_tensor, mob_adj_tensor, road_tensor

def standardize_matrix(matrix):
    # 计算每列的均值和标准差
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    
    # 对每列进行标准化
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix

def load_path_data():
    data_path = args.data_path
    path_p = np.load(data_path+'/path_p.npy')
    path_p = standardize_matrix(path_p)
    return path_p.reshape(1, 180, 180)

def load_graph_data():
    data_path = args.data_path
    adj_matrix = np.load(data_path+'/adjacency.npy')
    path_p = np.load(data_path+'/path_p.npy')
    adj_matrix_sparse = sp.coo_matrix(adj_matrix*path_p)
    edge_index, edge_value = from_scipy_sparse_matrix(adj_matrix_sparse)
    return edge_index,edge_value

def random_neighbor_sampling(edge_index, edge_attr, max_neighbors=5):
    """
    随机从每个节点的邻居中选择一定数量的边。

    参数：
    - edge_index (Tensor): 边的索引，形状为 (2, num_edges)
    - edge_attr (Tensor): 边的权重，形状为 (num_edges,)
    - max_neighbors (int): 每个节点最大邻居数量

    返回：
    - sampled_edge_index (Tensor): 采样后的边索引，形状为 (2, num_sampled_edges)
    - sampled_edge_attr (Tensor): 采样后的边权重，形状为 (num_sampled_edges,)
    """
    sampled_edge_index = []
    sampled_edge_attr = []

    # 遍历每个节点，进行随机邻居采样
    for node in range(180):  # 假设 edge_index 的第一个维度为 2（source 和 target）
        # 获取与当前节点相关的所有邻居的边
        neighbors = edge_index[1][edge_index[0] == node]

        # 如果邻居数量超过上限，则进行随机采样
        if len(neighbors) > max_neighbors:
            # 使用边权重（edge_attr）计算采样的概率
            probabilities = torch.softmax(edge_attr[edge_index[0] == node], dim=0)  # 使用边权重作为概率
            
            sampled_neighbors = torch.multinomial(probabilities, max_neighbors, replacement=False)
            # 获取采样到的边和对应的权重，并将源节点索引与目标节点索引拼接
            sampled_edge_index.append(torch.tensor([[node] * max_neighbors, neighbors[sampled_neighbors]], dtype=torch.long))
            sampled_edge_attr.append(edge_attr[edge_index[0] == node][sampled_neighbors])
        else:
            # 如果邻居数不超过最大数量，直接加入所有邻居
            for neighbor in neighbors:
                sampled_edge_index.append(torch.tensor([[node], [neighbor]], dtype=torch.long))
            sampled_edge_attr.append(edge_attr[edge_index[0] == node])

    # 使用 torch.cat 拼接采样的结果
    sampled_edge_index = torch.cat(sampled_edge_index, dim=1)  # 保证边索引的形状是 (2, num_sampled_edges)
    sampled_edge_attr = torch.cat(sampled_edge_attr, dim=0)  # 保证边权重的形状是 (num_sampled_edges,)

    return sampled_edge_index